"""
validation/playlist_validator.py — narrative-playlist-agent
Capa de validación matemática determinista.

Esta capa NO llama al LLM.
Todo lo que el LLM no puede garantizar con matemáticas, lo hace este módulo:
  - Que las transiciones entre tracks sean suaves (Δvalence/energy ≤ 0.3)
  - Que el arco tenga la pendiente correcta según la dirección solicitada
  - Que el score de coherencia sea calculable y reportable

Ver ARCHITECTURE.md sección 6: "¿Por qué la validación está fuera del agente?"
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats

from src.schemas.models import Track, ArcStatistics


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

# Máxima diferencia permitida entre tracks consecutivos
TRANSITION_DELTA_LIMIT = 0.3

# Pesos para el score de coherencia compuesto
SLOPE_WEIGHT = 0.6
TRANSITION_SMOOTHNESS_WEIGHT = 0.4

# Umbral mínimo de pendiente para considerar "positiva" o "negativa"
# Evita clasificar pendientes casi planas como direccionales
SLOPE_SIGNIFICANCE_THRESHOLD = 0.05


# ─────────────────────────────────────────────
# TIPOS DE DATOS
# ─────────────────────────────────────────────

ArcDirection = Literal["ascending", "descending", "arch", "valley", "flat"]

@dataclass
class TransitionViolation:
    """Una transición que supera el límite permitido."""
    position_from: int
    position_to: int
    track_from: str
    track_to: str
    delta_valence: float
    delta_energy: float
    exceeds_valence: bool
    exceeds_energy: bool

    def describe(self) -> str:
        parts = []
        if self.exceeds_valence:
            parts.append(f"Δvalence={self.delta_valence:.3f}")
        if self.exceeds_energy:
            parts.append(f"Δenergy={self.delta_energy:.3f}")
        return (
            f"Track {self.position_from}→{self.position_to} "
            f"({self.track_from} → {self.track_to}): {', '.join(parts)} supera {TRANSITION_DELTA_LIMIT}"
        )


@dataclass
class SlopeResult:
    """Resultado de la validación de pendiente para una dimensión."""
    dimension: str           # "valence" o "energy"
    slope: float             # pendiente de la regresión lineal
    r_squared: float         # bondad de ajuste (0.0–1.0)
    direction: Literal["positive", "negative", "neutral"]
    is_valid: bool           # ¿cumple la dirección requerida?
    values: list[float]      # valores originales (para debug)


@dataclass
class ValidationResult:
    """
    Resultado completo de la validación de una playlist.
    Retornado por validate_playlist() — el punto de entrada principal.
    """
    transitions_ok: bool
    slope_valence_ok: bool
    slope_energy_ok: bool
    coherence_score: float                          # 0.0–1.0
    violations: list[TransitionViolation] = field(default_factory=list)
    slope_valence: SlopeResult = None
    slope_energy: SlopeResult = None
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def is_fully_valid(self) -> bool:
        return self.transitions_ok and self.slope_valence_ok and self.slope_energy_ok

    def summary(self) -> str:
        lines = [
            f"Coherencia: {self.coherence_score:.0%}",
            f"Transiciones: {'✓' if self.transitions_ok else f'✗ ({len(self.violations)} violaciones)'}",
            f"Pendiente valence: {'✓' if self.slope_valence_ok else '✗'}",
            f"Pendiente energy:  {'✓' if self.slope_energy_ok else '✗'}",
        ]
        if self.warnings:
            lines.append(f"Advertencias: {len(self.warnings)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# FUNCIÓN 1: VALIDAR TRANSICIONES
# ─────────────────────────────────────────────

def validate_transitions(
    tracks: list[Track],
    delta_limit: float = TRANSITION_DELTA_LIMIT
) -> tuple[bool, list[TransitionViolation]]:
    """
    Verifica que el cambio entre tracks consecutivos no supere delta_limit
    en valence ni en energy.

    Args:
        tracks:      Lista de tracks en orden de posición
        delta_limit: Máximo cambio permitido (default: 0.3)

    Returns:
        (all_ok, violations_list)
        all_ok=True si no hay violaciones
    """
    if len(tracks) < 2:
        return True, []

    # Ordenar por posición para garantizar el orden correcto
    ordered = sorted(tracks, key=lambda t: t.position)
    violations = []

    for i in range(len(ordered) - 1):
        current = ordered[i]
        next_t = ordered[i + 1]

        # Redondear antes de comparar para evitar errores de precision float
        delta_v = round(abs(next_t.valence - current.valence), 4)
        delta_e = round(abs(next_t.energy - current.energy), 4)

        exceeds_v = delta_v > delta_limit
        exceeds_e = delta_e > delta_limit

        if exceeds_v or exceeds_e:
            violations.append(TransitionViolation(
                position_from=current.position,
                position_to=next_t.position,
                track_from=f"{current.track_name} — {current.artist}",
                track_to=f"{next_t.track_name} — {next_t.artist}",
                delta_valence=round(delta_v, 4),
                delta_energy=round(delta_e, 4),
                exceeds_valence=exceeds_v,
                exceeds_energy=exceeds_e
            ))

    return len(violations) == 0, violations


# ─────────────────────────────────────────────
# FUNCIÓN 2: VALIDAR PENDIENTE DEL ARCO
# ─────────────────────────────────────────────

def _classify_slope(slope: float, threshold: float = SLOPE_SIGNIFICANCE_THRESHOLD) -> str:
    """Clasifica una pendiente como positiva, negativa o neutral."""
    if slope > threshold:
        return "positive"
    elif slope < -threshold:
        return "negative"
    else:
        return "neutral"


def _validate_single_dimension(
    values: list[float],
    dimension: str,
    required_direction: ArcDirection
) -> SlopeResult:
    """
    Calcula la regresión lineal para una dimensión y valida
    que la pendiente sea coherente con la dirección del arco.
    """
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)

    result = stats.linregress(x, y)
    slope = float(result.slope)
    r_squared = float(result.rvalue ** 2)
    direction = _classify_slope(slope)

    # Determinar si la pendiente es válida según la dirección del arco
    if required_direction == "ascending":
        is_valid = direction == "positive"
    elif required_direction == "descending":
        is_valid = direction == "negative"
    elif required_direction == "flat":
        is_valid = direction == "neutral"
    else:
        # "arch" y "valley" se validan por segmentos — aquí siempre pass
        # la validación real de arch/valley está en validate_arc_slope
        is_valid = True

    return SlopeResult(
        dimension=dimension,
        slope=round(slope, 5),
        r_squared=round(r_squared, 4),
        direction=direction,
        is_valid=is_valid,
        values=values
    )


def validate_arc_slope(
    tracks: list[Track],
    direction: ArcDirection = "ascending"
) -> tuple[bool, SlopeResult, SlopeResult]:
    """
    Valida que el arco tenga la pendiente correcta en valence Y energy.

    Para arcos simples (ascending/descending/flat):
      Usa regresión lineal sobre toda la playlist.

    Para arcos compuestos (arch: sube luego baja / valley: baja luego sube):
      Divide la playlist en dos mitades y valida cada segmento por separado.

    Args:
        tracks:    Lista de tracks ordenados por posición
        direction: "ascending" | "descending" | "arch" | "valley" | "flat"

    Returns:
        (both_valid, slope_result_valence, slope_result_energy)
    """
    ordered = sorted(tracks, key=lambda t: t.position)
    valence_values = [t.valence for t in ordered]
    energy_values = [t.energy for t in ordered]

    if direction in ("ascending", "descending", "flat"):
        v_result = _validate_single_dimension(valence_values, "valence", direction)
        e_result = _validate_single_dimension(energy_values, "energy", direction)
        both_valid = v_result.is_valid and e_result.is_valid

    elif direction == "arch":
        # Primera mitad debe ser ascending, segunda mitad descending
        mid = len(ordered) // 2
        v1 = _validate_single_dimension(valence_values[:mid + 1], "valence", "ascending")
        v2 = _validate_single_dimension(valence_values[mid:], "valence", "descending")
        e1 = _validate_single_dimension(energy_values[:mid + 1], "energy", "ascending")
        e2 = _validate_single_dimension(energy_values[mid:], "energy", "descending")

        arch_valid = v1.is_valid and v2.is_valid and e1.is_valid and e2.is_valid
        # Para el return, usamos los results del segmento que cubre más tracks
        v_result = v1 if len(valence_values[:mid + 1]) >= len(valence_values[mid:]) else v2
        e_result = e1 if len(energy_values[:mid + 1]) >= len(energy_values[mid:]) else e2
        v_result.is_valid = arch_valid
        e_result.is_valid = arch_valid
        both_valid = arch_valid

    elif direction == "valley":
        # Primera mitad descending, segunda mitad ascending
        mid = len(ordered) // 2
        v1 = _validate_single_dimension(valence_values[:mid + 1], "valence", "descending")
        v2 = _validate_single_dimension(valence_values[mid:], "valence", "ascending")
        e1 = _validate_single_dimension(energy_values[:mid + 1], "energy", "descending")
        e2 = _validate_single_dimension(energy_values[mid:], "energy", "ascending")

        valley_valid = v1.is_valid and v2.is_valid and e1.is_valid and e2.is_valid
        v_result = v1
        e_result = e1
        v_result.is_valid = valley_valid
        e_result.is_valid = valley_valid
        both_valid = valley_valid

    else:
        raise ValueError(f"Dirección de arco no válida: {direction}. "
                         f"Usa: ascending | descending | arch | valley | flat")

    return both_valid, v_result, e_result


# ─────────────────────────────────────────────
# FUNCIÓN 3: SCORE DE COHERENCIA
# ─────────────────────────────────────────────

def calculate_coherence_score(
    tracks: list[Track],
    direction: ArcDirection = "ascending"
) -> float:
    """
    Score compuesto de coherencia del arco emocional.

    Fórmula:
        score = (slope_score * 0.6) + (transition_score * 0.4)

    slope_score:
        1.0 si ambas pendientes (valence y energy) son correctas
        0.5 si solo una es correcta
        0.0 si ninguna es correcta

    transition_score:
        1.0 - (violaciones / total_transiciones)
        Ej: 1 violación en 8 transiciones = 1 - (1/8) = 0.875

    Returns:
        float entre 0.0 y 1.0
    """
    if not tracks:
        return 0.0

    ordered = sorted(tracks, key=lambda t: t.position)

    # Componente 1: slope
    _, v_result, e_result = validate_arc_slope(ordered, direction)
    slope_components = [v_result.is_valid, e_result.is_valid]
    slope_score = sum(slope_components) / len(slope_components)

    # Componente 2: smoothness de transiciones
    total_transitions = max(len(ordered) - 1, 1)
    _, violations = validate_transitions(ordered)
    transition_score = 1.0 - (len(violations) / total_transitions)
    transition_score = max(0.0, transition_score)  # nunca negativo

    # Score compuesto
    score = (slope_score * SLOPE_WEIGHT) + (transition_score * TRANSITION_SMOOTHNESS_WEIGHT)
    return round(score, 4)


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────

def validate_playlist(
    tracks: list[Track],
    direction: ArcDirection = "ascending"
) -> ValidationResult:
    """
    Validación completa de la playlist.
    Llama a las 3 funciones anteriores y consolida el resultado.

    Llamado por agent.py después de recibir el output del LLM,
    antes de entregarlo al usuario.

    Args:
        tracks:    Lista de todos los tracks de la playlist (todas las fases)
        direction: Dirección del arco emocional

    Returns:
        ValidationResult con toda la información de validación
    """
    ordered = sorted(tracks, key=lambda t: t.position)

    # 1. Validar transiciones
    transitions_ok, violations = validate_transitions(ordered)

    # 2. Validar pendiente
    slope_ok, v_result, e_result = validate_arc_slope(ordered, direction)

    # 3. Calcular coherence score
    score = calculate_coherence_score(ordered, direction)

    # 4. Construir ArcStatistics
    valence_vals = [t.valence for t in ordered]
    energy_vals = [t.energy for t in ordered]

    arc_stats = ArcStatistics(
        valence_start=round(valence_vals[0], 4),
        valence_end=round(valence_vals[-1], 4),
        energy_start=round(energy_vals[0], 4),
        energy_end=round(energy_vals[-1], 4),
        valence_slope=v_result.direction,
        energy_slope=e_result.direction,
        arc_coherence_score=score,
        total_transitions_checked=len(ordered) - 1,
        transitions_violated=len(violations)
    )

    # 5. Generar warnings y recomendaciones
    warnings = []
    recommendations = []

    if violations:
        for v in violations:
            warnings.append(v.describe())
        recommendations.append(
            f"Reemplaza los tracks en posiciones: "
            f"{[v.position_to for v in violations]} con opciones de transición más suave."
        )

    if not v_result.is_valid:
        recommendations.append(
            f"La pendiente de valence es '{v_result.direction}' pero se esperaba "
            f"compatible con arco '{direction}'. "
            f"Revisa la distribución emocional de las fases."
        )

    if not e_result.is_valid:
        recommendations.append(
            f"La pendiente de energy es '{e_result.direction}' pero se esperaba "
            f"compatible con arco '{direction}'."
        )

    if score < 0.6:
        warnings.append(
            f"Score de coherencia bajo ({score:.0%}). "
            f"El arco emocional puede sentirse inconsistente."
        )

    return ValidationResult(
        transitions_ok=transitions_ok,
        slope_valence_ok=v_result.is_valid,
        slope_energy_ok=e_result.is_valid,
        coherence_score=score,
        violations=violations,
        slope_valence=v_result,
        slope_energy=e_result,
        warnings=warnings,
        recommendations=recommendations
    )
