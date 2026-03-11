"""
schemas/models.py — narrative-playlist-agent
Contratos de datos del sistema. Pydantic v2.
Todo output del agente pasa por estos modelos antes de llegar al usuario.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional


# ─────────────────────────────────────────────
# CAPA 1: Entidades base
# ─────────────────────────────────────────────

class Track(BaseModel):
    """
    Una canción individual dentro de una fase de la playlist.
    Todos los campos de audio features son obligatorios —
    si el LLM omite alguno, Pydantic rechaza el output completo.
    """
    position: int = Field(ge=1, description="Posición global en la playlist (1-based)")
    track_name: str = Field(min_length=1)
    artist: str = Field(min_length=1)
    valence: float = Field(ge=0.0, le=1.0, description="Positividad musical 0.0–1.0")
    energy: float = Field(ge=0.0, le=1.0, description="Intensidad 0.0–1.0")
    tempo: float = Field(gt=0.0, description="BPM")
    genre: str = Field(default="unknown")
    transition_note: str = Field(
        min_length=10,
        description="Por qué esta canción está en esta posición. Mínimo 10 chars."
    )

    model_config = {"extra": "ignore"}


class TransitionViolation(BaseModel):
    """
    Registra una violación de la constraint de transición suave.
    Generado por playlist_validator.py — no por el LLM.
    """
    position_from: int
    position_to: int
    track_from: str
    track_to: str
    delta_valence: float
    delta_energy: float
    exceeds_valence_limit: bool
    exceeds_energy_limit: bool


# ─────────────────────────────────────────────
# CAPA 2: Estructura de fases
# ─────────────────────────────────────────────

class Phase(BaseModel):
    """
    Una fase emocional dentro del arco narrativo.
    Contiene N tracks y su justificación paramétrica.
    """
    phase_number: int = Field(ge=1)
    phase_label: str = Field(
        min_length=3,
        description="Etiqueta de la emoción. Ej: 'Melancolía profunda'"
    )
    emotional_description: str = Field(
        min_length=20,
        description="Descripción del estado emocional de la fase"
    )
    valence_range: tuple[float, float] = Field(
        description="(min_valence, max_valence) usados en la consulta"
    )
    energy_range: tuple[float, float] = Field(
        description="(min_energy, max_energy) usados en la consulta"
    )
    tracks: list[Track] = Field(min_length=1)
    phase_explanation: str = Field(
        min_length=50,
        description="Justificación paramétrica OBLIGATORIA. "
                    "Explica cómo los rangos de valence/energy producen la emoción deseada."
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> "Phase":
        """Verifica que los rangos sean coherentes (min <= max)."""
        v_min, v_max = self.valence_range
        e_min, e_max = self.energy_range
        if v_min > v_max:
            raise ValueError(
                f"valence_range inválido: min ({v_min}) > max ({v_max})"
            )
        if e_min > e_max:
            raise ValueError(
                f"energy_range inválido: min ({e_min}) > max ({e_max})"
            )
        return self

    @model_validator(mode="after")
    def validate_track_positions(self) -> "Phase":
        """Verifica que los tracks tengan posiciones únicas."""
        positions = [t.position for t in self.tracks]
        if len(positions) != len(set(positions)):
            raise ValueError("Tracks con posiciones duplicadas dentro de la fase")
        return self


# ─────────────────────────────────────────────
# CAPA 3: Estadísticas del arco
# ─────────────────────────────────────────────

class ArcStatistics(BaseModel):
    """
    Estadísticas matemáticas del arco emocional completo.
    Calculadas por playlist_validator.py — no por el LLM.
    """
    valence_start: float = Field(ge=0.0, le=1.0)
    valence_end: float = Field(ge=0.0, le=1.0)
    energy_start: float = Field(ge=0.0, le=1.0)
    energy_end: float = Field(ge=0.0, le=1.0)
    valence_slope: Literal["positive", "negative", "neutral"]
    energy_slope: Literal["positive", "negative", "neutral"]
    arc_coherence_score: float = Field(
        ge=0.0, le=1.0,
        description="Score compuesto: 60% slope + 40% transition smoothness"
    )
    total_transitions_checked: int = Field(ge=0)
    transitions_violated: int = Field(ge=0)


# ─────────────────────────────────────────────
# CAPA 4: Output completo del agente
# ─────────────────────────────────────────────

class PlaylistOutput(BaseModel):
    """
    Output final del agente. Schema completo que recibe el frontend.
    Si cualquier campo obligatorio falta o es inválido,
    Pydantic lanza ValidationError antes de que el output llegue al usuario.
    """
    playlist_title: str = Field(min_length=3)
    user_prompt: str = Field(min_length=5)
    narrative_arc: str = Field(
        min_length=20,
        description="Descripción del arco en 1-2 oraciones"
    )
    total_tracks: int = Field(ge=1, le=20)
    phases: list[Phase] = Field(min_length=1)
    arc_statistics: Optional[ArcStatistics] = Field(
        default=None,
        description="Calculado por validator.py después de recibir el output del LLM. "
                    "None si la validación aún no se ejecutó."
    )
    arc_summary: str = Field(
        min_length=50,
        description="Párrafo final OBLIGATORIO: el LLM explica el arco completo."
    )
    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Advertencias del validator (no errores fatales)"
    )

    @model_validator(mode="after")
    def validate_total_tracks_count(self) -> "PlaylistOutput":
        """Verifica que total_tracks coincida con la suma real de tracks por fase."""
        real_count = sum(len(phase.tracks) for phase in self.phases)
        if real_count != self.total_tracks:
            raise ValueError(
                f"total_tracks ({self.total_tracks}) no coincide con "
                f"la suma real de tracks en fases ({real_count})"
            )
        return self

    @model_validator(mode="after")
    def validate_no_duplicate_tracks(self) -> "PlaylistOutput":
        """Verifica que no haya canciones duplicadas entre fases."""
        all_tracks = [
            f"{t.track_name.lower()}::{t.artist.lower()}"
            for phase in self.phases
            for t in phase.tracks
        ]
        if len(all_tracks) != len(set(all_tracks)):
            # Identifica los duplicados específicos
            seen = set()
            duplicates = []
            for name in all_tracks:
                if name in seen:
                    duplicates.append(name)
                seen.add(name)
            raise ValueError(
                f"Tracks duplicados detectados entre fases: {duplicates}"
            )
        return self

    def all_tracks_flat(self) -> list[Track]:
        """Retorna todos los tracks en orden de posición — útil para el validator."""
        tracks = [t for phase in self.phases for t in phase.tracks]
        return sorted(tracks, key=lambda t: t.position)

    def summary_for_cli(self) -> str:
        """Resumen compacto para imprimir en terminal."""
        lines = [
            f"\n🎵 {self.playlist_title}",
            f"📖 {self.narrative_arc}",
            f"🎧 {self.total_tracks} canciones en {len(self.phases)} fases\n",
        ]
        for phase in self.phases:
            lines.append(f"  ── {phase.phase_label} ({len(phase.tracks)} tracks)")
            for track in sorted(phase.tracks, key=lambda t: t.position):
                lines.append(
                    f"     {track.position:02d}. {track.track_name} — {track.artist} "
                    f"(V:{track.valence:.2f} E:{track.energy:.2f} T:{track.tempo:.0f}bpm)"
                )
        if self.arc_statistics:
            s = self.arc_statistics
            lines.append(f"\n📊 Arco: valence {s.valence_start:.2f}→{s.valence_end:.2f} "
                         f"| energy {s.energy_start:.2f}→{s.energy_end:.2f} "
                         f"| coherencia {s.arc_coherence_score:.0%}")
        lines.append(f"\n💬 {self.arc_summary}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# CAPA 5: Models internos del agente
# ─────────────────────────────────────────────

class QueryArgs(BaseModel):
    """
    Argumentos de query_song_database validados por Pydantic
    antes de construir el SQL. Previene rangos inválidos del LLM.
    """
    min_valence: float = Field(ge=0.0, le=1.0)
    max_valence: float = Field(ge=0.0, le=1.0)
    min_energy: float = Field(ge=0.0, le=1.0)
    max_energy: float = Field(ge=0.0, le=1.0)
    min_tempo: Optional[float] = Field(default=None, gt=0.0)
    max_tempo: Optional[float] = Field(default=None, gt=0.0)
    min_acousticness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    target_genres: Optional[list[str]] = Field(default=None)
    exclude_track_ids: Optional[list[str]] = Field(default=None)
    limit: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def validate_min_max(self) -> "QueryArgs":
        if self.min_valence > self.max_valence:
            raise ValueError(f"min_valence ({self.min_valence}) > max_valence ({self.max_valence})")
        if self.min_energy > self.max_energy:
            raise ValueError(f"min_energy ({self.min_energy}) > max_energy ({self.max_energy})")
        if self.min_tempo and self.max_tempo and self.min_tempo > self.max_tempo:
            raise ValueError(f"min_tempo ({self.min_tempo}) > max_tempo ({self.max_tempo})")
        return self


class QueryResult(BaseModel):
    """Resultado de una ejecución de query_song_database."""
    success: bool
    tracks: list[dict]
    attempts: int = Field(default=1, description="Número de intentos (1-3, por retry logic)")
    final_args: Optional[QueryArgs] = Field(
        default=None,
        description="Args finales usados (pueden diferir de los originales por retry)"
    )
    error: Optional[str] = None
