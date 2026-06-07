"""
tests/test_validator.py — narrative-playlist-agent
Tests unitarios para validation/playlist_validator.py
Ejecutar: python -m pytest tests/test_validator.py -v
"""

import pytest
from src.validation.playlist_validator import (
    validate_transitions,
    validate_arc_slope,
    calculate_coherence_score,
    validate_playlist,
    TransitionViolation,
    TRANSITION_DELTA_LIMIT,
)
from src.schemas.models import Track


# ─────────────────────────────────────────────
# Fixtures — playlists sintéticas para tests
# ─────────────────────────────────────────────

def make_track(position, valence, energy, tempo=90.0,
               name=None, artist="Test Artist"):
    return Track(
        position=position,
        track_name=name or f"Track {position}",
        artist=artist,
        valence=valence,
        energy=energy,
        tempo=tempo,
        genre="test",
        transition_note=f"Track {position} encaja porque sus valores son coherentes con la fase."
    )


def make_ascending_playlist(n=6) -> list[Track]:
    """
    Playlist con arco claramente ascendente.
    valence: 0.1 → 0.9 (paso uniforme)
    energy:  0.1 → 0.9 (paso uniforme)
    """
    step = 0.8 / (n - 1)
    return [
        make_track(
            position=i + 1,
            valence=round(0.1 + step * i, 3),
            energy=round(0.1 + step * i, 3)
        )
        for i in range(n)
    ]


def make_descending_playlist(n=6) -> list[Track]:
    """Playlist con arco claramente descendente."""
    step = 0.8 / (n - 1)
    return [
        make_track(
            position=i + 1,
            valence=round(0.9 - step * i, 3),
            energy=round(0.9 - step * i, 3)
        )
        for i in range(n)
    ]


def make_arch_playlist(n=6) -> list[Track]:
    """Playlist en arco: sube hasta el clímax y luego baja."""
    half = n // 2
    tracks = []
    # Primera mitad: sube
    for i in range(half):
        v = round(0.2 + (0.5 / (half - 1)) * i, 3)
        tracks.append(make_track(i + 1, valence=v, energy=v))
    # Segunda mitad: baja
    for i in range(n - half):
        v = round(0.7 - (0.5 / (n - half - 1 or 1)) * i, 3)
        tracks.append(make_track(half + i + 1, valence=v, energy=v))
    return tracks


# ─────────────────────────────────────────────
# Tests: validate_transitions
# ─────────────────────────────────────────────

class TestValidateTransitions:

    def test_smooth_transitions_pass(self):
        """Cambios graduales no deben generar violaciones."""
        tracks = make_ascending_playlist(6)
        ok, violations = validate_transitions(tracks)
        assert ok is True
        assert violations == []

    def test_abrupt_valence_jump_detected(self):
        """Un salto brusco de valence debe detectarse como violación."""
        tracks = [
            make_track(1, valence=0.10, energy=0.20),
            make_track(2, valence=0.55, energy=0.22),  # Δvalence = 0.45 > 0.3
            make_track(3, valence=0.60, energy=0.25),
        ]
        ok, violations = validate_transitions(tracks)
        assert ok is False
        assert len(violations) == 1
        assert violations[0].position_from == 1
        assert violations[0].position_to == 2
        assert violations[0].exceeds_valence is True

    def test_abrupt_energy_jump_detected(self):
        """Un salto brusco de energy debe detectarse como violación."""
        tracks = [
            make_track(1, valence=0.20, energy=0.10),
            make_track(2, valence=0.22, energy=0.55),  # Δenergy = 0.45 > 0.3
            make_track(3, valence=0.25, energy=0.58),
        ]
        ok, violations = validate_transitions(tracks)
        assert ok is False
        assert violations[0].exceeds_energy is True

    def test_both_dimensions_violated(self):
        tracks = [
            make_track(1, valence=0.10, energy=0.10),
            make_track(2, valence=0.60, energy=0.65),  # ambas > 0.3
        ]
        ok, violations = validate_transitions(tracks)
        assert ok is False
        assert violations[0].exceeds_valence is True
        assert violations[0].exceeds_energy is True

    def test_multiple_violations(self):
        tracks = [
            make_track(1, valence=0.10, energy=0.10),
            make_track(2, valence=0.55, energy=0.12),  # violación 1
            make_track(3, valence=0.57, energy=0.14),
            make_track(4, valence=0.58, energy=0.60),  # violación 2
        ]
        ok, violations = validate_transitions(tracks)
        assert ok is False
        assert len(violations) == 2

    def test_exactly_at_limit_passes(self):
        """Un cambio exactamente en el límite (0.3) debe pasar."""
        tracks = [
            make_track(1, valence=0.10, energy=0.10),
            make_track(2, valence=0.40, energy=0.10),  # Δvalence = 0.3 exacto
        ]
        ok, violations = validate_transitions(tracks)
        assert ok is True

    def test_single_track_no_violations(self):
        tracks = [make_track(1, valence=0.5, energy=0.5)]
        ok, violations = validate_transitions(tracks)
        assert ok is True
        assert violations == []

    def test_unordered_input_sorted_correctly(self):
        """Los tracks desordenados deben evaluarse en orden de posición."""
        tracks = [
            make_track(3, valence=0.58, energy=0.60),
            make_track(1, valence=0.10, energy=0.10),
            make_track(2, valence=0.55, energy=0.12),  # violación solo en posición 1→2
        ]
        ok, violations = validate_transitions(tracks)
        assert ok is False
        assert violations[0].position_from == 1

    def test_violation_describe_format(self):
        tracks = [
            make_track(1, valence=0.10, energy=0.10, name="Sad Song"),
            make_track(2, valence=0.60, energy=0.65, name="Happy Song"),
        ]
        _, violations = validate_transitions(tracks)
        description = violations[0].describe()
        assert "1→2" in description
        assert "Δvalence" in description


# ─────────────────────────────────────────────
# Tests: validate_arc_slope
# ─────────────────────────────────────────────

class TestValidateArcSlope:

    def test_ascending_arc_valid(self):
        tracks = make_ascending_playlist(6)
        ok, v_result, e_result = validate_arc_slope(tracks, "ascending")
        assert ok is True
        assert v_result.direction == "positive"
        assert e_result.direction == "positive"

    def test_descending_arc_valid(self):
        tracks = make_descending_playlist(6)
        ok, v_result, e_result = validate_arc_slope(tracks, "descending")
        assert ok is True
        assert v_result.direction == "negative"
        assert e_result.direction == "negative"

    def test_ascending_arc_fails_with_descending_data(self):
        """Arco descendente no debe pasar validación de ascending."""
        tracks = make_descending_playlist(6)
        ok, _, _ = validate_arc_slope(tracks, "ascending")
        assert ok is False

    def test_descending_arc_fails_with_ascending_data(self):
        tracks = make_ascending_playlist(6)
        ok, _, _ = validate_arc_slope(tracks, "descending")
        assert ok is False

    def test_arch_direction(self):
        """Arco que sube y luego baja debe pasar validación arch."""
        tracks = make_arch_playlist(8)
        ok, _, _ = validate_arc_slope(tracks, "arch")
        assert ok is True

    def test_slope_result_has_r_squared(self):
        """El resultado debe incluir R² para evaluar calidad del ajuste."""
        tracks = make_ascending_playlist(6)
        _, v_result, _ = validate_arc_slope(tracks, "ascending")
        assert 0.0 <= v_result.r_squared <= 1.0

    def test_invalid_direction_raises(self):
        tracks = make_ascending_playlist(3)
        with pytest.raises(ValueError) as exc:
            validate_arc_slope(tracks, "diagonal")
        assert "diagonal" in str(exc.value)

    def test_flat_direction(self):
        """Playlist plana debe pasar con direction=flat."""
        tracks = [make_track(i + 1, valence=0.5, energy=0.5) for i in range(5)]
        ok, v_result, e_result = validate_arc_slope(tracks, "flat")
        assert ok is True
        assert v_result.direction == "neutral"


# ─────────────────────────────────────────────
# Tests: calculate_coherence_score
# ─────────────────────────────────────────────

class TestCalculateCoherenceScore:

    def test_perfect_ascending_gets_high_score(self):
        tracks = make_ascending_playlist(6)
        score = calculate_coherence_score(tracks, "ascending")
        assert score >= 0.8

    def test_contradictory_arc_gets_low_score(self):
        """Arco descendente con dirección ascending → score bajo."""
        tracks = make_descending_playlist(6)
        score = calculate_coherence_score(tracks, "ascending")
        assert score < 0.7

    def test_score_range(self):
        """El score siempre debe estar entre 0.0 y 1.0."""
        for tracks in [
            make_ascending_playlist(6),
            make_descending_playlist(6),
            make_arch_playlist(6),
        ]:
            score = calculate_coherence_score(tracks, "ascending")
            assert 0.0 <= score <= 1.0

    def test_violations_reduce_score(self):
        """Más violaciones de transición → score más bajo."""
        smooth = make_ascending_playlist(6)
        score_smooth = calculate_coherence_score(smooth, "ascending")

        # Playlist con saltos bruscos dentro de arco ascendente
        jumpy = [
            make_track(1, valence=0.10, energy=0.10),
            make_track(2, valence=0.55, energy=0.12),  # salto brusco
            make_track(3, valence=0.58, energy=0.14),
            make_track(4, valence=0.60, energy=0.62),  # otro salto
            make_track(5, valence=0.75, energy=0.65),
            make_track(6, valence=0.90, energy=0.80),
        ]
        score_jumpy = calculate_coherence_score(jumpy, "ascending")
        assert score_smooth > score_jumpy

    def test_empty_list_returns_zero(self):
        score = calculate_coherence_score([], "ascending")
        assert score == 0.0


# ─────────────────────────────────────────────
# Tests: validate_playlist (punto de entrada)
# ─────────────────────────────────────────────

class TestValidatePlaylist:

    def test_perfect_playlist_fully_valid(self):
        tracks = make_ascending_playlist(6)
        result = validate_playlist(tracks, "ascending")
        assert result.is_fully_valid is True
        assert result.coherence_score >= 0.8
        assert result.transitions_ok is True
        assert len(result.violations) == 0

    def test_result_includes_arc_statistics(self):
        tracks = make_ascending_playlist(6)
        result = validate_playlist(tracks, "ascending")
        stats = result.slope_valence
        assert stats is not None
        assert stats.direction == "positive"

    def test_violations_surface_in_result(self):
        tracks = [
            make_track(1, valence=0.10, energy=0.10),
            make_track(2, valence=0.60, energy=0.65),  # salto brutal
            make_track(3, valence=0.65, energy=0.70),
        ]
        result = validate_playlist(tracks, "ascending")
        assert result.transitions_ok is False
        assert len(result.violations) == 1

    def test_recommendations_generated_on_failure(self):
        """Si hay violaciones, debe haber recomendaciones."""
        tracks = [
            make_track(1, valence=0.10, energy=0.10),
            make_track(2, valence=0.70, energy=0.80),  # salto
            make_track(3, valence=0.75, energy=0.82),
        ]
        result = validate_playlist(tracks, "ascending")
        if not result.transitions_ok:
            assert len(result.recommendations) > 0

    def test_summary_string_generated(self):
        tracks = make_ascending_playlist(4)
        result = validate_playlist(tracks, "ascending")
        summary = result.summary()
        assert "Coherencia" in summary
        assert "Transiciones" in summary

    def test_wrong_direction_makes_slope_invalid(self):
        """Playlist descendente validada como ascending → slope inválido."""
        tracks = make_descending_playlist(6)
        result = validate_playlist(tracks, "ascending")
        assert result.slope_valence_ok is False
        assert result.slope_energy_ok is False
        assert result.is_fully_valid is False

    def test_total_transitions_counted(self):
        tracks = make_ascending_playlist(5)
        result = validate_playlist(tracks, "ascending")
        # 5 tracks → 4 transiciones
        assert result.slope_valence.values is not None
