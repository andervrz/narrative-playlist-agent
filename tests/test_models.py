"""
tests/test_models.py — narrative-playlist-agent
Tests unitarios para schemas/models.py
Ejecutar: python -m pytest tests/test_models.py -v
"""

import pytest
from pydantic import ValidationError
from src.schemas.models import (
    Track, Phase, ArcStatistics, PlaylistOutput, QueryArgs, QueryResult
)


# ─────────────────────────────────────────────
# Fixtures — datos de prueba reutilizables
# ─────────────────────────────────────────────

def make_track(
    position=1,
    valence=0.2,
    energy=0.3,
    track_name="Test Song",
    artist="Test Artist",
    tempo=90.0,
    genre="indie",
    transition_note="Esta canción encaja por su bajo valence y energy suave.",
    **kwargs
):
    return Track(
        position=position,
        track_name=track_name,
        artist=artist,
        valence=valence,
        energy=energy,
        tempo=tempo,
        genre=genre,
        transition_note=transition_note,
        **kwargs
    )

def make_phase(phase_number=1, n_tracks=2):
    tracks = [make_track(position=i) for i in range(1, n_tracks + 1)]
    return Phase(
        phase_number=phase_number,
        phase_label="Melancolía profunda",
        emotional_description="Estado de tristeza contemplativa con energía muy baja.",
        valence_range=(0.0, 0.3),
        energy_range=(0.0, 0.4),
        tracks=tracks,
        phase_explanation=(
            "Se seleccionaron tracks con valence < 0.3 y energy < 0.4 "
            "para evocar introspección sin agresividad."
        )
    )


# ─────────────────────────────────────────────
# Tests: Track
# ─────────────────────────────────────────────

class TestTrack:

    def test_valid_track(self):
        track = make_track()
        assert track.valence == 0.2
        assert track.energy == 0.3

    def test_valence_out_of_range(self):
        with pytest.raises(ValidationError) as exc:
            make_track(valence=1.5)
        assert "valence" in str(exc.value)

    def test_energy_negative(self):
        with pytest.raises(ValidationError):
            make_track(energy=-0.1)

    def test_transition_note_too_short(self):
        with pytest.raises(ValidationError) as exc:
            make_track(transition_note="corto")
        assert "transition_note" in str(exc.value)

    def test_extra_fields_ignored(self):
        """Campos extra del LLM deben ignorarse silenciosamente."""
        track = Track(
            position=1,
            track_name="Song",
            artist="Artist",
            valence=0.5,
            energy=0.5,
            tempo=100.0,
            genre="pop",
            transition_note="Esta canción encaja porque sube el mood gradualmente.",
            campo_inventado_por_llm="este campo no debería romper nada"
        )
        assert track.track_name == "Song"


# ─────────────────────────────────────────────
# Tests: Phase
# ─────────────────────────────────────────────

class TestPhase:

    def test_valid_phase(self):
        phase = make_phase()
        assert phase.phase_number == 1
        assert len(phase.tracks) == 2

    def test_invalid_valence_range(self):
        """min > max debe fallar."""
        with pytest.raises(ValidationError) as exc:
            Phase(
                phase_number=1,
                phase_label="Test",
                emotional_description="Descripción de prueba larga para pasar validación.",
                valence_range=(0.8, 0.2),  # inválido
                energy_range=(0.0, 0.5),
                tracks=[make_track()],
                phase_explanation="Explicación de prueba que debe tener al menos 50 chars para pasar."
            )
        assert "valence_range" in str(exc.value)

    def test_phase_explanation_too_short(self):
        with pytest.raises(ValidationError) as exc:
            Phase(
                phase_number=1,
                phase_label="Test",
                emotional_description="Descripción suficientemente larga para la validación.",
                valence_range=(0.0, 0.3),
                energy_range=(0.0, 0.4),
                tracks=[make_track()],
                phase_explanation="corta"  # menos de 50 chars
            )
        assert "phase_explanation" in str(exc.value)

    def test_duplicate_track_positions(self):
        """Dos tracks con la misma posición deben fallar."""
        tracks = [make_track(position=1), make_track(position=1)]
        with pytest.raises(ValidationError) as exc:
            Phase(
                phase_number=1,
                phase_label="Test",
                emotional_description="Descripción suficientemente larga para la validación.",
                valence_range=(0.0, 0.3),
                energy_range=(0.0, 0.4),
                tracks=tracks,
                phase_explanation="Explicación de prueba que debe tener al menos 50 chars para pasar."
            )
        assert "duplicadas" in str(exc.value)


# ─────────────────────────────────────────────
# Tests: PlaylistOutput
# ─────────────────────────────────────────────

class TestPlaylistOutput:

    def _make_valid_output(self):
        phase1 = make_phase(phase_number=1, n_tracks=2)
        # Cambiar posiciones para que sean únicas globalmente
        phase1.tracks[0] = make_track(position=1, valence=0.2, energy=0.3)
        phase1.tracks[1] = make_track(position=2, valence=0.25, energy=0.35,
                                      track_name="Second Song")

        phase2 = Phase(
            phase_number=2,
            phase_label="Euforia",
            emotional_description="Estado de alegría intensa y energía máxima.",
            valence_range=(0.7, 1.0),
            energy_range=(0.7, 1.0),
            tracks=[
                make_track(position=3, valence=0.8, energy=0.85,
                           track_name="Third Song"),
            ],
            phase_explanation=(
                "Tracks con valence > 0.7 y energy > 0.7 para lograr "
                "el estado de euforia y celebración del arco narrativo."
            )
        )
        return PlaylistOutput(
            playlist_title="De la tristeza a la alegría",
            user_prompt="Quiero una playlist que vaya de triste a alegre",
            narrative_arc="Arco emocional ascendente de melancolía a euforia pura.",
            total_tracks=3,
            phases=[phase1, phase2],
            arc_summary=(
                "Esta playlist construye un viaje emocional desde la introspección "
                "profunda hasta la celebración pura, con una transición gradual "
                "en valence y energy a lo largo de las tres canciones."
            )
        )

    def test_valid_output(self):
        output = self._make_valid_output()
        assert output.total_tracks == 3
        assert len(output.phases) == 2

    def test_total_tracks_mismatch(self):
        """total_tracks que no coincide con la suma real debe fallar."""
        phase = make_phase(n_tracks=2)
        with pytest.raises(ValidationError) as exc:
            PlaylistOutput(
                playlist_title="Test",
                user_prompt="Test prompt",
                narrative_arc="Arco de prueba suficientemente descriptivo.",
                total_tracks=5,  # incorrecto — solo hay 2
                phases=[phase],
                arc_summary=(
                    "Resumen del arco suficientemente largo para pasar "
                    "la validación de longitud mínima de Pydantic."
                )
            )
        assert "total_tracks" in str(exc.value)

    def test_duplicate_tracks_across_phases(self):
        """La misma canción en dos fases diferentes debe fallar."""
        track_a = make_track(position=1, track_name="Same Song", artist="Same Artist")
        track_b = make_track(position=2, track_name="Same Song", artist="Same Artist")

        phase1 = Phase(
            phase_number=1,
            phase_label="Fase 1",
            emotional_description="Descripción de la fase 1 suficientemente larga.",
            valence_range=(0.0, 0.3),
            energy_range=(0.0, 0.4),
            tracks=[track_a],
            phase_explanation="Justificación de la fase 1 con al menos cincuenta caracteres necesarios."
        )
        phase2 = Phase(
            phase_number=2,
            phase_label="Fase 2",
            emotional_description="Descripción de la fase 2 suficientemente larga.",
            valence_range=(0.7, 1.0),
            energy_range=(0.7, 1.0),
            tracks=[track_b],
            phase_explanation="Justificación de la fase 2 con al menos cincuenta caracteres necesarios."
        )
        with pytest.raises(ValidationError) as exc:
            PlaylistOutput(
                playlist_title="Test",
                user_prompt="Test prompt",
                narrative_arc="Arco de prueba suficientemente descriptivo.",
                total_tracks=2,
                phases=[phase1, phase2],
                arc_summary=(
                    "Resumen del arco suficientemente largo para pasar "
                    "la validación de longitud mínima de Pydantic."
                )
            )
        assert "duplicados" in str(exc.value)

    def test_all_tracks_flat(self):
        output = self._make_valid_output()
        flat = output.all_tracks_flat()
        assert len(flat) == 3
        assert flat[0].position == 1
        assert flat[2].position == 3

    def test_summary_for_cli(self):
        output = self._make_valid_output()
        summary = output.summary_for_cli()
        assert "De la tristeza a la alegría" in summary
        assert "Melancolía" in summary


# ─────────────────────────────────────────────
# Tests: QueryArgs
# ─────────────────────────────────────────────

class TestQueryArgs:

    def test_valid_args(self):
        args = QueryArgs(
            min_valence=0.0, max_valence=0.3,
            min_energy=0.0, max_energy=0.4,
            limit=3
        )
        assert args.limit == 3

    def test_min_greater_than_max_valence(self):
        with pytest.raises(ValidationError) as exc:
            QueryArgs(
                min_valence=0.8, max_valence=0.2,
                min_energy=0.0, max_energy=0.5,
                limit=3
            )
        assert "min_valence" in str(exc.value)

    def test_optional_fields_default_none(self):
        args = QueryArgs(
            min_valence=0.0, max_valence=0.5,
            min_energy=0.0, max_energy=0.5,
            limit=3
        )
        assert args.min_tempo is None
        assert args.target_genres is None
        assert args.exclude_track_ids is None

    def test_limit_bounds(self):
        """Límite debe estar entre 1 y 10."""
        with pytest.raises(ValidationError):
            QueryArgs(
                min_valence=0.0, max_valence=0.5,
                min_energy=0.0, max_energy=0.5,
                limit=15  # > 10
            )


if __name__ == "__main__":
    # Permite correr directamente sin pytest
    import sys
    print("Corriendo tests básicos sin pytest...")

    # Test rápido manual
    try:
        track = make_track()
        print(f"✓ Track creado: {track.track_name}")

        phase = make_phase()
        print(f"✓ Phase creada: {phase.phase_label}")

        args = QueryArgs(min_valence=0.0, max_valence=0.3,
                        min_energy=0.0, max_energy=0.4, limit=3)
        print(f"✓ QueryArgs creado: V({args.min_valence}-{args.max_valence})")

        print("\n✅ Tests básicos pasados.")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
