"""
tests/test_week3.py — narrative-playlist-agent
Tests para los módulos de Semana 3:
  - system_prompt.py
  - output_tools.py
  - db_tools.py
  - agent.py (con mock del LLM)

Ejecutar: python -m pytest tests/test_week3.py -v
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.system_prompt import SYSTEM_PROMPT
from src.agents.output_tools import (
    generate_playlist_file,
    _slugify,
    _flatten_tracks,
    _compute_arc_statistics,
)
from src.agents.db_tools import (
    add_track_to_database,
    check_track_exists,
    get_user_added_tracks,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def db_path_with_source(tmp_path):
    """DB temporal con columna source."""
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            track_name TEXT,
            artist TEXT,
            valence REAL,
            energy REAL,
            tempo REAL,
            acousticness REAL,
            danceability REAL,
            instrumentalness REAL,
            track_genre TEXT,
            popularity INTEGER,
            source TEXT DEFAULT 'kaggle'
        )
    """)
    conn.execute(
        "INSERT INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("track_000001", "Existing Song", "Known Artist",
         0.2, 0.3, 80.0, 0.7, 0.4, 0.0, "indie", 50, "kaggle")
    )
    conn.commit()
    conn.close()
    return path


def make_sample_phases(n_tracks_per_phase=2):
    """Genera fases de muestra para tests."""
    phases = [
        {
            "phase_number": 1,
            "phase_label": "Melancolía profunda",
            "emotional_description": "Tristeza contemplativa",
            "valence_range": [0.0, 0.3],
            "energy_range": [0.0, 0.4],
            "tracks": [
                {
                    "position": i + 1,
                    "track_name": f"Sad Song {i+1}",
                    "artist": f"Artist {i+1}",
                    "valence": 0.1 + i * 0.05,
                    "energy": 0.2 + i * 0.05,
                    "tempo": 70.0,
                    "genre": "indie",
                    "transition_note": "Encaja por su bajo valence y energy suave."
                }
                for i in range(n_tracks_per_phase)
            ],
            "phase_explanation": "Tracks seleccionados con valence bajo para evocar tristeza."
        },
        {
            "phase_number": 2,
            "phase_label": "Épico",
            "emotional_description": "Clímax energético",
            "valence_range": [0.4, 0.6],
            "energy_range": [0.8, 1.0],
            "tracks": [
                {
                    "position": n_tracks_per_phase + i + 1,
                    "track_name": f"Epic Song {i+1}",
                    "artist": f"Epic Artist {i+1}",
                    "valence": 0.5,
                    "energy": 0.85 + i * 0.05,
                    "tempo": 130.0,
                    "genre": "rock",
                    "transition_note": "Alta energía para el clímax narrativo."
                }
                for i in range(n_tracks_per_phase)
            ],
            "phase_explanation": "Tracks épicos con alta energy para el punto culminante."
        }
    ]
    return phases


# ─────────────────────────────────────────────
# Tests: system_prompt.py
# ─────────────────────────────────────────────

class TestSystemPrompt:

    def test_system_prompt_is_string(self):
        assert isinstance(SYSTEM_PROMPT, str)

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 200

    def test_contains_tool_names(self):
        assert "query_song_database" in SYSTEM_PROMPT
        assert "generate_playlist_file" in SYSTEM_PROMPT
        assert "add_track_to_database" in SYSTEM_PROMPT

    def test_contains_emotional_mappings(self):
        """El mapeo emocional debe estar presente."""
        assert "valence" in SYSTEM_PROMPT
        assert "energy" in SYSTEM_PROMPT
        assert "Melancolía" in SYSTEM_PROMPT or "melanc" in SYSTEM_PROMPT.lower()

    def test_contains_no_hallucination_rule(self):
        """La regla anti-alucinación debe estar explícita."""
        assert "NUNCA" in SYSTEM_PROMPT or "nunca" in SYSTEM_PROMPT.lower()

    def test_contains_transition_constraint(self):
        assert "0.3" in SYSTEM_PROMPT

    def test_contains_tunemymusic_instructions(self):
        assert "tunemymusic" in SYSTEM_PROMPT.lower()


# ─────────────────────────────────────────────
# Tests: output_tools.py — utilidades
# ─────────────────────────────────────────────

class TestOutputToolsUtils:

    def test_slugify_basic(self):
        assert _slugify("De la Tristeza a la Paz") == "de_la_tristeza_a_la_paz"

    def test_slugify_special_chars(self):
        result = _slugify("Playlist: ¡Épica!")
        assert ":" not in result
        assert "¡" not in result

    def test_slugify_max_length(self):
        long_title = "a" * 100
        assert len(_slugify(long_title)) <= 50

    def test_flatten_tracks_order(self):
        phases = make_sample_phases(n_tracks_per_phase=2)
        tracks = _flatten_tracks(phases)
        positions = [t["position"] for t in tracks]
        assert positions == sorted(positions)

    def test_flatten_tracks_count(self):
        phases = make_sample_phases(n_tracks_per_phase=3)
        tracks = _flatten_tracks(phases)
        assert len(tracks) == 6

    def test_flatten_tracks_adds_phase_label(self):
        phases = make_sample_phases(n_tracks_per_phase=1)
        tracks = _flatten_tracks(phases)
        for track in tracks:
            assert "phase_label" in track

    def test_arc_statistics_ascending(self):
        tracks = [
            {"valence": 0.1, "energy": 0.2, "position": 1},
            {"valence": 0.5, "energy": 0.6, "position": 2},
            {"valence": 0.9, "energy": 0.9, "position": 3},
        ]
        stats = _compute_arc_statistics(tracks)
        assert stats["valence_slope"] == "positive"
        assert stats["energy_slope"] == "positive"
        assert stats["valence_start"] == 0.1
        assert stats["valence_end"] == 0.9

    def test_arc_statistics_descending(self):
        tracks = [
            {"valence": 0.9, "energy": 0.9, "position": 1},
            {"valence": 0.5, "energy": 0.5, "position": 2},
            {"valence": 0.1, "energy": 0.1, "position": 3},
        ]
        stats = _compute_arc_statistics(tracks)
        assert stats["valence_slope"] == "negative"

    def test_arc_statistics_empty(self):
        stats = _compute_arc_statistics([])
        assert stats == {}


# ─────────────────────────────────────────────
# Tests: generate_playlist_file
# ─────────────────────────────────────────────

class TestGeneratePlaylistFile:

    def test_generates_csv_and_json(self, tmp_path):
        phases = make_sample_phases()
        with patch("src.agents.output_tools.OUTPUT_DIR", tmp_path):
            result_str = generate_playlist_file({
                "playlist_title": "Test Playlist",
                "playlist_description": "Arco de prueba",
                "phases": phases,
                "arc_summary": "Este es el resumen del arco emocional de la playlist."
            })

        result = json.loads(result_str)
        assert result["success"] is True
        assert "csv_path" in result
        assert "json_path" in result
        assert Path(result["csv_path"]).exists()
        assert Path(result["json_path"]).exists()

    def test_csv_has_correct_headers(self, tmp_path):
        phases = make_sample_phases()
        with patch("src.agents.output_tools.OUTPUT_DIR", tmp_path):
            result_str = generate_playlist_file({
                "playlist_title": "CSV Test",
                "playlist_description": "Test",
                "phases": phases,
                "arc_summary": "Resumen del arco emocional suficientemente largo."
            })

        result = json.loads(result_str)
        csv_content = Path(result["csv_path"]).read_text()
        first_line = csv_content.strip().split("\n")[0]
        assert "Title" in first_line
        assert "Artist" in first_line

    def test_csv_track_count(self, tmp_path):
        phases = make_sample_phases(n_tracks_per_phase=3)
        with patch("src.agents.output_tools.OUTPUT_DIR", tmp_path):
            result_str = generate_playlist_file({
                "playlist_title": "Count Test",
                "playlist_description": "Test",
                "phases": phases,
                "arc_summary": "Resumen del arco emocional suficientemente largo."
            })

        result = json.loads(result_str)
        assert result["total_tracks"] == 6

    def test_json_contains_arc_summary(self, tmp_path):
        phases = make_sample_phases()
        arc_summary = "Viaje emocional de melancolía a épica gloria."
        with patch("src.agents.output_tools.OUTPUT_DIR", tmp_path):
            result_str = generate_playlist_file({
                "playlist_title": "JSON Test",
                "playlist_description": "Test",
                "phases": phases,
                "arc_summary": arc_summary
            })

        result = json.loads(result_str)
        json_data = json.loads(Path(result["json_path"]).read_text())
        assert json_data["arc_summary"] == arc_summary

    def test_empty_phases_returns_error(self, tmp_path):
        with patch("src.agents.output_tools.OUTPUT_DIR", tmp_path):
            result_str = generate_playlist_file({
                "playlist_title": "Empty",
                "playlist_description": "Test",
                "phases": [],
                "arc_summary": "Sin tracks."
            })
        result = json.loads(result_str)
        assert result["success"] is False
        assert "fases" in result["error"].lower()

    def test_tunemymusic_instructions_in_result(self, tmp_path):
        phases = make_sample_phases()
        with patch("src.agents.output_tools.OUTPUT_DIR", tmp_path):
            result_str = generate_playlist_file({
                "playlist_title": "Instructions Test",
                "playlist_description": "Test",
                "phases": phases,
                "arc_summary": "Resumen del arco emocional suficientemente largo."
            })
        result = json.loads(result_str)
        assert "tunemymusic" in result.get("tunemymusic_instructions", "").lower()


# ─────────────────────────────────────────────
# Tests: db_tools.py
# ─────────────────────────────────────────────

class TestCheckTrackExists:

    def test_existing_track_found(self, db_path_with_source):
        result_str = check_track_exists(
            {"track_name": "Existing Song", "artist": "Known Artist"},
            db_path=db_path_with_source
        )
        result = json.loads(result_str)
        assert result["found"] is True
        assert result["track_id"] == "track_000001"

    def test_case_insensitive_search(self, db_path_with_source):
        result_str = check_track_exists(
            {"track_name": "EXISTING SONG", "artist": "KNOWN ARTIST"},
            db_path=db_path_with_source
        )
        result = json.loads(result_str)
        assert result["found"] is True

    def test_missing_track_not_found(self, db_path_with_source):
        result_str = check_track_exists(
            {"track_name": "Nonexistent", "artist": "Nobody"},
            db_path=db_path_with_source
        )
        result = json.loads(result_str)
        assert result["found"] is False

    def test_db_not_found_returns_error(self):
        result_str = check_track_exists(
            {"track_name": "Song", "artist": "Artist"},
            db_path=Path("/fake/path.db")
        )
        result = json.loads(result_str)
        assert result["found"] is False
        assert "error" in result


class TestAddTrackToDatabase:

    def test_valid_track_inserted(self, db_path_with_source):
        result_str = add_track_to_database(
            {
                "track_name": "New Song",
                "artist": "New Artist",
                "valence": 0.4,
                "energy": 0.6,
                "tempo": 110.0,
                "acousticness": 0.3,
                "track_genre": "pop"
            },
            db_path=db_path_with_source
        )
        result = json.loads(result_str)
        assert result["success"] is True
        assert result["source"] == "user_added"
        assert result["track_id"].startswith("track_u")

    def test_track_retrievable_after_insert(self, db_path_with_source):
        add_track_to_database(
            {
                "track_name": "Retrievable Song",
                "artist": "Test Artist",
                "valence": 0.5,
                "energy": 0.5,
                "tempo": 100.0,
                "acousticness": 0.4
            },
            db_path=db_path_with_source
        )
        check_str = check_track_exists(
            {"track_name": "Retrievable Song", "artist": "Test Artist"},
            db_path=db_path_with_source
        )
        check = json.loads(check_str)
        assert check["found"] is True

    def test_invalid_valence_rejected(self, db_path_with_source):
        result_str = add_track_to_database(
            {
                "track_name": "Bad Song",
                "artist": "Bad Artist",
                "valence": 1.5,
                "energy": 0.5,
                "tempo": 100.0,
                "acousticness": 0.4
            },
            db_path=db_path_with_source
        )
        result = json.loads(result_str)
        assert result["success"] is False
        assert "valence" in result["error"].lower()

    def test_missing_required_field_rejected(self, db_path_with_source):
        result_str = add_track_to_database(
            {
                "track_name": "Incomplete Song",
                "artist": "Artist"
                # falta valence, energy, tempo, acousticness
            },
            db_path=db_path_with_source
        )
        result = json.loads(result_str)
        assert result["success"] is False

    def test_source_column_created_if_missing(self, tmp_path):
        """La columna source se crea automáticamente si no existe."""
        db = tmp_path / "no_source.db"
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE tracks (
                track_id TEXT PRIMARY KEY,
                track_name TEXT, artist TEXT,
                valence REAL, energy REAL, tempo REAL,
                acousticness REAL, danceability REAL,
                instrumentalness REAL, track_genre TEXT,
                popularity INTEGER
            )
        """)
        conn.commit()
        conn.close()

        result_str = add_track_to_database(
            {
                "track_name": "Auto Source",
                "artist": "Test",
                "valence": 0.3,
                "energy": 0.4,
                "tempo": 90.0,
                "acousticness": 0.5
            },
            db_path=db
        )
        result = json.loads(result_str)
        assert result["success"] is True


class TestGetUserAddedTracks:

    def test_returns_only_user_added(self, db_path_with_source):
        # Añadir un track de usuario
        add_track_to_database(
            {
                "track_name": "My Song",
                "artist": "Me",
                "valence": 0.5,
                "energy": 0.5,
                "tempo": 100.0,
                "acousticness": 0.3
            },
            db_path=db_path_with_source
        )

        result_str = get_user_added_tracks(db_path=db_path_with_source)
        result = json.loads(result_str)
        assert result["count"] == 1
        assert result["tracks"][0]["track_name"] == "My Song"

    def test_empty_when_no_user_tracks(self, db_path_with_source):
        result_str = get_user_added_tracks(db_path=db_path_with_source)
        result = json.loads(result_str)
        assert result["count"] == 0
