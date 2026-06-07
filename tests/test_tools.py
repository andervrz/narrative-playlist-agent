"""
tests/test_tools.py — narrative-playlist-agent
Tests unitarios e integración para agent/tools.py
Ejecutar: python -m pytest tests/test_tools.py -v
"""

import json
import pytest
from pathlib import Path

from src.agent.tools import (
    build_query,
    execute_query,
    relax_args,
    query_with_retry,
    query_song_database,
    check_db_ready,
    TOOL_SCHEMA,
)
from src.schemas.models import QueryArgs


# ─────────────────────────────────────────────
# Fixtures locales
# ─────────────────────────────────────────────

def make_args(**kwargs) -> QueryArgs:
    defaults = dict(
        min_valence=0.0, max_valence=0.3,
        min_energy=0.0,  max_energy=0.4,
        limit=3
    )
    defaults.update(kwargs)
    return QueryArgs(**defaults)


# ─────────────────────────────────────────────
# Tests: build_query
# ─────────────────────────────────────────────

class TestBuildQuery:

    def test_basic_query_structure(self):
        args = make_args()
        sql, params = build_query(args)
        assert "valence BETWEEN ? AND ?" in sql
        assert "energy BETWEEN ? AND ?" in sql
        assert "ORDER BY RANDOM()" in sql
        assert "LIMIT ?" in sql

    def test_params_order(self):
        """Los parámetros deben estar en el orden correcto para SQLite."""
        args = make_args(min_valence=0.1, max_valence=0.3,
                         min_energy=0.2, max_energy=0.5, limit=5)
        sql, params = build_query(args)
        # valence min, valence max, energy min, energy max, ..., limit
        assert params[0] == 0.1
        assert params[1] == 0.3
        assert params[2] == 0.2
        assert params[3] == 0.5
        assert params[-1] == 5

    def test_optional_tempo_included(self):
        args = make_args(min_tempo=120.0, max_tempo=160.0)
        sql, params = build_query(args)
        assert "tempo >= ?" in sql
        assert "tempo <= ?" in sql
        assert 120.0 in params
        assert 160.0 in params

    def test_optional_acousticness(self):
        args = make_args(min_acousticness=0.7)
        sql, params = build_query(args)
        assert "acousticness >= ?" in sql
        assert 0.7 in params

    def test_target_genres_single(self):
        args = make_args(target_genres=["rock"])
        sql, params = build_query(args)
        assert "track_genre IN" in sql
        assert "rock" in params

    def test_target_genres_multiple(self):
        args = make_args(target_genres=["rock", "classical", "indie"])
        sql, params = build_query(args)
        assert "track_genre IN (?,?,?)" in sql

    def test_target_genres_normalized_lowercase(self):
        """Los géneros deben ir en lowercase independiente del input."""
        args = make_args(target_genres=["ROCK", "Classical"])
        sql, params = build_query(args)
        assert "rock" in params
        assert "classical" in params

    def test_exclude_track_ids(self):
        args = make_args(exclude_track_ids=["track_000001", "track_000002"])
        sql, params = build_query(args)
        assert "track_id NOT IN" in sql
        assert "track_000001" in params

    def test_no_optional_fields(self):
        """Sin campos opcionales, no deben aparecer en el SQL."""
        args = make_args()
        sql, params = build_query(args)
        assert "tempo >= ?" not in sql
        assert "tempo <= ?" not in sql
        assert "acousticness >= ?" not in sql
        assert "track_genre IN" not in sql
        assert "track_id NOT IN" not in sql


# ─────────────────────────────────────────────
# Tests: relax_args
# ─────────────────────────────────────────────

class TestRelaxArgs:

    def test_ranges_expand(self):
        args = make_args(min_valence=0.2, max_valence=0.3,
                         min_energy=0.2, max_energy=0.3)
        relaxed = relax_args(args, step=0.1)
        assert relaxed.min_valence == pytest.approx(0.1)
        assert relaxed.max_valence == pytest.approx(0.4)
        assert relaxed.min_energy  == pytest.approx(0.1)
        assert relaxed.max_energy  == pytest.approx(0.4)

    def test_clamped_at_zero(self):
        """min no puede bajar de 0.0."""
        args = make_args(min_valence=0.05, max_valence=0.2,
                         min_energy=0.0, max_energy=0.2)
        relaxed = relax_args(args, step=0.1)
        assert relaxed.min_valence == pytest.approx(0.0)
        assert relaxed.min_energy  == pytest.approx(0.0)

    def test_clamped_at_one(self):
        """max no puede superar 1.0."""
        args = make_args(min_valence=0.8, max_valence=0.95,
                         min_energy=0.8, max_energy=0.95)
        relaxed = relax_args(args, step=0.1)
        assert relaxed.max_valence == pytest.approx(1.0)
        assert relaxed.max_energy  == pytest.approx(1.0)

    def test_preserves_other_fields(self):
        """Los campos no afectados por relaxation deben conservarse."""
        args = make_args(min_tempo=120.0, target_genres=["rock"], limit=5)
        relaxed = relax_args(args, step=0.1)
        assert relaxed.min_tempo == 120.0
        assert relaxed.target_genres == ["rock"]
        assert relaxed.limit == 5


# ─────────────────────────────────────────────
# Tests: execute_query (requiere DB fixture)
# ─────────────────────────────────────────────

class TestExecuteQuery:

    def test_returns_tracks_melancolía(self, db_path):
        args = make_args(min_valence=0.0, max_valence=0.3,
                         min_energy=0.0, max_energy=0.4, limit=3)
        results = execute_query(args, db_path)
        assert len(results) == 3
        for track in results:
            assert 0.0 <= track["valence"] <= 0.3
            assert 0.0 <= track["energy"] <= 0.4

    def test_returns_tracks_euforia(self, db_path):
        args = make_args(min_valence=0.7, max_valence=1.0,
                         min_energy=0.7, max_energy=1.0, limit=3)
        results = execute_query(args, db_path)
        assert len(results) >= 1
        for track in results:
            assert track["valence"] >= 0.7
            assert track["energy"] >= 0.7

    def test_empty_result_impossible_range(self, db_path):
        """Parámetros imposibles deben retornar lista vacía, no error."""
        args = make_args(min_valence=0.99, max_valence=1.0,
                         min_energy=0.0, max_energy=0.01, limit=3)
        results = execute_query(args, db_path)
        assert results == []

    def test_respects_limit(self, db_path):
        args = make_args(min_valence=0.0, max_valence=1.0,
                         min_energy=0.0, max_energy=1.0, limit=5)
        results = execute_query(args, db_path)
        assert len(results) <= 5

    def test_genre_filter(self, db_path):
        args = make_args(min_valence=0.0, max_valence=1.0,
                         min_energy=0.0, max_energy=1.0,
                         target_genres=["jazz"], limit=5)
        results = execute_query(args, db_path)
        for track in results:
            assert track["track_genre"] == "jazz"

    def test_exclude_ids(self, db_path):
        # Primera query — obtener algunos tracks
        args = make_args(min_valence=0.0, max_valence=1.0,
                         min_energy=0.0, max_energy=1.0, limit=3)
        first_batch = execute_query(args, db_path)
        ids_to_exclude = [t["track_id"] for t in first_batch]

        # Segunda query — excluir esos IDs
        args2 = make_args(min_valence=0.0, max_valence=1.0,
                          min_energy=0.0, max_energy=1.0,
                          exclude_track_ids=ids_to_exclude, limit=3)
        second_batch = execute_query(args2, db_path)
        second_ids = [t["track_id"] for t in second_batch]

        for excluded_id in ids_to_exclude:
            assert excluded_id not in second_ids

    def test_db_not_found_raises(self):
        args = make_args()
        with pytest.raises(FileNotFoundError) as exc:
            execute_query(args, Path("/nonexistent/path/tracks.db"))
        assert "load_dataset" in str(exc.value)

    def test_result_contains_required_fields(self, db_path):
        args = make_args(limit=1)
        results = execute_query(args, db_path)
        assert len(results) == 1
        track = results[0]
        for field in ["track_id", "track_name", "artist", "valence", "energy", "tempo"]:
            assert field in track


# ─────────────────────────────────────────────
# Tests: query_with_retry
# ─────────────────────────────────────────────

class TestQueryWithRetry:

    def test_success_first_attempt(self, db_path):
        args = make_args(min_valence=0.0, max_valence=0.3,
                         min_energy=0.0, max_energy=0.4, limit=3)
        result = query_with_retry(args, db_path=db_path)
        assert result.success is True
        assert result.attempts == 1
        assert len(result.tracks) == 3

    def test_retry_on_empty_result(self, db_path):
        """
        Rango que no tiene resultados directos pero sí con relaxation.
        valence BETWEEN 0.34 AND 0.34 → 0 resultados
        Después de relax: 0.24–0.44 → encuentra tracks medios
        """
        args = make_args(min_valence=0.34, max_valence=0.34,
                         min_energy=0.34, max_energy=0.34, limit=2)
        result = query_with_retry(args, db_path=db_path)
        assert result.success is True
        assert result.attempts > 1

    def test_fail_after_max_attempts(self, db_path):
        """Parámetros imposibles incluso después de retry → success=False."""
        args = make_args(min_valence=0.995, max_valence=0.999,
                         min_energy=0.001, max_energy=0.002, limit=3)
        result = query_with_retry(args, db_path=db_path, max_attempts=3)
        assert result.success is False
        assert result.attempts == 3
        assert result.error is not None

    def test_db_not_found_returns_error_result(self):
        args = make_args()
        result = query_with_retry(args, db_path=Path("/fake/path.db"))
        assert result.success is False
        assert "load_dataset" in result.error


# ─────────────────────────────────────────────
# Tests: query_song_database (entry point público)
# ─────────────────────────────────────────────

class TestQuerySongDatabase:

    def test_returns_valid_json(self, db_path):
        raw_args = {
            "min_valence": 0.0, "max_valence": 0.3,
            "min_energy": 0.0, "max_energy": 0.4,
            "limit": 3
        }
        result_str = query_song_database(raw_args, db_path=db_path)
        result = json.loads(result_str)
        assert result["success"] is True
        assert "tracks" in result
        assert "count" in result

    def test_invalid_args_returns_error_json(self, db_path):
        """Args inválidos del LLM → JSON de error, sin crash."""
        raw_args = {
            "min_valence": 1.5,  # fuera de rango
            "max_valence": 0.3,
            "min_energy": 0.0,
            "max_energy": 0.4,
            "limit": 3
        }
        result_str = query_song_database(raw_args, db_path=db_path)
        result = json.loads(result_str)
        assert result["success"] is False
        assert "error" in result

    def test_min_greater_than_max_returns_error(self, db_path):
        raw_args = {
            "min_valence": 0.8, "max_valence": 0.2,  # inválido
            "min_energy": 0.0, "max_energy": 0.5,
            "limit": 3
        }
        result_str = query_song_database(raw_args, db_path=db_path)
        result = json.loads(result_str)
        assert result["success"] is False

    def test_note_mentions_retry_when_relaxed(self, db_path):
        """Si se necesitó retry, el note debe mencionarlo."""
        raw_args = {
            "min_valence": 0.34, "max_valence": 0.34,
            "min_energy": 0.34, "max_energy": 0.34,
            "limit": 2
        }
        result_str = query_song_database(raw_args, db_path=db_path)
        result = json.loads(result_str)
        if result["success"] and result["attempts"] > 1:
            assert "relajados" in result["note"]


# ─────────────────────────────────────────────
# Tests: TOOL_SCHEMA estructura
# ─────────────────────────────────────────────

class TestToolSchema:

    def test_schema_has_required_keys(self):
        assert TOOL_SCHEMA["type"] == "function"
        assert "function" in TOOL_SCHEMA
        fn = TOOL_SCHEMA["function"]
        assert fn["name"] == "query_song_database"
        assert "description" in fn
        assert "parameters" in fn

    def test_required_fields_defined(self):
        required = TOOL_SCHEMA["function"]["parameters"]["required"]
        assert "min_valence" in required
        assert "max_valence" in required
        assert "min_energy" in required
        assert "max_energy" in required
        assert "limit" in required

    def test_optional_fields_exist(self):
        props = TOOL_SCHEMA["function"]["parameters"]["properties"]
        assert "min_tempo" in props
        assert "target_genres" in props
        assert "exclude_track_ids" in props


# ─────────────────────────────────────────────
# Tests: check_db_ready
# ─────────────────────────────────────────────

class TestCheckDbReady:

    def test_valid_db_returns_true(self, db_path):
        ok, msg = check_db_ready(db_path)
        assert ok is True
        assert "tracks disponibles" in msg

    def test_missing_db_returns_false(self):
        ok, msg = check_db_ready(Path("/nonexistent/tracks.db"))
        assert ok is False
        assert "load_dataset" in msg
