"""
tests/test_skills.py — narrative-playlist-agent
Tests para el sistema de skills.
Ejecutar: python -m pytest tests/test_skills.py -v
"""

import pytest
from pathlib import Path

from skills.loader import (
    load_skill,
    load_all_skills,
    select_skill,
    get_skill_context,
    get_skills_index,
    _parse_frontmatter,
)


# ─────────────────────────────────────────────
# Tests: _parse_frontmatter
# ─────────────────────────────────────────────

class TestParseFrontmatter:

    def test_extracts_name(self):
        content = "---\nname: test-skill\nversion: 1.0.0\n---\n## Body"
        meta, body = _parse_frontmatter(content)
        assert meta["name"] == "test-skill"

    def test_extracts_version(self):
        content = "---\nname: skill\nversion: 2.1.0\n---\nBody"
        meta, body = _parse_frontmatter(content)
        assert meta["version"] == "2.1.0"

    def test_body_separated_from_frontmatter(self):
        content = "---\nname: skill\n---\n## Instructions\nDo this."
        meta, body = _parse_frontmatter(content)
        assert "## Instructions" in body
        assert "name" not in body

    def test_no_frontmatter_returns_full_content(self):
        content = "## Just a body\nNo frontmatter here."
        meta, body = _parse_frontmatter(content)
        assert meta == {}
        assert "Just a body" in body


# ─────────────────────────────────────────────
# Tests: load_skill
# ─────────────────────────────────────────────

class TestLoadSkill:

    def test_load_playlist_generation(self):
        skill = load_skill("playlist_generation")
        assert skill is not None
        assert skill["name"] == "playlist_generation"
        assert "content" in skill
        assert len(skill["content"]) > 100

    def test_load_track_ingestion(self):
        skill = load_skill("track_ingestion")
        assert skill is not None
        assert "content" in skill

    def test_load_arc_validation(self):
        skill = load_skill("arc_validation")
        assert skill is not None

    def test_nonexistent_skill_returns_none(self):
        skill = load_skill("nonexistent_skill_xyz")
        assert skill is None

    def test_skill_has_metadata(self):
        skill = load_skill("playlist_generation")
        assert "metadata" in skill
        assert skill["metadata"].get("name") is not None

    def test_playlist_skill_contains_mapeo(self):
        skill = load_skill("playlist_generation")
        assert "valence" in skill["content"]
        assert "energy" in skill["content"]

    def test_track_skill_contains_steps(self):
        skill = load_skill("track_ingestion")
        assert "Paso" in skill["content"] or "paso" in skill["content"].lower()

    def test_validation_skill_contains_score(self):
        skill = load_skill("arc_validation")
        assert "score" in skill["content"].lower() or "coherencia" in skill["content"].lower()


# ─────────────────────────────────────────────
# Tests: load_all_skills
# ─────────────────────────────────────────────

class TestLoadAllSkills:

    def test_returns_three_skills(self):
        skills = load_all_skills()
        assert len(skills) == 3

    def test_all_have_name(self):
        skills = load_all_skills()
        for skill in skills:
            assert "name" in skill
            assert skill["name"]

    def test_all_have_description(self):
        skills = load_all_skills()
        for skill in skills:
            assert "description" in skill
            assert len(skill["description"]) > 10

    def test_all_have_version(self):
        skills = load_all_skills()
        for skill in skills:
            assert "version" in skill


# ─────────────────────────────────────────────
# Tests: select_skill
# ─────────────────────────────────────────────

class TestSelectSkill:

    def test_playlist_prompt_selects_generation(self):
        result = select_skill("hazme una playlist de tristeza a euforia")
        assert result == "playlist_generation"

    def test_add_track_prompt_selects_ingestion(self):
        result = select_skill("agrega 'Papaoutai' de Stromae al dataset")
        assert result == "track_ingestion"

    def test_validation_prompt_selects_validation(self):
        result = select_skill("qué coherencia tiene mi playlist")
        assert result == "arc_validation"

    def test_english_playlist_prompt(self):
        result = select_skill("create a playlist for me with songs")
        assert result == "playlist_generation"

    def test_unrelated_prompt_returns_none(self):
        result = select_skill("hola cómo estás")
        assert result is None

    def test_strongest_signal_wins(self):
        """Cuando hay múltiples signals, la más fuerte gana."""
        result = select_skill(
            "agrega esta canción al dataset para usarla en una playlist"
        )
        # Tiene señales de ambas — la que tenga más keywords gana
        assert result in ("track_ingestion", "playlist_generation")


# ─────────────────────────────────────────────
# Tests: get_skill_context
# ─────────────────────────────────────────────

class TestGetSkillContext:

    def test_returns_content_for_playlist_prompt(self):
        context = get_skill_context("hazme una playlist épica")
        assert len(context) > 50
        assert "Skill activa" in context

    def test_returns_empty_for_unrelated_prompt(self):
        context = get_skill_context("qué hora es")
        assert context == ""

    def test_context_contains_skill_instructions(self):
        context = get_skill_context("quiero una playlist de tristeza a paz")
        assert "valence" in context or "energy" in context


# ─────────────────────────────────────────────
# Tests: get_skills_index
# ─────────────────────────────────────────────

class TestGetSkillsIndex:

    def test_returns_non_empty_string(self):
        index = get_skills_index()
        assert isinstance(index, str)
        assert len(index) > 0

    def test_contains_skill_names(self):
        index = get_skills_index()
        assert "playlist" in index.lower()

    def test_contains_descriptions(self):
        index = get_skills_index()
        assert "**" in index  # las skills usan bold en el índice
