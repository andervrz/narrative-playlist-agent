"""
skills/loader.py — narrative-playlist-agent
Carga y gestiona las skills disponibles para el agente.

Las skills se cargan bajo demanda (progressive disclosure):
  - Tier 1: nombre + descripción de cada skill (siempre en contexto)
  - Tier 2: contenido completo del SKILL.md (cuando es relevante)

El agente decide qué skill cargar basado en el prompt del usuario.
"""

import re
from pathlib import Path
from typing import Optional

SKILLS_DIR = Path(__file__).resolve().parent


# ─────────────────────────────────────────────
# CARGA DE SKILLS
# ─────────────────────────────────────────────

def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Extrae el frontmatter YAML y el cuerpo de un SKILL.md.
    Retorna (metadata_dict, body_text).
    """
    metadata = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            raw_meta = parts[1].strip()
            body = parts[2].strip()

            # Parser YAML manual — sin dependencia externa
            lines = raw_meta.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if value == ">":
                        # Recoger líneas indentadas siguientes como description
                        parts = []
                        i += 1
                        while i < len(lines) and (lines[i].startswith("  ") or lines[i].strip() == ""):
                            parts.append(lines[i].strip())
                            i += 1
                        metadata[key] = " ".join(p for p in parts if p)
                        continue
                    if value:
                        metadata[key] = value
                i += 1

    return metadata, body


def load_skill(skill_name: str) -> Optional[dict]:
    """
    Carga el contenido completo de una skill por nombre.

    Args:
        skill_name: nombre de la carpeta de la skill
                    (ej: 'playlist_generation', 'track_ingestion')

    Returns:
        dict con metadata y contenido, o None si no existe
    """
    skill_path = SKILLS_DIR / skill_name / "SKILL.md"

    if not skill_path.exists():
        return None

    content = skill_path.read_text(encoding="utf-8")
    metadata, body = _parse_frontmatter(content)

    return {
        "name": skill_name,
        "metadata": metadata,
        "content": body,
        "full_path": str(skill_path)
    }


def load_all_skills() -> list[dict]:
    """
    Carga el índice de todas las skills disponibles.
    Solo carga nombre y descripción (Tier 1 — liviano).
    """
    skills = []

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        content = skill_md.read_text(encoding="utf-8")
        metadata, body = _parse_frontmatter(content)

        # Extraer descripción limpia (remover saltos YAML)
        description = metadata.get("description", "")
        description = " ".join(description.split())

        skills.append({
            "name": skill_dir.name,
            "display_name": metadata.get("name", skill_dir.name),
            "description": description,
            "version": metadata.get("version", "1.0.0"),
            "tools": metadata.get("tools", "").split(", ") if metadata.get("tools") else []
        })

    return skills


# ─────────────────────────────────────────────
# SELECCIÓN DE SKILL
# ─────────────────────────────────────────────

# Keywords que activan cada skill
SKILL_TRIGGERS = {
    "playlist_generation": [
        "playlist", "canciones", "songs", "arco", "emocional",
        "viaje", "de triste", "de oscuro", "épico", "melancolía",
        "euforia", "relajar", "generar", "crear", "hazme",
        "quiero una", "necesito música"
    ],
    "track_ingestion": [
        "agrega", "añade", "incluye", "agregar", "añadir",
        "no está en", "falta", "quiero que esté", "add",
        "guardar canción", "nueva canción", "este track"
    ],
    "arc_validation": [
        "coherencia", "score", "validación", "transición",
        "violación", "brusco", "mejorar calidad", "análisis del arco",
        "qué tan buena", "es correcta", "funciona el arco",
        "tiene coherencia", "coherencia tiene", "validar", "revisar arco"
    ]
}


def select_skill(user_prompt: str) -> Optional[str]:
    """
    Selecciona la skill más relevante para el prompt del usuario.

    Returns:
        nombre de la carpeta de la skill, o None si no hay match claro
    """
    prompt_lower = user_prompt.lower()
    scores = {}

    for skill_name, keywords in SKILL_TRIGGERS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            scores[skill_name] = score

    if not scores:
        return None

    return max(scores, key=scores.get)


def get_skill_context(user_prompt: str) -> str:
    """
    Retorna el contenido de la skill relevante para inyectar
    en el contexto del agente.

    Si no hay skill relevante, retorna string vacío.
    """
    skill_name = select_skill(user_prompt)

    if not skill_name:
        return ""

    skill = load_skill(skill_name)
    if not skill:
        return ""

    return (
        f"\n\n## Skill activa: {skill['metadata'].get('name', skill_name)}\n\n"
        f"{skill['content']}"
    )


def get_skills_index() -> str:
    """
    Retorna un índice compacto de todas las skills disponibles.
    Para inyectar en el system prompt como referencia.
    """
    skills = load_all_skills()
    if not skills:
        return ""

    lines = ["\n## Skills disponibles\n"]
    for skill in skills:
        lines.append(f"- **{skill['display_name']}**: {skill['description'][:100]}...")

    return "\n".join(lines)
