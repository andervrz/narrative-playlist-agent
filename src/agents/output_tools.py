"""
agent/output_tools.py — narrative-playlist-agent
Tool: generate_playlist_file

Genera dos archivos de output a partir de la playlist del agente:
  - CSV  → para subir a TuneMyMusic (compatible con cualquier plataforma)
  - JSON → registro completo con narrativa, features y arc_statistics

Esta tool es llamada por el agente al final del flujo,
cuando todas las fases tienen tracks asignados y validados.
"""

import json
import csv
import re
import unicodedata
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Schema JSON de la tool para pasarla al LLM
GENERATE_PLAYLIST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_playlist_file",
        "description": (
            "Genera los archivos de output de la playlist: "
            "CSV para TuneMyMusic y JSON con la narrativa completa. "
            "Llamar SOLO cuando todas las fases tengan tracks asignados. "
            "Retorna los paths de los archivos generados."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "playlist_title": {
                    "type": "string",
                    "description": "Título de la playlist generado por el agente."
                },
                "playlist_description": {
                    "type": "string",
                    "description": "Descripción breve del arco emocional."
                },
                "phases": {
                    "type": "array",
                    "description": "Lista de fases con sus tracks y explicaciones.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "phase_number": {"type": "integer"},
                            "phase_label": {"type": "string"},
                            "emotional_description": {"type": "string"},
                            "valence_range": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
                            "energy_range": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
                            "tracks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "position": {"type": "integer"},
                                        "track_name": {"type": "string"},
                                        "artist": {"type": "string"},
                                        "valence": {"type": "number"},
                                        "energy": {"type": "number"},
                                        "tempo": {"type": "number"},
                                        "genre": {"type": "string"},
                                        "transition_note": {"type": "string"}
                                    }
                                }
                            },
                            "phase_explanation": {"type": "string"}
                        },
                        "required": [
                            "phase_number", "phase_label", "tracks",
                            "phase_explanation"
                        ]
                    }
                },
                "arc_summary": {
                    "type": "string",
                    "description": (
                        "Párrafo final que describe el viaje emocional completo. "
                        "Obligatorio — mínimo 50 caracteres."
                    )
                }
            },
            "required": [
                "playlist_title", "playlist_description",
                "phases", "arc_summary"
            ]
        }
    }
}


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convierte un título en slug ASCII para nombres de archivo."""
    text = text.lower().strip()
    # Normaliza y elimina acentos: "épica" → "epica", "canción" → "cancion"
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    text = text.strip('_')
    return text[:50] or "playlist"


def _build_filename(title: str) -> str:
    """Genera nombre de archivo con timestamp."""
    slug = _slugify(title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"playlist_{slug}_{timestamp}"


def _flatten_tracks(phases: list[dict]) -> list[dict]:
    """Extrae todos los tracks de todas las fases en orden de posición."""
    tracks = []
    for phase in phases:
        for track in phase.get("tracks", []):
            track["phase_label"] = phase.get("phase_label", "")
            tracks.append(track)
    return sorted(tracks, key=lambda t: t.get("position", 0))


# ─────────────────────────────────────────────
# GENERADOR CSV — para TuneMyMusic
# ─────────────────────────────────────────────

def _write_csv(tracks: list[dict], filepath: Path) -> None:
    """
    Genera CSV compatible con TuneMyMusic.
    Formato: Title, Artist (dos columnas — el mínimo requerido)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Artist"])
        for track in tracks:
            writer.writerow([
                track.get("track_name", ""),
                track.get("artist", "")
            ])


# ─────────────────────────────────────────────
# GENERADOR JSON — registro completo
# ─────────────────────────────────────────────

def _compute_arc_statistics(tracks: list[dict]) -> dict:
    """Calcula estadísticas del arco desde los tracks."""
    if not tracks:
        return {}

    valence_vals = [t.get("valence", 0) for t in tracks]
    energy_vals = [t.get("energy", 0) for t in tracks]

    def slope_direction(vals: list[float]) -> str:
        if len(vals) < 2:
            return "neutral"
        delta = vals[-1] - vals[0]
        if delta > 0.1:
            return "positive"
        elif delta < -0.1:
            return "negative"
        return "neutral"

    return {
        "valence_start": round(valence_vals[0], 3),
        "valence_end": round(valence_vals[-1], 3),
        "energy_start": round(energy_vals[0], 3),
        "energy_end": round(energy_vals[-1], 3),
        "valence_slope": slope_direction(valence_vals),
        "energy_slope": slope_direction(energy_vals),
        "total_tracks": len(tracks)
    }


def _write_json(
    title: str,
    description: str,
    phases: list[dict],
    arc_summary: str,
    tracks: list[dict],
    filepath: Path
) -> None:
    """Genera JSON con el registro completo de la playlist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    arc_stats = _compute_arc_statistics(tracks)

    payload = {
        "playlist_title": title,
        "playlist_description": description,
        "generated_at": datetime.now().isoformat(),
        "total_tracks": len(tracks),
        "phases": phases,
        "arc_statistics": arc_stats,
        "arc_summary": arc_summary,
        "tracks_flat": [
            {
                "position": t.get("position"),
                "track_name": t.get("track_name"),
                "artist": t.get("artist"),
                "valence": t.get("valence"),
                "energy": t.get("energy"),
                "tempo": t.get("tempo"),
                "genre": t.get("genre"),
                "phase": t.get("phase_label"),
                "transition_note": t.get("transition_note", "")
            }
            for t in tracks
        ]
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# ENTRY POINT PÚBLICO
# ─────────────────────────────────────────────

def generate_playlist_file(raw_args: dict) -> str:
    """
    Función principal llamada por el agente.
    Genera CSV + JSON y retorna JSON string con los paths.

    Args:
        raw_args: dict con los argumentos del LLM

    Returns:
        JSON string con resultado — nunca lanza excepción al agente
    """
    try:
        title = raw_args.get("playlist_title", "playlist")
        description = raw_args.get("playlist_description", "")
        phases = raw_args.get("phases", [])
        arc_summary = raw_args.get("arc_summary", "")

        if not phases:
            return json.dumps({
                "success": False,
                "error": "No hay fases en la playlist. "
                         "Asegúrate de llamar query_song_database primero."
            })

        tracks = _flatten_tracks(phases)
        if not tracks:
            return json.dumps({
                "success": False,
                "error": "No hay tracks en las fases. "
                         "Verifica que query_song_database retornó resultados."
            })

        filename = _build_filename(title)
        csv_path = OUTPUT_DIR / f"{filename}.csv"
        json_path = OUTPUT_DIR / f"{filename}.json"

        _write_csv(tracks, csv_path)
        _write_json(title, description, phases, arc_summary, tracks, json_path)

        return json.dumps({
            "success": True,
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "total_tracks": len(tracks),
            "total_phases": len(phases),
            "tunemymusic_instructions": (
                "1. Ve a tunemymusic.com → selecciona File como fuente "
                "→ sube el archivo .csv "
                "→ elige tu plataforma de streaming favorita."
            )
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error generando archivos: {e}"
        })
