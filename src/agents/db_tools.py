"""
agent/db_tools.py — narrative-playlist-agent
Tool: add_track_to_database

Permite al usuario enriquecer el dataset local con canciones nuevas.
El LLM estima los audio features — el usuario los confirma antes del INSERT.

Flujo:
  1. Verificar que el track no existe ya en la DB
  2. LLM propone los features estimados
  3. Usuario confirma o ajusta
  4. INSERT en SQLite con source='user_added'
"""

import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "tracks.db"

# Schema JSON para el LLM
ADD_TRACK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_track_to_database",
        "description": (
            "Añade una canción al dataset local con sus audio features. "
            "Usar cuando el usuario pide una canción que no está en la DB. "
            "El LLM debe estimar los features basado en su conocimiento musical "
            "y presentarlos al usuario para confirmación antes de llamar esta tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "track_name": {
                    "type": "string",
                    "description": "Nombre exacto de la canción."
                },
                "artist": {
                    "type": "string",
                    "description": "Nombre del artista o banda."
                },
                "valence": {
                    "type": "number",
                    "description": (
                        "Positividad musical (0.0–1.0). "
                        "0.0=muy triste/oscuro, 1.0=muy alegre/eufórico."
                    )
                },
                "energy": {
                    "type": "number",
                    "description": (
                        "Intensidad (0.0–1.0). "
                        "0.0=suave/acústico, 1.0=ruidoso/intenso."
                    )
                },
                "tempo": {
                    "type": "number",
                    "description": "BPM estimado de la canción."
                },
                "acousticness": {
                    "type": "number",
                    "description": (
                        "Nivel acústico (0.0–1.0). "
                        "1.0=completamente acústico, 0.0=completamente electrónico."
                    )
                },
                "danceability": {
                    "type": "number",
                    "description": "Aptitud para el baile (0.0–1.0)."
                },
                "instrumentalness": {
                    "type": "number",
                    "description": (
                        "Probabilidad de ausencia de vocales (0.0–1.0). "
                        "Valores > 0.5 sugieren tracks instrumentales."
                    )
                },
                "track_genre": {
                    "type": "string",
                    "description": (
                        "Género musical en lowercase con guiones. "
                        "Ej: 'pop', 'rock', 'hip-hop', 'classical', 'electronic'."
                    )
                }
            },
            "required": [
                "track_name", "artist",
                "valence", "energy", "tempo", "acousticness"
            ]
        }
    }
}

# Schema para verificar si existe
CHECK_TRACK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_track_exists",
        "description": (
            "Verifica si una canción ya existe en la base de datos local. "
            "Llamar ANTES de add_track_to_database para evitar duplicados."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "track_name": {"type": "string"},
                "artist": {"type": "string"}
            },
            "required": ["track_name", "artist"]
        }
    }
}


# ─────────────────────────────────────────────
# FUNCIONES CORE
# ─────────────────────────────────────────────

def _get_next_track_id(conn: sqlite3.Connection) -> str:
    """Genera el próximo track_id para tracks añadidos por el usuario."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM tracks WHERE source = 'user_added'"
    )
    count = cursor.fetchone()[0]
    # Formato: track_u000001, track_u000002, ...
    return f"track_u{str(count + 1).zfill(6)}"


def check_track_exists(raw_args: dict, db_path: Path = DEFAULT_DB_PATH) -> str:
    """
    Verifica si un track ya existe en la DB.
    Retorna JSON con found: bool y track_id si existe.
    """
    try:
        track_name = raw_args.get("track_name", "").strip().lower()
        artist = raw_args.get("artist", "").strip().lower()

        if not db_path.exists():
            return json.dumps({
                "found": False,
                "error": "DB no encontrada. Ejecuta load_dataset.py primero."
            })

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT track_id, track_name, artist, valence, energy, tempo
            FROM tracks
            WHERE LOWER(track_name) = ? AND LOWER(artist) = ?
            LIMIT 1
            """,
            (track_name, artist)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return json.dumps({
                "found": True,
                "track_id": row[0],
                "track_name": row[1],
                "artist": row[2],
                "valence": row[3],
                "energy": row[4],
                "tempo": row[5],
                "message": f"'{raw_args['track_name']}' de {raw_args['artist']} "
                           f"ya existe en la DB con ID {row[0]}."
            })

        return json.dumps({
            "found": False,
            "message": f"'{raw_args['track_name']}' de {raw_args['artist']} "
                       f"no está en la DB. Puedes añadirla con add_track_to_database."
        })

    except Exception as e:
        return json.dumps({"found": False, "error": str(e)})


def add_track_to_database(
    raw_args: dict,
    db_path: Path = DEFAULT_DB_PATH
) -> str:
    """
    Inserta un track nuevo en la DB SQLite con source='user_added'.

    El LLM debe haber:
      1. Verificado con check_track_exists que no existe
      2. Estimado los features y presentado al usuario para confirmación
      3. Recibido confirmación del usuario antes de llamar esta función

    Retorna JSON string con resultado.
    """
    try:
        # Validar campos obligatorios
        required = ["track_name", "artist", "valence", "energy", "tempo", "acousticness"]
        missing = [f for f in required if f not in raw_args]
        if missing:
            return json.dumps({
                "success": False,
                "error": f"Campos obligatorios faltantes: {missing}"
            })

        # Validar rangos
        range_fields = ["valence", "energy", "acousticness"]
        for field in range_fields:
            val = raw_args.get(field)
            if val is not None and not (0.0 <= float(val) <= 1.0):
                return json.dumps({
                    "success": False,
                    "error": f"{field} debe estar entre 0.0 y 1.0. "
                             f"Recibido: {val}"
                })

        if not db_path.exists():
            return json.dumps({
                "success": False,
                "error": "DB no encontrada. Ejecuta load_dataset.py primero."
            })

        conn = sqlite3.connect(db_path)

        # Verificar columna source — añadirla si no existe
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(tracks)")
        columns = {row[1] for row in cursor.fetchall()}
        if "source" not in columns:
            cursor.execute(
                "ALTER TABLE tracks ADD COLUMN source TEXT DEFAULT 'kaggle'"
            )
            conn.commit()

        track_id = _get_next_track_id(conn)

        cursor.execute(
            """
            INSERT INTO tracks (
                track_id, track_name, artist,
                valence, energy, tempo,
                acousticness, danceability, instrumentalness,
                track_genre, popularity, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                track_id,
                raw_args["track_name"].strip(),
                raw_args["artist"].strip(),
                float(raw_args["valence"]),
                float(raw_args["energy"]),
                float(raw_args["tempo"]),
                float(raw_args.get("acousticness", 0.5)),
                float(raw_args.get("danceability", 0.5)),
                float(raw_args.get("instrumentalness", 0.0)),
                raw_args.get("track_genre", "unknown").lower().strip(),
                0,           # popularity — desconocida para tracks nuevos
                "user_added"
            )
        )
        conn.commit()
        conn.close()

        return json.dumps({
            "success": True,
            "track_id": track_id,
            "track_name": raw_args["track_name"],
            "artist": raw_args["artist"],
            "source": "user_added",
            "message": (
                f"'{raw_args['track_name']}' de {raw_args['artist']} "
                f"añadida al dataset con ID {track_id}. "
                f"Ya disponible para futuros arcos narrativos."
            )
        })

    except sqlite3.IntegrityError:
        return json.dumps({
            "success": False,
            "error": f"El track '{raw_args.get('track_name')}' de "
                     f"'{raw_args.get('artist')}' ya existe en la DB."
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error al añadir track: {e}"
        })


def get_user_added_tracks(db_path: Path = DEFAULT_DB_PATH) -> str:
    """
    Lista todos los tracks añadidos por el usuario.
    Útil para que el agente informe al usuario qué ha añadido.
    """
    try:
        if not db_path.exists():
            return json.dumps({"tracks": [], "count": 0})

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Verificar si la columna source existe
        cursor.execute("PRAGMA table_info(tracks)")
        columns = {row[1] for row in cursor.fetchall()}
        if "source" not in columns:
            return json.dumps({"tracks": [], "count": 0})

        cursor.execute(
            """
            SELECT track_id, track_name, artist, valence, energy, tempo
            FROM tracks
            WHERE source = 'user_added'
            ORDER BY track_id
            """
        )
        rows = cursor.fetchall()
        conn.close()

        tracks = [
            {
                "track_id": r[0],
                "track_name": r[1],
                "artist": r[2],
                "valence": r[3],
                "energy": r[4],
                "tempo": r[5]
            }
            for r in rows
        ]

        return json.dumps({
            "tracks": tracks,
            "count": len(tracks)
        })

    except Exception as e:
        return json.dumps({"tracks": [], "error": str(e)})
