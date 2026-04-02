"""
agent/tools.py — narrative-playlist-agent
Implementación de query_song_database: la única herramienta del agente.

Responsabilidades:
  - Recibir los argumentos del LLM (validados por QueryArgs)
  - Construir la query SQL de forma segura
  - Ejecutar contra tracks.db
  - Activar retry_logic si hay 0 resultados
  - Retornar QueryResult con los tracks encontrados

El LLM NO construye SQL directamente.
El LLM llama a esta función con parámetros estructurados.
Python construye el SQL. Esto es deliberado — ver ARCHITECTURE.md sección 6.
"""

import sqlite3
import random
import json
from pathlib import Path
from typing import Optional

from src.schemas.models import QueryArgs, QueryResult


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "tracks.db"

# Columnas que retorna la tool al LLM
# Solo las necesarias — no enviamos columnas internas innecesarias
RETURN_COLUMNS = [
    "track_id", "track_name", "artist",
    "valence", "energy", "tempo",
    "danceability", "acousticness", "track_genre"
]

# Cuánto ampliar los rangos en cada intento de retry
RETRY_RELAXATION_STEP = 0.1
MAX_RETRY_ATTEMPTS = 3


# ─────────────────────────────────────────────
# TOOL SCHEMA — define la herramienta para el LLM
# Este dict se pasa directamente a la Groq/OpenAI API
# ─────────────────────────────────────────────

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_song_database",
        "description": (
            "Busca canciones en la base de datos local usando rangos de audio features. "
            "Retorna tracks reales de la DB. "
            "REGLA CRÍTICA: Solo usar canciones retornadas por esta herramienta. "
            "Nunca inventar tracks que no hayan sido retornados aquí."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "min_valence": {
                    "type": "number",
                    "description": "Positividad musical mínima (0.0–1.0). 0.0=muy triste, 1.0=muy alegre."
                },
                "max_valence": {
                    "type": "number",
                    "description": "Positividad musical máxima (0.0–1.0)."
                },
                "min_energy": {
                    "type": "number",
                    "description": "Intensidad mínima (0.0–1.0). 0.0=suave/acústico, 1.0=ruidoso/rápido."
                },
                "max_energy": {
                    "type": "number",
                    "description": "Intensidad máxima (0.0–1.0)."
                },
                "min_tempo": {
                    "type": "number",
                    "description": "BPM mínimo. Opcional. Usar cuando el arco requiera control de ritmo."
                },
                "max_tempo": {
                    "type": "number",
                    "description": "BPM máximo. Opcional."
                },
                "min_acousticness": {
                    "type": "number",
                    "description": "Acousticness mínima (0.0–1.0). Útil para fases de paz o melancolía."
                },
                "target_genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Géneros para filtrar. Ej: ['rock', 'classical']. Omitir = todos los géneros."
                },
                "exclude_track_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs de tracks ya en la playlist. Evita duplicados entre fases."
                },
                "limit": {
                    "type": "integer",
                    "description": "Cantidad de canciones a retornar. Default: 3. Máximo: 10.",
                    "default": 3
                }
            },
            "required": ["min_valence", "max_valence", "min_energy", "max_energy", "limit"]
        }
    }
}


# ─────────────────────────────────────────────
# CORE: CONSTRUCCIÓN DEL SQL
# ─────────────────────────────────────────────

def build_query(args: QueryArgs) -> tuple[str, list]:
    """
    Construye la query SQL y los parámetros de forma segura (parametrized).
    Nunca interpola valores directamente en el string SQL.

    Retorna:
        (sql_string, params_list) para usar con cursor.execute(sql, params)
    """
    conditions = []
    params = []

    # Filtros obligatorios
    conditions.append("valence BETWEEN ? AND ?")
    params.extend([args.min_valence, args.max_valence])

    conditions.append("energy BETWEEN ? AND ?")
    params.extend([args.min_energy, args.max_energy])

    # Filtros opcionales
    if args.min_tempo is not None:
        conditions.append("tempo >= ?")
        params.append(args.min_tempo)

    if args.max_tempo is not None:
        conditions.append("tempo <= ?")
        params.append(args.max_tempo)

    if args.min_acousticness is not None:
        conditions.append("acousticness >= ?")
        params.append(args.min_acousticness)

    if args.target_genres:
        # Géneros normalizados a lowercase antes de comparar
        normalized = [g.lower().strip() for g in args.target_genres]
        placeholders = ",".join("?" * len(normalized))
        conditions.append(f"track_genre IN ({placeholders})")
        params.extend(normalized)

    if args.exclude_track_ids:
        placeholders = ",".join("?" * len(args.exclude_track_ids))
        conditions.append(f"track_id NOT IN ({placeholders})")
        params.extend(args.exclude_track_ids)

    # Columnas a retornar
    cols = ", ".join(RETURN_COLUMNS)
    where_clause = " AND ".join(conditions)

    # ORDER BY RANDOM() para variedad — el agente no recibe siempre las mismas canciones
    sql = f"""
        SELECT {cols}
        FROM tracks
        WHERE {where_clause}
        ORDER BY RANDOM()
        LIMIT ?
    """
    params.append(args.limit)

    return sql.strip(), params


def relax_args(args: QueryArgs, step: float = RETRY_RELAXATION_STEP) -> QueryArgs:
    """
    Amplía los rangos de valence y energy en ±step para el retry.
    Clampea en [0.0, 1.0] para no salir del rango válido.
    Conserva todos los demás filtros (tempo, genre, etc.) sin cambios.
    """
    new_min_v = max(0.0, round(args.min_valence - step, 3))
    new_max_v = min(1.0, round(args.max_valence + step, 3))
    new_min_e = max(0.0, round(args.min_energy - step, 3))
    new_max_e = min(1.0, round(args.max_energy + step, 3))

    return QueryArgs(
        min_valence=new_min_v,
        max_valence=new_max_v,
        min_energy=new_min_e,
        max_energy=new_max_e,
        min_tempo=args.min_tempo,
        max_tempo=args.max_tempo,
        min_acousticness=args.min_acousticness,
        target_genres=args.target_genres,
        exclude_track_ids=args.exclude_track_ids,
        limit=args.limit
    )


# ─────────────────────────────────────────────
# CORE: EJECUCIÓN DE QUERY
# ─────────────────────────────────────────────

def execute_query(args: QueryArgs, db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """
    Ejecuta la query SQL contra SQLite.
    Retorna lista de dicts con la metadata de los tracks.
    Retorna lista vacía si no hay resultados (el llamador maneja esto).
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"Base de datos no encontrada: {db_path}\n"
            f"Ejecuta primero: python src/ingestion/load_dataset.py"
        )

    sql, params = build_query(args)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Permite acceder por nombre de columna
    cursor = conn.cursor()

    try:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        # Convertir sqlite3.Row a dict para serialización JSON
        results = [dict(row) for row in rows]
    finally:
        conn.close()

    return results


# ─────────────────────────────────────────────
# CORE: RETRY LOGIC
# ─────────────────────────────────────────────

def query_with_retry(
    args: QueryArgs,
    db_path: Path = DEFAULT_DB_PATH,
    max_attempts: int = MAX_RETRY_ATTEMPTS
) -> QueryResult:
    """
    Ejecuta la query con retry automático si hay 0 resultados.

    Estrategia de retry:
      Intento 1: args originales
      Intento 2: amplía rangos en ±0.1
      Intento 3: amplía rangos en ±0.2 (acumulado)

    Si después de max_attempts sigue vacío:
      Retorna QueryResult con success=False y error descriptivo.
    """
    current_args = args
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            tracks = execute_query(current_args, db_path)

            if tracks:
                return QueryResult(
                    success=True,
                    tracks=tracks,
                    attempts=attempt,
                    final_args=current_args
                )

            # 0 resultados — preparar siguiente intento
            if attempt < max_attempts:
                current_args = relax_args(args, step=RETRY_RELAXATION_STEP * attempt)
            else:
                last_error = (
                    f"0 resultados después de {max_attempts} intentos. "
                    f"Rangos finales: V({current_args.min_valence:.2f}–{current_args.max_valence:.2f}), "
                    f"E({current_args.min_energy:.2f}–{current_args.max_energy:.2f}). "
                    f"Considera ajustar la emoción de esta fase."
                )

        except FileNotFoundError as e:
            return QueryResult(
                success=False,
                tracks=[],
                attempts=attempt,
                error=str(e)
            )
        except sqlite3.Error as e:
            last_error = f"Error SQLite en intento {attempt}: {e}"

    return QueryResult(
        success=False,
        tracks=[],
        attempts=max_attempts,
        final_args=current_args,
        error=last_error
    )


# ─────────────────────────────────────────────
# ENTRY POINT PÚBLICO — lo que llama el agente
# ─────────────────────────────────────────────

def query_song_database(raw_args: dict, db_path: Path = DEFAULT_DB_PATH) -> str:
    """
    Función principal que ejecuta el agente cuando el LLM hace un tool call.

    Args:
        raw_args: dict con los argumentos del LLM (sin validar)
        db_path:  path a la DB SQLite

    Returns:
        JSON string — el agente recibe este string como tool_result
        Siempre retorna JSON válido, nunca lanza excepción al agente.
    """
    try:
        # Validar argumentos con Pydantic antes de tocar la DB
        args = QueryArgs(**raw_args)
    except Exception as e:
        return json.dumps({
            "success": False,
            "tracks": [],
            "error": f"Argumentos inválidos: {e}. Revisa los rangos de valence y energy."
        })

    result = query_with_retry(args, db_path=db_path)

    if result.success:
        return json.dumps({
            "success": True,
            "tracks": result.tracks,
            "count": len(result.tracks),
            "attempts": result.attempts,
            "note": (
                f"Se encontraron {len(result.tracks)} tracks."
                + (f" (parámetros relajados en intento {result.attempts})" if result.attempts > 1 else "")
            )
        })
    else:
        return json.dumps({
            "success": False,
            "tracks": [],
            "count": 0,
            "attempts": result.attempts,
            "error": result.error
        })


# ─────────────────────────────────────────────
# UTILIDAD: verificar DB disponible
# ─────────────────────────────────────────────

def check_db_ready(db_path: Path = DEFAULT_DB_PATH) -> tuple[bool, str]:
    """
    Verifica que la DB esté disponible y tenga el schema correcto.
    Llamado al iniciar el agente — falla rápido con mensaje claro.
    """
    if not db_path.exists():
        return False, (
            f"DB no encontrada: {db_path}\n"
            f"Ejecuta: python src/ingestion/load_dataset.py"
        )

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks")
        count = cursor.fetchone()[0]

        # Verificar columnas requeridas
        cursor.execute("PRAGMA table_info(tracks)")
        cols = {row[1] for row in cursor.fetchall()}
        required = {"track_id", "track_name", "artist", "valence", "energy", "tempo", "track_genre"}
        missing = required - cols

        conn.close()

        if missing:
            return False, f"Columnas faltantes en la DB: {missing}"

        return True, f"DB lista: {count:,} tracks disponibles"

    except sqlite3.Error as e:
        return False, f"Error al verificar DB: {e}"
