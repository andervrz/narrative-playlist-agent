"""
ingestion/load_dataset.py — narrative-playlist-agent
Lee el CSV de Kaggle, lo limpia y lo carga en SQLite con índices.
Script de una sola ejecución — no forma parte del agent loop.

Uso:
    python src/ingestion/load_dataset.py
    python src/ingestion/load_dataset.py --csv data/raw/mi_archivo.csv
    python src/ingestion/load_dataset.py --verify-only
"""

import sqlite3
import argparse
import sys
import os
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint

console = Console()

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "tracks.db"

# Columnas que necesita el agente (subset del dataset de Kaggle)
REQUIRED_COLUMNS = [
    "track_name", "artists", "track_genre",
    "valence", "energy", "danceability",
    "acousticness", "instrumentalness", "tempo", "popularity"
]

# Mapeo de nombres de columnas del CSV de Kaggle a los nombres internos
COLUMN_RENAME_MAP = {
    "artists": "artist",
}

# Géneros a normalizar: reemplazos conocidos
GENRE_NORMALIZATIONS = {
    "hip-hop": "hip-hop",
    "hip_hop": "hip-hop",
    "r&b": "r-and-b",
    "r-n-b": "r-and-b",
    "rnb": "r-and-b",
    "k-pop": "k-pop",
    "k_pop": "k-pop",
    "alt-rock": "alt-rock",
    "alternative rock": "alt-rock",
}


# ─────────────────────────────────────────────
# PASO 1: CARGA DEL CSV
# ─────────────────────────────────────────────

def load_csv(csv_path: Path) -> pd.DataFrame:
    """Carga el CSV y valida que las columnas requeridas existan."""
    console.rule("[bold blue]PASO 1 — Carga del CSV[/bold blue]")

    if not csv_path.exists():
        console.print(f"\n[red]❌ Archivo no encontrado:[/red] {csv_path}")
        console.print("\n[yellow]📥 Descarga el dataset aquí:[/yellow]")
        console.print("   https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
        console.print(f"\n   Luego colócalo en: [cyan]{DEFAULT_CSV_PATH}[/cyan]")
        sys.exit(1)

    console.print(f"[green]✓[/green] Leyendo {csv_path}...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Cargando CSV...", total=None)
        df = pd.read_csv(csv_path, low_memory=False)
        progress.update(task, description=f"CSV cargado: {len(df):,} filas")

    console.print(f"[green]✓[/green] {len(df):,} registros cargados")
    console.print(f"[green]✓[/green] Columnas disponibles: {list(df.columns)}\n")

    # Validar columnas requeridas
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        console.print(f"[red]❌ Columnas faltantes en el CSV:[/red] {missing}")
        console.print("[yellow]Verifica que estás usando el dataset correcto de Kaggle.[/yellow]")
        sys.exit(1)

    console.print(f"[green]✓[/green] Todas las columnas requeridas presentes\n")
    return df


# ─────────────────────────────────────────────
# PASO 2: LIMPIEZA DE DATOS
# ─────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza el DataFrame.
    - Filtra solo las columnas necesarias
    - Renombra columnas
    - Dropea nulls en columnas críticas
    - Normaliza géneros
    - Filtra rangos válidos de audio features
    """
    console.rule("[bold blue]PASO 2 — Limpieza de datos[/bold blue]")
    initial_count = len(df)

    # Filtrar solo columnas necesarias
    df = df[REQUIRED_COLUMNS].copy()

    # Renombrar columnas
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Drop nulls en columnas críticas
    critical_cols = ["track_name", "artist", "valence", "energy", "tempo", "track_genre"]
    before_drop = len(df)
    df = df.dropna(subset=critical_cols)
    dropped_nulls = before_drop - len(df)
    console.print(f"[green]✓[/green] Nulls removidos: {dropped_nulls:,} filas")

    # Convertir tipos numéricos
    float_cols = ["valence", "energy", "danceability", "acousticness", "instrumentalness"]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)

    # Filtrar rangos válidos de audio features (0.0 a 1.0)
    before_range = len(df)
    for col in float_cols:
        df = df[(df[col] >= 0.0) & (df[col] <= 1.0)]
    df = df[df["tempo"] > 0]
    dropped_range = before_range - len(df)
    console.print(f"[green]✓[/green] Filas con rangos inválidos removidas: {dropped_range:,}")

    # Drop nulls generados por conversión
    df = df.dropna()

    # Normalizar géneros
    df["track_genre"] = df["track_genre"].str.lower().str.strip()
    df["track_genre"] = df["track_genre"].replace(GENRE_NORMALIZATIONS)
    genre_count = df["track_genre"].nunique()
    console.print(f"[green]✓[/green] Géneros únicos después de normalización: {genre_count}")

    # Generar track_id único
    df = df.reset_index(drop=True)
    df.insert(0, "track_id", ["track_" + str(i).zfill(6) for i in df.index])

    # Limpiar strings
    df["track_name"] = df["track_name"].str.strip()
    df["artist"] = df["artist"].str.strip()

    # Eliminar duplicados exactos (mismo track_name + artist)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["track_name", "artist"], keep="first")
    dropped_dedup = before_dedup - len(df)
    console.print(f"[green]✓[/green] Duplicados removidos: {dropped_dedup:,}")

    final_count = len(df)
    console.print(f"\n[bold green]Resumen de limpieza:[/bold green]")
    console.print(f"  Registros originales : {initial_count:>8,}")
    console.print(f"  Registros finales    : {final_count:>8,}")
    console.print(f"  Removidos (total)    : {initial_count - final_count:>8,}")
    console.print(f"  Tasa de retención    : {final_count/initial_count:.1%}\n")

    return df


# ─────────────────────────────────────────────
# PASO 3: CARGA EN SQLITE
# ─────────────────────────────────────────────

def load_to_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    """
    Carga el DataFrame en SQLite y crea índices para queries rápidas.
    Los índices en valence, energy y tempo son críticos
    para el rendimiento de las queries BETWEEN en 600k rows.
    """
    console.rule("[bold blue]PASO 3 — Carga en SQLite[/bold blue]")

    # Crear directorio si no existe
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Si la DB ya existe, advertir y sobreescribir
    if db_path.exists():
        console.print(f"[yellow]⚠️  DB existente detectada — será sobreescrita[/yellow]")
        db_path.unlink()

    console.print(f"[green]✓[/green] Creando DB en {db_path}...")

    conn = sqlite3.connect(db_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Cargando datos...", total=None)

        # Cargar datos
        df.to_sql("tracks", conn, if_exists="replace", index=False)
        progress.update(task, description=f"{len(df):,} tracks cargados en SQLite")

    # Crear índices — crítico para rendimiento
    console.print("[green]✓[/green] Creando índices...")
    indexes = [
        ("idx_valence", "CREATE INDEX IF NOT EXISTS idx_valence ON tracks(valence)"),
        ("idx_energy", "CREATE INDEX IF NOT EXISTS idx_energy ON tracks(energy)"),
        ("idx_tempo", "CREATE INDEX IF NOT EXISTS idx_tempo ON tracks(tempo)"),
        ("idx_genre", "CREATE INDEX IF NOT EXISTS idx_genre ON tracks(track_genre)"),
        ("idx_acousticness", "CREATE INDEX IF NOT EXISTS idx_acousticness ON tracks(acousticness)"),
        ("idx_danceability", "CREATE INDEX IF NOT EXISTS idx_danceability ON tracks(danceability)"),
        # Índice compuesto para las queries más comunes del agente
        ("idx_val_energy", "CREATE INDEX IF NOT EXISTS idx_val_energy ON tracks(valence, energy)"),
    ]

    for name, sql in indexes:
        conn.execute(sql)
        console.print(f"  [dim]✓ {name}[/dim]")

    conn.commit()
    conn.close()

    # Tamaño del archivo generado
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    console.print(f"\n[green]✓[/green] DB creada: {db_path}")
    console.print(f"[green]✓[/green] Tamaño: {db_size_mb:.1f} MB\n")


# ─────────────────────────────────────────────
# PASO 4: VERIFICACIÓN
# ─────────────────────────────────────────────

def verify_db(db_path: Path) -> bool:
    """
    Verifica que la DB esté correctamente construida.
    Ejecuta queries de prueba que simulan lo que hará el agente.
    """
    console.rule("[bold blue]PASO 4 — Verificación[/bold blue]")

    if not db_path.exists():
        console.print(f"[red]❌ DB no encontrada: {db_path}[/red]")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Conteo total
    cursor.execute("SELECT COUNT(*) FROM tracks")
    total = cursor.fetchone()[0]
    console.print(f"[green]✓[/green] Total de tracks: {total:,}")

    # 2. Verificar columnas
    cursor.execute("PRAGMA table_info(tracks)")
    columns = [row[1] for row in cursor.fetchall()]
    console.print(f"[green]✓[/green] Columnas: {columns}")

    # 3. Distribución de audio features
    cursor.execute("""
        SELECT
            ROUND(AVG(valence), 3)     as avg_valence,
            ROUND(AVG(energy), 3)      as avg_energy,
            ROUND(AVG(tempo), 1)       as avg_tempo,
            ROUND(AVG(danceability), 3) as avg_dance,
            ROUND(AVG(acousticness), 3) as avg_acoustic
        FROM tracks
    """)
    row = cursor.fetchone()
    table = Table(title="Estadísticas del dataset")
    table.add_column("Métrica", style="cyan")
    table.add_column("Promedio", style="green")
    metrics = ["valence", "energy", "tempo", "danceability", "acousticness"]
    for metric, val in zip(metrics, row):
        table.add_row(metric, str(val))
    console.print(table)

    # 4. Queries de prueba que simulan al agente
    console.print("\n[bold]Queries de prueba (simulando al agente):[/bold]")

    test_queries = [
        ("Melancolía (V:0-0.3, E:0-0.4)",
         "SELECT COUNT(*) FROM tracks WHERE valence BETWEEN 0.0 AND 0.3 AND energy BETWEEN 0.0 AND 0.4"),
        ("Intensidad (V:0-0.4, E:0.8-1.0)",
         "SELECT COUNT(*) FROM tracks WHERE valence BETWEEN 0.0 AND 0.4 AND energy BETWEEN 0.8 AND 1.0"),
        ("Euforia (V:0.7-1.0, E:0.7-1.0)",
         "SELECT COUNT(*) FROM tracks WHERE valence BETWEEN 0.7 AND 1.0 AND energy BETWEEN 0.7 AND 1.0"),
        ("Paz (V:0.5-1.0, E:0-0.3)",
         "SELECT COUNT(*) FROM tracks WHERE valence BETWEEN 0.5 AND 1.0 AND energy BETWEEN 0.0 AND 0.3"),
    ]

    all_ok = True
    for label, sql in test_queries:
        cursor.execute(sql)
        count = cursor.fetchone()[0]
        status = "[green]✓[/green]" if count > 10 else "[red]⚠️[/red]"
        console.print(f"  {status} {label}: {count:,} canciones disponibles")
        if count < 10:
            all_ok = False

    # 5. Géneros disponibles
    cursor.execute("""
        SELECT track_genre, COUNT(*) as cnt
        FROM tracks
        GROUP BY track_genre
        ORDER BY cnt DESC
        LIMIT 10
    """)
    genres = cursor.fetchall()
    console.print(f"\n[green]✓[/green] Top 10 géneros:")
    for genre, count in genres:
        console.print(f"  [dim]{genre}: {count:,}[/dim]")

    conn.close()

    if all_ok:
        console.print(f"\n[bold green]✅ Base de datos lista para el agente.[/bold green]")
        console.print(f"   Path: [cyan]{db_path}[/cyan]\n")
    else:
        console.print(f"\n[yellow]⚠️  Algunas queries retornan pocos resultados. "
                      f"El agente podría tener dificultades en ciertas emociones.[/yellow]\n")

    return all_ok


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Carga el dataset de Kaggle en SQLite para narrative-playlist-agent"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path al CSV de Kaggle (default: {DEFAULT_CSV_PATH})"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path de la DB SQLite a generar (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Solo verifica una DB existente, no procesa el CSV"
    )
    args = parser.parse_args()

    console.print("\n[bold cyan]narrative-playlist-agent[/bold cyan] — Data Ingestion\n")

    if args.verify_only:
        verify_db(args.db)
        return

    # Pipeline completo
    df = load_csv(args.csv)
    df = clean_dataframe(df)
    load_to_sqlite(df, args.db)
    verify_db(args.db)


if __name__ == "__main__":
    main()
