"""
tests/conftest.py — narrative-playlist-agent
Fixtures compartidos entre todos los test files.

El fixture db_path crea una SQLite en memoria con 50 tracks sintéticos
que cubren todos los rangos emocionales del agente.
Esto permite testear tools.py sin depender del dataset real de Kaggle.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest


# Tracks sintéticos diseñados para cubrir todos los rangos emocionales
SYNTHETIC_TRACKS = [
    # Melancolía (valence 0.0–0.3, energy 0.0–0.4)
    ("track_000001", "Sad Song 1",    "Artist A", 0.10, 0.15, 72.0,  0.40, 0.75, "indie"),
    ("track_000002", "Sad Song 2",    "Artist B", 0.15, 0.20, 80.0,  0.35, 0.70, "indie"),
    ("track_000003", "Sad Song 3",    "Artist C", 0.20, 0.25, 68.0,  0.45, 0.80, "classical"),
    ("track_000004", "Sad Song 4",    "Artist D", 0.25, 0.30, 75.0,  0.38, 0.65, "classical"),
    ("track_000005", "Sad Song 5",    "Artist E", 0.12, 0.18, 70.0,  0.50, 0.72, "ambient"),

    # Tensión / oscuridad (valence 0.1–0.3, energy 0.4–0.7)
    ("track_000006", "Dark Song 1",   "Artist F", 0.15, 0.50, 95.0,  0.55, 0.30, "rock"),
    ("track_000007", "Dark Song 2",   "Artist G", 0.20, 0.55, 100.0, 0.60, 0.25, "rock"),
    ("track_000008", "Dark Song 3",   "Artist H", 0.25, 0.60, 105.0, 0.65, 0.20, "metal"),
    ("track_000009", "Dark Song 4",   "Artist I", 0.18, 0.65, 98.0,  0.58, 0.28, "metal"),
    ("track_000010", "Dark Song 5",   "Artist J", 0.22, 0.58, 102.0, 0.62, 0.22, "electronic"),

    # Épico / climático (valence 0.4–0.6, energy 0.8–1.0)
    ("track_000011", "Epic Song 1",   "Artist K", 0.45, 0.85, 130.0, 0.70, 0.10, "rock"),
    ("track_000012", "Epic Song 2",   "Artist L", 0.50, 0.88, 135.0, 0.72, 0.08, "metal"),
    ("track_000013", "Epic Song 3",   "Artist M", 0.55, 0.90, 128.0, 0.75, 0.05, "electronic"),
    ("track_000014", "Epic Song 4",   "Artist N", 0.42, 0.92, 140.0, 0.68, 0.12, "rock"),
    ("track_000015", "Epic Song 5",   "Artist O", 0.48, 0.87, 132.0, 0.71, 0.09, "metal"),

    # Furia / intensidad (valence 0.0–0.4, energy 0.8–1.0, tempo > 120)
    ("track_000016", "Rage Song 1",   "Artist P", 0.10, 0.92, 150.0, 0.65, 0.05, "metal"),
    ("track_000017", "Rage Song 2",   "Artist Q", 0.15, 0.95, 155.0, 0.60, 0.03, "metal"),
    ("track_000018", "Rage Song 3",   "Artist R", 0.20, 0.90, 145.0, 0.70, 0.07, "electronic"),
    ("track_000019", "Rage Song 4",   "Artist S", 0.12, 0.93, 148.0, 0.63, 0.04, "rock"),
    ("track_000020", "Rage Song 5",   "Artist T", 0.08, 0.97, 160.0, 0.58, 0.02, "metal"),

    # Euforia / alegría (valence 0.7–1.0, energy 0.7–1.0)
    ("track_000021", "Happy Song 1",  "Artist U", 0.80, 0.80, 120.0, 0.85, 0.10, "pop"),
    ("track_000022", "Happy Song 2",  "Artist V", 0.85, 0.85, 125.0, 0.88, 0.08, "pop"),
    ("track_000023", "Happy Song 3",  "Artist W", 0.90, 0.90, 115.0, 0.90, 0.05, "dance"),
    ("track_000024", "Happy Song 4",  "Artist X", 0.75, 0.75, 118.0, 0.82, 0.12, "pop"),
    ("track_000025", "Happy Song 5",  "Artist Y", 0.88, 0.82, 122.0, 0.87, 0.07, "dance"),

    # Paz / relajación (valence 0.5–1.0, energy 0.0–0.3, acousticness > 0.8)
    ("track_000026", "Calm Song 1",   "Artist Z", 0.60, 0.20, 65.0,  0.30, 0.90, "classical"),
    ("track_000027", "Calm Song 2",   "Artist AA",0.65, 0.25, 70.0,  0.35, 0.88, "ambient"),
    ("track_000028", "Calm Song 3",   "Artist BB",0.70, 0.15, 60.0,  0.28, 0.92, "classical"),
    ("track_000029", "Calm Song 4",   "Artist CC",0.75, 0.22, 68.0,  0.32, 0.87, "ambient"),
    ("track_000030", "Calm Song 5",   "Artist DD",0.55, 0.18, 62.0,  0.25, 0.95, "classical"),

    # Tracks medios — zona de transición (para probar suavidad de arcos)
    ("track_000031", "Mid Song 1",    "Artist EE",0.35, 0.45, 88.0,  0.50, 0.40, "indie"),
    ("track_000032", "Mid Song 2",    "Artist FF",0.40, 0.50, 92.0,  0.55, 0.35, "indie"),
    ("track_000033", "Mid Song 3",    "Artist GG",0.45, 0.55, 95.0,  0.60, 0.30, "pop"),
    ("track_000034", "Mid Song 4",    "Artist HH",0.50, 0.60, 98.0,  0.58, 0.25, "pop"),
    ("track_000035", "Mid Song 5",    "Artist II",0.55, 0.65, 100.0, 0.62, 0.20, "electronic"),

    # Tracks adicionales para probar géneros específicos
    ("track_000036", "Jazz Song 1",   "Artist JJ",0.55, 0.35, 80.0,  0.60, 0.50, "jazz"),
    ("track_000037", "Jazz Song 2",   "Artist KK",0.60, 0.40, 85.0,  0.58, 0.48, "jazz"),
    ("track_000038", "Hip-hop 1",     "Artist LL",0.50, 0.70, 90.0,  0.75, 0.15, "hip-hop"),
    ("track_000039", "Hip-hop 2",     "Artist MM",0.55, 0.72, 95.0,  0.78, 0.12, "hip-hop"),
    ("track_000040", "R&B Song 1",    "Artist NN",0.65, 0.55, 85.0,  0.70, 0.20, "r-and-b"),

    # Tracks edge case: parámetros extremos
    ("track_000041", "Ultra Sad",     "Artist OO",0.02, 0.05, 55.0,  0.20, 0.95, "ambient"),
    ("track_000042", "Ultra Happy",   "Artist PP",0.98, 0.98, 130.0, 0.95, 0.02, "dance"),
    ("track_000043", "Ultra Slow",    "Artist QQ",0.50, 0.05, 45.0,  0.15, 0.97, "classical"),
    ("track_000044", "Ultra Fast",    "Artist RR",0.50, 0.99, 180.0, 0.80, 0.01, "electronic"),
    ("track_000045", "Ultra Angry",   "Artist SS",0.05, 0.99, 170.0, 0.70, 0.02, "metal"),

    # Tracks para probar retry (rango muy estrecho que necesita relaxation)
    ("track_000046", "Rare Mood 1",   "Artist TT",0.35, 0.35, 88.0,  0.55, 0.35, "indie"),
    ("track_000047", "Rare Mood 2",   "Artist UU",0.36, 0.36, 90.0,  0.54, 0.36, "indie"),
    ("track_000048", "Rare Mood 3",   "Artist VV",0.37, 0.37, 91.0,  0.53, 0.37, "indie"),
    ("track_000049", "Rare Mood 4",   "Artist WW",0.38, 0.38, 92.0,  0.52, 0.38, "indie"),
    ("track_000050", "Rare Mood 5",   "Artist XX",0.39, 0.39, 93.0,  0.51, 0.39, "indie"),
]


@pytest.fixture
def db_path(tmp_path):
    """
    Crea una SQLite temporal con tracks sintéticos.
    Se destruye automáticamente al terminar el test.
    """
    path = tmp_path / "test_tracks.db"
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            track_name TEXT NOT NULL,
            artist TEXT NOT NULL,
            valence REAL NOT NULL,
            energy REAL NOT NULL,
            tempo REAL NOT NULL,
            danceability REAL NOT NULL,
            acousticness REAL NOT NULL,
            track_genre TEXT NOT NULL
        )
    """)

    cursor.executemany(
        "INSERT INTO tracks VALUES (?,?,?,?,?,?,?,?,?)",
        SYNTHETIC_TRACKS
    )

    # Crear los mismos índices que load_dataset.py
    cursor.execute("CREATE INDEX idx_valence ON tracks(valence)")
    cursor.execute("CREATE INDEX idx_energy  ON tracks(energy)")
    cursor.execute("CREATE INDEX idx_genre   ON tracks(track_genre)")
    cursor.execute("CREATE INDEX idx_val_energy ON tracks(valence, energy)")

    conn.commit()
    conn.close()

    return path
