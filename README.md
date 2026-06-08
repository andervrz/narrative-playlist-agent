# narrative-playlist-agent

> Traduce arcos emocionales en lenguaje natural a playlists matemáticamente validadas, exportadas en CSV para cualquier plataforma de streaming.

---

## Demo

```
→ "lista oscura a legendaria"

✅ Playlist "Oscura a Legendaria" — 6 canciones en 2 fases
📊 Arco: valence 0.077→0.583 | energy 0.517→0.805
🎯 Coherencia: 92%
📄 playlist_oscura_a_legendaria.csv → listo para TuneMyMusic
```

---

## Qué hace

El agente recibe un prompt narrativo libre y ejecuta autónomamente:

1. **Descompone el arco emocional** en fases con parámetros numéricos (`valence`, `energy`, `tempo`)
2. **Busca canciones reales** en una base de datos local de 81,000+ tracks con audio features verificados
3. **Valida matemáticamente** el arco con regresión lineal y constraints de transición suave (Δvalence ≤ 0.3)
4. **Genera un CSV** compatible con TuneMyMusic para publicar en Spotify, Apple Music, YouTube Music o cualquier plataforma

---

## Por qué no es un script más

La mayoría de repos de "playlist AI" en GitHub hacen esto:

```
LLM inventa nombres de canciones → busca en Spotify → crea playlist
```

**El problema:** el LLM alucina tracks que no existen o con features incorrectos.

Este agente hace lo contrario:

```
LLM traduce emoción → rangos numéricos
SQLite retorna tracks REALES con features verificados
Python valida el arco matemáticamente
LLM nunca inventa ninguna canción
```

---

## Stack

```
LLM:         Groq (LLaMA 3.3-70b) — via OpenAI SDK
Base datos:  SQLite con 81,000+ tracks (Kaggle Spotify Dataset)
Validación:  scipy.stats.linregress — Python puro, sin LLM
Output:      CSV para TuneMyMusic + JSON con narrativa completa
Skills:      3 SKILL.md composables (generación, ingesta, validación)
```

---

## Setup

### 1. Clonar e instalar

```bash
git clone https://github.com/andervrz/narrative-playlist-agent
cd narrative-playlist-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar

```bash
cp .env.example .env
# Editar .env y añadir tu GROQ_API_KEY
# Obtener gratis en: console.groq.com
```

### 3. Preparar la base de datos

Descargar el dataset:
- **Kaggle:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- **HuggingFace (sin cuenta):** https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/blob/main/dataset.csv

```bash
# Colocar el CSV en:
data/raw/dataset.csv

# Generar SQLite:
python3 src/ingestion/load_dataset.py
```

### 4. Correr

```bash
python3 src/main.py
```

---

## Uso

```bash
# Interactivo
python3 src/main.py

# Con prompt directo
python3 src/main.py --prompt "Playlist de 9 canciones: oscura → épica → paz"
```

Ejemplos de prompts:
```
"De tristeza profunda a euforia pura, 8 canciones"
"Playlist para estudiar, concentrada y tranquila"
"Viaje emocional: melancolía → furia → aceptación"
"Música épica para entrenar, terminar relajado"
```

---

## Output

Por cada ejecución el agente genera dos archivos en `output/`:

| Archivo | Uso |
|---|---|
| `playlist_[titulo]_[fecha].csv` | Subir a tunemymusic.com → cualquier plataforma |
| `playlist_[titulo]_[fecha].json` | Registro completo con narrativa, features y arco |

### Publicar en tu plataforma

1. Ve a [tunemymusic.com](https://tunemymusic.com)
2. Selecciona **File** como fuente
3. Sube el `.csv`
4. Elige: Spotify / Apple Music / YouTube Music / Tidal / Deezer
5. La playlist aparece en tu cuenta

---

## Arquitectura

```
src/
├── ingestion/load_dataset.py      → CSV → SQLite con índices
├── schemas/models.py              → Pydantic v2 (contratos de datos)
├── agent/
│   ├── tools.py                   → query_song_database + retry_logic
│   ├── db_tools.py                → add_track_to_database
│   ├── output_tools.py            → generate_playlist_file (CSV + JSON)
│   ├── system_prompt.py           → instrucciones del agente
│   └── agent.py                   → loop LLM ↔ tools (harness)
├── validation/playlist_validator.py → validación matemática determinista
└── main.py                        → CLI con Rich

skills/
├── playlist_generation/SKILL.md   → cuándo y cómo generar playlists
├── track_ingestion/SKILL.md       → cómo añadir tracks al dataset
├── arc_validation/SKILL.md        → cómo interpretar el score del arco
└── loader.py                      → carga dinámica de skills
```

### Principio central

```
LLM      → razona (traduce emoción a números, elige qué buscar)
SQLite   → verdad (tracks reales con audio features verificados)
Python   → valida (matemáticas, constraints, arco coherente)
```

---

## Tests

```bash
pytest tests/ -v
# 142 tests — 0 fallos
```

---

## Añadir canciones al dataset

Si una canción no está en el dataset, el agente puede añadirla:

```
→ "Agrega 'Papaoutai' de Stromae"

Agente: Estimé valence=0.28, energy=0.62, tempo=122. ¿Correcto?
Tú:     Sí
Agente: ✅ Añadida con ID track_u000001. Ya disponible para futuros arcos.
```

---

## Por qué Spotify deprecó los audio features

En noviembre 2024, Spotify deprecó los endpoints `GET /audio-features` y `GET /recommendations` para nuevas apps. En febrero 2026 añadió restricciones adicionales que requieren cuenta Premium para el modo de desarrollo.

Este proyecto usa el dataset de Kaggle como fuente de verdad para los audio features — lo que lo hace independiente de las APIs de streaming y más robusto que los sistemas que dependen de endpoints externos.

---

## Roadmap

- [ ] v2: Streamlit UI con visualización gráfica del arco emocional
- [ ] v2: Feedback loop — registrar qué canciones saltas para mejorar el arco
- [ ] v3: Integración directa con Spotify (requiere cuenta Premium)

---

## Autor

**Ander Vasques** — [@andervrz](https://github.com/andervrz)

AI Engineer — Venezuela
