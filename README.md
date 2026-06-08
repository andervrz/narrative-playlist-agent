# narrative-playlist-agent

> Translates emotional narratives in natural language into mathematically validated playlists, exported as CSV for any streaming platform.

---

## Demo

```
→ "dark playlist that builds into legendary"

✅ Playlist "Dark to Legendary" — 6 songs in 2 phases
📊 Arc: valence 0.077→0.583 | energy 0.517→0.805
🎯 Coherence score: 92%
📄 playlist_dark_to_legendary.csv → ready for TuneMyMusic
```

---

## What it does

The agent takes a free-form narrative prompt and autonomously:

1. **Decomposes the emotional arc** into phases with explicit numeric parameters (`valence`, `energy`, `tempo`)
2. **Retrieves real songs** from a local database of 81,000+ tracks with verified audio features
3. **Mathematically validates** the arc using linear regression and smooth transition constraints (Δvalence ≤ 0.3)
4. **Generates a CSV** compatible with TuneMyMusic to publish on Spotify, Apple Music, YouTube Music, or any platform

---

## Why this is different

Most "AI playlist" repos on GitHub do this:

```
LLM hallucinates song names → search on Spotify → create playlist
```

**The problem:** The LLM invents tracks that don't exist or assigns wrong audio features.

This agent does the opposite:

```
LLM translates emotion → numeric ranges
SQLite returns REAL tracks with verified audio features
Python validates the arc mathematically
The LLM never invents a single song
```

---

## Stack

```
LLM:        Groq (LLaMA 3.3-70b) — via OpenAI SDK
Database:   SQLite with 81,000+ tracks (Kaggle Spotify Dataset)
Validation: scipy.stats.linregress — pure Python, no LLM
Output:     CSV for TuneMyMusic + JSON with full narrative
Skills:     3 composable SKILL.md files (generation, ingestion, validation)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/andervrz/narrative-playlist-agent
cd narrative-playlist-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get one free at: console.groq.com
```

### 3. Prepare the database

Download the dataset:
- **Kaggle:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- **HuggingFace (no account needed):** https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/blob/main/dataset.csv

```bash
# Place the CSV here:
data/raw/dataset.csv

# Generate SQLite database:
python3 src/ingestion/load_dataset.py
```

### 4. Run

```bash
python3 src/main.py
```

---

## Usage

```bash
# Interactive mode
python3 src/main.py

# Direct prompt
python3 src/main.py --prompt "9-song playlist: dark → epic → peace"
```

Prompt examples:
```
"From deep sadness to pure euphoria, 8 songs"
"Focus playlist for studying, calm and concentrated"
"Emotional journey: melancholy → rage → acceptance"
"Epic workout music that ends with total relaxation"
"dark to legendary"
```

---

## Output

Each run generates two files in `output/`:

| File | Use |
|---|---|
| `playlist_[title]_[date].csv` | Upload to tunemymusic.com → any platform |
| `playlist_[title]_[date].json` | Full record with narrative, audio features, arc stats |

### Publish to your platform

1. Go to [tunemymusic.com](https://tunemymusic.com)
2. Select **File** as source
3. Upload the `.csv`
4. Choose: Spotify / Apple Music / YouTube Music / Tidal / Deezer
5. Playlist appears in your account

---

## Architecture

```
src/
├── ingestion/load_dataset.py       → CSV → SQLite with indexes
├── schemas/models.py               → Pydantic v2 (data contracts)
├── agent/
│   ├── tools.py                    → query_song_database + retry_logic
│   ├── db_tools.py                 → add_track_to_database
│   ├── output_tools.py             → generate_playlist_file (CSV + JSON)
│   ├── system_prompt.py            → agent instructions as Python constant
│   └── agent.py                    → LLM ↔ tools loop (harness)
├── validation/playlist_validator.py → deterministic math validation layer
└── main.py                         → CLI with Rich

skills/
├── playlist_generation/SKILL.md    → when and how to generate playlists
├── track_ingestion/SKILL.md        → how to add tracks to the dataset
├── arc_validation/SKILL.md         → how to interpret the arc score
└── loader.py                       → dynamic skill loading
```

### Core principle

```
LLM      → reasons  (translates emotion to numbers, decides what to search)
SQLite   → truth    (real tracks with verified audio features)
Python   → validates (math, constraints, coherent arc)
```

---

## Tests

```bash
pytest tests/ -v
# 142 tests — 0 failures
```

---

## Adding songs to the dataset

If a song isn't in the dataset, the agent can add it interactively:

```
→ "Add 'Papaoutai' by Stromae"

Agent: I estimated valence=0.28, energy=0.62, tempo=122. Correct?
You:   Yes
Agent: ✅ Added with ID track_u000001. Available for future arcs.
```

---

## Why Spotify deprecated audio features

In November 2024, Spotify deprecated the `GET /audio-features` and `GET /recommendations` endpoints for new apps. In February 2026 they added further restrictions requiring a Premium account for developer mode.

This project uses the Kaggle dataset as the source of truth for audio features — making it independent of streaming APIs and more robust than systems that rely on external endpoints that can change or disappear.

---

## Roadmap

- [ ] v2: Streamlit UI with visual arc chart (valence/energy per track)
- [ ] v2: Feedback loop — log skipped tracks to improve future arcs
- [ ] v3: Direct Spotify integration (requires Premium account)

---

## Author

**Ander Vasquez** — [@andervrz](https://github.com/andervrz)

AI Engineer — Venezuela
