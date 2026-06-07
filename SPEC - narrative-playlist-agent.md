# SPEC.md — narrative-playlist-agent
**Versión:** 1.1 (corregida)  
**Estado:** Listo para arquitectura  
**Autor:** Ander Vazques  
**Fecha:** Marzo 2026

---

## 1. Visión General del Proyecto

**Objetivo:**  
Construir un agente de IA capaz de traducir una narrativa emocional en lenguaje natural (ej. *"de la profunda tristeza a la furia, terminando en aceptación pacífica"*) en una lista de reproducción secuencial, coherente y matemáticamente justificada.

**Diferenciador Técnico:**  
A diferencia de los sistemas de recomendación tradicionales basados en filtrado colaborativo, este agente utiliza un LLM para realizar razonamiento secuencial. El sistema no solo recupera canciones de una base de datos local, sino que las ordena matemáticamente y genera una explicación paso a paso de por qué los valores de `valence`, `energy` y `tempo` de cada transición logran el arco emocional solicitado.

**Por qué no usamos la Spotify API para audio features:**  
Spotify deprecó los endpoints `GET /audio-features`, `GET /recommendations` y `GET /audio-analysis` en noviembre 2024. En febrero 2026 introdujo restricciones adicionales que limitan gravemente el acceso en Development Mode. El sistema utiliza un dataset local como fuente de verdad para los audio features, y la Spotify API únicamente para publicar la playlist en la cuenta del usuario (endpoint `POST /playlists/{id}/tracks`, aún disponible) — esta funcionalidad es opcional y corresponde a v3.

---

## 2. Dataset — Fuente de Datos Oficial

**Dataset:** `Spotify Tracks Dataset`  
**Fuente:** [Kaggle — maharshipandya/spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)  
**Tamaño:** ~600,000 tracks  
**Formato:** CSV  
**Licencia:** CC0 (dominio público — sin restricciones de uso)

### Columnas utilizadas por el agente

| Columna | Tipo | Rango | Descripción |
|---|---|---|---|
| `track_name` | string | — | Nombre de la canción |
| `artists` | string | — | Nombre del artista |
| `valence` | float | 0.0 – 1.0 | Positividad musical. 0.0 = muy triste/tenso, 1.0 = muy alegre/eufórico |
| `energy` | float | 0.0 – 1.0 | Intensidad y actividad. 0.0 = suave/acústico, 1.0 = ruidoso/rápido |
| `danceability` | float | 0.0 – 1.0 | Aptitud para el baile |
| `acousticness` | float | 0.0 – 1.0 | Confianza en que el track es acústico |
| `instrumentalness` | float | 0.0 – 1.0 | Probabilidad de ausencia de vocales |
| `tempo` | float | BPM | Pulsos por minuto estimados |
| `track_genre` | string | ~114 valores | Género musical (ver nota de limpieza) |
| `popularity` | int | 0 – 100 | Popularidad en Spotify al momento de la captura |

### Nota crítica sobre `track_genre`
El dataset contiene ~114 géneros con inconsistencias (ej. `"hip-hop"` y `"hip_hop"` como valores distintos). Durante la fase de ingesta de datos se aplicará normalización: lowercase, reemplazo de guiones bajos por guiones, y deduplicación. El campo `target_genres` del tool schema filtra contra los valores normalizados.

---

## 3. Flujo de Experiencia del Usuario

El agente opera en v1 a través de CLI, y en v2 a través de interfaz web (Streamlit).

```
[1] INPUT
    Usuario ingresa prompt narrativo en texto libre.
    Ejemplo: "Hazme una playlist de 10 canciones que empiece oscura
    y melancólica, suba a algo épico para entrenar, y termine
    relajándome por completo."

[2] PHASE DECOMPOSITION (LLM — invisible para el usuario)
    El LLM divide el arco en N fases lógicas con etiquetas y
    parámetros numéricos asignados por el System Prompt.

[3] TOOL EXECUTION (Function Calling — invisible para el usuario)
    El agente ejecuta query_song_database por cada fase.
    Si una consulta retorna 0 resultados → relajación
    autónoma de parámetros y reintento (máx. 3 intentos).

[4] VALIDATION LAYER (Python determinista — invisible para el usuario)
    Función playlist_validator.py verifica:
    - Constraint de transición: Δvalence ≤ 0.3, Δenergy ≤ 0.3
      entre tracks consecutivos, salvo que la narrativa lo exija.
    - Coherencia de arco: pendiente de regresión lineal de
      valence y energy según dirección del arco solicitado.
    Si falla → el agente reordena o reemplaza tracks problemáticos.

[5] OUTPUT FINAL
    - Playlist estructurada (ver Output Schema en sección 6).
    - Bloque explicativo generado por el LLM.
    - Estadísticas del arco (valence inicial vs final, etc.).
```

---

## 4. Versiones del Producto (Scope Control)

| Versión | Alcance | Entregable |
|---|---|---|
| **v1 — MVP** | CLI + SQLite + 1 tool + output explicado en texto | Repo público en GitHub |
| **v2** | Streamlit UI + visualización gráfica del arco emocional | Demo deployable |
| **v3** | Spotify OAuth para publicar playlist real en cuenta del usuario | Integración opcional |

**Todo el spec actual define v1.** Las funcionalidades de v2 y v3 se documentan en el README como roadmap pero no se incluyen en el desarrollo inicial.

---

## 5. Arquitectura del Agente

### Motor de base de datos
**SQLite** (no ChromaDB/FAISS para v1).

**Justificación:** Las consultas del agente son filtros por rangos numéricos continuos (`valence BETWEEN 0.1 AND 0.3`). SQL es determinista, predecible y no requiere infraestructura externa. Una base vectorial está justificada solo si la búsqueda es semántica sobre texto — lo cual no es el caso aquí. ChromaDB se puede añadir en v2 como capa complementaria para búsqueda por descripción textual de mood.

### Core del agente
LLM configurado con System Prompt estricto que restringe sus salidas a Tool Calls hasta que la playlist esté completa. El agente no puede inventar canciones — toda pista en el output debe ser un registro real retornado por `query_song_database`.

### LLM recomendado
**Groq (LLaMA 3.3-70b)** para v1 — gratis, rápido, compatible con OpenAI SDK. Intercambiable con GPT-4o o Claude vía variable de entorno `LLM_PROVIDER`.

---

## 6. Ingeniería de Contexto — System Prompt

El éxito del agente depende de la calidad del System Prompt. Este es el fragmento central:

```
Eres un ingeniero de curación musical. Tu única función es construir
playlists narrativas usando la herramienta query_song_database.

REGLAS ABSOLUTAS:
1. Nunca inventes canciones. Toda pista debe venir de un tool call exitoso.
2. Divide el arco emocional del usuario en entre 2 y 5 fases lógicas.
3. Asigna parámetros numéricos a cada fase usando el siguiente mapeo:

   Tristeza / Melancolía:  valence 0.0–0.3,  energy 0.0–0.4,  acousticness > 0.6
   Tensión / Oscuridad:    valence 0.0–0.3,  energy 0.4–0.7,  tempo > 100
   Furia / Intensidad:     valence 0.0–0.4,  energy 0.8–1.0,  tempo > 120
   Épico / Climático:      valence 0.4–0.6,  energy 0.8–1.0,  tempo > 115
   Alegría / Euforia:      valence 0.7–1.0,  energy 0.7–1.0,  danceability > 0.6
   Paz / Relajación:       valence 0.5–1.0,  energy 0.0–0.3,  acousticness > 0.8

4. Asegura que el Δvalence y Δenergy entre tracks consecutivos no supere 0.3,
   salvo que el usuario pida un cambio abrupto explícito.

5. Si query_song_database retorna 0 resultados:
   - Amplía los rangos en ±0.1 y reintenta.
   - Máximo 3 intentos por fase.
   - Si al tercer intento no hay resultados, documéntalo en el output.

6. No emitas el output final hasta que TODAS las fases tengan tracks asignados.
```

---

## 7. Especificación de Herramientas (Function Calling Schema)

```json
{
  "name": "query_song_database",
  "description": "Busca en la base de datos local de canciones usando rangos de audio features para encontrar tracks que coincidan con un estado emocional específico. Retorna una lista de diccionarios con metadata de las canciones. IMPORTANTE: Solo usar canciones retornadas por esta herramienta — nunca inventar tracks.",
  "parameters": {
    "type": "object",
    "properties": {
      "min_valence": {
        "type": "number",
        "description": "Valor mínimo de positividad musical (0.0 a 1.0). 0.0 = muy triste, 1.0 = muy alegre."
      },
      "max_valence": {
        "type": "number",
        "description": "Valor máximo de positividad musical (0.0 a 1.0)."
      },
      "min_energy": {
        "type": "number",
        "description": "Valor mínimo de intensidad (0.0 a 1.0). 0.0 = suave/acústico, 1.0 = ruidoso/rápido."
      },
      "max_energy": {
        "type": "number",
        "description": "Valor máximo de intensidad (0.0 a 1.0)."
      },
      "min_tempo": {
        "type": "number",
        "description": "BPM mínimo. Opcional. Usar cuando el arco requiera control de ritmo explícito."
      },
      "max_tempo": {
        "type": "number",
        "description": "BPM máximo. Opcional."
      },
      "min_acousticness": {
        "type": "number",
        "description": "Valor mínimo de acousticness (0.0 a 1.0). Opcional. Útil para fases de paz o melancolía acústica."
      },
      "target_genres": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Lista opcional de géneros normalizados (lowercase, con guiones). Ej: ['rock', 'classical', 'electronic']. Si se omite, busca en todos los géneros."
      },
      "exclude_track_ids": {
        "type": "array",
        "items": { "type": "string" },
        "description": "IDs de tracks ya asignados a la playlist. Evita duplicados entre fases."
      },
      "limit": {
        "type": "integer",
        "description": "Número de canciones a recuperar. Default: 3. Máximo recomendado: 10.",
        "default": 3
      }
    },
    "required": ["min_valence", "max_valence", "min_energy", "max_energy", "limit"]
  }
}
```

**Corrección vs versión anterior:** Se añadieron `min_tempo`, `max_tempo`, `min_acousticness` (para coherencia con las reglas del System Prompt) y `exclude_track_ids` (para evitar duplicados entre fases sin lógica adicional en el agente).

---

## 8. Output Schema (Estructura de Datos del Output Final)

Este schema es obligatorio. El LLM debe generar el output final en este formato JSON para que el frontend pueda renderizarlo de forma consistente.

```json
{
  "playlist_title": "string — título generado por el LLM para la playlist",
  "user_prompt": "string — prompt original del usuario",
  "narrative_arc": "string — descripción del arco en 1-2 oraciones",
  "total_tracks": "integer",
  "phases": [
    {
      "phase_number": 1,
      "phase_label": "string — ej: 'Melancolía profunda'",
      "emotional_description": "string — descripción breve del estado emocional",
      "valence_range": [0.1, 0.25],
      "energy_range": [0.2, 0.35],
      "tempo_range": [60, 90],
      "tracks": [
        {
          "position": 1,
          "track_name": "string",
          "artist": "string",
          "valence": 0.18,
          "energy": 0.28,
          "tempo": 72.4,
          "genre": "string",
          "transition_note": "string — por qué esta canción está en esta posición"
        }
      ],
      "phase_explanation": "string — justificación paramétrica de la fase completa"
    }
  ],
  "arc_statistics": {
    "valence_start": 0.18,
    "valence_end": 0.82,
    "energy_start": 0.28,
    "energy_end": 0.75,
    "valence_slope": "positive | negative | neutral",
    "energy_slope": "positive | negative | neutral",
    "arc_coherence_score": 0.91
  },
  "arc_summary": "string — párrafo final del LLM explicando el arco completo"
}
```

---

## 9. Capa de Validación Determinista (playlist_validator.py)

Esta capa es independiente del LLM. Es código Python puro que verifica la coherencia matemática del output antes de entregarlo al usuario.

### Función 1: `validate_transitions(tracks)`
Verifica que el Δvalence y Δenergy entre tracks consecutivos no supere 0.3.
- Si encuentra una violación: marca el track problemático y lo reporta.
- El agente puede usar esta información para reemplazar el track.

### Función 2: `validate_arc_slope(tracks, direction)`
Calcula la pendiente de regresión lineal sobre `valence` Y `energy` de la playlist completa.
- Para arcos **ascendentes** (tristeza → alegría): requiere `m > 0` en ambas dimensiones.
- Para arcos **descendentes** (euforia → paz): requiere `m < 0` en ambas dimensiones.
- Para arcos **en V** o **en arco** (tristeza → clímax → paz): valida por segmentos.

### Función 3: `calculate_arc_coherence_score(tracks)`
Devuelve un float entre 0.0 y 1.0 que mide qué tan bien la secuencia de tracks representa el arco solicitado. Este score se incluye en `arc_statistics` del output.

---

## 10. Criterios de Aceptación — MVP (v1)

Para considerar v1 listo para publicación en GitHub, el sistema debe cumplir:

| Criterio | Descripción | Verificación |
|---|---|---|
| **Sin alucinaciones** | Cada track del output existe en la DB local | `track_id` presente en SQLite |
| **Coherencia matemática** | Arco ascendente → pendiente `m > 0` en `valence` Y `energy` | `playlist_validator.validate_arc_slope()` |
| **Constraint de transición** | Δvalence ≤ 0.3 y Δenergy ≤ 0.3 entre consecutivos | `playlist_validator.validate_transitions()` |
| **Manejo de vacíos** | 0 resultados → relajación de parámetros + reintento autónomo (máx. 3) | Test con parámetros extremos |
| **Explicabilidad obligatoria** | Output rechazado si no incluye `phase_explanation` y `arc_summary` | Validación de schema Pydantic |
| **Output schema válido** | JSON del output pasa validación Pydantic sin errores | Model `PlaylistOutput` |

---

## 11. Edge Cases Definidos

| Caso | Comportamiento esperado |
|---|---|
| Arco de 1 sola emoción ("solo música triste") | El agente crea 1 fase, selecciona tracks con variación progresiva interna dentro de los rangos de esa emoción |
| Parámetros imposibles (valence < 0.1 AND energy > 0.95) | 3 intentos con relajación de ±0.1 por intento; si falla, documenta en output |
| Usuario pide géneros que no existen en el dataset | El agente lo reporta, omite el filtro de género y procede |
| Playlist solicitada > 20 tracks | El agente advierte y genera máximo 20 tracks (límite de v1) |
| Arco contradictorio ("alegre pero oscuro y triste al mismo tiempo") | El LLM interpreta la tensión como una emoción compuesta (ej. `valence 0.3–0.5`, `energy 0.5–0.7`) y lo documenta en el output |

---

## 12. Stack Tecnológico

| Capa | Tecnología | Versión target | Justificación |
|---|---|---|---|
| Lenguaje | Python | 3.11+ | Base del curso, ecosistema AI |
| LLM Provider | Groq API (LLaMA 3.3-70b) | — | Gratis, rápido, compatible OpenAI SDK |
| LLM SDK | `openai` | 1.x | Compatible con múltiples providers |
| Base de datos | SQLite | built-in | Sin infraestructura, determinista para rangos |
| ORM / queries | `sqlite3` (stdlib) o `pandas` | — | Simple para v1 |
| Data processing | `pandas` | 2.x | Ingesta y limpieza del CSV |
| Validación | `pydantic` | 2.x | Schema del output, evita alucinaciones de formato |
| Math validation | `scipy` / `numpy` | — | Regresión lineal para validar arco |
| Variables de entorno | `python-dotenv` | — | Gestión de API keys |
| CLI (v1) | `rich` | — | Output visual en terminal |
| Frontend (v2) | `streamlit` | — | Demo visual del arco emocional |
| IDE / Deploy | GitHub Codespaces | — | Sin dependencia de hardware local |

---

## 13. Estructura del Repositorio

```
narrative-playlist-agent/
├── .env.example                    # Template de variables de entorno
├── .gitignore
├── requirements.txt
├── README.md
├── SPEC.md                         # Este documento
│
├── data/
│   ├── raw/                        # CSV original de Kaggle (no commiteado)
│   └── db/
│       └── tracks.db               # SQLite generado por el script de ingesta
│
├── src/
│   ├── ingestion/
│   │   └── load_dataset.py         # Limpia CSV y carga en SQLite
│   │
│   ├── agent/
│   │   ├── system_prompt.py        # System Prompt como constante
│   │   ├── tools.py                # Implementación de query_song_database
│   │   └── agent.py                # Loop principal del agente (tool calling)
│   │
│   ├── validation/
│   │   └── playlist_validator.py   # Validación matemática determinista
│   │
│   ├── schemas/
│   │   └── models.py               # Pydantic models (PlaylistOutput, Phase, Track)
│   │
│   └── main.py                     # Entry point CLI
│
├── frontend/                       # v2
│   └── app.py                      # Streamlit UI
│
└── tests/
    ├── test_tools.py
    ├── test_validator.py
    └── test_agent_integration.py
```

---

## 14. Variables de Entorno (.env.example)

```bash
# LLM Provider
LLM_PROVIDER=groq                        # groq | openai | anthropic
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Opcional

# Spotify (solo para v3 — publicación de playlist)
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8080/callback

# Config del agente
MAX_TRACKS_PER_PLAYLIST=20
MAX_RETRY_ATTEMPTS=3
DEFAULT_TRACKS_PER_PHASE=3
```

---

*Spec v1.1 — Listo para fase de arquitectura.*
