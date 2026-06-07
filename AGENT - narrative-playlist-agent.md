# AGENT.md — narrative-playlist-agent (Spotify Premium)
**Versión:** 1.0
**Plataforma:** Spotify (cuenta Premium obligatoria)
**Estado:** Especificación lista para implementación
**Referencia:** SPEC.md v1.1 + ARCHITECTURE.md v1.0 + PROJECT_HANDOFF.md

---

## 1. Por qué Spotify Premium cambia el scope del agente

Con una cuenta Spotify Free el agente puede crear playlists y añadir
tracks pero no puede controlar la reproducción programáticamente.
Con Premium, el agente pasa de "creador de listas" a "DJ autónomo":

```
FREE:   genera playlist → la publicas → tú le das play manualmente
PREMIUM: genera playlist → la publica → inicia reproducción automática
         → puede pausar, saltar, controlar volumen sin intervención tuya
```

Esta diferencia es la que convierte el proyecto en un agente con
acciones ejecutables reales en el mundo, no solo en un generador
de contenido.

---

## 2. Capacidades disponibles con Spotify Premium

### Lo que el agente PUEDE hacer (endpoints vivos post-deprecaciones)

#### Búsqueda y catálogo
```
GET  /search                     Buscar tracks, artistas, álbumes, playlists
GET  /tracks/{id}                Info de un track específico
GET  /artists/{id}               Info de un artista
GET  /artists/{id}/albums        Álbumes de un artista
GET  /albums/{id}/tracks         Tracks de un álbum
```

#### Playlists (escritura)
```
POST /me/playlists               Crear playlist en la cuenta del usuario
POST /playlists/{id}/items       Añadir tracks a playlist
PUT  /playlists/{id}/items       Reordenar tracks en playlist
DELETE /playlists/{id}/items     Eliminar tracks de playlist
GET  /me/playlists               Listar playlists del usuario
GET  /playlists/{id}             Contenido de una playlist específica
```

#### Playback (EXCLUSIVO Premium)
```
PUT  /me/player/play             Iniciar/reanudar reproducción
PUT  /me/player/pause            Pausar reproducción
POST /me/player/next             Saltar al siguiente track
POST /me/player/previous         Volver al track anterior
PUT  /me/player/volume           Controlar volumen (0-100)
PUT  /me/player/seek             Saltar a posición en el track
GET  /me/player                  Estado actual del reproductor
GET  /me/player/devices          Dispositivos disponibles (móvil, PC, etc.)
PUT  /me/player                  Transferir reproducción a otro dispositivo
POST /me/player/queue            Añadir track a la cola actual
```

#### Biblioteca y datos del usuario
```
GET  /me/player/recently-played  Historial reciente de reproducción
GET  /me/tracks                  Liked Songs del usuario
PUT  /me/library                 Guardar tracks en biblioteca
GET  /me/library/contains        Verificar si un track está en biblioteca
```

### Lo que el agente NO PUEDE hacer (deprecado desde nov 2024)
```
❌ GET /audio-features/{id}     valence, energy, tempo por ID de Spotify
❌ GET /recommendations         recomendaciones algorítmicas
❌ GET /audio-analysis/{id}     análisis de estructura rítmica
```

**Nota crítica:** Por esta razón el dataset de Kaggle con audio features
sigue siendo la fuente de verdad para el razonamiento emocional del agente.
Spotify no provee esos datos — nosotros los tenemos localmente.

---

## 3. Arquitectura del Agente con Spotify Premium

```
┌─────────────────────────────────────────────────────────────────┐
│                         USUARIO                                  │
│    "Playlist oscura que sube épica y termine en paz. 9 tracks"  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      agent.py (Loop)                             │
│                                                                  │
│  LLM (Groq LLaMA 3.3-70b)                                       │
│  ├── Tool 1: query_song_database     → SQLite (Kaggle features) │
│  ├── Tool 2: search_spotify_track    → Spotify Search API       │
│  ├── Tool 3: create_spotify_playlist → Spotify Playlists API    │
│  ├── Tool 4: start_playback          → Spotify Player API       │
│  └── Tool 5: add_track_to_database   → SQLite (user_added)      │
│                          │                                       │
│  playlist_validator.py   │ (validación matemática fuera del LLM)│
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
┌─────────────────────┐   ┌───────────────────────────┐
│   SQLite (local)    │   │   Spotify Web API          │
│   600k tracks       │   │   + MCP Server             │
│   con audio features│   │   (autenticación OAuth)    │
└─────────────────────┘   └───────────────────────────┘
```

---

## 4. MCP Server para Spotify

### El servidor recomendado

**`modelcontextprotocol/spotify-mcp`**
- Mantenido por la organización oficial de MCP
- Última actualización: 29 marzo 2026
- Stack: TypeScript
- Auth: OAuth 2.0 con PKCE

### Instalación y configuración

```json
// claude_desktop_config.json o equivalente
{
  "mcpServers": {
    "spotify": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/spotify-mcp"],
      "env": {
        "SPOTIFY_CLIENT_ID": "your_client_id",
        "SPOTIFY_CLIENT_SECRET": "your_client_secret",
        "SPOTIFY_REDIRECT_URI": "http://127.0.0.1:8888/callback",
        "SPOTIFY_PERSIST_TOKENS": "true"
      }
    }
  }
}
```

### Tools que expone el MCP

```
spotify_play              Iniciar reproducción
spotify_pause             Pausar
spotify_next              Saltar track
spotify_previous          Track anterior
spotify_seek              Saltar a posición
spotify_set_volume        Controlar volumen
spotify_get_playback_state Estado actual del reproductor
spotify_get_devices       Dispositivos disponibles
spotify_transfer_playback Cambiar dispositivo de reproducción
spotify_search            Buscar en catálogo
spotify_create_playlist   Crear playlist
spotify_add_to_playlist   Añadir tracks a playlist
spotify_get_playlist      Obtener contenido de playlist
spotify_queue_track       Añadir a cola
```

### Relación MCP ↔ API directa

```
MCP = wrapper conveniente sobre la misma Spotify Web API
      Mismas credenciales, mismas reglas, mismos endpoints
      Ventaja: no implementas OAuth manualmente
      Decisión: usar MCP para acciones de playback y playlist
                usar spotipy directamente si necesitas control fino
```

---

## 5. Tools del Agente — Especificación Completa

### Tool 1: `query_song_database` ✅ IMPLEMENTADA

Consulta SQLite local con audio features del dataset de Kaggle.
Ver `src/agent/tools.py` — implementación completa con retry_logic.

```json
{
  "name": "query_song_database",
  "parameters": {
    "min_valence": number,        "max_valence": number,
    "min_energy": number,         "max_energy": number,
    "min_tempo": number,          "max_tempo": number,
    "min_acousticness": number,
    "target_genres": [string],
    "exclude_track_ids": [string],
    "limit": integer (default 3, max 10)
  }
}
```

**Retry logic:** si 0 resultados → amplía rangos ±0.1 por intento (máx 3).

---

### Tool 2: `search_spotify_track` ⏳ PENDIENTE

Valida que un track del dataset exista en el catálogo de Spotify
y obtiene su Spotify URI para poder añadirlo a playlists.

```json
{
  "name": "search_spotify_track",
  "description": "Busca un track en Spotify por nombre y artista.
                  Retorna el Spotify URI si existe, null si no.",
  "parameters": {
    "track_name": string,
    "artist": string
  }
}
```

**Implementación:**
```python
import spotipy
from spotipy.oauth2 import SpotifyOAuth

def search_spotify_track(track_name: str, artist: str) -> dict:
    sp = get_spotify_client()
    query = f"track:{track_name} artist:{artist}"
    results = sp.search(q=query, type='track', limit=1)
    tracks = results['tracks']['items']
    if not tracks:
        return {"found": False, "uri": None}
    track = tracks[0]
    return {
        "found": True,
        "uri": track['uri'],
        "spotify_id": track['id'],
        "name": track['name'],
        "artist": track['artists'][0]['name'],
        "preview_url": track.get('preview_url')
    }
```

**Nota:** si el track del dataset no se encuentra en Spotify,
el agente usa el siguiente resultado de `query_song_database`.
Nunca falla — siempre tiene alternativas desde la DB.

---

### Tool 3: `create_and_populate_playlist` ⏳ PENDIENTE

Crea la playlist en Spotify con todos los tracks validados.

```json
{
  "name": "create_and_populate_playlist",
  "description": "Crea una playlist en Spotify y añade los tracks.
                  Requiere los Spotify URIs obtenidos por search_spotify_track.",
  "parameters": {
    "playlist_title": string,
    "playlist_description": string,
    "track_uris": [string],
    "public": boolean (default false)
  }
}
```

**Implementación:**
```python
def create_and_populate_playlist(
    playlist_title: str,
    playlist_description: str,
    track_uris: list[str],
    public: bool = False
) -> dict:
    sp = get_spotify_client()
    user_id = sp.current_user()['id']
    playlist = sp.user_playlist_create(
        user=user_id,
        name=playlist_title,
        public=public,
        description=playlist_description
    )
    sp.playlist_add_items(playlist['id'], track_uris)
    return {
        "playlist_id": playlist['id'],
        "playlist_url": playlist['external_urls']['spotify'],
        "tracks_added": len(track_uris)
    }
```

---

### Tool 4: `start_playback` ⏳ PENDIENTE (Premium exclusivo)

Inicia la reproducción de la playlist recién creada en el
dispositivo activo del usuario.

```json
{
  "name": "start_playback",
  "description": "Inicia la reproducción de la playlist en Spotify.
                  Requiere cuenta Premium y un dispositivo activo.",
  "parameters": {
    "playlist_uri": string,
    "device_id": string (opcional — usa dispositivo activo si se omite)
  }
}
```

**Implementación:**
```python
def start_playback(playlist_uri: str, device_id: str = None) -> dict:
    sp = get_spotify_client()
    try:
        kwargs = {"context_uri": playlist_uri}
        if device_id:
            kwargs["device_id"] = device_id
        sp.start_playback(**kwargs)
        return {"success": True, "message": "Reproducción iniciada"}
    except spotipy.SpotifyException as e:
        if "Premium required" in str(e):
            return {
                "success": False,
                "error": "premium_required",
                "message": "Reproducción automática requiere Spotify Premium. "
                           "La playlist fue creada — puedes reproducirla manualmente."
            }
        return {"success": False, "error": str(e)}
```

**Fallback graceful:** si falla por falta de Premium, el agente
informa al usuario y entrega el link de la playlist igualmente.

---

### Tool 5: `get_playback_devices` ⏳ PENDIENTE

Lista los dispositivos Spotify disponibles del usuario para
permitir que el agente elija dónde reproducir.

```json
{
  "name": "get_playback_devices",
  "description": "Lista los dispositivos Spotify activos del usuario.",
  "parameters": {}
}
```

**Implementación:**
```python
def get_playback_devices() -> dict:
    sp = get_spotify_client()
    devices = sp.devices()
    return {
        "devices": [
            {
                "id": d['id'],
                "name": d['name'],
                "type": d['type'],
                "is_active": d['is_active'],
                "volume": d['volume_percent']
            }
            for d in devices['devices']
        ]
    }
```

---

### Tool 6: `add_track_to_database` ⏳ PENDIENTE (v2)

Permite al usuario enriquecer la DB con canciones nuevas.

```json
{
  "name": "add_track_to_database",
  "description": "Añade una canción al dataset local con sus audio features.
                  El LLM estima los features — el usuario los confirma.",
  "parameters": {
    "track_name": string,
    "artist": string,
    "valence": number,
    "energy": number,
    "tempo": number,
    "acousticness": number,
    "danceability": number (opcional),
    "genre": string (opcional)
  }
}
```

**Flujo conversacional:**
```
Usuario: "Agrega 'Papaoutai' de Stromae al dataset"
Agente:  [search_spotify_track → confirma existencia]
         "Basado en mi conocimiento musical, estimo:
          valence=0.28, energy=0.62, tempo=122, acousticness=0.15
          ¿Estos valores te parecen correctos? Puedes ajustar cualquiera."
Usuario: "Sí, correcto"
Agente:  [add_track_to_database → INSERT con source='user_added']
         "Track añadido. ID: track_600001. Ya disponible para futuros arcos."
```

---

## 6. Flujo Completo del Agente — Spotify Premium

```
PASO 1 — INPUT
  Usuario: "Playlist 9 canciones: oscura → épica → paz"

PASO 2 — PHASE DECOMPOSITION (LLM)
  Fase 1: Oscuridad   (3 tracks) → valence 0.0–0.2, energy 0.0–0.4
  Fase 2: Épico       (3 tracks) → valence 0.4–0.6, energy 0.8–1.0
  Fase 3: Paz         (3 tracks) → valence 0.5–0.8, energy 0.0–0.3

PASO 3 — QUERY DATABASE (por cada fase)
  Tool: query_song_database(min_valence=0.0, max_valence=0.2, ...)
  → retorna tracks reales con features verificados
  → retry automático si 0 resultados

PASO 4 — VALIDATION LAYER (Python, fuera del LLM)
  validate_transitions() → Δvalence ≤ 0.3, Δenergy ≤ 0.3
  validate_arc_slope()   → pendiente positiva en valence Y energy
  coherence_score()      → score 0.0–1.0

PASO 5 — SEARCH EN SPOTIFY (por cada track validado)
  Tool: search_spotify_track("Nils Frahm", "Says")
  → obtiene Spotify URI: "spotify:track:4iV5W9uY..."
  → si no existe: usa siguiente track de la DB

PASO 6 — CREAR PLAYLIST
  Tool: create_and_populate_playlist(
    title="De la oscuridad a la paz",
    track_uris=["spotify:track:xxx", ...]
  )
  → retorna playlist_url

PASO 7 — INICIAR REPRODUCCIÓN (Premium)
  Tool: get_playback_devices() → identifica dispositivo activo
  Tool: start_playback(playlist_uri, device_id)
  → música empieza a sonar

PASO 8 — OUTPUT AL USUARIO
  "✅ Playlist creada y reproduciendo en tu iPhone.
   🎵 9 tracks | Arco: valence 0.15→0.65 | Score: 89%

   Fase 1 — Oscuridad profunda (tracks 1-3):
   Seleccioné tracks con valence 0.10–0.18 y energy 0.20–0.35
   para evocar introspección sin agresividad...

   [link a la playlist en Spotify]"
```

---

## 7. Autenticación Spotify OAuth

### Setup inicial (una vez)

```bash
pip install spotipy python-dotenv
```

```python
# src/agent/spotify_client.py
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

SCOPES = [
    "user-read-playback-state",      # ver estado del reproductor
    "user-modify-playback-state",    # controlar reproducción (Premium)
    "user-read-currently-playing",   # track actual
    "playlist-modify-private",       # crear playlists privadas
    "playlist-modify-public",        # crear playlists públicas
    "user-read-recently-played",     # historial
    "user-library-read",             # liked songs
    "user-library-modify",           # guardar en biblioteca
]

def get_spotify_client() -> spotipy.Spotify:
    """
    Retorna cliente autenticado.
    Primer uso: abre browser para OAuth.
    Usos siguientes: usa token cacheado en .spotify_cache
    """
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
        scope=" ".join(SCOPES),
        cache_path=".spotify_cache"
    ))
```

### Variables de entorno (.env)

```bash
# Spotify
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8080/callback

# LLM
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here

# Agent config
MAX_TRACKS_PER_PLAYLIST=20
MAX_RETRY_ATTEMPTS=3
DEFAULT_TRACKS_PER_PHASE=3
```

### Registro de la app en Spotify Developer Dashboard

```
1. developer.spotify.com → Dashboard → Create App
2. App name: narrative-playlist-agent
3. Redirect URI: http://127.0.0.1:8080/callback
4. APIs used: Web API
5. Guardar CLIENT_ID y CLIENT_SECRET en .env
6. REQUISITO: cuenta Spotify Premium activa del dueño de la app
```

---

## 8. Estructura de Archivos — Versión Spotify

```
narrative-playlist-agent/
├── .env.example
├── .spotify_cache             ← tokens OAuth (no commiteado)
├── requirements.txt
├── SPEC.md                    ✅ v1.1
├── ARCHITECTURE.md            ✅ v1.0
├── PROJECT_HANDOFF.md         ✅ referencia general
├── AGENT.md                   ✅ este documento
│
├── data/
│   ├── raw/dataset.csv        ← Kaggle (no commiteado)
│   └── db/tracks.db           ← SQLite generado
│
├── src/
│   ├── ingestion/
│   │   └── load_dataset.py    ✅ implementado
│   │
│   ├── agent/
│   │   ├── tools.py           ✅ query_song_database
│   │   ├── spotify_client.py  ⏳ OAuth + cliente spotipy
│   │   ├── spotify_tools.py   ⏳ search, create_playlist,
│   │   │                         start_playback, get_devices
│   │   ├── db_tools.py        ⏳ add_track_to_database (v2)
│   │   ├── system_prompt.py   ⏳ System Prompt completo
│   │   └── agent.py           ⏳ Loop principal
│   │
│   ├── validation/
│   │   └── playlist_validator.py  ✅ implementado
│   │
│   ├── schemas/
│   │   └── models.py          ✅ implementado
│   │
│   └── main.py                ⏳ Entry point CLI
│
├── frontend/
│   └── app.py                 ⏳ Streamlit (v2)
│
└── tests/
    ├── conftest.py            ✅ SQLite fixture
    ├── test_models.py         ✅ 18 tests
    ├── test_tools.py          ✅ 36 tests
    ├── test_validator.py      ✅ 27 tests
    ├── test_spotify_tools.py  ⏳ con mocks de spotipy
    └── test_agent_integration.py ⏳
```

---

## 9. System Prompt — Versión Spotify

```
Eres un ingeniero de curación musical con acceso a Spotify Premium.

Tu objetivo es crear playlists narrativas que cuenten una historia
emocional a través de la música y reproducirlas automáticamente.

HERRAMIENTAS DISPONIBLES:
1. query_song_database      → busca canciones por audio features numéricos
2. search_spotify_track     → valida existencia en catálogo Spotify
3. create_and_populate_playlist → crea y llena la playlist
4. get_playback_devices     → lista dispositivos disponibles
5. start_playback           → inicia reproducción (requiere Premium)
6. add_track_to_database    → añade canciones nuevas al dataset local

REGLAS ABSOLUTAS:
1. Nunca inventes canciones. Todos los tracks vienen de query_song_database.
2. Siempre valida con search_spotify_track antes de añadir a la playlist.
   Si un track no existe en Spotify, usa el siguiente de la DB.
3. Divide el arco emocional en 2-5 fases con parámetros numéricos.
4. No emitas el output final hasta que todas las fases tengan tracks
   confirmados en Spotify Y la playlist esté creada.

MAPEO EMOCIONAL:
  Melancolía:  valence 0.0–0.3,  energy 0.0–0.4,  acousticness > 0.6
  Tensión:     valence 0.0–0.3,  energy 0.4–0.7,  tempo > 100
  Furia:       valence 0.0–0.4,  energy 0.8–1.0,  tempo > 120
  Épico:       valence 0.4–0.6,  energy 0.8–1.0,  tempo > 115
  Euforia:     valence 0.7–1.0,  energy 0.7–1.0,  danceability > 0.6
  Paz:         valence 0.5–1.0,  energy 0.0–0.3,  acousticness > 0.8

CONSTRAINT DE TRANSICIÓN:
  Δvalence ≤ 0.3 y Δenergy ≤ 0.3 entre tracks consecutivos.
  Si el usuario pide un cambio abrupto explícito, documéntalo.

MANEJO DE ERRORES:
  - 0 resultados en DB → amplía rangos ±0.1, máx 3 intentos
  - Track no encontrado en Spotify → siguiente de la DB
  - Premium no disponible para playback → entrega playlist con link
  - Sin dispositivos activos → informa al usuario y entrega link
```

---

## 10. Output Schema — Versión Spotify

Extiende `PlaylistOutput` con campos específicos de Spotify:

```python
class SpotifyPlaylistOutput(PlaylistOutput):
    spotify_playlist_id: str
    spotify_playlist_url: str
    playback_started: bool
    playback_device: Optional[str]     # nombre del dispositivo
    tracks_not_found_in_spotify: list[str]  # tracks de DB sin match en Spotify
```

---

## 11. Criterios de Aceptación MVP Spotify

| Criterio | Verificación |
|---|---|
| Sin alucinaciones de tracks | Todo track en output existe en SQLite Y en Spotify |
| Arco matemático válido | validate_arc_slope() pasa en valence Y energy |
| Transiciones suaves | validate_transitions() sin violaciones |
| Playlist creada en Spotify | `spotify_playlist_id` presente en output |
| Reproducción iniciada | `playback_started: True` O error graceful documentado |
| Explicación paramétrica | `phase_explanation` y `arc_summary` presentes |
| Fallback sin Premium | Si playback falla, entrega link funcional |

---

## 12. Dependencias Adicionales vs Versión YouTube Music

```
# Agregar a requirements.txt
spotipy>=2.23.0         # cliente oficial Spotify Web API
```

El resto del stack es idéntico al documentado en PROJECT_HANDOFF.md.

---

## 13. Comparativa Final: Spotify Premium vs YouTube Music

| Aspecto | Spotify Premium | YouTube Music (ytmusicapi) |
|---|---|---|
| Reproducción automática | ✅ API oficial | ❌ no disponible |
| Crear playlist | ✅ oficial | ✅ no oficial |
| Añadir tracks | ✅ oficial | ✅ no oficial |
| Control de volumen | ✅ oficial | ❌ no disponible |
| Cambiar dispositivo | ✅ oficial | ❌ no disponible |
| Costo acceso | Premium obligatorio (~$10/mes) | Gratis (cuenta Google) |
| Estabilidad API | Alta — oficial documentada | Media — puede romperse |
| Audio features | ❌ deprecados | ❌ nunca existieron |
| Requiere dataset Kaggle | ✅ sí (features deprecados) | ✅ sí (nunca existieron) |

**Conclusión:** Spotify Premium es la versión más poderosa del agente
porque permite reproducción automática — el agente actúa completamente
sin intervención del usuario. YouTube Music es la versión más accesible
económicamente pero sin control de playback.

---

*AGENT.md v1.0 — Versión Spotify Premium*
*Referencia cruzada: SPEC.md v1.1 | ARCHITECTURE.md v1.0 | PROJECT_HANDOFF.md*
