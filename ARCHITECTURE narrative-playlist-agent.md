# ARCHITECTURE.md — narrative-playlist-agent
**Versión:** 1.0  
**Estado:** Aprobado para desarrollo  
**Referencia:** SPEC.md v1.1  
**Fecha:** Marzo 2026

---

## 1. Visión Arquitectónica

El sistema es un **agente de IA con tool calling** que opera sobre una base de datos local. No es un chatbot. No es un wrapper de API. Es un sistema autónomo que toma decisiones secuenciales, ejecuta herramientas, valida sus propios resultados y produce outputs estructurados y explicados.

### Principio de diseño central
```
LLM = Motor de razonamiento
SQLite = Fuente de verdad
Python = Capa de control determinista
```

El LLM no tiene acceso directo a los datos. Solo puede interactuar con ellos a través de la herramienta `query_song_database`. Todo lo que el LLM no puede hacer de forma confiable (matemáticas, validación de rangos, detección de duplicados) lo hace Python.

---

## 2. Diagrama de Arquitectura — Vista General

```
┌─────────────────────────────────────────────────────────────────┐
│                         USUARIO                                  │
│                  (CLI en v1 / Streamlit en v2)                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ prompt narrativo (texto libre)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        main.py                                   │
│                     Entry Point / CLI                            │
│   - Recibe input del usuario                                     │
│   - Llama al agente                                              │
│   - Imprime output con Rich                                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     agent/agent.py                               │
│                   AGENT LOOP (núcleo)                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  1. Construye messages[] con system_prompt + user input  │    │
│  │  2. Llama al LLM (Groq API)                              │    │
│  │  3. LLM responde con tool_call → query_song_database     │    │
│  │  4. Python ejecuta la tool y devuelve resultados         │    │
│  │  5. Resultado se añade al context como tool_result       │    │
│  │  6. LLM evalúa y hace siguiente tool_call (o termina)    │    │
│  │  7. Cuando el LLM emite texto final → fin del loop       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────┬──────────────────────────────────────────┬───────────────┘
       │ tool calls                               │ output JSON
       ▼                                          ▼
┌──────────────────┐                 ┌────────────────────────────┐
│   agent/tools.py │                 │  validation/               │
│                  │                 │  playlist_validator.py     │
│  query_song_     │                 │                            │
│  database()      │                 │  - validate_transitions()  │
│                  │                 │  - validate_arc_slope()    │
│  Construye SQL   │                 │  - coherence_score()       │
│  → ejecuta       │                 └────────────┬───────────────┘
│  → retorna rows  │                              │ output validado
└──────┬───────────┘                              │
       │ SQL queries                              ▼
       ▼                              ┌────────────────────────────┐
┌──────────────────┐                  │   schemas/models.py        │
│   data/db/       │                  │                            │
│   tracks.db      │                  │  PlaylistOutput (Pydantic) │
│   (SQLite)       │                  │  Phase                     │
│                  │                  │  Track                     │
│   ~600k tracks   │                  │  ArcStatistics             │
└──────────────────┘                  └────────────────────────────┘
       ▲
       │ generado por
┌──────────────────┐
│  ingestion/      │
│  load_dataset.py │
│                  │
│  CSV (Kaggle)    │
│  → limpieza      │
│  → SQLite        │
└──────────────────┘
```

---

## 3. Flujo de Datos Detallado — Secuencia Completa

```
PASO 1 — Input
  Usuario escribe:
  "Playlist que empiece oscura, suba épica, termine relajada. 9 canciones."

PASO 2 — Agent Loop: Primera llamada al LLM
  messages = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user",   content: "Playlist que empiece oscura..." }
  ]
  → LLM recibe contexto y genera tool_call #1:
  {
    "name": "query_song_database",
    "arguments": {
      "min_valence": 0.0, "max_valence": 0.3,
      "min_energy": 0.0,  "max_energy": 0.4,
      "min_acousticness": 0.6,
      "limit": 3
    }
  }

PASO 3 — Tool Execution
  tools.py recibe los argumentos
  → construye: SELECT * FROM tracks WHERE valence BETWEEN 0.0 AND 0.3
               AND energy BETWEEN 0.0 AND 0.4
               AND acousticness >= 0.6
               ORDER BY RANDOM() LIMIT 3
  → ejecuta contra tracks.db
  → retorna 3 rows como lista de dicts

PASO 4 — Tool Result al contexto
  messages.append({
    role: "tool",
    content: [{ track_name: "...", artist: "...", valence: 0.18, ... }, ...]
  })

  PASO 4b — ¿0 resultados?
    → tools.py activa retry_logic:
       Intento 1: rangos originales → 0 resultados
       Intento 2: amplía ±0.1      → evalúa
       Intento 3: amplía ±0.2      → evalúa
       Intento 4 (si falla todo):  → documenta en output

PASO 5 — Agent Loop: Segunda y tercera llamada (fases 2 y 3)
  El LLM repite el proceso para cada fase del arco.
  Cada tool_result se acumula en messages[].
  El LLM tiene visibilidad completa del contexto acumulado.
  → Usa exclude_track_ids para evitar duplicados entre fases.

PASO 6 — Output Final del LLM
  Cuando todas las fases tienen tracks asignados, el LLM
  emite su respuesta final en formato JSON (Output Schema del SPEC).

PASO 7 — Validation Layer
  playlist_validator.py recibe el JSON
  → validate_transitions(): verifica Δvalence ≤ 0.3, Δenergy ≤ 0.3
  → validate_arc_slope():   verifica pendiente correcta
  → coherence_score():      calcula score 0.0–1.0
  → Si hay violaciones: devuelve lista de tracks problemáticos al agente
    para reemplazo (máx. 1 iteración de corrección)

PASO 8 — Pydantic Validation
  El JSON pasa por PlaylistOutput (Pydantic model)
  → Si el schema no es válido: error capturado, se loggea, no llega al usuario

PASO 9 — Output al usuario
  CLI (Rich): tabla de tracks + explicación por fase + arc_summary
  (v2: Streamlit renderiza el arco con gráfica de valence/energy)
```

---

## 4. Componentes — Responsabilidad de Cada Módulo

### 4.1 `ingestion/load_dataset.py`
**Responsabilidad:** Única vez que se ejecuta. Prepara la base de datos.

```
Funciones:
  download_instructions()  → imprime link de Kaggle (no auto-descarga)
  clean_dataframe(df)      → normaliza géneros, dropea nulls, renombra columnas
  load_to_sqlite(df, path) → crea tracks.db con índices en valence, energy, tempo
  verify_db(path)          → cuenta rows, valida columnas, imprime resumen

Índices SQLite creados:
  CREATE INDEX idx_valence ON tracks(valence);
  CREATE INDEX idx_energy  ON tracks(energy);
  CREATE INDEX idx_tempo   ON tracks(tempo);
  CREATE INDEX idx_genre   ON tracks(track_genre);
  → Sin índices, las queries con BETWEEN en 600k rows son lentas.
```

### 4.2 `agent/system_prompt.py`
**Responsabilidad:** Define el comportamiento del agente como constante Python.

```python
SYSTEM_PROMPT = """
Eres un ingeniero de curación musical...
[contenido completo del System Prompt del SPEC sección 6]
"""
```

No es un archivo de texto. Es una constante Python importable. Esto permite versionarla, testearla y modificarla sin tocar la lógica del agente.

### 4.3 `agent/tools.py`
**Responsabilidad:** Implementación real de `query_song_database`.

```
Funciones:
  build_query(args)          → construye SQL string desde los argumentos del LLM
  execute_query(sql, db)     → ejecuta contra SQLite, retorna lista de dicts
  retry_with_relaxed_params  → amplía rangos ±0.1 por intento, máx. 3 intentos
  format_tool_result(rows)   → convierte rows a formato esperado por el LLM

TOOL_SCHEMA (dict)           → definición JSON del tool para pasarla a la API
```

### 4.4 `agent/agent.py`
**Responsabilidad:** Loop principal. Coordina LLM ↔ tools ↔ validación.

```
Clase: NarrativePlaylistAgent
  __init__(llm_client, db_path)
  run(user_prompt) → PlaylistOutput
    - Construye messages[]
    - Loop: llama LLM → detecta tool_calls → ejecuta → acumula en messages
    - Cuando LLM emite texto final → parsea JSON → valida con Pydantic
    - Llama a playlist_validator
    - Retorna PlaylistOutput validado

  _execute_tool(tool_call)   → despacha al tools.py correcto
  _parse_final_output(text)  → extrae JSON del texto del LLM
  _handle_validation_errors  → reintento de corrección si validator falla
```

### 4.5 `validation/playlist_validator.py`
**Responsabilidad:** Validación matemática determinista. No llama al LLM.

```
Funciones:
  validate_transitions(tracks)
    → Itera pares consecutivos
    → Retorna lista de violaciones: [{position, delta_valence, delta_energy}]

  validate_arc_slope(tracks, direction)
    → direction: "ascending" | "descending" | "arch" | "valley"
    → Usa scipy.stats.linregress sobre valence[] y energy[]
    → Retorna {valence_slope_ok: bool, energy_slope_ok: bool, details: dict}

  calculate_coherence_score(tracks, phases)
    → Score compuesto: 60% slope + 40% transition smoothness
    → Retorna float 0.0–1.0

  ValidationResult (dataclass)
    → transitions_ok: bool
    → slope_ok: bool
    → coherence_score: float
    → violations: list
    → recommendations: list[str]
```

### 4.6 `schemas/models.py`
**Responsabilidad:** Contratos de datos. Evita que el LLM entregue formatos inconsistentes.

```python
class Track(BaseModel):
    position: int
    track_name: str
    artist: str
    valence: float = Field(ge=0.0, le=1.0)
    energy: float  = Field(ge=0.0, le=1.0)
    tempo: float
    genre: str
    transition_note: str

class Phase(BaseModel):
    phase_number: int
    phase_label: str
    emotional_description: str
    valence_range: tuple[float, float]
    energy_range: tuple[float, float]
    tracks: list[Track]
    phase_explanation: str     # OBLIGATORIO — si falta, Pydantic rechaza

class ArcStatistics(BaseModel):
    valence_start: float
    valence_end: float
    energy_start: float
    energy_end: float
    valence_slope: Literal["positive", "negative", "neutral"]
    energy_slope: Literal["positive", "negative", "neutral"]
    arc_coherence_score: float

class PlaylistOutput(BaseModel):
    playlist_title: str
    user_prompt: str
    narrative_arc: str
    total_tracks: int
    phases: list[Phase]
    arc_statistics: ArcStatistics
    arc_summary: str           # OBLIGATORIO
```

### 4.7 `main.py`
**Responsabilidad:** Entry point. Orquesta el flujo completo para CLI.

```
flujo:
  1. Lee input del usuario (Rich prompt)
  2. Instancia NarrativePlaylistAgent
  3. Llama agent.run(user_prompt)
  4. Recibe PlaylistOutput
  5. Imprime con Rich:
     - Tabla de tracks por fase
     - Explicación de cada fase
     - Arc statistics
     - Arc summary
```

---

## 5. Diagrama de Secuencia — Agent Loop

```
main.py          agent.py         LLM (Groq)       tools.py        tracks.db
   │                │                 │                │                │
   │──run(prompt)──►│                 │                │                │
   │                │──messages[]────►│                │                │
   │                │                 │                │                │
   │                │◄──tool_call #1──│                │                │
   │                │                 │                │                │
   │                │──execute_tool()─────────────────►│                │
   │                │                 │                │──SQL query────►│
   │                │                 │                │◄──rows─────────│
   │                │◄────────────────────────────rows─│                │
   │                │                 │                │                │
   │                │──tool_result───►│                │                │
   │                │                 │                │                │
   │                │◄──tool_call #2──│  (fase 2)      │                │
   │                │──execute_tool()─────────────────►│                │
   │                │◄────────────────────────────rows─│                │
   │                │──tool_result───►│                │                │
   │                │                 │                │                │
   │                │◄──tool_call #3──│  (fase 3)      │                │
   │                │──execute_tool()─────────────────►│                │
   │                │◄────────────────────────────rows─│                │
   │                │──tool_result───►│                │                │
   │                │                 │                │                │
   │                │◄──final JSON────│  (todas fases  │                │
   │                │                 │   completas)   │                │
   │                │                 │                │                │
   │                │──validate()     │                │                │
   │                │──parse Pydantic │                │                │
   │                │                 │                │                │
   │◄──PlaylistOutput│                │                │                │
   │                │                 │                │                │
```

---

## 6. Decisiones de Arquitectura Justificadas

### ¿Por qué SQLite y no ChromaDB?
Las consultas del agente son filtros por rangos numéricos: `valence BETWEEN 0.1 AND 0.3`. SQL es la herramienta correcta para esto. Una base vectorial resuelve búsqueda semántica sobre texto — no es el problema aquí. Añadir ChromaDB en v1 sería complejidad sin valor.

**ChromaDB entra en v2** si se añade búsqueda por descripción textual de mood: *"canciones que suenen como lluvia en ventana"* → embedding → vector search. Ese es el caso de uso correcto.

### ¿Por qué el agente no construye el SQL directamente?
El LLM construye SQL directamente en Text-to-SQL clásico. Aquí el LLM llama a una tool con parámetros estructurados, y Python construye el SQL. Esto es más seguro (sin SQL injection), más predecible (el agente no puede inventar columnas que no existen) y más fácil de testear unitariamente.

### ¿Por qué la validación está fuera del agente?
Porque el LLM no es determinista con matemáticas. Decirle al LLM "asegúrate de que la pendiente sea positiva" no garantiza nada. La capa de validación en Python sí lo garantiza. Esta separación entre *razonamiento* (LLM) y *verificación* (Python) es un patrón estándar en sistemas de agentes en producción.

### ¿Por qué Pydantic para el output?
El LLM puede devolver JSON con campos faltantes, tipos incorrectos o estructura diferente en cada ejecución. Pydantic convierte eso en un error capturado con un mensaje claro, en lugar de un crash silencioso o un output corrupto que llega al usuario.

---

## 7. Manejo de Errores por Capa

```
CAPA              ERROR                         RESPUESTA
─────────────────────────────────────────────────────────────
tools.py          0 resultados SQL              retry_logic (3 intentos)
tools.py          DB no encontrada              FileNotFoundError con mensaje claro
tools.py          Columna faltante en DB        schema check al iniciar el agente

agent.py          LLM no retorna tool_call      re-prompt con instrucción explícita
agent.py          Loop infinito (>10 calls)     max_iterations guard → error limpio
agent.py          JSON inválido en output LLM   intento de reparación + Pydantic

validator.py      Violación de transición       lista de tracks problemáticos
validator.py      Pendiente incorrecta          reporte detallado al agente

schemas/models.py Campo obligatorio faltante    ValidationError con campo específico

main.py           Groq API key faltante         mensaje claro + link a .env.example
main.py           DB no inicializada            instrucción para correr load_dataset.py
```

---

## 8. Configuración del Entorno

### Setup inicial (una sola vez)
```bash
# 1. Clonar repo
git clone https://github.com/andervrz/narrative-playlist-agent
cd narrative-playlist-agent

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# → editar .env con tu GROQ_API_KEY

# 4. Descargar dataset de Kaggle y poner en data/raw/
# Link: kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

# 5. Generar SQLite
python src/ingestion/load_dataset.py

# 6. Correr el agente
python src/main.py
```

### requirements.txt
```
openai>=1.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pandas>=2.0.0
scipy>=1.11.0
numpy>=1.24.0
rich>=13.0.0
streamlit>=1.30.0      # v2
```

---

## 9. Orden de Desarrollo (Secuencia de Implementación)

El orden importa. Construir en secuencia equivocada genera re-trabajo.

```
SEMANA 1 — Base de datos y datos
  [ ] load_dataset.py → limpiar CSV → generar tracks.db
  [ ] Verificar que las columnas existen y los índices funcionan
  [ ] Test manual: query SQL básica en terminal SQLite

SEMANA 1 — Schemas y contratos
  [ ] schemas/models.py → Track, Phase, ArcStatistics, PlaylistOutput
  [ ] Test: instanciar modelos con datos de prueba, verificar validaciones

SEMANA 2 — Tools
  [ ] tools.py → build_query(), execute_query(), retry_logic()
  [ ] TOOL_SCHEMA dict
  [ ] Test unitario: query con parámetros conocidos → resultado esperado
  [ ] Test: parámetros imposibles → retry hasta relajar → resultado

SEMANA 2 — Validación
  [ ] playlist_validator.py → validate_transitions(), validate_arc_slope()
  [ ] Test: playlist con violación conocida → detectada correctamente
  [ ] Test: playlist con arco ascendente → pendiente positiva confirmada

SEMANA 3 — Agent Loop
  [ ] system_prompt.py → System Prompt completo
  [ ] agent.py → loop básico con tool calling
  [ ] Test de integración: prompt simple → tool call → resultado
  [ ] Test completo: prompt narrativo → playlist completa → JSON válido

SEMANA 3 — CLI y presentación
  [ ] main.py → input usuario → output Rich formateado
  [ ] README.md → instrucciones de setup + demo GIF
  [ ] Subir a GitHub → repo público

SEMANA 4 — v2 (opcional)
  [ ] frontend/app.py → Streamlit UI
  [ ] Gráfica de arco emocional (valence/energy por track)
```

---

## 10. Lo que este proyecto demuestra en entrevistas

Cada componente mapea a una competencia de AI Engineer:

| Componente | Competencia demostrada |
|---|---|
| Agent Loop con tool calling | Agentic AI — el core del curso de Ed Donner |
| System Prompt con reglas numéricas | Context Engineering real |
| Tool schema JSON bien definido | Spec Driven Development |
| SQLite con índices | Integración de bases de datos con LLMs |
| Pydantic en el output | Structured outputs — patrón de producción |
| playlist_validator.py | Separación de razonamiento vs verificación |
| retry_logic en tools | Resiliencia de agentes — manejo de fallos |
| Arquitectura en capas | Pensamiento de sistemas |

---

*ARCHITECTURE.md v1.0 — Referencia: SPEC.md v1.1*  
*Siguiente paso: implementación en orden definido en sección 9.*
