---
name: playlist-generation
description: >
  Genera playlists narrativas con arco emocional matemáticamente validado.
  Usar cuando el usuario describa un viaje emocional, pida una playlist
  con narrativa, mencione emociones en secuencia, o use frases como:
  "playlist que vaya de X a Y", "arco emocional", "de triste a alegre",
  "para entrenar luego relajarme", "cuéntame una historia con música".
  NO usar para preguntas simples sobre música o artistas específicos.
version: 1.0.0
author: andervrz
tools:
  - query_song_database
  - generate_playlist_file
---

## Objetivo

Traducir un arco emocional en lenguaje natural a una playlist
secuencial con tracks reales de la base de datos local, validada
matemáticamente y exportada en CSV para TuneMyMusic.

---

## Paso 1 — Analizar el prompt del usuario

Lee el prompt e identifica:

- **Número de fases:** cuántos estados emocionales distintos hay
- **Tracks totales:** si el usuario lo especifica, distribuye equitativamente
  entre fases. Si no lo especifica, usa 3 tracks por fase.
- **Géneros preferidos:** si el usuario menciona géneros, úsalos en
  `target_genres` de la query.

Ejemplos de análisis:

```
"De tristeza a euforia, 6 canciones"
→ Fase 1: Tristeza (3 tracks) | Fase 2: Euforia (3 tracks)

"Oscura → épica → paz, 9 canciones"
→ Fase 1: Oscuridad (3) | Fase 2: Épico (3) | Fase 3: Paz (3)

"Para estudiar toda la tarde"
→ Fase única: Concentración (6 tracks, sin arco)
```

---

## Paso 2 — Mapeo emocional obligatorio

Convertir cada emoción a parámetros numéricos exactos:

| Emoción | min_valence | max_valence | min_energy | max_energy | Extras |
|---|---|---|---|---|---|
| Melancolía / Tristeza | 0.0 | 0.3 | 0.0 | 0.4 | min_acousticness: 0.6 |
| Tensión / Oscuridad | 0.0 | 0.3 | 0.4 | 0.7 | min_tempo: 100 |
| Furia / Intensidad | 0.0 | 0.4 | 0.8 | 1.0 | min_tempo: 120 |
| Épico / Climático | 0.4 | 0.6 | 0.8 | 1.0 | min_tempo: 115 |
| Alegría / Euforia | 0.7 | 1.0 | 0.7 | 1.0 | — |
| Paz / Relajación | 0.5 | 1.0 | 0.0 | 0.3 | min_acousticness: 0.8 |
| Concentración | 0.3 | 0.6 | 0.3 | 0.6 | — |
| Nostalgia | 0.2 | 0.5 | 0.1 | 0.4 | min_acousticness: 0.5 |

Para emociones ambiguas o mixtas: usa rangos intermedios y
documenta la tensión en `phase_explanation`.

---

## Paso 3 — Ejecutar queries por fase

Para cada fase:

1. Llama `query_song_database` con los parámetros de la fase
2. Incluye `exclude_track_ids` con todos los IDs ya asignados
   para evitar duplicados entre fases
3. Si retorna 0 resultados → el sistema hace retry automático
   (no necesitas hacer nada, la tool lo maneja internamente)
4. Acumula los track_ids retornados para la siguiente fase

```
Ejemplo de secuencia de queries:

Fase 1 → query(valence 0-0.3, energy 0-0.4)
         → retorna [track_001, track_002, track_003]

Fase 2 → query(valence 0.4-0.6, energy 0.8-1.0,
                exclude_track_ids=["track_001","track_002","track_003"])
         → retorna [track_045, track_089, track_112]
```

---

## Paso 4 — Verificar transiciones suaves

Antes de generar el output, mentalmente verifica que entre
el último track de una fase y el primero de la siguiente,
el salto de valence y energy no supere 0.3.

Si detectas un salto brusco:
- Reordena los tracks dentro de la fase para suavizar la transición
- O solicita un track adicional con valores intermedios

El validador matemático (`playlist_validator.py`) verificará esto
formalmente — pero anticiparlo reduce iteraciones.

---

## Paso 5 — Generar archivos

Cuando TODAS las fases tienen tracks asignados, llama
`generate_playlist_file` con:

```python
{
  "playlist_title": "título creativo que refleje el arco",
  "playlist_description": "descripción breve del viaje emocional",
  "phases": [lista completa de fases con tracks],
  "arc_summary": "párrafo de 2-3 oraciones que describe el viaje
                  emocional completo, mencionando cómo valence y
                  energy evolucionan a lo largo de la playlist"
}
```

**El arc_summary es obligatorio.** Mínimo 50 caracteres.
Debe mencionar la evolución paramétrica, no solo la emoción.

---

## Paso 6 — Output al usuario

Después de que `generate_playlist_file` retorne success=True:

1. Confirma que los archivos fueron generados (muestra los paths)
2. Presenta la playlist de forma legible:
   - Por fase con su etiqueta emocional
   - Cada track con: posición, nombre, artista, V/E/T
3. Explica el arco en lenguaje natural
4. Da las instrucciones de TuneMyMusic

---

## Manejo de errores comunes

**El usuario pide una canción específica que no está en la DB:**
→ Usa la skill `track-ingestion` para añadirla primero,
  luego continúa con la generación de la playlist.

**Arco imposible (ej: "alegre y oscuro al mismo tiempo"):**
→ Interpreta como emoción compuesta (valence 0.3-0.5, energy 0.5-0.7)
  y documenta la ambigüedad en el phase_explanation.

**Usuario pide > 20 tracks:**
→ Informa que el límite es 20 tracks en v1 y ajusta la distribución.

**3 intentos de query sin resultados:**
→ Informa al usuario, sugiere ajustar el género o la emoción de esa fase.
