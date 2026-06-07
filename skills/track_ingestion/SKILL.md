---
name: track-ingestion
description: >
  Añade canciones nuevas al dataset local cuando el usuario las pide
  y no existen en la base de datos. Usar cuando el usuario diga:
  "agrega X canción", "añade X al dataset", "quiero incluir X",
  o cuando durante la generación de una playlist se detecta que
  una canción solicitada no está en la DB.
  SIEMPRE verificar existencia antes de añadir.
  SIEMPRE pedir confirmación del usuario antes del INSERT.
version: 1.0.0
author: andervrz
tools:
  - check_track_exists
  - add_track_to_database
---

## Objetivo

Enriquecer el dataset local con canciones que el usuario solicita,
estimando sus audio features con conocimiento musical y
guardándolas con `source='user_added'` para trazabilidad.

---

## Paso 1 — Verificar existencia

SIEMPRE empieza por llamar `check_track_exists` antes de cualquier
estimación o INSERT.

```python
check_track_exists({
    "track_name": "nombre exacto",
    "artist": "nombre del artista"
})
```

**Si retorna found=True:**
→ Informa al usuario que ya existe y muestra sus features actuales.
→ Pregunta si quiere actualizar los valores o usarla tal como está.
→ No hagas INSERT.

**Si retorna found=False:**
→ Continúa con la estimación de features.

---

## Paso 2 — Estimar audio features

Usa tu conocimiento musical para estimar los valores.
Presenta la estimación al usuario ANTES de guardar.

### Guía de estimación

**valence (positividad 0.0–1.0):**
- Canciones en modo menor, letras de pérdida → 0.1–0.3
- Canciones ambiguas, melancolía con ritmo → 0.3–0.5
- Canciones neutras, pop estándar → 0.5–0.7
- Canciones alegres, celebración → 0.7–0.9
- Euforia pura, dance → 0.8–1.0

**energy (intensidad 0.0–1.0):**
- Baladas acústicas, piano solo → 0.1–0.25
- Pop suave, indie folk → 0.3–0.5
- Pop/rock estándar → 0.5–0.7
- Rock, electrónica intensa → 0.7–0.85
- Metal, drum & bass, hardstyle → 0.85–1.0

**tempo (BPM):**
- Baladas lentas → 60–80 BPM
- Pop estándar → 90–120 BPM
- Dance/electrónica → 120–135 BPM
- Metal/drum&bass → 140–180+ BPM

**acousticness (0.0–1.0):**
- Producción completamente electrónica → 0.0–0.15
- Pop con instrumentos → 0.15–0.4
- Guitarra acústica + producción → 0.4–0.7
- Completamente acústico → 0.8–1.0

### Ejemplos de referencia

```
Stromae — Papaoutai:
  valence: 0.28  (melancólico a pesar del ritmo)
  energy: 0.62   (beat constante, moderadamente intenso)
  tempo: 122     (tempo del bombo característico)
  acousticness: 0.12 (producción electrónica)

Nils Frahm — Says:
  valence: 0.12  (contemplativo, introspectivo)
  energy: 0.18   (minimalista, suave)
  tempo: 78      (lento, pausado)
  acousticness: 0.35 (piano + electrónica suave)

Metallica — Enter Sandman:
  valence: 0.30  (oscuro, tenso)
  energy: 0.92   (muy intenso)
  tempo: 123     (riff rápido)
  acousticness: 0.02 (completamente eléctrico)
```

---

## Paso 3 — Presentar estimación al usuario

Antes del INSERT, muestra los valores estimados y pide confirmación:

```
Estimé los siguientes valores para "[canción]" de [artista]:

  valence:      0.28  (melancólico con ritmo)
  energy:       0.62  (moderadamente intenso)
  tempo:        122   BPM
  acousticness: 0.12  (producción electrónica)
  género:       pop

¿Estos valores te parecen correctos?
Puedes ajustar cualquiera antes de guardar.
(responde "sí" para confirmar o indica qué cambiar)
```

**No hagas el INSERT hasta recibir confirmación explícita.**

---

## Paso 4 — INSERT en la base de datos

Solo después de confirmación del usuario, llama
`add_track_to_database` con los valores confirmados:

```python
add_track_to_database({
    "track_name": "nombre exacto",
    "artist": "artista",
    "valence": 0.28,
    "energy": 0.62,
    "tempo": 122.0,
    "acousticness": 0.12,
    "danceability": 0.75,       # opcional
    "instrumentalness": 0.0,    # opcional
    "track_genre": "pop"        # lowercase, con guiones si aplica
})
```

---

## Paso 5 — Confirmar al usuario

Después del INSERT exitoso:

```
✅ "[canción]" de [artista] añadida al dataset.
   ID asignado: track_u000001
   Fuente: user_added

Ya está disponible para tus próximas playlists.
```

Si el usuario estaba generando una playlist cuando hizo la solicitud,
continúa con la generación incluyendo el track recién añadido.

---

## Casos especiales

**Canción en otro idioma:**
→ Estima normalmente. Los features son universales independiente
  del idioma de las letras.

**Artista muy nuevo o desconocido:**
→ Informa que la estimación puede ser menos precisa para artistas
  fuera del conocimiento del modelo. Sugiere que el usuario ajuste
  los valores si tiene referencia del sonido.

**Usuario pide agregar múltiples canciones a la vez:**
→ Procesa una por una, verificando existencia y pidiendo
  confirmación para cada una individualmente.

**Usuario quiere actualizar features de una canción existente:**
→ Esta funcionalidad no está en v1. Informa al usuario y
  documenta el caso para v2.
