---
name: arc-validation
description: >
  Interpreta y explica los resultados de la validación matemática
  del arco emocional. Usar cuando el usuario pregunte por la calidad
  de su playlist, pida explicación del score de coherencia, quiera
  saber por qué hay violaciones de transición, o solicite mejorar
  una playlist ya generada.
  También se activa automáticamente cuando playlist_validator
  detecta violaciones en el arco.
version: 1.0.0
author: andervrz
---

## Objetivo

Traducir los resultados técnicos del validador matemático
(`playlist_validator.py`) a lenguaje natural que el usuario
pueda entender y actuar sobre ellos.

---

## Métricas que maneja esta skill

### Score de coherencia (0.0 – 1.0)

```
≥ 0.85  → Excelente. El arco es matemáticamente sólido.
0.70–0.84 → Bueno. Pequeñas irregularidades no perceptibles al oído.
0.55–0.69 → Aceptable. Algunas transiciones pueden sentirse bruscas.
< 0.55  → Bajo. El arco emocional no es coherente matemáticamente.
```

**Cómo explicarlo al usuario:**

```
Score 0.91:
"Tu playlist tiene una coherencia del 91%. El viaje emocional
es matemáticamente suave — cada canción lleva naturalmente a la
siguiente sin saltos bruscos."

Score 0.62:
"La coherencia es del 62%. Hay algunas transiciones que pueden
sentirse un poco abruptas. Te muestro cuáles son y cómo mejorarlas."
```

### Pendiente del arco (slope)

| Valor | Significa | Cuándo es válido |
|---|---|---|
| `positive` | valence/energy sube a lo largo de la playlist | arcos ascendentes |
| `negative` | valence/energy baja | arcos descendentes |
| `neutral` | sin cambio significativo | playlists de mood constante |

### Violaciones de transición

Una violación ocurre cuando Δvalence > 0.3 o Δenergy > 0.3
entre dos tracks consecutivos.

---

## Cómo interpretar y comunicar violaciones

Cuando el validador reporta violaciones, explícalas así:

```
Violación entre Track 3 y Track 4:
  Track 3: "Dark Song" — valence: 0.15, energy: 0.35
  Track 4: "Happy Song" — valence: 0.72, energy: 0.80
  Δvalence: 0.57 (supera el límite de 0.3)
  Δenergy:  0.45 (supera el límite de 0.3)

→ Explicación al usuario:
"El salto entre 'Dark Song' y 'Happy Song' es muy abrupto —
 es como pasar de luto a fiesta sin transición. Necesitamos
 un track intermedio que conecte ambos estados."
```

---

## Acciones correctivas disponibles

### Acción 1 — Reemplazar track problemático

Si hay una violación en la posición N:

```
1. Identifica el rango intermedio necesario
   Ej: valence 0.15→0.72 necesita un puente en 0.40–0.50

2. Llama query_song_database con ese rango intermedio
   query(min_valence=0.35, max_valence=0.55,
         min_energy=0.50, max_energy=0.65)

3. Inserta el track puente entre los dos conflictivos

4. Re-evalúa las transiciones
```

### Acción 2 — Reordenar tracks dentro de una fase

Si los tracks están en orden subóptimo dentro de una fase,
reorganízalos de menor a mayor valence/energy para suavizar
la progresión interna.

### Acción 3 — Informar sin corregir

Si el usuario pidió el cambio abrupto explícitamente
(ej: "quiero un contraste fuerte entre las fases"),
documenta la violación pero no la corrijas.

```
"Detecté un salto brusco entre la Fase 1 y la Fase 2
(Δvalence=0.52), tal como solicitaste para el contraste dramático.
Lo he documentado en el JSON de la playlist."
```

---

## Formato del reporte al usuario

Cuando el validador retorna resultados, presenta así:

```
📊 ANÁLISIS DEL ARCO EMOCIONAL
───────────────────────────────
Coherencia global:    91% ✅
Pendiente valence:    positiva ✅  (0.12 → 0.85)
Pendiente energy:     positiva ✅  (0.20 → 0.88)
Transiciones válidas: 8/8 ✅

El arco matemáticamente representa un viaje ascendente
desde la introversión hasta la celebración.
```

Si hay violaciones:

```
📊 ANÁLISIS DEL ARCO EMOCIONAL
───────────────────────────────
Coherencia global:    68% ⚠️
Pendiente valence:    positiva ✅
Pendiente energy:     positiva ✅
Transiciones válidas: 6/8 ⚠️

Violaciones detectadas:
  Track 3→4: Δvalence=0.45 (límite: 0.3)
  Track 6→7: Δenergy=0.38  (límite: 0.3)

¿Quieres que corrija estas transiciones automáticamente?
```

---

## Nota sobre el validador

`playlist_validator.py` es código Python determinista —
no es el LLM quien valida. El LLM interpreta y comunica
los resultados del validador. Esta separación es intencional:
Python garantiza la matemática, el LLM garantiza la comunicación.
