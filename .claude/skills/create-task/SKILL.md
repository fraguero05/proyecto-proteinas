---
name: create-task
description: Arrancar un paso nuevo del pipeline de proteínas. Ubica en qué Hito y fase estamos, arma un plan corto y espera el visto bueno del desarrollador antes de tocar código, como exige el PROMPT.md.
allowed-tools: Read, Grep, Glob, Bash
---

# Create Task — proyecto_proteinas

Flujo para arrancar un paso nuevo del pipeline de diseño computacional de proteínas
(Python 3.11 / uv / typer / pytest). Es la implementación de una tesis de grado, así
que el trabajo se organiza en **Hitos**, no en tickets sueltos.

## Cuándo usar este skill

Usarlo cuando:

- El desarrollador pide arrancar un hito, una fase o una funcionalidad nueva.
- Se retoma el proyecto después de un rato y hay que ubicarse ("¿en qué quedamos?").
- Hace falta entender el alcance antes de tocar código.

---

## Instrucciones

1. Ubicarse: qué hito está en curso
2. Identificar el alcance
3. Plan corto y **esperar visto bueno**
4. Programar

### Step 1 — Ubicarse

Antes de proponer nada, averiguar dónde quedó el proyecto. En orden:

```bash
git log --oneline | head -20        # los scopes (fase1, fase2, ...) dicen mucho
grep -n "Estado actual" README.md   # el README declara el hito en curso
grep -n "^## Cómo reproducir" README.md  # qué hitos ya están cerrados y documentados
```

Leer también el `PROMPT.md`, que es el contrato del proyecto: define las cinco
fases, los seis hitos (0 a 5) y las restricciones que no se negocian.

Los hitos y su estado se leen así:

| Hito | Fase | Qué entrega |
|---|---|---|
| 0 | — | Andamiaje: repo, config con pydantic, logging, CLI typer, `run_manifest.json`, pytest |
| 1 | 1 | Clientes RCSB PDB y UniProt, filtros, SQLite, `fetch` + `curate`, bioseguridad |
| 2 | 2 | **A:** AlphaFold DB + pLDDT desde B-factor · **B:** `SequenceDesigner` + ProteinMPNN en CPU |
| 3 | 3 | GROMACS por subprocess, minimización/NVT/NPT/producción, análisis con MDAnalysis |
| 4 | 4 | Dataset DSSP de estructura secundaria, BiLSTM en PyTorch + baseline, métricas Q3 |
| 5 | 5 | RMSD/TM-score vs referencia, reporte final en HTML y Markdown |

### Step 2 — Identificar el alcance

Mapa del código (`src/pdpipe/`):

| Carpeta | Contenido |
|---|---|
| `cli.py` | Comandos de typer: `fetch`, `curate`, `predict`, `design`, `simulate`, `analyze`, `report` |
| `config.py` | Validación estricta del `config.yaml` con pydantic |
| `phase1_data/` | Clientes de API, filtros de curación, esquema SQLite, **verificación de bioseguridad** |
| `phase2_design/` | AlphaFold DB, pLDDT, diseño de secuencias |
| `phase3_md/` | Preparación del sistema y dinámica molecular |
| `phase4_ml/` | Datasets, modelos, entrenamiento, evaluación |
| `phase5_analysis/` | Métricas contra la referencia experimental, reportes |
| `utils/` | Logging, checksums, `run_manifest.json` |

Usar Grep/Glob para encontrar el módulo o la función concreta antes de asumir dónde va el cambio.

### Step 3 — Plan corto y esperar el visto bueno

**Esta es la regla más importante del proyecto.** El `PROMPT.md` dice textualmente:
*"Antes de cada hito, mostrame un plan corto y esperá mi visto bueno"* y *"hacé uno
por vez y parate a que yo lo pruebe"*.

Entonces:

- Presentar un plan **corto** (no un documento): qué se va a implementar, en qué
  archivos, y qué tests lo van a cubrir.
- **Explicar brevemente cada decisión técnica antes de implementarla** — el
  desarrollador es usuario intermedio de Python y no tiene experiencia previa con
  GROMACS ni Rosetta.
- **Parar ahí.** No empezar a escribir código hasta que confirme.

### Step 4 — Programar

Una vez aprobado:

- **Tests junto con el código, no después** (`PROMPT.md`).
- **Los tests no pueden depender de la red**: van contra fixtures grabadas en el repo.
- **Docstrings que citen la fase de la tesis** que implementa el módulo, para poder
  referenciarlo en el documento escrito.
- Nada hardcodeado: rutas, IDs y parámetros salen del `config.yaml`.
- Semillas fijas y registradas; cada corrida escribe su `run_manifest.json`.
- Al cerrar el hito, agregar al README la sección **"Cómo reproducir el Hito N"**
  con los comandos exactos (activa el skill `[[write-docs]]`).
- Si el desarrollador quiere commitear, activa el skill `[[git-commit]]`.

## Reglas

- **No arrancar a codear sin plan aprobado.** Es un pedido explícito del `PROMPT.md`.
- **No hacer dos hitos de una.** Uno por vez, y parar para que lo pruebe.
- **Restricciones que no se negocian** (del `PROMPT.md`):
  - Todo corre en **laptop sin GPU** en modo por defecto. Lo que requiera GPU va a
    un notebook de Colab aparte, con fallback local.
  - **Nada de Rosetta** en la primera versión — ProteinMPNN sobre CPU, detrás de la
    interfaz abstracta `SequenceDesigner` para poder enchufar Rosetta después.
  - **No instalar AlphaFold2 completo local.** Orden de preferencia: AlphaFold DB por
    API → ColabFold en notebook → ESMFold vía ESM Atlas.
  - **Solo proteínas de uso académico estándar** (lisozima, ubiquitina, GFP,
    hemoglobina, enzimas metabólicas bien caracterizadas). Nada de toxinas ni
    patógenos de la lista de agentes seleccionados — y la fase 1 tiene una
    verificación explícita que los rechaza. Si una tarea propone tocar eso, frenar
    y avisar.
- Si una dependencia científica no se puede instalar en el entorno, decirlo y
  proponer alternativa — nunca dejar código que no corre.
- Si la tarea es ambigua sobre qué fase o qué hito toca, preguntar antes de tocar código.
