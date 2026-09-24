# Prompt para Claude Code — Pipeline de diseño computacional de proteínas

> Cómo usarlo: creá una carpeta vacía, entrá con `claude` y pegá el bloque de abajo como primer mensaje.
> Si preferís, guardá este archivo como `PROMPT.md` dentro del repo y escribí solo: "Leé PROMPT.md y empezá por el Hito 0".

---

## Prompt

Quiero que construyas conmigo, paso a paso, un pipeline reproducible de diseño computacional de proteínas. Es la implementación práctica de mi tesis de grado de Ingeniería en Informática (FP-UNA). Trabajá de forma incremental: no generes todo el proyecto de una sola vez.

### Contexto y objetivo

El pipeline debe permitir pasar de una estructura depositada en el PDB a una variante rediseñada, evaluada energéticamente y caracterizada con modelos predictivos. Cinco fases, iguales a las de la tesis:

1. **Recolección de datos** — descarga y curación desde RCSB PDB y UniProt.
2. **Diseño computacional** — predicción de estructura y generación de variantes de secuencia.
3. **Validación estructural** — minimización, equilibración y dinámica molecular corta.
4. **Modelos de IA** — predicción de estructura secundaria y de propiedades a partir de secuencia.
5. **Análisis y comparación** — métricas frente a la estructura experimental de referencia.

Soy usuario intermedio de Python y no tengo experiencia previa con GROMACS ni con Rosetta. Explicá brevemente cada decisión técnica antes de implementarla.

### Restricciones importantes

- **Todo debe correr en una laptop sin GPU dedicada** en modo por defecto. Para lo que requiera GPU (AlphaFold completo, MD larga) generá un notebook de Google Colab aparte y dejá el path local como fallback.
- Nada de Rosetta en la primera versión: requiere licencia y compilación larga. Usá **ProteinMPNN** (pesos públicos, corre en CPU) para el diseño de secuencia y dejá una interfaz `SequenceDesigner` abstracta para enchufar Rosetta después.
- Para predicción de estructura usá, en este orden de preferencia: (a) descarga directa desde la **AlphaFold Protein Structure Database** vía API cuando el UniProt ID existe, (b) **ColabFold** en notebook, (c) **ESMFold** vía API de ESM Atlas si está disponible. No intentes instalar AlphaFold2 completo localmente.
- Dinámica molecular con **GROMACS** y campo de fuerza AMBER99SB-ILDN, agua TIP3P, caja cúbica, iones para neutralizar. Simulaciones cortas (1–5 ns) como prueba de concepto, con parámetros configurables.
- Todo parametrizable desde un único `config.yaml`. Nada de rutas ni IDs hardcodeados.
- Semillas aleatorias fijas y registradas. Cada ejecución debe escribir un `run_manifest.json` con versiones de software, parámetros, semillas, timestamp y hashes de los archivos de entrada.
- Tests con pytest desde el principio. Los tests no deben requerir descargas de red: usá fixtures con archivos PDB pequeños incluidos en el repo.
- Trabajá solo con proteínas de uso académico estándar (lisozima, ubiquitina, GFP, hemoglobina, enzimas metabólicas bien caracterizadas). No incluyas toxinas ni proteínas de patógenos de la lista de agentes seleccionados; agregá una verificación explícita en la fase de recolección que rechace esos casos y lo documente en el README.

### Stack

Python 3.11, gestión de entorno con `uv` (o `conda` si `uv` no alcanza para las dependencias científicas), Biopython, requests, pandas, NumPy, SciPy, MDAnalysis, PyTorch, scikit-learn, matplotlib, pytest, typer para la CLI, pydantic para validar el config, SQLite para la base local.

### Estructura de repositorio que quiero

```
protein-design-pipeline/
├── README.md
├── pyproject.toml
├── config.yaml
├── src/pdpipe/
│   ├── cli.py
│   ├── config.py
│   ├── phase1_data/      # PDB/UniProt clients, filtros, SQLite
│   ├── phase2_design/    # predicción de estructura + ProteinMPNN
│   ├── phase3_md/        # preparación, GROMACS, análisis de trayectoria
│   ├── phase4_ml/        # datasets, modelos, entrenamiento, evaluación
│   ├── phase5_analysis/  # RMSD/TM-score vs referencia, reportes
│   └── utils/            # logging, manifest, checksums
├── tests/
├── notebooks/            # ColabFold y MD con GPU
└── data/                 # raw/ interim/ processed/ (en .gitignore)
```

CLI esperada:

```
pdpipe fetch --pdb-id 1UBQ
pdpipe curate --resolution-max 2.0 --organism "Homo sapiens"
pdpipe predict --uniprot P0CG48
pdpipe design --input data/processed/1UBQ.pdb --n-sequences 8
pdpipe simulate --input designs/1UBQ_var03.pdb --ns 2
pdpipe analyze --run-id <id>
pdpipe report --run-id <id>
```

### Hitos — hacé uno por vez y parate a que yo lo pruebe

**Hito 0 — Andamiaje.** Repo, pyproject, config.yaml con pydantic, logging, CLI vacía con typer, `run_manifest.json`, pytest corriendo, README con instrucciones de instalación. Nada de ciencia todavía.

**Hito 1 — Fase 1.** Cliente del RCSB PDB (search API + descarga de mmCIF/PDB) y de UniProt REST. Filtros por resolución, organismo, método experimental y longitud. Esquema SQLite con las tablas `proteinas`, `funciones`, `interacciones`, `ptms` tal como están en la tesis. Comando `fetch` y `curate` funcionando end-to-end con 1UBQ y 1LYZ. Verificación de biosafety incluida.

**Hito 2 — Fase 2.** Descarga de modelos desde AlphaFold DB por UniProt ID, con pLDDT parseado desde el campo B-factor y reportado por residuo. Interfaz `SequenceDesigner` + implementación con ProteinMPNN sobre CPU. Salida: FASTA de variantes + tabla con score por secuencia y posiciones mutadas respecto del original.

**Hito 3 — Fase 3.** Preparación del sistema (limpieza de aguas y heteroátomos, `pdb2gmx`, caja, solvatación, iones), minimización, NVT, NPT y producción. Envoltorio en Python que llame a GROMACS por subprocess con manejo de errores legible. Análisis con MDAnalysis: RMSD, RMSF, radio de giro, SASA, puentes de hidrógeno, con gráficos guardados en PNG. Si GROMACS no está instalado, el comando debe fallar con un mensaje claro y un script de instalación sugerido.

**Hito 4 — Fase 4.** Dataset de estructura secundaria a partir de DSSP sobre las estructuras curadas, codificación one-hot y ventanas deslizantes. Modelo BiLSTM en PyTorch y un baseline (regresión logística o random forest). División train/val/test controlando redundancia de secuencia con CD-HIT o, si no está disponible, un clustering por identidad con Biopython. Métricas Q3, precisión, recall, F1 y matriz de confusión.

**Hito 5 — Fase 5.** Comparación de cada variante contra la referencia experimental: RMSD tras superposición, TM-score, diferencias en energía estimada, resumen de estabilidad de la MD. Reporte final en HTML y Markdown con todas las figuras y el manifiesto de la corrida.

### Cómo quiero que trabajes

- Antes de cada hito, mostrame un plan corto y esperá mi visto bueno.
- Escribí los tests junto con el código, no después.
- Commits pequeños y descriptivos, en español.
- Si una dependencia científica no se puede instalar en mi entorno, decímelo y proponé alternativa en vez de dejar código que no corre.
- Cada módulo con docstrings que citen la fase de la tesis que implementa, para que pueda referenciarlo en el documento.
- Cuando termines un hito, dejá en el README una sección "Cómo reproducir el Hito N" con los comandos exactos.

Empezá por el Hito 0.

---

## Sugerencias para después

- Segunda iteración: agregar RFdiffusion para generar esqueletos nuevos, y Rosetta como segundo `SequenceDesigner` si conseguís la licencia académica.
- Para el capítulo de resultados de la tesis: el `run_manifest.json` y los reportes HTML de la Fase 5 son exactamente el material que necesitás para las tablas y figuras.
- Si el entorno local se te complica, pedile a Claude Code que genere primero el notebook de Colab del Hito 2 y trabajá desde ahí.
