# pdpipe — Pipeline de diseño computacional de proteínas

Implementación práctica de la tesis de grado de Ingeniería en Informática (FP-UNA).

El pipeline lleva una estructura depositada en el **PDB** hasta una **variante rediseñada**,
validada energéticamente por dinámica molecular y caracterizada con modelos predictivos.
Está pensado para correr en una **laptop sin GPU dedicada**: lo que requiere GPU
(AlphaFold completo, MD larga) se delega a notebooks de Google Colab, con un camino
local de menor costo como alternativa.

## Las cinco fases

| Fase | Módulo | Qué hace | Hito |
|------|--------|----------|------|
| 1. Recolección de datos | `phase1_data` | Descarga y curación desde RCSB PDB y UniProt, base SQLite local | 1 |
| 2. Diseño computacional | `phase2_design` | Estructura predicha (AlphaFold DB) + variantes con ProteinMPNN | 2 |
| 3. Validación estructural | `phase3_md` | Minimización, equilibración y MD corta con GROMACS | 3 |
| 4. Modelos de IA | `phase4_ml` | Estructura secundaria a partir de secuencia (BiLSTM vs. baseline) | 4 |
| 5. Análisis y comparación | `phase5_analysis` | RMSD, TM-score y reportes contra la referencia experimental | 5 |

**Estado actual: Hito 0 completo** (andamiaje). Los comandos de fase validan sus
argumentos y la configuración, pero todavía no ejecutan ciencia: informan en qué hito
se implementan y terminan con código de salida `2`.

## Instalación

Requiere **Python 3.11** (no 3.12+ ni 3.14: varias dependencias científicas, sobre todo
PyTorch, todavía no publican wheels para las versiones más nuevas).

La forma recomendada es con [`uv`](https://docs.astral.sh/uv/), que puede instalar y
aislar el propio Python 3.11 sin tocar el del sistema:

```bash
# 1. Instalar uv (una sola vez)
#    Windows PowerShell:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#    Linux / macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clonar y entrar
git clone https://github.com/fraguero05/proyecto-proteinas.git
cd proyecto-proteinas

# 3. Crear el entorno con Python 3.11 e instalar el paquete
uv python install 3.11
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Activación del entorno:

```powershell
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
```
```bash
source .venv/bin/activate       # Linux / macOS / Git Bash
```

### Extras por fase

Las dependencias pesadas están separadas para que la instalación base sea rápida:

```bash
uv pip install -e ".[dev]"        # andamiaje + pytest          (Hito 0)
uv pip install -e ".[sci,dev]"    # Biopython, NumPy, MDAnalysis (Hitos 1, 3, 5)
uv pip install -e ".[ml,sci,dev]" # + PyTorch, scikit-learn      (Hito 4)
```

### Dependencias externas (no son paquetes de Python)

| Herramienta | Necesaria desde | Instalación |
|-------------|-----------------|-------------|
| **GROMACS** | Hito 3 | `sudo apt install gromacs` (Ubuntu/WSL) |
| **DSSP** | Hito 4 | `sudo apt install dssp` |
| **CD-HIT** | Hito 4 (opcional) | `sudo apt install cd-hit` — si falta, se usa un clustering por identidad con Biopython |

`pdpipe info` muestra cuáles están disponibles en tu máquina.

## Uso

```bash
pdpipe --help                                        # ayuda general
pdpipe info                                          # diagnóstico del entorno
pdpipe fetch --pdb-id 1UBQ                           # Fase 1
pdpipe curate --resolution-max 2.0 --organism "Homo sapiens"
pdpipe predict --uniprot P0CG48                      # Fase 2
pdpipe design --input data/processed/1UBQ.pdb --n-sequences 8
pdpipe simulate --input designs/1UBQ_var03.pdb --ns 2   # Fase 3
pdpipe analyze --run-id <id>                         # Fase 5
pdpipe report --run-id <id>
```

Opciones globales: `--config` (ruta al `config.yaml`; por defecto lo busca hacia arriba
desde el directorio actual) y `--log-level`.

## Configuración

Todo se parametriza desde un único **`config.yaml`** en la raíz: rutas, semilla, criterios
de curación, parámetros de MD, hiperparámetros del modelo y opciones de reporte. No hay
rutas ni identificadores hardcodeados en el código.

La validación es **estricta** (pydantic con `extra="forbid"`): una clave mal escrita corta
la ejecución con un error claro en vez de convertirse en un valor por defecto silencioso
que después aparece como un resultado raro a mitad de una simulación de dos horas.

## Reproducibilidad

Cada corrida escribe un **`run_manifest.json`** en `runs/<run_id>/` con:

- **`run_id`** ordenable cronológicamente (`AAAAMMDDTHHMMSSZ-xxxxxx`) y timestamps en UTC
- **Versiones** del intérprete, el sistema operativo, cada paquete relevante y GROMACS
- **Semillas** efectivamente aplicadas a `random`, NumPy y PyTorch — incluyendo cuáles
  *no* se pudieron aplicar, que es información igual de relevante
- **Configuración y parámetros** completos de esa corrida
- **Hashes SHA-256** de los archivos de entrada y de salida

Junto con los reportes de la Fase 5, es el material directo para las tablas y figuras del
capítulo de resultados.

## Bioseguridad

El pipeline trabaja **solo con proteínas de uso académico estándar**: lisozima, ubiquitina,
GFP, hemoglobina y enzimas metabólicas bien caracterizadas.

La Fase 1 incluye una **verificación explícita** (`data.biosafety_check` en el `config.yaml`,
activada por defecto) que rechaza estructuras correspondientes a toxinas y a proteínas de
patógenos de la lista de agentes seleccionados. Cuando una estructura es rechazada, el
motivo queda registrado en el `run_manifest.json` de la corrida.

Esa verificación se implementa en el Hito 1; esta sección documenta el criterio desde ya.

## Desarrollo

```bash
pytest                      # toda la suite
pytest -v                   # con el detalle de cada test
pytest --cov=pdpipe         # con cobertura
pytest tests/test_config.py # un archivo puntual
```

**Los tests nunca tocan la red.** Lo que necesita datos usa archivos chicos incluidos en
`tests/fixtures/`. Si en algún momento hace falta un test con red, se marca con
`@pytest.mark.network` y queda fuera de la corrida por defecto.

### Estructura

```
proyecto-proteinas/
├── config.yaml              # única fuente de parámetros
├── pyproject.toml
├── src/pdpipe/
│   ├── cli.py               # comandos (typer)
│   ├── config.py            # validación del config (pydantic)
│   ├── phase1_data/ … phase5_analysis/
│   └── utils/               # logging, checksums, manifiesto
├── tests/
├── notebooks/               # ColabFold y MD con GPU
└── data/                    # raw/ interim/ processed/ (gitignored)
```

---

## Cómo reproducir el Hito 0

Andamiaje: repositorio, configuración validada, logging, CLI, manifiesto de corrida y
tests. Sin ciencia todavía.

```bash
# Entorno
uv python install 3.11
uv venv --python 3.11
uv pip install -e ".[dev]"

# 1. La suite de tests pasa entera
pytest -v

# 2. La CLI responde y lista los siete comandos
pdpipe --help
pdpipe --version

# 3. Diagnóstico del entorno: versiones, config detectado y herramientas externas
pdpipe info

# 4. Se genera un run_manifest.json real en runs/<run_id>/
pdpipe info --save

# 5. Los comandos de fase validan argumentos y anuncian su hito (exit code 2)
pdpipe fetch --pdb-id 1UBQ
pdpipe predict --uniprot P0CG48
pdpipe design --input tests/fixtures/mini.pdb --n-sequences 8

# 6. Un error de configuración se detecta al arrancar (exit code 1)
pdpipe --config tests/fixtures/no_existe.yaml info
```

**Qué mirar:** que `pytest` pase entero, que `runs/<run_id>/run_manifest.json` exista y
contenga versiones, semillas y config, y que los comandos de fase terminen con código `2`
diciendo en qué hito se implementan.
