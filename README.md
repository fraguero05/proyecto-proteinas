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

**Estado actual: Hito 1 completo.** `fetch`, `curate` y `db-stats` funcionan end-to-end.
Los comandos de las fases 2 a 5 validan sus argumentos y la configuración, pero todavía
no ejecutan ciencia: informan en qué hito se implementan y terminan con código de
salida `2`.

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
uv pip install -e ".[dev]"        # andamiaje + pytest       (Hito 0)
uv pip install -e ".[data,dev]"   # requests, Biopython      (Hito 1)
uv pip install -e ".[md,dev]"     # NumPy, MDAnalysis        (Hitos 3 y 5)
uv pip install -e ".[ml,dev]"     # PyTorch, scikit-learn    (Hito 4)
uv pip install -e ".[sci,dev]"    # todo lo científico de una
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
pdpipe db-stats                                      # contenido de la base
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

`pdpipe.phase1_data.biosafety` implementa una **verificación explícita**
(`data.biosafety_check` en el `config.yaml`, activada por defecto) con tres capas
independientes:

1. **Organismo** — contra una lista de patógenos y agentes seleccionados (HHS/USDA).
2. **Keywords de UniProt** — anotación curada (`Toxin`, `Neurotoxin`, `Virulence`…), la
   señal más confiable de las tres.
3. **Texto libre** — nombre de la proteína y título de la estructura, para casos sin
   anotar en UniProt.

La comparación de texto es por **palabra completa**, y hay una lista de excepciones, para
que una *antitoxina* o una enzima de *detoxificación* no se rechacen por contener la
subcadena "toxin".

**Falla cerrado:** si no hay datos suficientes para verificar (UniProt caído, estructura
sin accesión), se rechaza. Un falso positivo cuesta una línea en la lista; un falso
negativo mete en el pipeline algo que no debería estar.

Cuando una estructura es rechazada, **no se descargan sus coordenadas**, y el motivo queda
registrado en la tabla `proteinas` (`motivo_rechazo`) y en el `run_manifest.json`. Un
rechazo por bioseguridad no se revierte aflojando los criterios de `curate`.

Las listas son deliberadamente conservadoras: pueden rechazar proteínas inocuas de
organismos patógenos. Si necesitás una, ajustá `ORGANISMOS_EXCLUIDOS` en
`src/pdpipe/phase1_data/biosafety.py` y dejá constancia del criterio en la tesis.

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

---

## Cómo reproducir el Hito 1

Fase 1: descarga desde RCSB PDB y UniProt, curación y base SQLite.

```bash
uv pip install -e ".[data,dev]"

# 1. La suite completa pasa (sin red: todo contra fixtures grabadas)
pytest -v

# 2. Descarga de las dos estructuras de referencia
pdpipe fetch --pdb-id 1UBQ --pdb-id 1LYZ

# 3. Estado de la base
pdpipe db-stats

# 4. Curación con los criterios del config.yaml
pdpipe curate

# 5. Curación más estricta: 1LYZ (2.00 Å) queda afuera, 1UBQ (1.80 Å) pasa.
#    No se re-descarga nada: curate trabaja sobre lo que ya está.
pdpipe curate --resolution-max 1.9

# 6. La verificación de bioseguridad rechaza una neurotoxina botulínica
pdpipe fetch --pdb-id 3BTA
```

**Qué mirar:**

- En el paso 3, `1UBQ` con **longitud 76** y `1LYZ` con **129** — la longitud de la cadena
  cristalizada, no la de la proteína completa de UniProt (P0CG48 tiene 685 residuos).
- En el paso 5, que `1LYZ` se rechace con el motivo explícito y que no haya tráfico de red.
- En el paso 6, que `3BTA` se rechace por organismo, que **no** aparezca `data/raw/3BTA.pdb`
  en el disco, y que el motivo quede en el `run_manifest.json`.

### Esquema de la base

```
proteinas(id, pdb_id, uniprot_id, nombre, organismo, tax_id, metodo,
          resolucion, longitud, secuencia, archivo_path, sha256,
          fecha_descarga, curada, motivo_rechazo)
funciones(id, proteina_id→, tipo, termino_go, descripcion, evidencia)
interacciones(id, proteina_id→, partner_uniprot, partner_gen, tipo, fuente,
              n_experimentos, evidencia)
ptms(id, proteina_id→, posicion, posicion_fin, tipo, descripcion, evidencia)
```

Las tres tablas hijas cuelgan de `proteinas` con `ON DELETE CASCADE`. Las proteínas
**rechazadas se conservan** con su `motivo_rechazo`: saber qué quedó afuera y por qué es
parte del resultado de la curación, y es lo que permite justificar en la tesis el tamaño
del conjunto final.

De dónde sale cada tabla:

| Tabla | Fuente |
|---|---|
| `proteinas` | RCSB (método, resolución) + entidad polimérica (longitud, secuencia, organismo) + UniProt (nombre) |
| `funciones` | UniProt: términos GO (tres ontologías), keywords y el comentario `FUNCTION` |
| `interacciones` | UniProt: comentarios `INTERACTION` (IntAct) y `SUBUNIT` |
| `ptms` | UniProt: features `Modified residue`, `Glycosylation`, `Disulfide bond`, `Cross-link`, `Lipidation`… |

### Refrescar las fixtures de test

Los tests corren contra respuestas reales grabadas. Si una API cambia de forma:

```bash
python tests/fixtures/capturar_fixtures.py
git diff tests/fixtures/    # revisar qué cambió antes de commitear
```
