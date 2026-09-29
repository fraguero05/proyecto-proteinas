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

**Estado actual: Hitos 1 y 2 completos, Hito 3 en curso.** Funcionan `fetch`,
`curate`, `db-stats`, `predict` (AlphaFold DB + pLDDT), `design` (variantes con
ProteinMPNN) y `md-analyze` (análisis de trayectorias). Dentro del Hito 2 quedan
pendientes ColabFold y ESMFold como fuentes alternativas de estructura. Del Hito 3
está hecha la parte C (análisis); las partes A y B requieren GROMACS, que no se
puede instalar en el entorno de desarrollo — ver "Cómo reproducir el Hito 3". Faltan
las fases 4 y 5, cuyos comandos validan argumentos y configuración pero informan en
qué hito se implementan y terminan con código de salida `2`.

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
| **ProteinMPNN** | Hito 2, parte B | `git clone https://github.com/dauparas/ProteinMPNN tools/ProteinMPNN` — no está en PyPI, trae los pesos adentro |
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
pdpipe simulate --input designs/1UBQ_var03.pdb --ns 2   # Fase 3 (requiere GROMACS)
pdpipe md-analyze --topology md/sistema.gro --trajectory md/prod.xtc   # Fase 3
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

---

## Cómo reproducir el Hito 2, parte A

Fase 2, primer paso: estructura predicha desde AlphaFold DB y confianza por residuo.

```bash
uv pip install -e ".[data,dev]"     # todavía no hace falta PyTorch

pytest -v

# Descarga el modelo de AlphaFold para la lisozima y reporta su pLDDT
pdpipe predict --uniprot P00698
```

**Qué mirar:**

- `Media 93.89` contra `Media según la API 93.88`. Esa fila existe a propósito: el pLDDT
  se calcula leyendo el campo B-factor del modelo, y contrastarlo con el valor que
  reporta la propia API es lo que prueba que se está leyendo la columna correcta. El
  centésimo de diferencia es redondeo — el formato PDB guarda el B-factor con dos
  decimales y la API calcula sobre la precisión completa.
- La distribución por banda debe dar `1.4% / 9.5% / 0.7% / 88.4%`, idéntica a los campos
  `fractionPlddt*` de la API.
- En `data/processed/AF-P00698-F1_plddt.csv`, que los residuos 1 a 18 tengan pLDDT bajo y
  del 19 en adelante salte a >90: ese corte es el péptido señal de la lisozima, que es
  flexible. Si el pLDDT fuera uniforme, estaría mal leído.

Fuentes alternativas (todavía no implementadas, salen con código `2` y un mensaje que
explica la alternativa):

```bash
pdpipe predict --uniprot P00698 --source colabfold   # va por notebook, requiere GPU
pdpipe predict --uniprot P00698 --source esmfold     # pendiente
```

### pLDDT: qué es y por qué se lee del B-factor

AlphaFold no escribe la confianza en un campo propio: la guarda en la **columna del
B-factor**, que en una estructura experimental significa otra cosa (el factor de
temperatura). Por eso un archivo de AlphaFold no se puede interpretar como uno del PDB
sin saberlo: un "B-factor" de 90 sería pésimo en un cristal y es excelente en una
predicción.

Las bandas son las que define DeepMind y usa AlphaFold DB para colorear sus modelos:

| Banda | pLDDT | Interpretación |
|---|---|---|
| muy baja | < 50 | probablemente desordenado |
| baja | 50–70 | poco confiable |
| confiable | 70–90 | buena confianza en el plegamiento |
| muy alta | ≥ 90 | calidad experimental |

`design.plddt_min` en el `config.yaml` (70 por defecto) dispara un **aviso**, no un error:
un modelo de confianza media puede seguir sirviendo si las regiones que importan están
bien resueltas. Quien decide es quien mira los números.

### Refrescar las fixtures de test

Los tests corren contra respuestas reales grabadas. Si una API cambia de forma:

```bash
python tests/fixtures/capturar_fixtures.py
git diff tests/fixtures/    # revisar qué cambió antes de commitear
```

---

## Cómo reproducir el Hito 2, parte B

Fase 2, segundo paso: dado el esqueleto, qué secuencias podrían plegarse en él.

ProteinMPNN **no es un paquete de PyPI**: es un repositorio que trae los pesos adentro,
así que se clona y el pipeline lo invoca por subprocess. La carpeta `tools/` está en el
`.gitignore` justamente por eso.

```bash
# 1. Clonar ProteinMPNN (trae los pesos adentro) e instalar PyTorch CPU
git clone https://github.com/dauparas/ProteinMPNN tools/ProteinMPNN
uv pip install -e ".[data,ml,dev]"

# 2. La suite completa pasa. Sin el clon, el test que corre ProteinMPNN de
#    verdad se saltea y los otros 33 del diseño siguen corriendo.
pytest -q

# 3. Generar variantes sobre la ubiquitina
pdpipe design --input tests/fixtures/1UBQ.pdb --n-sequences 8
```

Si ProteinMPNN no está clonado, `design` sale con código `2` y el mensaje trae el
`git clone` exacto. Si lo clonaste en otro lado, apuntá `design.proteinmpnn_home` del
`config.yaml` a esa ruta.

**Qué mirar** (números de una corrida real sobre 1UBQ con `seed: 42`):

- La tabla de variantes: `global_score`, cuántas mutaciones tiene cada una respecto de
  la original y en qué posiciones. La mejor va resaltada en verde.
- `data/processed/1UBQ_variantes.fasta`, con la secuencia original primero para poder
  alinear sin ir a buscarla a otro archivo, y
  `data/processed/1UBQ_variantes.csv` con el detalle por variante.
- **La identidad ronda el 55%, no el 95%.** Es lo esperable y no un error: ProteinMPNN
  propone secuencias compatibles con el esqueleto, no copias de la original. Recuperar
  la mitad de la secuencia nativa es el orden de magnitud que reporta el método.
- **Las mutaciones evitan el núcleo hidrofóbico.** En esa corrida, 28 de 273 mutaciones
  (10%) cayeron en los 16 residuos del núcleo, que son el 21% de la proteína: la mitad
  de lo que daría el azar. Los residuos enterrados están más restringidos por el
  esqueleto, así que el modelo los conserva.
- El efecto de `--temperature` es real pero moderado: la identidad media pasa de 55.1%
  a `0.1` a 50.5% a `0.5`. Sube la diversidad entre variantes más que la distancia al
  original.

Los scores exactos dependen del modelo de pesos, de la semilla y de la temperatura, así
que no hay un número fijo que deba salir. Lo que sí es reproducible: dos corridas con la
misma semilla y los mismos parámetros dan un CSV idéntico. La semilla sale de `seed` en
el `config.yaml` y queda registrada en el `run_manifest.json`.

### Diseño inverso: qué hace ProteinMPNN

La predicción de estructura va de secuencia a estructura. El diseño va al revés: se fija
el esqueleto y se busca qué secuencias podrían plegarse en él. Es el paso que convierte
una estructura depositada en una variante propia.

| Columna | Qué significa |
|---|---|
| `global_score` | Log-verosimilitud negativa de toda la secuencia. **Más bajo es mejor**, al revés de lo que sugiere la palabra "score". |
| `score` | Lo mismo, restringido a las posiciones diseñadas. |
| `recuperación` | Fracción de la secuencia original que el modelo reprodujo por su cuenta. Alta = el esqueleto determina fuertemente la secuencia. |
| `identidad` | Fracción de residuos iguales al original, calculada por el pipeline. |

> **Advertencia:** una variante con buen score es una hipótesis computacional, no una
> proteína que se sepa que funciona. Validar experimentalmente es otro trabajo; el
> pipeline solo la propone y la caracteriza.

### Posiciones fijas: van en numeración del PDB

`design.fixed_positions` lista los residuos que no se deben mutar (un sitio catalítico,
por ejemplo) **en la numeración del PDB**, que es la que se ve en un visualizador y la
que usa el CSV de pLDDT de la parte A.

ProteinMPNN, en cambio, numera las posiciones fijas `1..N` sobre la cadena diseñada. El
pipeline traduce entre las dos. La distinción importa porque confundirlas no da error:
congela el residuo equivocado y el resultado parece correcto. Por eso, una posición que
no exista en la estructura corta la ejecución en vez de ignorarse en silencio.

```yaml
design:
  fixed_positions: [35, 52]   # numeración del PDB, no índice de secuencia
```

### ProteinMPNN en Windows

ProteinMPNN deriva rutas con `ruta.rfind("/")` en dos lugares: para encontrar sus pesos
(`protein_mpnn_run.py`) y para nombrar el archivo de salida (`protein_mpnn_utils.py`).
En Windows las rutas vienen con barras invertidas, `rfind` no encuentra separador y las
dos deducciones salen mal — busca los pesos bajo `protein_mpnn_run.p` y quiere escribir
`seqs/C:\...\1UBQ.fa`.

El pipeline lo esquiva solo: pasa `--path_to_model_weights` explícito y le da las rutas
con barras normales, que Windows acepta igual. No hay que hacer nada, pero si lo corrés
a mano desde `tools/ProteinMPNN` vas a encontrarte con esos dos errores.

---

## Cómo reproducir el Hito 3, parte C

Fase 3, análisis de trayectorias: qué hizo la proteína durante la simulación.

Esta parte **no necesita GROMACS**. Trabaja sobre una trayectoria ya simulada, venga
de donde venga — de `pdpipe simulate` si tenés GROMACS, o de un notebook de Colab.

```bash
uv pip install -e ".[data,md,dev]"

# 1. La suite completa
pytest -q

# 2. Análisis sobre la trayectoria de ejemplo incluida en el repo
pdpipe md-analyze --topology tests/fixtures/traj_1UBQ.pdb

# 3. Sobre una simulación real: topología y trayectoria por separado
pdpipe md-analyze --topology md/sistema.gro --trajectory md/produccion.xtc
```

**Qué mirar:**

- La tabla de medidas, con media, desvío, valor inicial, final y **deriva**. En el
  RMSD la deriva es la señal más directa de si el sistema se estabilizó: si sigue
  creciendo al final de la simulación, la producción fue demasiado corta.
- Los residuos más móviles del RMSF. Sobre la trayectoria de ejemplo salen el 76 y el
  75 — la cola C-terminal de la ubiquitina, que es justamente su región flexible.
- Las advertencias. La trayectoria de ejemplo dispara dos, a propósito: que no declara
  paso de tiempo y que no se pueden contar puentes de hidrógeno.
- Las cinco figuras en `data/processed/figuras/`, más el panel `_resumen.png` con
  todo junto, que es la que suele ir al documento.

### Qué mide cada cosa

| Medida | Pregunta que responde | Si sube sostenidamente |
|---|---|---|
| **RMSD** | ¿Cuánto se alejó de la estructura de partida? | El sistema todavía no se equilibró |
| **RMSF** | ¿Dónde está la flexibilidad, residuo por residuo? | (es un perfil, no una serie) |
| **Radio de giro** | ¿Qué tan compacta está? | Se está desplegando |
| **SASA** | ¿Cuánta superficie queda expuesta al solvente? | Se está abriendo; acompaña al radio de giro |
| **Puentes de hidrógeno** | ¿Cuántos hay cuadro a cuadro? | Si *bajan*, se pierde estructura secundaria |

### Dos decisiones que conviene conocer

**La SASA no la calcula MDAnalysis.** El plan original decía "análisis con MDAnalysis:
RMSD, RMSF, radio de giro, SASA, puentes de hidrógeno", pero MDAnalysis no tiene módulo
de superficie: no existe `MDAnalysis.analysis.sasa` ni equivalente. Se usa la
implementación de Shrake-Rupley de Biopython (`Bio.PDB.SASA.ShrakeRupley`), que ya era
dependencia del proyecto por la Fase 1. Es además la medida más cara de las cinco, así
que `md.sasa_stride` permite analizar uno de cada N cuadros.

**El RMSF exige alinear la trayectoria antes; el RMSD no.** El RMSD superpone cada
cuadro contra la referencia por su cuenta, así que una proteína que rota o se traslada
dentro de la caja no lo afecta. El RMSF, en cambio, mide la dispersión de cada átomo
alrededor de su posición media: si el conjunto rota, esa rotación aparece como
flexibilidad en todos los residuos por igual.

> **Advertencia:** un RMSF calculado sin alinear no da error — da números plausibles y
> equivocados. Sobre la trayectoria de ejemplo, que rota 4° por cuadro a propósito, el
> RMSF sin alinear da 2.06 Å contra 0.76 Å alineado: casi el triple. El pipeline alinea
> siempre antes del RMSF, y hay un test que lo fija.

### Leer las salidas en PowerShell

Las tablas y los JSON se escriben en UTF-8 sin BOM. `Get-Content` de Windows PowerShell
5.1 los lee en la codepage ANSI del sistema, así que los acentos y los símbolos como
`Å` aparecen como `Ã…`. El archivo está bien; es el comando:

```powershell
Get-Content data\processed	raj_1UBQ_md_resumen.json -Encoding UTF8
```

El RMSF se reporta **promediado por residuo**, no por átomo. MDAnalysis lo devuelve por
átomo; con la selección por defecto hay un carbono alfa por residuo y coinciden, pero
`--selection backbone` son cuatro átomos por residuo y sin promediar el perfil repetiría
cada número de residuo cuatro veces.

### Las fixtures de trayectoria

Los tests no tocan la red ni requieren GROMACS. Las dos trayectorias son sintéticas y
están construidas para que la respuesta correcta se conozca de antemano:

| Fixture | Qué es | Respuesta esperada |
|---|---|---|
| `traj_1UBQ.pdb` | 11 cuadros de la ubiquitina, con ruido proporcional a la distancia al centro y una rotación rígida de 4° por cuadro | El RMSD ignora la rotación; el RMSF la acusa si no se alinea |
| `aguas_hbond.pdb` | Dos aguas con hidrógenos explícitos: puente lineal a 2.80 Å en el primer cuadro, separadas a 6.00 Å en el segundo | Exactamente `[1, 0]` puentes |

Son PDB multi-modelo y no `.xtc` a propósito: el `.gitignore` excluye las trayectorias
binarias por peso, y un PDB de texto se puede revisar en un diff.

> **Ojo:** son fixtures para probar el *código de análisis*, no simulaciones reales. No
> tienen física atrás. La validación contra una trayectoria de verdad llega con la
> parte B.

### Partes A y B: el estado de GROMACS

Las partes A (preparación del sistema) y B (simulación) necesitan GROMACS, y **no está
instalado en el entorno de desarrollo**: es una máquina Windows administrada, sin
permisos de administrador, y GROMACS no publica binarios para Windows nativo ni está
en conda-forge para esa plataforma. WSL2 requiere elevación.

Eso no bloquea el hito: `pdb2gmx`, la caja, la solvatación, los iones y las cuatro
etapas de simulación se ejecutan desde un notebook de Colab, que es exactamente lo que
el plan original previó para lo que no entra en una laptop. El código del envoltorio es
el mismo en los dos lados; lo único que cambia es dónde existe el ejecutable `gmx`.

Mientras tanto, `pdpipe simulate` falla con un mensaje claro y la sugerencia de
instalación, que es el comportamiento especificado desde el principio.
