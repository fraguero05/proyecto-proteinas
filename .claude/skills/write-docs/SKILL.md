---
name: write-docs
description: Escribir y mantener la documentación de proyecto_proteinas. La documentación vive en el README.md (no hay carpeta docs/ ni generador de sitio) y en los docstrings que citan la fase de la tesis. Usar al cerrar un hito o cuando el comportamiento documentado quedó desactualizado.
allowed-tools: Bash, Read, Grep, Glob, Edit, Write
---

# Write Docs — proyecto_proteinas

Escribe o actualiza la documentación del pipeline de diseño computacional de
proteínas. Es la implementación de una tesis de grado de Ingeniería en Informática
(FP-UNA), así que la documentación tiene dos destinatarios: quien reproduce el
pipeline, y el propio documento de tesis.

**Dónde vive la documentación:**

| Dónde | Qué |
|---|---|
| `README.md` | Todo: instalación, uso, configuración, reproducibilidad, bioseguridad, y una sección "Cómo reproducir el Hito N" por hito cerrado |
| Docstrings en `src/pdpipe/` | Qué fase de la tesis implementa cada módulo, para poder citarlo en el documento escrito |
| `PROMPT.md` | **No se toca.** Es el contrato original del proyecto, un documento histórico |

No hay carpeta `docs/` ni generador de sitio (VitePress, MkDocs). Un solo `README.md`,
Markdown plano, se lee desde el repo o desde la web del hosting de git.

## Cuándo usar este skill

Usarlo cuando:

- Se cierra un hito y hay que agregar su sección **"Cómo reproducir el Hito N"**
  (lo pide explícitamente el `PROMPT.md`).
- Un comando nuevo de la CLI queda funcionando y hay que sumarlo a la sección de uso.
- Cambia el `config.yaml`, el esquema de la base o el comportamiento de bioseguridad.
- La documentación existente quedó desfasada respecto al código actual.

**No usarlo para:**

- Editar el `PROMPT.md` — es el pedido original, queda como está.
- Documentar cosas que todavía no funcionan.

---

## Estructura actual del README

En este orden (verificarlo con `grep -n "^#" README.md` antes de insertar algo):

```
# pdpipe — Pipeline de diseño computacional de proteínas
  Estado actual: qué hito está completo y qué está en curso
## Las cinco fases
## Instalación
### Extras por fase
### Dependencias externas (no son paquetes de Python)
## Uso
## Configuración
## Reproducibilidad
## Bioseguridad
## Desarrollo
### Estructura
## Cómo reproducir el Hito 0
## Cómo reproducir el Hito 1
## Cómo reproducir el Hito 2, parte A
   ### pLDDT: qué es y por qué se lee del B-factor
   ### Refrescar las fixtures de test
```

Las secciones "Cómo reproducir" van **al final y en orden de hito**. Las secciones
conceptuales de cada hito (como "pLDDT: qué es y por qué se lee del B-factor")
van como subsección del hito al que pertenecen.

---

## Instrucciones

1. Leer el código antes de escribir
2. Ubicar la sección
3. Escribir siguiendo las convenciones
4. Actualizar el estado del proyecto
5. Verificar que los comandos documentados corren de verdad

### Step 1 — Leer el código

Nunca documentar de memoria ni desde lo que se conversó. Leer el módulo, el
comando de typer en `cli.py`, el modelo de pydantic en `config.py` o el test que
lo cubre. Si algo no se pudo verificar, decirlo explícitamente en el texto en
vez de asumir.

### Step 2 — Ubicar la sección

| Qué cambió | Dónde va |
|---|---|
| Se cerró un hito | Sección nueva `## Cómo reproducir el Hito N` al final, en orden |
| Comando nuevo de la CLI | Bloque de la sección `## Uso` |
| Parámetro nuevo del `config.yaml` | `## Configuración` |
| Esquema de la base, tablas, campos | Subsección del hito correspondiente (ej. `### Esquema de la base`) |
| Regla de bioseguridad | `## Bioseguridad` |
| Dependencia externa (GROMACS, DSSP, CD-HIT) | `### Dependencias externas` |
| Concepto científico que hace falta para entender la salida | Subsección `###` dentro del hito que lo introduce |

Si ya existe una sección relevante, **editarla** en vez de crear una paralela.

### Step 3 — Convenciones de escritura

#### Idioma y tono

- Todo en **español**, claro y directo.
- Nombres de comandos, archivos, clases, campos y tablas: exactamente como están
  en el código (`pdpipe predict --uniprot P0CG48`, `run_manifest.json`, `phase2_design`).

#### Las secciones "Cómo reproducir el Hito N"

Son el corazón del README: tienen que permitir que alguien reproduzca el hito
desde cero. El formato que ya usa el repo es un bloque de shell con los comandos
**numerados y comentados**, en el orden en que se corren:

````markdown
## Cómo reproducir el Hito 2, parte A

Fase 2, primer paso: estructura predicha desde AlphaFold DB y confianza por residuo.

```bash
# 1. La suite completa pasa (sin red: todo contra fixtures grabadas)
uv run pytest -q

# 2. Descarga el modelo de AlphaFold para la lisozima y reporta su pLDDT
uv run pdpipe predict --uniprot P00698
```
````

Reglas de esos bloques:

- Comandos **exactos y copiables**, con el prefijo `uv run` que usa el proyecto.
- Un comentario por paso explicando qué demuestra ese comando.
- Si un paso demuestra un fallo esperado (validación, config inválida, bioseguridad),
  decir cuál es el **exit code** — el README ya lo hace (`exit code 2`, `exit code 1`).
- Usar los IDs de referencia del proyecto: `1UBQ` y `1LYZ` para estructuras,
  `P00698` (lisozima) y `P0CG48` (ubiquitina) para UniProt.

#### Alertas

Bloques de cita estándar de Markdown (no hay build que soporte contenedores especiales):

```markdown
> **Advertencia:** el pLDDT se lee de la columna B-factor del PDB de AlphaFold, no
> es un factor de temperatura real. Leer la columna equivocada daría 1.0 en vez de 93.89.
```

#### Tablas

Para campos, parámetros y esquemas:

```markdown
| Campo | Tipo | Descripción |
|---|---|---|
| `resolution` | float | Resolución en Å reportada por el RCSB. |
```

### Step 4 — Actualizar el estado del proyecto

El README declara arriba de todo en qué hito está el proyecto. Al cerrar un hito
(o una parte de uno), **actualizar esa línea** — si no, queda mintiendo:

```markdown
**Estado actual: Hito 1 completo, Hito 2 en curso.** Funcionan `fetch`, `curate`,
`db-stats` y `predict` (AlphaFold DB + pLDDT). Falta la parte B del Hito 2 (ProteinMPNN)
```

### Step 5 — Verificar

No hay build que valide nada. Verificar a mano:

- **Correr los comandos que se documentan.** Si una sección "Cómo reproducir" tiene
  un comando que falla, la sección está mal. Esto no es opcional.
- Que los bloques de código y las tablas estén bien cerrados.
- Que los enlaces internos (`#cómo-reproducir-el-hito-2-parte-a`) resuelvan.

## Reglas

- **Escribir siempre en español.**
- **Leer el código antes de documentar** — nunca inventar comportamiento. Si algo no
  se pudo verificar, decirlo en el texto.
- **Correr los comandos antes de publicarlos** en una sección "Cómo reproducir".
- **No documentar características no implementadas.** El README describe lo que
  funciona hoy; lo que falta se nombra en la línea de estado, no en una sección propia.
- **Actualizar, no duplicar** — editar la sección existente en lugar de crear una paralela.
- **No tocar `PROMPT.md`.**
- Los docstrings de cada módulo deben citar la fase de la tesis que implementan —
  si se agrega un módulo sin docstring, sumarlo acá también.
