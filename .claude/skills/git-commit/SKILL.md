---
name: git-commit
description: Redactar y aplicar un mensaje de commit (Conventional Commits, en español) para proyecto_proteinas. El repo usa git y los commits los hace Claude Code cuando el desarrollador lo pide.
allowed-tools: Read, Grep, Glob, Bash
---

# Git Commit — proyecto_proteinas

Redacta un mensaje de commit siguiendo Conventional Commits, adaptado al pipeline
de proteínas (Python / uv / typer / pytest, control de versiones en **git**).

## Cuándo usar este skill

Usarlo cuando:

- El desarrollador pide commitear los cambios que se acaban de hacer.
- El desarrollador pide redactar/armar un mensaje de commit.

**No usarlo para:**

- Commitear cambios a medio terminar — esperar a que el hito o el paso esté cerrado
  y los tests pasen.
- Hacer `git push` — nunca por defecto, solo si el desarrollador lo pide explícitamente.

---

## Instrucciones

1. Analizar los cambios
2. Determinar tipo y scope
3. Redactar el mensaje
4. Commitear (si corresponde)

### Step 1 — Analizar los cambios

Correr `git status --short` y `git diff` (o `git diff --staged` si ya hay algo
en el índice) para ver qué cambió realmente. No confiar solo en lo que se
conversó: el diff manda.

Si hay cambios de varias cosas distintas mezcladas, proponer separarlos en más
de un commit antes de redactar — este repo mantiene un commit por paso lógico
(ver el historial: cliente HTTP, luego clientes de API, luego bioseguridad,
luego esquema SQLite, cada uno por separado).

**Nunca commitear** archivos que el `.gitignore` excluye a propósito: `data/raw/`,
`data/interim/`, `data/processed/`, `runs/`, `*.sqlite`, modelos entrenados
(`*.pt`, `*.pkl`), trayectorias de dinámica molecular (`*.xtc`, `*.trr`, ...).
Si aparece uno de esos en `git status`, es señal de que algo se generó donde no
debía — avisar en vez de agregarlo con `-f`.

### Step 2 — Tipo

| Type | Cuándo |
|---|---|
| `feat` | Nueva funcionalidad (comando de CLI, cliente de API, filtro, modelo) |
| `fix` | Corrección de un error o bug |
| `refactor` | Restructuración de código sin cambio de comportamiento |
| `test` | Tests nuevos o corregidos |
| `docs` | Cambios en el `README.md`, el `PROMPT.md` o docstrings |
| `chore` | Mantenimiento: estructura, dependencias, configuración, limpieza |
| `perf` | Mejora de rendimiento |

### Step 3 — Scope

Scopes de proyecto_proteinas — inferir de qué carpeta de `src/pdpipe/` se tocó:

**Por fase del pipeline (preferido si el cambio pertenece a una fase concreta):**

| Scope | Carpeta | Contenido |
|---|---|---|
| `fase1` | `src/pdpipe/phase1_data/` | RCSB PDB, UniProt, curación, SQLite, bioseguridad |
| `fase2` | `src/pdpipe/phase2_design/` | AlphaFold DB, pLDDT, diseño de secuencias (ProteinMPNN) |
| `fase3` | `src/pdpipe/phase3_md/` | Dinámica molecular |
| `fase4` | `src/pdpipe/phase4_ml/` | Machine learning |
| `fase5` | `src/pdpipe/phase5_analysis/` | Análisis y figuras |

**Transversales:**

| Scope | Qué toca |
|---|---|
| `cli` | `src/pdpipe/cli.py` — comandos de typer |
| `config` | `src/pdpipe/config.py`, `config.yaml` |
| `utils` | `src/pdpipe/utils/` — logging, checksums, `run_manifest.json` |
| `deps` | `pyproject.toml`, `uv.lock` |

**Reglas:**

- Si el cambio cae dentro de una sola fase → usar esa fase (ej. `feat(fase2)`).
- Si es transversal a todo el proyecto → usar el scope transversal que corresponda,
  o **omitirlo**. En este repo `docs:`, `test:` y `chore:` suelen ir sin scope
  cuando abarcan todo (ej. `docs: documentar el Hito 1`).

### Step 4 — Título

Mirar el historial real antes de escribir (`git log --oneline`). La convención
de este repo es:

- **Idioma:** siempre **español**.
- **Minúscula inicial**, sin punto final, máximo ~72 caracteres.
- **Frase nominal o infinitivo**, describiendo qué trae el commit — no imperativo:
  - ✅ `feat(fase2): extracción del pLDDT desde el campo B-factor`
  - ✅ `feat(fase1): comandos fetch, curate y db-stats`
  - ✅ `docs: documentar el Hito 1`
  - ✅ `test(fase1): 185 tests nuevos contra respuestas reales grabadas`
  - ❌ `feat(fase2): Agregar extracción del pLDDT` (imperativo + mayúscula)
- Si el commit cierra o avanza un hito del `PROMPT.md`, nombrarlo (`Hito 2`).

### Step 5 — Body

- **Idioma:** español.
- **Prosa, no bullets.** Párrafos cortos separados por línea en blanco, ~80
  caracteres por línea.
- Explicar **el qué y sobre todo el porqué**: qué decisión se tomó y qué se
  rompería si se hubiera hecho de otra forma. El historial de este repo usa el
  cuerpo para dejar asentado el razonamiento científico, no para listar archivos.

  Ejemplo del repo (`test(fase2)`):

  > El test que sostiene todo lo demás contrasta la media y las cuatro fracciones
  > por banda que calculamos del B-factor contra las que reporta la propia API:
  > leer la columna equivocada daría 1.0 en vez de 93.89.

- Si el cambio toca **bioseguridad** (`phase1_data`, verificación en tres capas),
  decirlo explícitamente en el cuerpo — es el punto más sensible del proyecto.
- Si el cambio altera resultados reproducibles (esquema de la base, semillas,
  parámetros del `config.yaml`), aclarar si hay que regenerar datos.

### Step 6 — Footer

Cerrar el mensaje con la línea de atribución que indique la sesión, en el formato
que ya usa el historial:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Si la sesión indica otro nombre de modelo, usar ese — pero mantener siempre la
línea, porque todos los commits del repo la tienen.

### Step 7 — Entregar o commitear

Mostrar el mensaje final en un bloque de código. Si el desarrollador pidió
commitear, aplicarlo con un heredoc (nunca con `-m` de una línea, que pierde el
cuerpo):

```bash
git commit -F - <<'EOF'
feat(fase2): diseño de secuencias con ProteinMPNN sobre CPU

La interfaz SequenceDesigner queda separada de la implementación para poder
cambiar de modelo sin tocar el comando de CLI.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
```

Después del commit, mostrar `git log -1 --stat` para confirmar qué entró.

## Reglas

- **Nunca `git push`** salvo pedido explícito del desarrollador.
- **Nunca `--no-verify`** ni saltear hooks.
- **Nunca `git add -A` a ciegas** — revisar `git status` primero y agregar
  solo lo que corresponde al commit.
- **Nunca `git add -f`** sobre algo que el `.gitignore` excluye.
- Si los tests no pasan, decirlo antes de commitear y preguntar si igual se
  sigue — no commitear en silencio sobre una suite roja.
- Si el repo está en `main` y el cambio es grande o experimental, ofrecer crear
  una rama antes — pero este proyecto viene trabajando directo sobre `main`,
  así que no imponerlo.
