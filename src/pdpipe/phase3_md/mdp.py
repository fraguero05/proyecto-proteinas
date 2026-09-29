"""Generación de los archivos ``.mdp`` de GROMACS (Fase 3).

Un ``.mdp`` es el archivo de parámetros de una etapa de simulación. Acá se
generan desde el ``config.yaml`` en vez de guardarlos como plantillas fijas,
para que cambiar la temperatura o la duración sea editar el config y no cuatro
archivos a mano.

Las cuatro etapas y para qué sirve cada una:

1. **Minimización** — baja la energía del sistema recién armado. Sin esto,
   un choque entre dos átomos mal ubicados hace explotar la simulación en los
   primeros pasos.
2. **NVT** — equilibra la temperatura a volumen constante, con la proteína
   sujeta por restricciones de posición para que el solvente se acomode
   alrededor sin arrastrarla.
3. **NPT** — equilibra la presión, y con ella la densidad del agua. Sigue con
   restricciones de posición.
4. **Producción** — la simulación de la que se saca la trayectoria, ya sin
   restricciones.

Los parámetros siguen el uso estándar para AMBER99SB-ILDN con agua TIP3P:
PME para la electrostática, cortes a 1.0 nm, LINCS sobre los enlaces con
hidrógeno (que es lo que permite el paso de 2 fs).
"""

from __future__ import annotations

from pathlib import Path

from pdpipe.config import MDConfig
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

ETAPAS = ("minim", "nvt", "npt", "prod")

# Bloques comunes. Se repiten en las tres etapas de dinámica y se dejan acá
# una sola vez para que no se desincronicen entre etapas.
_VECINOS = """; Lista de vecinos y cortes
cutoff-scheme            = Verlet
ns_type                  = grid
nstlist                  = 10
rcoulomb                 = 1.0
rvdw                     = 1.0
"""

_ELECTROSTATICA = """; Electrostática de largo alcance
coulombtype              = PME
pme_order                = 4
fourierspacing           = 0.16
"""

_RESTRICCIONES = """; Restricciones de enlace
continuation             = {continuacion}
constraint_algorithm     = lincs
constraints              = h-bonds
lincs_iter               = 1
lincs_order              = 4
"""


def pasos(ps: float, timestep_fs: float) -> int:
    """Cuántos pasos de integración hacen falta para simular ``ps``.

    Se redondea al entero más cercano: GROMACS cuenta pasos, no tiempo, y
    pedir 100 ps con un paso de 2 fs son exactamente 50 000.
    """
    dt_ps = timestep_fs / 1000.0
    if dt_ps <= 0:
        raise ValueError(f"El paso de tiempo tiene que ser positivo, no {timestep_fs}")
    return max(1, round(ps / dt_ps))


def _frecuencia_salida(md: MDConfig) -> int:
    """Cada cuántos pasos se guarda un cuadro de la trayectoria.

    Guardar cada paso llenaría el disco sin aportar nada: los cuadros
    consecutivos están correlacionados. Se guarda cada ``output_every_ps``.
    """
    return pasos(md.output_every_ps, md.timestep_fs)


def contenido_minim(md: MDConfig) -> str:
    """Minimización de energía por descenso más pronunciado."""
    return f"""; Minimización de energía — generado por pdpipe desde config.yaml
integrator               = steep
emtol                    = 1000.0
emstep                   = 0.01
nsteps                   = {md.minimization_steps}

{_VECINOS}
{_ELECTROSTATICA}
pbc                      = xyz
"""


def contenido_nvt(md: MDConfig, seed: int) -> str:
    """Equilibración a volumen y temperatura constantes."""
    n = pasos(md.nvt_ps, md.timestep_fs)
    salida = _frecuencia_salida(md)
    return f"""; Equilibración NVT — generado por pdpipe desde config.yaml
integrator               = md
dt                       = {md.timestep_fs / 1000.0}
nsteps                   = {n}          ; {md.nvt_ps} ps

; La proteína queda sujeta mientras el solvente se acomoda a su alrededor.
define                   = -DPOSRES

{_RESTRICCIONES.format(continuacion="no")}
; Frecuencia de escritura
nstxout-compressed       = {salida}
nstenergy                = {salida}
nstlog                   = {salida}

{_VECINOS}
{_ELECTROSTATICA}
; Termostato
tcoupl                   = V-rescale
tc-grps                  = Protein Non-Protein
tau_t                    = 0.1     0.1
ref_t                    = {md.temperature_k}   {md.temperature_k}

pcoupl                   = no
pbc                      = xyz
DispCorr                 = EnerPres

; Velocidades iniciales desde la distribución de Maxwell, con semilla fija
; para que la corrida sea reproducible.
gen_vel                  = yes
gen_temp                 = {md.temperature_k}
gen_seed                 = {seed}
"""


def contenido_npt(md: MDConfig) -> str:
    """Equilibración a presión constante: ajusta la densidad del agua."""
    n = pasos(md.npt_ps, md.timestep_fs)
    salida = _frecuencia_salida(md)
    return f"""; Equilibración NPT — generado por pdpipe desde config.yaml
integrator               = md
dt                       = {md.timestep_fs / 1000.0}
nsteps                   = {n}          ; {md.npt_ps} ps

define                   = -DPOSRES

{_RESTRICCIONES.format(continuacion="yes")}
; Frecuencia de escritura
nstxout-compressed       = {salida}
nstenergy                = {salida}
nstlog                   = {salida}

{_VECINOS}
{_ELECTROSTATICA}
; Termostato
tcoupl                   = V-rescale
tc-grps                  = Protein Non-Protein
tau_t                    = 0.1     0.1
ref_t                    = {md.temperature_k}   {md.temperature_k}

; Barostato
pcoupl                   = C-rescale
pcoupltype               = isotropic
tau_p                    = 2.0
ref_p                    = {md.pressure_bar}
compressibility          = 4.5e-5
; Escala el centro de masa de las restricciones junto con la caja; sin esto
; las restricciones de posición pelean contra el barostato.
refcoord_scaling         = com

pbc                      = xyz
DispCorr                 = EnerPres

; Se continúan las velocidades del NVT en vez de generarlas de nuevo.
gen_vel                  = no
"""


def contenido_prod(md: MDConfig) -> str:
    """Producción: la etapa de la que sale la trayectoria a analizar."""
    n = pasos(md.production_ns * 1000.0, md.timestep_fs)
    salida = _frecuencia_salida(md)
    cuadros = max(1, n // salida)
    return f"""; Producción — generado por pdpipe desde config.yaml
integrator               = md
dt                       = {md.timestep_fs / 1000.0}
nsteps                   = {n}          ; {md.production_ns} ns

; Sin restricciones de posición: acá la proteína se mueve libre.
define                   =

{_RESTRICCIONES.format(continuacion="yes")}
; Frecuencia de escritura — un cuadro cada {md.output_every_ps} ps, ~{cuadros} cuadros
nstxout-compressed       = {salida}
nstenergy                = {salida}
nstlog                   = {salida}

{_VECINOS}
{_ELECTROSTATICA}
; Termostato
tcoupl                   = V-rescale
tc-grps                  = Protein Non-Protein
tau_t                    = 0.1     0.1
ref_t                    = {md.temperature_k}   {md.temperature_k}

; Barostato de Parrinello-Rahman: más adecuado que C-rescale para producción,
; donde interesa muestrear bien el ensamble y no solo llegar a la densidad.
pcoupl                   = Parrinello-Rahman
pcoupltype               = isotropic
tau_p                    = 2.0
ref_p                    = {md.pressure_bar}
compressibility          = 4.5e-5

pbc                      = xyz
DispCorr                 = EnerPres
gen_vel                  = no
"""


def escribir_mdp(etapa: str, md: MDConfig, destino: str | Path, seed: int = 42) -> Path:
    """Escribe el ``.mdp`` de una etapa y devuelve su ruta.

    Args:
        etapa: ``minim``, ``nvt``, ``npt`` o ``prod``.
        md: sección ``md`` del config.
        destino: carpeta donde escribirlo.
        seed: semilla para las velocidades iniciales del NVT.

    Raises:
        ValueError: si la etapa no existe.
    """
    if etapa not in ETAPAS:
        raise ValueError(f"Etapa desconocida: '{etapa}'. Opciones: {list(ETAPAS)}")

    generadores = {
        "minim": lambda: contenido_minim(md),
        "nvt": lambda: contenido_nvt(md, seed),
        "npt": lambda: contenido_npt(md),
        "prod": lambda: contenido_prod(md),
    }

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{etapa}.mdp"
    ruta.write_text(generadores[etapa](), encoding="utf-8")
    logger.info("Parámetros de %s: %s", etapa, ruta)
    return ruta


def escribir_todos(md: MDConfig, destino: str | Path, seed: int = 42) -> dict[str, Path]:
    """Escribe los cuatro ``.mdp``, indexados por etapa."""
    return {etapa: escribir_mdp(etapa, md, destino, seed) for etapa in ETAPAS}


__all__ = [
    "ETAPAS",
    "contenido_minim",
    "contenido_npt",
    "contenido_nvt",
    "contenido_prod",
    "escribir_mdp",
    "escribir_todos",
    "pasos",
]
