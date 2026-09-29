"""Fase 3 — Validación estructural por dinámica molecular.

Preparación del sistema (limpieza de aguas y heteroátomos, ``pdb2gmx``, caja
cúbica, solvatación TIP3P, iones de neutralización), minimización de energía,
equilibración NVT y NPT, y producción corta (1–5 ns) con GROMACS y el campo
de fuerza AMBER99SB-ILDN, invocado por subprocess.

El análisis de la trayectoria (RMSD, RMSF, radio de giro, SASA, puentes de
hidrógeno) se hace con MDAnalysis y produce figuras PNG.

Estado: parte C (análisis de trayectorias) implementada. Las partes A
(preparación) y B (simulación) requieren GROMACS, que no está disponible
en el entorno local; ver el README, "Cómo reproducir el Hito 3".
"""

from pdpipe.phase3_md.analysis import (
    ErrorDeAnalisis,
    alinear,
    calcular_puentes_de_hidrogeno,
    calcular_radio_de_giro,
    calcular_rmsd,
    calcular_rmsf,
    calcular_sasa,
    cargar,
    tiene_hidrogenos,
)
from pdpipe.phase3_md.figures import generar_figuras
from pdpipe.phase3_md.models import (
    PerfilPorResiduo,
    ResultadoAnalisisMD,
    SerieTemporal,
)
from pdpipe.phase3_md.pipeline import analizar

__all__ = [
    "ErrorDeAnalisis",
    "PerfilPorResiduo",
    "ResultadoAnalisisMD",
    "SerieTemporal",
    "alinear",
    "analizar",
    "calcular_puentes_de_hidrogeno",
    "calcular_radio_de_giro",
    "calcular_rmsd",
    "calcular_rmsf",
    "calcular_sasa",
    "cargar",
    "generar_figuras",
    "tiene_hidrogenos",
]
