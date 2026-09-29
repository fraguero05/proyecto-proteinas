"""Fase 3 — Validación estructural por dinámica molecular.

Preparación del sistema (limpieza de aguas y heteroátomos, ``pdb2gmx``, caja
cúbica, solvatación TIP3P, iones de neutralización), minimización de energía,
equilibración NVT y NPT, y producción corta (1–5 ns) con GROMACS y el campo
de fuerza AMBER99SB-ILDN, invocado por subprocess.

El análisis de la trayectoria (RMSD, RMSF, radio de giro, SASA, puentes de
hidrógeno) se hace con MDAnalysis y produce figuras PNG.

Estado: las tres partes implementadas. La limpieza de la estructura y la
generación de los .mdp corren en cualquier máquina; los pasos de GROMACS
requieren el binario, que en el entorno de desarrollo no está disponible
y se ejecuta desde notebooks/md_gromacs.ipynb.
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
from pdpipe.phase3_md.gromacs import (
    ClienteGromacs,
    ErrorDeGromacs,
    GromacsNoDisponible,
)
from pdpipe.phase3_md.mdp import escribir_mdp, escribir_todos, pasos
from pdpipe.phase3_md.models import (
    EtapaSimulacion,
    PerfilPorResiduo,
    ResultadoAnalisisMD,
    ResultadoLimpieza,
    ResultadoSimulacion,
    SerieTemporal,
    SistemaPreparado,
)
from pdpipe.phase3_md.pipeline import analizar
from pdpipe.phase3_md.preparation import limpiar_estructura, preparar_sistema
from pdpipe.phase3_md.simulation import correr_etapa, simular

__all__ = [
    "ClienteGromacs",
    "ErrorDeAnalisis",
    "ErrorDeGromacs",
    "EtapaSimulacion",
    "GromacsNoDisponible",
    "PerfilPorResiduo",
    "ResultadoAnalisisMD",
    "ResultadoLimpieza",
    "ResultadoSimulacion",
    "SerieTemporal",
    "SistemaPreparado",
    "alinear",
    "analizar",
    "calcular_puentes_de_hidrogeno",
    "calcular_radio_de_giro",
    "calcular_rmsd",
    "calcular_rmsf",
    "calcular_sasa",
    "cargar",
    "correr_etapa",
    "escribir_mdp",
    "escribir_todos",
    "generar_figuras",
    "limpiar_estructura",
    "pasos",
    "preparar_sistema",
    "simular",
    "tiene_hidrogenos",
]
