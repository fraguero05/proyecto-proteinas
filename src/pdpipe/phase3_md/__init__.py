"""Fase 3 — Validación estructural por dinámica molecular.

Preparación del sistema (limpieza de aguas y heteroátomos, ``pdb2gmx``, caja
cúbica, solvatación TIP3P, iones de neutralización), minimización de energía,
equilibración NVT y NPT, y producción corta (1–5 ns) con GROMACS y el campo
de fuerza AMBER99SB-ILDN, invocado por subprocess.

El análisis de la trayectoria (RMSD, RMSF, radio de giro, SASA, puentes de
hidrógeno) se hace con MDAnalysis y produce figuras PNG.

Estado: pendiente — se implementa en el Hito 3.
"""

__all__: list[str] = []
