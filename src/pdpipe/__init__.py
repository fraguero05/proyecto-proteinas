"""pdpipe — pipeline reproducible de diseño computacional de proteínas.

Implementación práctica de la tesis de grado de Ingeniería en Informática
(FP-UNA). El pipeline lleva una estructura depositada en el PDB hasta una
variante rediseñada, validada por dinámica molecular y caracterizada con
modelos predictivos, en cinco fases:

1. Recolección de datos      -> :mod:`pdpipe.phase1_data`
2. Diseño computacional      -> :mod:`pdpipe.phase2_design`
3. Validación estructural    -> :mod:`pdpipe.phase3_md`
4. Modelos de IA             -> :mod:`pdpipe.phase4_ml`
5. Análisis y comparación    -> :mod:`pdpipe.phase5_analysis`
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
