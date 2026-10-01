"""Fase 4 — Modelos de IA.

Predicción de estructura secundaria a partir de secuencia: dataset construido
con DSSP sobre las estructuras curadas en la Fase 1, codificación one-hot y
ventanas deslizantes centradas en el residuo.

Se comparan un BiLSTM en PyTorch contra dos baselines clásicos (regresión
logística y random forest). La división train/val/test controla la redundancia
de secuencia agrupando por identidad con Biopython —CD-HIT no está disponible
en el entorno— para que el test no mida memorización.

Métricas: Q3, precisión, recall, F1 y matriz de confusión.

Módulos:

* :mod:`~pdpipe.phase4_ml.dssp` — etiquetas Q3 con la implementación de DSSP de MDTraj.
* :mod:`~pdpipe.phase4_ml.dataset` — one-hot, ventanas y secuencias completas.
* :mod:`~pdpipe.phase4_ml.splits` — agrupamiento por identidad y reparto por grupos.
* :mod:`~pdpipe.phase4_ml.pipeline` — de la base de la Fase 1 al dataset (parte A).
* :mod:`~pdpipe.phase4_ml.classifiers` — BiLSTM y baselines con una interfaz común.
* :mod:`~pdpipe.phase4_ml.training` — entrenamiento y evaluación (parte B).
* :mod:`~pdpipe.phase4_ml.metrics` y :mod:`~pdpipe.phase4_ml.figures` — números y figuras.
"""

__all__: list[str] = []
