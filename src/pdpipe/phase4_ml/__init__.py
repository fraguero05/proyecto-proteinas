"""Fase 4 — Modelos de IA.

Predicción de estructura secundaria a partir de secuencia: dataset construido
con DSSP sobre las estructuras curadas en la Fase 1, codificación one-hot y
ventanas deslizantes centradas en el residuo.

Se comparan un BiLSTM en PyTorch contra un baseline clásico (regresión
logística o random forest). La división train/val/test controla la redundancia
de secuencia (CD-HIT, o clustering por identidad con Biopython como
alternativa) para que el test no mida memorización.

Métricas: Q3, precisión, recall, F1 y matriz de confusión.

Estado: pendiente — se implementa en el Hito 4.
"""

__all__: list[str] = []
