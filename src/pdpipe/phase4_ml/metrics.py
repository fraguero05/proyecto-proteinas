"""Métricas de predicción de estructura secundaria (Fase 4).

Q3, precisión, recall, F1 por clase y matriz de confusión. Se calculan acá con
numpy en vez de pedírselas a scikit-learn para que el BiLSTM y el baseline se
midan con exactamente el mismo código, y para que la definición de cada número
quede escrita en un lugar citable.

Todas las métricas son **por residuo**: cada residuo de cada proteína del
conjunto cuenta una vez, sin promediar primero por proteína. Es la convención
del Q3 en la literatura (Rost y Sander, 1993).
"""

from __future__ import annotations

import numpy as np

from pdpipe.phase4_ml.dssp import CLASES_Q3
from pdpipe.phase4_ml.models import Metricas


def matriz_de_confusion(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Conteos ``(real, predicho)`` en el orden de :data:`CLASES_Q3`."""
    n = len(CLASES_Q3)
    y_real = np.asarray(y_real, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_real.shape != y_pred.shape:
        raise ValueError(
            f"{len(y_real)} etiquetas reales y {len(y_pred)} predicciones"
        )
    matriz = np.zeros((n, n), dtype=np.int64)
    np.add.at(matriz, (y_real, y_pred), 1)
    return matriz


def evaluar(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    modelo: str,
    conjunto: str = "test",
    clase_base: int | None = None,
) -> Metricas:
    """Calcula todas las métricas de una evaluación.

    Args:
        y_real: índices de clase reales, uno por residuo.
        y_pred: índices de clase predichos, alineados con ``y_real``.
        modelo: nombre del modelo, para el reporte.
        conjunto: nombre del conjunto evaluado.
        clase_base: clase que predeciría el modelo trivial. Lo correcto es
            pasar la mayoritaria **del entrenamiento**, que es lo único que un
            modelo trivial podría conocer; si no se pasa, se usa la
            mayoritaria del propio conjunto evaluado, que es un piso algo
            optimista.

    Una clase que nunca se predice tiene precisión 0 y no indefinida: es un
    fallo del modelo y tiene que verse en la tabla, no desaparecer como NaN.
    """
    matriz = matriz_de_confusion(y_real, y_pred)
    total = int(matriz.sum())

    reales = matriz.sum(axis=1)
    predichos = matriz.sum(axis=0)
    aciertos = np.diag(matriz)

    if clase_base is None:
        clase_base = int(np.argmax(reales)) if total else 0

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    for i, clase in enumerate(CLASES_Q3):
        p = aciertos[i] / predichos[i] if predichos[i] else 0.0
        r = aciertos[i] / reales[i] if reales[i] else 0.0
        precision[clase] = round(float(p), 4)
        recall[clase] = round(float(r), 4)
        f1[clase] = round(float(2 * p * r / (p + r)) if p + r else 0.0, 4)

    return Metricas(
        modelo=modelo,
        conjunto=conjunto,
        n_residuos=total,
        q3=round(float(aciertos.sum() / total), 4) if total else 0.0,
        q3_base=round(float(reales[clase_base] / total), 4) if total else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=matriz.tolist(),
    )


def f1_macro(metricas: Metricas) -> float:
    """Promedio simple del F1 de las tres clases.

    Complementa al Q3: un modelo que nunca predice hebra (E, la clase
    minoritaria) puede tener un Q3 decente y un F1 macro malo.
    """
    if not metricas.f1:
        return 0.0
    return round(sum(metricas.f1.values()) / len(metricas.f1), 4)


__all__ = ["evaluar", "f1_macro", "matriz_de_confusion"]
