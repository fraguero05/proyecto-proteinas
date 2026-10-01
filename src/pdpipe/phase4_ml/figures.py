"""Figuras de la evaluación de modelos (Fase 4).

Una matriz de confusión por modelo y la curva de aprendizaje del BiLSTM. Con
matplotlib en modo sin pantalla, igual que las de la Fase 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pdpipe.phase4_ml.dssp import CLASES_Q3
from pdpipe.phase4_ml.models import Metricas

DPI = 150

NOMBRES = {"H": "hélice (H)", "E": "hebra (E)", "C": "coil (C)"}


def _plt():
    """Importa matplotlib forzando el backend sin pantalla."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depende del extra `md`
        raise ImportError(
            'matplotlib no está instalado. Instalá el extra: uv pip install -e ".[md]"'
        ) from exc

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def figura_confusion(metricas: Metricas, destino: Path) -> Path:
    """Matriz de confusión normalizada por fila, con los conteos anotados.

    Se normaliza por fila (por clase real) para que la diagonal sea el recall
    de cada clase: así una clase minoritaria mal predicha se ve igual de mal
    que una mayoritaria, en vez de quedar como una celda pálida.
    """
    plt = _plt()
    conteos = np.array(metricas.confusion, dtype=float)
    totales = conteos.sum(axis=1, keepdims=True)
    fracciones = np.divide(conteos, totales, out=np.zeros_like(conteos), where=totales > 0)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    imagen = ax.imshow(fracciones, cmap="Blues", vmin=0, vmax=1)
    etiquetas = [NOMBRES[c] for c in CLASES_Q3]
    ax.set_xticks(range(len(CLASES_Q3)), labels=etiquetas)
    ax.set_yticks(range(len(CLASES_Q3)), labels=etiquetas)
    ax.set_xlabel("predicho")
    ax.set_ylabel("real (DSSP)")
    ax.set_title(f"{metricas.modelo} — {metricas.conjunto}, Q3 = {metricas.q3:.3f}")

    for i in range(len(CLASES_Q3)):
        for j in range(len(CLASES_Q3)):
            ax.text(
                j, i,
                f"{fracciones[i, j]:.2f}\n({int(conteos[i, j])})",
                ha="center", va="center", fontsize="small",
                color="white" if fracciones[i, j] > 0.5 else "black",
            )

    fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destino, dpi=DPI)
    plt.close(fig)
    return destino


def figura_aprendizaje(historial: dict, destino: Path, q3_base: float | None = None) -> Path:
    """Pérdida de entrenamiento y Q3 de validación por época."""
    plt = _plt()
    epocas = [e["numero"] for e in historial["epocas"]]
    perdidas = [e["perdida_train"] for e in historial["epocas"]]
    q3 = [e["q3_val"] for e in historial["epocas"]]

    fig, (izq, der) = plt.subplots(1, 2, figsize=(9, 3.6))
    izq.plot(epocas, perdidas, marker="o", markersize=3, color="#1f77b4")
    izq.set_xlabel("época")
    izq.set_ylabel("entropía cruzada (train)")
    izq.set_title("Pérdida")
    izq.grid(alpha=0.3)

    der.plot(epocas, q3, marker="o", markersize=3, color="#2ca02c", label="Q3 validación")
    der.axvline(
        historial["mejor_epoca"], color="gray", linestyle="--", linewidth=0.8,
        label=f"mejor época ({historial['mejor_epoca']})",
    )
    if q3_base is not None:
        der.axhline(
            q3_base, color="#d62728", linestyle=":", linewidth=1,
            label=f"clase mayoritaria ({q3_base:.3f})",
        )
    der.set_xlabel("época")
    der.set_ylabel("Q3")
    der.set_title("Validación")
    der.legend(loc="lower right", fontsize="small")
    der.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(destino, dpi=DPI)
    plt.close(fig)
    return destino


__all__ = ["figura_aprendizaje", "figura_confusion"]
