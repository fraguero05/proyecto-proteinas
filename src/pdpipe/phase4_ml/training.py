"""Entrenamiento y evaluación de los modelos (Fase 4, parte B).

Parte del dataset ya construido por :mod:`pdpipe.phase4_ml.pipeline` —no
vuelve a correr DSSP ni a dividir— y entrena uno o más modelos sobre exactamente
el mismo reparto train/val/test. Que todos vean las mismas proteínas es lo que
hace comparables sus números.

Protocolo:

1. Se entrena con ``train``.
2. ``val`` se usa para elegir la época del BiLSTM (corte temprano). Los
   baselines no eligen nada con validación; solo se reporta.
3. ``test`` se mira una sola vez, al final, con el modelo ya elegido.

El piso de comparación (``q3_base``) es la clase mayoritaria **del
entrenamiento** aplicada al conjunto evaluado: lo que acertaría un modelo que
no mira la secuencia.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pdpipe.config import Config
from pdpipe.phase4_ml import classifiers as clf
from pdpipe.phase4_ml.metrics import evaluar, f1_macro
from pdpipe.phase4_ml.models import Metricas, Proteina, Split
from pdpipe.phase4_ml.pipeline import ErrorDeDataset, cargar_dataset
from pdpipe.utils.logging import get_logger
from pdpipe.utils.manifest import RunManifest

logger = get_logger(__name__)


@dataclass
class ResultadoModelo:
    """Lo que produjo entrenar y evaluar un modelo."""

    nombre: str
    val: Metricas
    test: Metricas
    historial: dict
    segundos: float
    archivo: Path | None = None
    figuras: list[Path] = field(default_factory=list)

    def a_dict(self) -> dict:
        return {
            "modelo": self.nombre,
            "segundos_entrenamiento": round(self.segundos, 1),
            "archivo": str(self.archivo) if self.archivo else None,
            "val": {**self.val.model_dump(), "f1_macro": f1_macro(self.val)},
            "test": {**self.test.model_dump(), "f1_macro": f1_macro(self.test)},
            "historial": self.historial,
        }


def repartir(proteinas: list[Proteina], division: Split) -> dict[str, list[Proteina]]:
    """Separa las proteínas según el split, conservando el orden del split."""
    por_id = {p.identificador: p for p in proteinas}
    faltan = [i for i in division.train + division.val + division.test if i not in por_id]
    if faltan:
        raise ErrorDeDataset(
            f"El split menciona {len(faltan)} proteínas que no están en el "
            f"dataset (p. ej. {faltan[0]}). Reconstruilo con: pdpipe dataset"
        )
    return {
        nombre: [por_id[i] for i in getattr(division, nombre)]
        for nombre in ("train", "val", "test")
    }


def clase_mayoritaria(proteinas: list[Proteina]) -> int:
    """Índice de la clase más frecuente, en residuos."""
    y = clf.etiquetas(proteinas)
    if len(y) == 0:
        return 0
    return int(np.bincount(y, minlength=3).argmax())


def entrenar_modelos(
    config: Config,
    modelos: list[str],
    destino: Path,
    epocas: int | None = None,
    con_figuras: bool = True,
    manifest: RunManifest | None = None,
) -> list[ResultadoModelo]:
    """Entrena y evalúa cada modelo pedido sobre el dataset guardado.

    Args:
        config: configuración del pipeline.
        modelos: nombres de :data:`classifiers.MODELOS`.
        destino: directorio de la corrida; ahí quedan modelos, métricas y figuras.
        epocas: pisa ``ml.epochs`` del config (útil para una prueba rápida).
        con_figuras: generar las matrices de confusión y la curva de aprendizaje.
        manifest: si se pasa, se le registran entradas y salidas.

    Raises:
        ErrorDeDataset: si no hay dataset construido o está inconsistente.
        clf.ErrorDeModelo: si un modelo no se puede crear o entrenar.
    """
    desconocidos = [m for m in modelos if m not in clf.MODELOS]
    if desconocidos:
        raise clf.ErrorDeModelo(
            f"Modelo desconocido: {', '.join(desconocidos)}. "
            f"Opciones: {', '.join(clf.MODELOS)}"
        )

    procesados = Path(config.resolved_paths()["data_processed"])
    proteinas, division = cargar_dataset(procesados)
    conjuntos = repartir(proteinas, division)
    if not conjuntos["train"] or not conjuntos["test"]:
        raise ErrorDeDataset(
            "El split deja vacío el entrenamiento o el test. "
            "Hacen falta más proteínas: pdpipe fetch --search 500"
        )

    if manifest:
        manifest.add_input(procesados / "dataset_ss.json", key="dataset_ss.json")
        manifest.add_input(procesados / "dataset_ss_split.json", key="dataset_ss_split.json")

    if epocas is not None:
        config = config.model_copy(
            update={"ml": config.ml.model_copy(update={"epochs": epocas})}
        )

    base = clase_mayoritaria(conjuntos["train"])
    y_val = clf.etiquetas(conjuntos["val"])
    y_test = clf.etiquetas(conjuntos["test"])

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    resultados: list[ResultadoModelo] = []
    for nombre in modelos:
        modelo = clf.crear(nombre, config)
        inicio = time.perf_counter()
        historial = modelo.entrenar(conjuntos["train"], conjuntos["val"])
        segundos = time.perf_counter() - inicio

        resultado = ResultadoModelo(
            nombre=nombre,
            val=evaluar(y_val, modelo.predecir(conjuntos["val"]), nombre, "val", base),
            test=evaluar(y_test, modelo.predecir(conjuntos["test"]), nombre, "test", base),
            historial=historial.a_dict(),
            segundos=segundos,
            archivo=modelo.guardar(destino),
        )
        logger.info(
            "%s: Q3 test %.4f (piso %.4f), %.1f s",
            nombre, resultado.test.q3, resultado.test.q3_base, segundos,
        )

        if con_figuras:
            resultado.figuras = _figuras(resultado, destino)
        if manifest:
            manifest.add_output(resultado.archivo, key=resultado.archivo.name)
            for figura in resultado.figuras:
                manifest.add_output(figura, key=figura.name)
            manifest.add_note(
                f"{nombre}: Q3 test {resultado.test.q3:.4f}, "
                f"F1 macro {f1_macro(resultado.test):.4f}"
            )
            if getattr(modelo, "dispositivo", None):
                manifest.add_note(f"{nombre}: dispositivo {modelo.dispositivo}")

        resultados.append(resultado)

    archivo = escribir_metricas(resultados, division, destino)
    if manifest:
        manifest.add_output(archivo, key=archivo.name)
    return resultados


def _figuras(resultado: ResultadoModelo, destino: Path) -> list[Path]:
    from pdpipe.phase4_ml import figures

    figuras = [
        figures.figura_confusion(
            resultado.test, destino / f"confusion_{resultado.nombre}.png"
        )
    ]
    if len(resultado.historial["epocas"]) > 1:
        figuras.append(
            figures.figura_aprendizaje(
                resultado.historial,
                destino / f"aprendizaje_{resultado.nombre}.png",
                q3_base=resultado.val.q3_base,
            )
        )
    return figuras


def escribir_metricas(
    resultados: list[ResultadoModelo], division: Split, destino: Path
) -> Path:
    """Guarda todas las métricas en un JSON, que es lo que lee la Fase 5."""
    archivo = Path(destino) / "metricas_ss.json"
    archivo.write_text(
        json.dumps(
            {
                "split": {
                    "train": len(division.train),
                    "val": len(division.val),
                    "test": len(division.test),
                    "identidad_max": division.identidad_max,
                    "n_grupos": division.n_grupos,
                },
                "modelos": [r.a_dict() for r in resultados],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return archivo


__all__ = [
    "ResultadoModelo",
    "clase_mayoritaria",
    "entrenar_modelos",
    "escribir_metricas",
    "repartir",
]
