"""Orquestación del dataset de estructura secundaria (Fase 4, parte A).

Toma las estructuras curadas en la Fase 1, les corre DSSP, arma las ventanas
y las reparte en train/val/test sin redundancia entre conjuntos.

La cadena completa es: ``fetch --search`` baja estructuras, ``curate`` las
filtra, y esto las convierte en un dataset. Que el dataset salga de la propia
curación del proyecto —y no de un archivo descargado de otro lado— es lo que
hace que los criterios del ``config.yaml`` se vean reflejados en el resultado.
"""

from __future__ import annotations

import json
from pathlib import Path

from pdpipe.config import Config
from pdpipe.phase1_data.database import abrir_base
from pdpipe.phase4_ml import dataset as ds
from pdpipe.phase4_ml import splits as sp
from pdpipe.phase4_ml.dssp import CLASES_Q3, disponible, motivo_no_disponible
from pdpipe.phase4_ml.models import Proteina, Split
from pdpipe.utils.logging import get_logger
from pdpipe.utils.manifest import RunManifest

logger = get_logger(__name__)


class ErrorDeDataset(Exception):
    """No se pudo construir el dataset."""


def construir_dataset(
    config: Config,
    solo_curadas: bool = True,
    longitud_min: int | None = None,
    manifest: RunManifest | None = None,
) -> tuple[list[Proteina], Split, dict]:
    """Arma el dataset completo y lo deja en disco.

    Args:
        config: configuración del pipeline.
        solo_curadas: usar únicamente las estructuras que pasaron la curación.
            Ponerlo en ``False`` incluye también las rechazadas, lo que sirve
            para probar con pocos datos pero no para un resultado publicable.
        longitud_min: descarta cadenas más cortas; por defecto sale de
            ``data.length_min``.
        manifest: si se pasa, se le registran las salidas.

    Raises:
        ErrorDeDataset: si falta DSSP, no hay estructuras, o ninguna sobrevive
            al procesamiento.
    """
    if not disponible():
        raise ErrorDeDataset(motivo_no_disponible())

    rutas = config.resolved_paths()
    minimo = longitud_min if longitud_min is not None else config.data.length_min

    with abrir_base(rutas["database"]) as base:
        registros = base.listar_proteinas(solo_curadas=solo_curadas)

    # Con --all entran también las no curadas, pero nunca las rechazadas por
    # bioseguridad. Hoy esas no tienen coordenadas en disco, pero no alcanza
    # con eso: una estructura bajada antes de que la lista la incluyera sí las
    # tendría.
    vetadas = [r.pdb_id for r in registros if r.rechazada_por_bioseguridad]
    if vetadas:
        logger.info("%d estructuras excluidas por bioseguridad", len(vetadas))
        registros = [r for r in registros if not r.rechazada_por_bioseguridad]

    if not registros:
        raise ErrorDeDataset(
            "No hay estructuras en la base"
            + (" que hayan pasado la curación" if solo_curadas else "")
            + ".\n\nBajá un conjunto con:  pdpipe fetch --search 500\n"
            "y curalo con:              pdpipe curate"
        )

    estructuras: dict[str, Path] = {}
    sin_archivo: list[str] = []
    for registro in registros:
        ruta = getattr(registro, "archivo_path", None)
        if not ruta or not Path(ruta).is_file():
            sin_archivo.append(registro.pdb_id)
            continue
        estructuras[registro.pdb_id] = Path(ruta)

    if sin_archivo:
        logger.warning(
            "%d estructuras figuran en la base pero no están en disco: %s",
            len(sin_archivo),
            ", ".join(sin_archivo[:10]) + ("…" if len(sin_archivo) > 10 else ""),
        )

    if not estructuras:
        raise ErrorDeDataset(
            "Ninguna de las estructuras de la base está en disco. "
            "¿Se borró data/raw?"
        )

    proteinas, descartadas = ds.construir(estructuras, longitud_min=minimo)
    if not proteinas:
        raise ErrorDeDataset(
            f"DSSP no pudo procesar ninguna de las {len(estructuras)} "
            "estructuras. Revisá los motivos en el JSON del dataset."
        )

    division = sp.dividir(
        proteinas,
        val_size=config.ml.val_size,
        test_size=config.ml.test_size,
        identidad_max=config.ml.identity_threshold,
        seed=config.seed,
    )

    # La verificación es el control de calidad del split: si encuentra pares
    # redundantes entre conjuntos, el Q3 del test va a estar inflado.
    redundantes = sp.verificar(proteinas, division, config.ml.identity_threshold)

    resumen = _resumen(proteinas, division, descartadas, redundantes, config)
    salidas = _escribir(proteinas, division, resumen, rutas["data_processed"])

    if manifest:
        for ruta in salidas.values():
            manifest.add_output(ruta, key=ruta.name)
        manifest.add_note(
            f"Dataset: {len(proteinas)} proteínas, "
            f"{resumen['n_residuos']} residuos, {division.n_grupos} grupos"
        )
        if redundantes:
            manifest.add_note(
                f"advertencia: {len(redundantes)} pares redundantes entre conjuntos"
            )

    resumen["archivos"] = {k: str(v) for k, v in salidas.items()}
    return proteinas, division, resumen


def _resumen(
    proteinas: list[Proteina],
    division: Split,
    descartadas: list[str],
    redundantes: list[tuple[str, str, float]],
    config: Config,
) -> dict:
    """Arma el resumen que va al JSON y a la pantalla."""
    por_conjunto = {}
    for nombre in ("train", "val", "test"):
        ids = set(getattr(division, nombre))
        subconjunto = [p for p in proteinas if p.identificador in ids]
        por_conjunto[nombre] = {
            "n_proteinas": len(subconjunto),
            "n_residuos": sum(len(p) for p in subconjunto),
            "composicion": ds.composicion(subconjunto),
        }

    return {
        "n_proteinas": len(proteinas),
        "n_residuos": sum(len(p) for p in proteinas),
        "n_grupos": division.n_grupos,
        "identidad_max": division.identidad_max,
        "ventana": config.ml.window_size,
        "clases": list(CLASES_Q3),
        "composicion": ds.composicion(proteinas),
        "conjuntos": por_conjunto,
        "descartadas": descartadas,
        "pares_redundantes": [
            {"a": a, "b": b, "identidad": v} for a, b, v in redundantes
        ],
    }


def _escribir(
    proteinas: list[Proteina], division: Split, resumen: dict, destino: Path
) -> dict[str, Path]:
    """Guarda el dataset en disco.

    Las proteínas van en JSON con su secuencia y su asignación de DSSP, no las
    matrices de ventanas: estas últimas se reconstruyen en un segundo y pesan
    órdenes de magnitud más. Lo que no se puede reconstruir sin volver a correr
    DSSP sobre 500 archivos es justamente lo que se guarda.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    archivo_proteinas = destino / "dataset_ss.json"
    archivo_proteinas.write_text(
        json.dumps(
            [
                {
                    "id": p.identificador,
                    "secuencia": p.secuencia,
                    "estructura": p.estructura,
                    "numeros": p.numeros,
                }
                for p in proteinas
            ],
            indent=1,
        ),
        encoding="utf-8",
    )

    archivo_split = destino / "dataset_ss_split.json"
    archivo_split.write_text(
        json.dumps(division.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    archivo_resumen = destino / "dataset_ss_resumen.json"
    archivo_resumen.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Dataset: %s", archivo_proteinas)
    return {
        "proteinas": archivo_proteinas,
        "split": archivo_split,
        "resumen": archivo_resumen,
    }


def cargar_dataset(destino: Path) -> tuple[list[Proteina], Split]:
    """Relee un dataset ya construido, para no rehacer DSSP en cada corrida."""
    destino = Path(destino)
    archivo_proteinas = destino / "dataset_ss.json"
    archivo_split = destino / "dataset_ss_split.json"

    if not archivo_proteinas.is_file() or not archivo_split.is_file():
        raise ErrorDeDataset(
            f"No hay un dataset construido en {destino}.\n\n"
            "Armalo con:  pdpipe dataset"
        )

    crudas = json.loads(archivo_proteinas.read_text(encoding="utf-8"))
    proteinas = [
        Proteina(
            identificador=d["id"],
            secuencia=d["secuencia"],
            estructura=d["estructura"],
            numeros=d.get("numeros", []),
        )
        for d in crudas
    ]
    division = Split(**json.loads(archivo_split.read_text(encoding="utf-8")))
    return proteinas, division


__all__ = ["ErrorDeDataset", "cargar_dataset", "construir_dataset"]
