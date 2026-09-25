"""Orquestación de la Fase 2: ``predict`` y ``design``.

**Parte A — ``predict``.** Obtiene la estructura predicha para una accesión
UniProt y reporta su confianza por residuo. La salida son tres archivos en
``data/processed/``:

* el modelo de coordenadas tal como vino,
* un CSV con el pLDDT de cada residuo,
* un JSON con el resumen y los metadatos de procedencia.

El CSV por residuo es el insumo de la parte B: las posiciones con pLDDT bajo
son justamente las que no conviene tomar como referencia rígida al diseñar.

**Parte B — ``design``.** Toma un esqueleto y genera variantes de secuencia
detrás de la interfaz :class:`~pdpipe.phase2_design.designer.SequenceDesigner`.
Deja un FASTA con las variantes y una tabla con el score y las mutaciones de
cada una respecto de la original.
"""

from __future__ import annotations

from pathlib import Path

import csv
import json

from pdpipe.config import Config
from pdpipe.phase1_data.http_client import ClienteHTTP
from pdpipe.phase2_design.alphafold import ClienteAlphaFold, SinModeloPredicho
from pdpipe.phase2_design.designer import obtener_disenador
from pdpipe.phase2_design.models import ModeloPredicho, ResultadoDiseno
from pdpipe.phase2_design.plddt import anotar_modelo
from pdpipe.utils.logging import get_logger
from pdpipe.utils.manifest import RunManifest

logger = get_logger(__name__)


class FuenteNoDisponible(Exception):
    """La fuente de estructura pedida todavía no está implementada."""


def predecir(
    config: Config,
    uniprot_id: str,
    fuente: str | None = None,
    formato: str = "pdb",
    forzar: bool = False,
    manifest: RunManifest | None = None,
) -> ModeloPredicho:
    """Obtiene la estructura predicha de una accesión y la caracteriza.

    Args:
        config: configuración del pipeline.
        uniprot_id: accesión UniProt a modelar.
        fuente: sobrescribe ``design.structure_source`` del config.
        formato: ``pdb`` o ``cif``.
        forzar: ignora la caché y vuelve a descargar.
        manifest: si se pasa, se le registran los archivos producidos.

    Raises:
        SinModeloPredicho: la accesión no está en AlphaFold DB.
        FuenteNoDisponible: se pidió ColabFold o ESMFold.
    """
    rutas = config.resolved_paths()
    elegida = (fuente or config.design.structure_source.value).lower()

    if elegida == "colabfold":
        raise FuenteNoDisponible(
            "ColabFold requiere GPU y corre en notebook, no localmente. "
            "Usá notebooks/colabfold.ipynb (Hito 2, parte C)."
        )
    if elegida == "esmfold":
        raise FuenteNoDisponible(
            "ESMFold todavía no está implementado. Usá --source alphafold_db."
        )
    if elegida != "alphafold_db":
        raise FuenteNoDisponible(f"Fuente desconocida: '{elegida}'")

    http = ClienteHTTP(
        timeout_s=config.data.http.timeout_s,
        max_retries=config.data.http.max_retries,
        cache_dir=rutas["data_raw"],
        usar_cache=config.data.http.cache and not forzar,
    )
    try:
        cliente = ClienteAlphaFold(http)
        modelo = cliente.descargar(
            uniprot_id, destino=rutas["data_processed"], formato=formato
        )
        modelo = anotar_modelo(modelo, modelo.archivo)
    finally:
        http.close()

    _avisar_confianza_baja(modelo, config.design.plddt_min)

    csv_path, json_path = _escribir_salidas(modelo, rutas["data_processed"])

    if manifest:
        manifest.add_output(modelo.archivo, key=modelo.archivo.name)
        manifest.add_output(csv_path, key=csv_path.name)
        manifest.add_output(json_path, key=json_path.name)
        if modelo.resumen:
            manifest.add_note(
                f"{modelo.uniprot_id}: pLDDT medio {modelo.resumen.media}, "
                f"{modelo.resumen.fraccion_confiable:.1%} de residuos con pLDDT >= 70"
            )

    return modelo


def _avisar_confianza_baja(modelo: ModeloPredicho, umbral: float) -> None:
    """Avisa si el modelo no llega al pLDDT mínimo configurado.

    Es un aviso, no un error: un modelo de confianza baja puede seguir siendo
    útil si las regiones que importan están bien resueltas. Quien decide es
    quien mira los números, no el pipeline.
    """
    if modelo.resumen is None:
        return

    if modelo.resumen.media < umbral:
        logger.warning(
            "pLDDT medio %.2f por debajo del mínimo configurado (%.1f). "
            "El modelo puede no ser confiable para diseñar sobre él.",
            modelo.resumen.media,
            umbral,
        )

    bajos = modelo.residuos_bajo(umbral)
    if bajos:
        logger.info(
            "%d de %d residuos por debajo de %.1f (p. ej. %s)",
            len(bajos),
            modelo.resumen.n_residuos,
            umbral,
            ", ".join(str(r.numero) for r in bajos[:10])
            + ("…" if len(bajos) > 10 else ""),
        )


def _escribir_salidas(modelo: ModeloPredicho, destino: Path) -> tuple[Path, Path]:
    """Escribe el CSV por residuo y el JSON de resumen."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    base = modelo.entry_id or modelo.uniprot_id

    csv_path = destino / f"{base}_plddt.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        escritor = csv.writer(handle)
        escritor.writerow(["residuo", "aminoacido", "plddt", "banda"])
        for r in modelo.residuos:
            escritor.writerow([r.numero, r.aminoacido or "X", f"{r.plddt:.2f}", r.banda.value])

    json_path = destino / f"{base}_resumen.json"
    datos = modelo.model_dump(mode="json", exclude={"residuos"})
    datos["archivo"] = str(modelo.archivo) if modelo.archivo else None
    datos["plddt_csv"] = str(csv_path)
    json_path.write_text(
        json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("pLDDT por residuo: %s", csv_path)
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Parte B — design
# ---------------------------------------------------------------------------


def disenar(
    config: Config,
    estructura: str | Path,
    n_secuencias: int | None = None,
    temperatura: float | None = None,
    seed: int | None = None,
    manifest: RunManifest | None = None,
) -> ResultadoDiseno:
    """Genera variantes de secuencia sobre un esqueleto.

    Args:
        config: configuración del pipeline.
        estructura: PDB de partida (el de ``predict`` o uno experimental).
        n_secuencias: sobrescribe ``design.n_sequences``.
        temperatura: sobrescribe ``design.temperature``.
        seed: sobrescribe la semilla global del config.
        manifest: si se pasa, se le registran los archivos producidos.

    Raises:
        DisenadorNoDisponible: la herramienta de diseño no está instalada.
        ErrorDeDiseno: la corrida falló o su salida no se pudo interpretar.
    """
    rutas = config.resolved_paths()
    estructura = Path(estructura)

    n = n_secuencias if n_secuencias is not None else config.design.n_sequences
    t = temperatura if temperatura is not None else config.design.temperature
    semilla = seed if seed is not None else config.seed

    raiz = config.source_path.parent if config.source_path else Path.cwd()
    home = config.design.proteinmpnn_home
    if not home.is_absolute():
        home = raiz / home

    disenador = obtener_disenador(
        config.design.designer.value,
        home=home,
        modelo=config.design.proteinmpnn_model,
    )

    resultado = disenador.disenar(
        estructura=estructura,
        n_secuencias=n,
        temperatura=t,
        posiciones_fijas=config.design.fixed_positions,
        seed=semilla,
    )

    fasta_path, tabla_path = _escribir_diseno(resultado, rutas["data_processed"])
    resultado = resultado.model_copy(
        update={"fasta": fasta_path, "tabla": tabla_path}
    )

    if manifest:
        manifest.add_input(estructura, key=estructura.name)
        manifest.add_output(fasta_path, key=fasta_path.name)
        manifest.add_output(tabla_path, key=tabla_path.name)
        mejor = resultado.mejor()
        if mejor:
            manifest.add_note(
                f"{estructura.stem}: {resultado.n_variantes} variantes, "
                f"mejor global_score {mejor.global_score} ({mejor.n_mutaciones} "
                f"mutaciones, {mejor.identidad:.1%} de identidad)"
            )

    return resultado


def _escribir_diseno(
    resultado: ResultadoDiseno, destino: Path
) -> tuple[Path, Path]:
    """Escribe el FASTA de variantes y la tabla de scores y mutaciones."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    base = resultado.estructura.stem

    fasta_path = destino / f"{base}_variantes.fasta"
    with fasta_path.open("w", encoding="utf-8", newline="\n") as handle:
        # La original va primero para que el FASTA sirva de insumo directo a un
        # alineamiento sin tener que ir a buscarla a otro archivo.
        handle.write(f">{base}_original\n{resultado.secuencia_original}\n")
        for v in resultado.variantes:
            handle.write(
                f">{base}_{v.id} global_score={v.global_score} "
                f"n_mutaciones={v.n_mutaciones} "
                f"mutaciones={v.notacion_mutaciones() or '-'}\n{v.secuencia}\n"
            )

    tabla_path = destino / f"{base}_variantes.csv"
    with tabla_path.open("w", encoding="utf-8", newline="") as handle:
        escritor = csv.writer(handle)
        escritor.writerow(
            [
                "variante", "score", "global_score", "recuperacion",
                "n_mutaciones", "identidad", "mutaciones", "secuencia",
            ]
        )
        for v in resultado.variantes:
            escritor.writerow(
                [
                    v.id, v.score, v.global_score, v.recuperacion,
                    v.n_mutaciones, v.identidad, v.notacion_mutaciones(), v.secuencia,
                ]
            )

    logger.info("Variantes: %s", fasta_path)
    logger.info("Tabla de diseño: %s", tabla_path)
    return fasta_path, tabla_path
