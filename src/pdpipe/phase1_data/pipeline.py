"""Orquestación de la Fase 1: ``fetch`` y ``curate``.

Separa los dos pasos de forma deliberada:

* ``fetch`` baja y persiste, aplicando **solo** la verificación de
  bioseguridad, que no es negociable.
* ``curate`` aplica los criterios de calidad (resolución, método, organismo,
  longitud) sobre lo ya descargado, sin volver a la red.

Esa división permite ajustar los umbrales de curación y volver a correr
``curate`` las veces que haga falta sin re-descargar nada, que es lo que hace
usable la exploración de criterios para el capítulo de resultados.
"""

from __future__ import annotations

from pathlib import Path

from pdpipe.config import Config
from pdpipe.phase1_data import biosafety
from pdpipe.phase1_data.database import BaseDatos
from pdpipe.phase1_data.filters import aplicar_filtros
from pdpipe.phase1_data.http_client import ClienteHTTP, ErrorDeRed
from pdpipe.phase1_data.models import Proteina
from pdpipe.phase1_data.rcsb import ClienteRCSB, validar_pdb_id
from pdpipe.phase1_data.uniprot import ClienteUniProt
from pdpipe.utils.checksums import file_sha256
from pdpipe.utils.logging import get_logger
from pdpipe.utils.manifest import RunManifest

logger = get_logger(__name__)


class ResultadoFetch:
    """Resumen de una corrida de ``fetch``."""

    def __init__(self) -> None:
        self.descargadas: list[str] = []
        self.rechazadas: list[tuple[str, str]] = []  # (pdb_id, motivo)
        self.fallidas: list[tuple[str, str]] = []    # (pdb_id, error)

    @property
    def total(self) -> int:
        return len(self.descargadas) + len(self.rechazadas) + len(self.fallidas)


class ResultadoCurate:
    """Resumen de una corrida de ``curate``."""

    def __init__(self) -> None:
        self.aceptadas: list[str] = []
        self.rechazadas: list[tuple[str, str]] = []

    @property
    def total(self) -> int:
        return len(self.aceptadas) + len(self.rechazadas)


def construir_proteina(
    entrada,
    registro=None,
    archivo: Path | None = None,
) -> Proteina:
    """Cruza los metadatos de RCSB con la anotación de UniProt.

    **La longitud y la secuencia salen de la entidad polimérica del PDB, no de
    UniProt.** Son cosas distintas: 1UBQ contiene 76 residuos de ubiquitina,
    mientras que su UniProt (P0CG48, poliubiquitina-C) tiene 685. Filtrar la
    estructura por 685 la descartaría por larga cuando es corta.

    UniProt aporta lo que el PDB no tiene: nombre, anotación funcional,
    interacciones y PTMs. Si la entidad no trae organismo, se usa el de
    UniProt como respaldo.
    """
    entidad = entrada.entidad_principal

    return Proteina(
        pdb_id=entrada.pdb_id,
        uniprot_id=registro.accesion if registro else None,
        nombre=(registro.nombre if registro else None) or entrada.titulo,
        organismo=(entidad.organismo if entidad else None)
        or (registro.organismo if registro else None),
        tax_id=(entidad.tax_id if entidad else None)
        or (registro.tax_id if registro else None),
        metodo=entrada.metodo,
        resolucion=entrada.resolucion,
        longitud=entidad.longitud if entidad else None,
        secuencia=entidad.secuencia if entidad else None,
        archivo_path=str(archivo) if archivo else None,
        sha256=file_sha256(archivo) if archivo and archivo.is_file() else None,
        funciones=registro.funciones if registro else [],
        interacciones=registro.interacciones if registro else [],
        ptms=registro.ptms if registro else [],
    )


def fetch(
    config: Config,
    pdb_ids: list[str] | None = None,
    uniprot_ids: list[str] | None = None,
    formato: str = "pdb",
    forzar: bool = False,
    manifest: RunManifest | None = None,
) -> ResultadoFetch:
    """Descarga estructuras y anotaciones, y las persiste en la base.

    Aplica la verificación de bioseguridad antes de guardar nada. Una
    estructura rechazada se registra en la base con su motivo, pero su archivo
    de coordenadas se elimina del disco.

    Args:
        config: configuración del pipeline.
        pdb_ids: códigos PDB a descargar.
        uniprot_ids: accesiones UniProt; se resuelven a sus códigos PDB.
        formato: ``pdb`` o ``cif``.
        forzar: ignora la caché y vuelve a descargar.
        manifest: si se pasa, se le registran los archivos y las notas.
    """
    rutas = config.resolved_paths()
    resultado = ResultadoFetch()

    http = ClienteHTTP(
        timeout_s=config.data.http.timeout_s,
        max_retries=config.data.http.max_retries,
        cache_dir=rutas["data_raw"],
        usar_cache=config.data.http.cache and not forzar,
    )
    rcsb = ClienteRCSB(http)
    uniprot = ClienteUniProt(http)

    objetivos = [validar_pdb_id(p) for p in (pdb_ids or [])]

    # Una accesión UniProt se expande a todas sus estructuras experimentales.
    for accesion in uniprot_ids or []:
        try:
            registro = uniprot.obtener(accesion)
        except ErrorDeRed as exc:
            resultado.fallidas.append((accesion, str(exc)))
            continue
        if not registro.pdb_ids:
            resultado.fallidas.append(
                (accesion, "la accesión no tiene estructuras en el PDB")
            )
            continue
        logger.info(
            "%s aporta %d estructuras", accesion, len(registro.pdb_ids)
        )
        objetivos.extend(registro.pdb_ids)

    objetivos = list(dict.fromkeys(objetivos))  # sin duplicados, orden estable

    with BaseDatos(rutas["database"]) as db:
        for pdb_id in objetivos:
            try:
                _procesar_una(
                    pdb_id=pdb_id,
                    rcsb=rcsb,
                    uniprot=uniprot,
                    db=db,
                    config=config,
                    destino=rutas["data_raw"],
                    formato=formato,
                    resultado=resultado,
                    manifest=manifest,
                )
            except ErrorDeRed as exc:
                logger.error("%s: %s", pdb_id, exc)
                resultado.fallidas.append((pdb_id, str(exc)))
            except Exception as exc:  # noqa: BLE001 - una falla no corta el lote
                logger.exception("%s: error inesperado", pdb_id)
                resultado.fallidas.append((pdb_id, f"{type(exc).__name__}: {exc}"))

    http.close()
    return resultado


def _procesar_una(
    pdb_id: str,
    rcsb: ClienteRCSB,
    uniprot: ClienteUniProt,
    db: BaseDatos,
    config: Config,
    destino: Path,
    formato: str,
    resultado: ResultadoFetch,
    manifest: RunManifest | None,
) -> None:
    """Descarga, verifica y persiste una sola estructura."""
    logger.info("Procesando %s", pdb_id)
    entrada = rcsb.obtener_entrada(pdb_id)

    registro = None
    if entrada.uniprot_ids:
        try:
            registro = uniprot.obtener(entrada.uniprot_ids[0])
        except ErrorDeRed as exc:
            logger.warning("Sin anotación UniProt para %s: %s", pdb_id, exc)

    # Bioseguridad primero, antes de escribir el archivo al disco.
    if config.data.biosafety_check:
        veredicto = biosafety.verificar(
            organismo=registro.organismo if registro else None,
            keywords=registro.keywords if registro else None,
            nombre=registro.nombre if registro else None,
            titulo=entrada.titulo,
        )
        if not veredicto.permitida:
            motivo = f"[bioseguridad/{veredicto.capa}] {veredicto.motivo}"
            logger.warning("%s RECHAZADA por bioseguridad: %s", pdb_id, veredicto.motivo)

            proteina = construir_proteina(entrada, registro, archivo=None)
            db.guardar_proteina(
                proteina.model_copy(update={"curada": False, "motivo_rechazo": motivo})
            )
            resultado.rechazadas.append((pdb_id, motivo))
            if manifest:
                manifest.add_note(f"{pdb_id} rechazada por bioseguridad: {veredicto.motivo}")
            return

    archivo = rcsb.descargar_estructura(pdb_id, destino, formato=formato)
    proteina = construir_proteina(entrada, registro, archivo=archivo)
    db.guardar_proteina(proteina)
    resultado.descargadas.append(pdb_id)

    if manifest:
        manifest.add_output(archivo, key=f"{pdb_id}{archivo.suffix}")


def curate(
    config: Config,
    resolucion_max: float | None = None,
    organismos: list[str] | None = None,
    metodos: list[str] | None = None,
    longitud_min: int | None = None,
    longitud_max: int | None = None,
    manifest: RunManifest | None = None,
) -> ResultadoCurate:
    """Aplica los criterios de calidad sobre lo ya descargado. No usa la red.

    Los argumentos que no sean ``None`` sobrescriben lo que diga el
    ``config.yaml``; el resto se toma de ahí.
    """
    rutas = config.resolved_paths()
    resultado = ResultadoCurate()

    # Overrides de la CLI sobre la sección `data` del config.
    overrides = {
        "resolution_max": resolucion_max,
        "organisms": organismos,
        "experimental_methods": metodos,
        "length_min": longitud_min,
        "length_max": longitud_max,
    }
    efectivos = {k: v for k, v in overrides.items() if v is not None}
    criterios = (
        config.data.model_copy(update=efectivos) if efectivos else config.data
    )

    if efectivos:
        logger.info("Criterios sobrescritos desde la CLI: %s", efectivos)
    if manifest:
        manifest.params["criterios"] = criterios.model_dump(mode="json")

    with BaseDatos(rutas["database"]) as db:
        proteinas = db.listar_proteinas()
        if not proteinas:
            logger.warning(
                "No hay proteínas en la base. Corré `pdpipe fetch` primero."
            )
            return resultado

        for proteina in proteinas:
            # Un rechazo por bioseguridad no se revisa: se mantiene.
            if proteina.rechazada_por_bioseguridad:
                resultado.rechazadas.append((proteina.pdb_id, proteina.motivo_rechazo))
                continue

            veredicto = aplicar_filtros(proteina, criterios)
            db.marcar_curada(proteina.pdb_id, veredicto.aceptada, veredicto.motivo)

            if veredicto.aceptada:
                resultado.aceptadas.append(proteina.pdb_id)
                logger.info("%s aceptada", proteina.pdb_id)
            else:
                resultado.rechazadas.append((proteina.pdb_id, veredicto.motivo or ""))
                logger.info("%s rechazada: %s", proteina.pdb_id, veredicto.motivo)

    return resultado
