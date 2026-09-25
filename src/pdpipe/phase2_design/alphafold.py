"""Cliente de la AlphaFold Protein Structure Database (Fase 2).

Primera opción del orden de preferencia para obtener una estructura predicha:
si el UniProt ID ya tiene un modelo depositado, bajarlo cuesta una petición y
no requiere GPU. ColabFold y ESMFold quedan para los casos que no están en la
base.

La API (``alphafold.ebi.ac.uk/api/prediction/{accesion}``) devuelve metadatos y
las URLs de los archivos; el pLDDT se extrae después del B-factor del modelo
descargado (ver :mod:`pdpipe.phase2_design.plddt`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pdpipe.phase1_data.http_client import (
    ClienteHTTP,
    ErrorDeRed,
    RecursoNoEncontrado,
)
from pdpipe.phase2_design.models import ModeloPredicho
from pdpipe.utils.checksums import file_sha256
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

URL_API = "https://alphafold.ebi.ac.uk/api/prediction"

FORMATOS = {"pdb": "pdbUrl", "cif": "cifUrl"}


class SinModeloPredicho(Exception):
    """La accesión no tiene modelo en AlphaFold DB."""


class ClienteAlphaFold:
    """Descarga modelos predichos de AlphaFold DB."""

    def __init__(self, http: ClienteHTTP) -> None:
        self.http = http

    def obtener_metadatos(self, accesion: str) -> dict[str, Any]:
        """Metadatos crudos de la predicción para una accesión UniProt.

        Raises:
            SinModeloPredicho: si la accesión no está en la base.
        """
        accesion = accesion.strip().upper()
        try:
            respuesta = self.http.get_json(
                f"{URL_API}/{accesion}",
                nombre_cache=f"alphafold_{accesion}.json",
            )
        except RecursoNoEncontrado as exc:
            raise SinModeloPredicho(
                f"{accesion} no tiene modelo en AlphaFold DB. "
                f"Probá con ColabFold (notebooks/) o ESMFold."
            ) from exc

        # La API devuelve una lista; el cliente HTTP la entrega tal cual.
        predicciones = respuesta if isinstance(respuesta, list) else [respuesta]
        if not predicciones:
            raise SinModeloPredicho(f"{accesion} devolvió una respuesta vacía")
        return predicciones[0]

    def descargar(
        self,
        accesion: str,
        destino: str | Path,
        formato: str = "pdb",
    ) -> ModeloPredicho:
        """Descarga el modelo y devuelve sus metadatos, sin el pLDDT todavía.

        El pLDDT se agrega después con
        :func:`pdpipe.phase2_design.plddt.anotar_modelo`, que necesita el
        archivo ya en disco.
        """
        accesion = accesion.strip().upper()
        if formato not in FORMATOS:
            raise ValueError(
                f"Formato '{formato}' desconocido. Opciones: {sorted(FORMATOS)}"
            )

        metadatos = self.obtener_metadatos(accesion)
        url = metadatos.get(FORMATOS[formato])
        if not url:
            raise SinModeloPredicho(
                f"{accesion} no tiene archivo en formato {formato}"
            )

        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        entry_id = metadatos.get("entryId") or f"AF-{accesion}-F1"
        version = metadatos.get("latestVersion")
        nombre = f"{entry_id}-model_v{version}.{formato}" if version else f"{entry_id}.{formato}"
        ruta = destino / nombre

        if ruta.is_file() and self.http.usar_cache:
            logger.info("%s ya estaba descargado en %s", accesion, ruta)
        else:
            logger.info("Descargando modelo de AlphaFold para %s (v%s)", accesion, version)
            ruta.write_bytes(self.http.get_bytes(url))

        return parsear_metadatos(metadatos).model_copy(
            update={"archivo": ruta, "sha256": file_sha256(ruta)}
        )


def parsear_metadatos(data: dict[str, Any]) -> ModeloPredicho:
    """Convierte la respuesta de la API en un :class:`ModeloPredicho`."""
    return ModeloPredicho(
        uniprot_id=data.get("uniprotAccession", ""),
        entry_id=data.get("entryId"),
        version=data.get("latestVersion"),
        fuente="alphafold_db",
        nombre=data.get("uniprotDescription"),
        organismo=data.get("organismScientificName"),
        tax_id=data.get("taxId"),
        secuencia=data.get("uniprotSequence"),
        metrica_global_api=data.get("globalMetricValue"),
    )


__all__ = [
    "ClienteAlphaFold",
    "ErrorDeRed",
    "SinModeloPredicho",
    "parsear_metadatos",
]
