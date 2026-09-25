"""Fase 1 — Recolección de datos.

Descarga y curación de estructuras desde el RCSB PDB y de anotaciones
funcionales desde UniProt, con filtros por resolución, organismo, método
experimental y longitud, y persistencia en una base SQLite local
(tablas ``proteinas``, ``funciones``, ``interacciones``, ``ptms``).

Incluye la verificación de bioseguridad que rechaza toxinas y proteínas de
patógenos de la lista de agentes seleccionados (ver :mod:`.biosafety`).

Estado: implementada en el Hito 1.
"""

from pdpipe.phase1_data.database import BaseDatos, abrir_base
from pdpipe.phase1_data.http_client import (
    ClienteHTTP,
    ErrorDeRed,
    RecursoNoEncontrado,
)
from pdpipe.phase1_data.models import (
    EntradaPDB,
    Funcion,
    Interaccion,
    PTM,
    Proteina,
    RegistroUniProt,
    ResultadoFiltro,
)
from pdpipe.phase1_data.pipeline import curate, fetch
from pdpipe.phase1_data.rcsb import ClienteRCSB, PDBIDInvalido, validar_pdb_id
from pdpipe.phase1_data.uniprot import ClienteUniProt

__all__ = [
    "BaseDatos",
    "ClienteHTTP",
    "ClienteRCSB",
    "ClienteUniProt",
    "EntradaPDB",
    "ErrorDeRed",
    "Funcion",
    "Interaccion",
    "PDBIDInvalido",
    "PTM",
    "Proteina",
    "RecursoNoEncontrado",
    "RegistroUniProt",
    "ResultadoFiltro",
    "abrir_base",
    "curate",
    "fetch",
    "validar_pdb_id",
]
