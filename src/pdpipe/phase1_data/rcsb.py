"""Cliente del RCSB PDB: Search API, Data API y descarga de estructuras (Fase 1).

Tres endpoints distintos, que es una fuente habitual de confusión:

* ``search.rcsb.org``  — busca entradas por criterios, devuelve solo IDs
* ``data.rcsb.org``    — metadatos de una entrada (método, resolución, título)
* ``files.rcsb.org``   — los archivos de coordenadas (PDB y mmCIF)

Las formas de los campos están verificadas contra respuestas reales, grabadas
en ``tests/fixtures/rcsb_entry_*.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import re

from pdpipe.phase1_data.http_client import ClienteHTTP
from pdpipe.phase1_data.models import EntidadPolimerica, EntradaPDB
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

URL_BUSQUEDA = "https://search.rcsb.org/rcsbsearch/v2/query"

#: Máximo de resultados que la Search API devuelve por pedido. Para recorrer
#: los ~23.600 grupos no redundantes del PDB hacen falta unas cinco páginas.
FILAS_POR_PAGINA = 5000
URL_DATOS = "https://data.rcsb.org/rest/v1/core"
URL_ARCHIVOS = "https://files.rcsb.org/download"

# Un código PDB son cuatro caracteres que empiezan con un dígito.
_PATRON_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")

FORMATOS = {"pdb": ".pdb", "cif": ".cif", "mmcif": ".cif"}


class PDBIDInvalido(ValueError):
    """El identificador no tiene la forma de un código PDB."""


def validar_pdb_id(pdb_id: str) -> str:
    """Normaliza un código PDB a mayúsculas y valida su forma.

    Se valida antes de salir a la red: así un typo falla al instante con un
    mensaje claro, en vez de como un 404 después de tres reintentos.
    """
    limpio = pdb_id.strip().upper()
    if not _PATRON_PDB_ID.match(limpio):
        raise PDBIDInvalido(
            f"'{pdb_id}' no es un código PDB válido "
            f"(son 4 caracteres que empiezan con un dígito, por ejemplo 1UBQ)"
        )
    return limpio


class ClienteRCSB:
    """Acceso al RCSB PDB."""

    def __init__(self, http: ClienteHTTP) -> None:
        self.http = http

    # ------------------------------------------------------------ metadatos

    def obtener_entrada_cruda(self, pdb_id: str) -> dict[str, Any]:
        """JSON crudo de los metadatos de una entrada."""
        pdb_id = validar_pdb_id(pdb_id)
        return self.http.get_json(
            f"{URL_DATOS}/entry/{pdb_id}",
            nombre_cache=f"rcsb_entry_{pdb_id}.json",
        )

    def obtener_entrada(self, pdb_id: str) -> EntradaPDB:
        """Metadatos parseados de una entrada, con sus entidades poliméricas."""
        pdb_id = validar_pdb_id(pdb_id)
        crudo = self.obtener_entrada_cruda(pdb_id)
        entrada = parsear_entrada(crudo)
        return entrada.model_copy(
            update={"entidades": self.obtener_entidades(pdb_id, crudo)}
        )

    def obtener_entidades(
        self, pdb_id: str, entrada_cruda: dict[str, Any] | None = None
    ) -> list[EntidadPolimerica]:
        """Entidades poliméricas de una entrada.

        El endpoint ``entry`` no las trae: hay que consultar cada una por
        separado. De acá salen la longitud, la secuencia y el organismo de la
        cadena efectivamente cristalizada, que es lo que corresponde usar para
        filtrar una estructura.

        Si una entidad falla, se la omite en vez de cortar: una estructura sin
        anotación sigue siendo utilizable.
        """
        pdb_id = validar_pdb_id(pdb_id)
        if entrada_cruda is None:
            try:
                entrada_cruda = self.obtener_entrada_cruda(pdb_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("No se pudo leer la entrada %s: %s", pdb_id, exc)
                return []

        ids_entidades = entrada_cruda.get(
            "rcsb_entry_container_identifiers", {}
        ).get("polymer_entity_ids", [])

        entidades: list[EntidadPolimerica] = []
        for entity_id in ids_entidades:
            try:
                cruda = self.http.get_json(
                    f"{URL_DATOS}/polymer_entity/{pdb_id}/{entity_id}",
                    nombre_cache=f"rcsb_entity_{pdb_id}_{entity_id}.json",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "No se pudo leer la entidad %s de %s: %s", entity_id, pdb_id, exc
                )
                continue
            entidades.append(parsear_entidad(cruda, entity_id=str(entity_id)))

        return entidades

    def obtener_uniprot_ids(self, pdb_id: str) -> list[str]:
        """Accesiones UniProt de las entidades polipeptídicas de una entrada."""
        entidades = self.obtener_entidades(pdb_id)
        todas = [acc for e in entidades for acc in e.uniprot_ids]
        return list(dict.fromkeys(todas))

    # ------------------------------------------------------------ descarga

    def descargar_estructura(
        self,
        pdb_id: str,
        destino: str | Path,
        formato: str = "pdb",
    ) -> Path:
        """Descarga el archivo de coordenadas y devuelve la ruta al archivo.

        Si ya existe en ``destino`` y la caché está activa, no vuelve a bajarlo.
        """
        pdb_id = validar_pdb_id(pdb_id)
        if formato not in FORMATOS:
            raise ValueError(
                f"Formato '{formato}' desconocido. Opciones: {sorted(FORMATOS)}"
            )

        extension = FORMATOS[formato]
        nombre = f"{pdb_id}{extension}"
        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        ruta_final = destino / nombre

        if ruta_final.is_file() and self.http.usar_cache:
            logger.info("%s ya estaba descargado en %s", pdb_id, ruta_final)
            return ruta_final

        logger.info("Descargando %s (%s)", pdb_id, formato)
        contenido = self.http.get_bytes(f"{URL_ARCHIVOS}/{nombre}")
        ruta_final.write_bytes(contenido)
        return ruta_final

    # ------------------------------------------------------------ búsqueda

    def buscar(
        self,
        resolucion_max: float | None = None,
        organismos: list[str] | None = None,
        metodos: list[str] | None = None,
        longitud_min: int | None = None,
        longitud_max: int | None = None,
        limite: int = 100,
        identidad_max: int | None = None,
    ) -> list[str]:
        """Busca entradas que cumplan los criterios y devuelve sus códigos PDB.

        Con ``identidad_max`` se agrupa por identidad de secuencia y se
        devuelve un representante por grupo, que es la unica forma de obtener
        un conjunto diverso: el orden por defecto del PDB es alfabetico y
        arranca con decenas de mutantes de mioglobina y lisozima T4.
        """
        codigos: list[str] = []
        inicio = 0
        while len(codigos) < limite:
            consulta = construir_consulta(
                resolucion_max=resolucion_max,
                organismos=organismos,
                metodos=metodos,
                longitud_min=longitud_min,
                longitud_max=longitud_max,
                limite=min(FILAS_POR_PAGINA, limite - len(codigos)),
                identidad_max=identidad_max,
                inicio=inicio,
            )
            respuesta = self.http.post_json(URL_BUSQUEDA, consulta)
            nuevos = extraer_codigos(respuesta)
            if not nuevos:
                break
            codigos.extend(nuevos)
            inicio += FILAS_POR_PAGINA
            logger.info("Búsqueda: %d códigos acumulados", len(codigos))

        return list(dict.fromkeys(codigos))[:limite]


# ---------------------------------------------------------------------------
# Parsers y constructores — funciones puras, testeables sin red
# ---------------------------------------------------------------------------


def extraer_codigos(respuesta: dict[str, Any]) -> list[str]:
    """Saca los códigos PDB de una respuesta de la Search API.

    Una búsqueda por entradas devuelve identificadores como ``1UBQ``, pero una
    agrupada por identidad devuelve entidades como ``1UBQ_1``: el sufijo es la
    entidad polimérica dentro de la entrada. Se corta en el guion bajo y se
    quitan repetidos, porque dos entidades distintas pueden venir del mismo
    archivo.
    """
    codigos: list[str] = []
    for item in respuesta.get("result_set", []):
        identificador = item.get("identifier", "") if isinstance(item, dict) else str(item)
        codigo = identificador.split("_", 1)[0].strip().upper()
        if codigo:
            codigos.append(codigo)
    return list(dict.fromkeys(codigos))


def parsear_entrada(data: dict[str, Any]) -> EntradaPDB:
    """Convierte la respuesta del Data API en un :class:`EntradaPDB`."""
    info = data.get("rcsb_entry_info", {})

    # resolution_combined es una lista: una estructura puede combinar varios
    # experimentos. Se toma la mejor (la menor).
    resoluciones = info.get("resolution_combined") or []
    resolucion = min(resoluciones) if resoluciones else None

    metodos = [m.get("method") for m in data.get("exptl", []) if m.get("method")]

    return EntradaPDB(
        pdb_id=data.get("rcsb_id", "").upper(),
        titulo=data.get("struct", {}).get("title"),
        metodo=metodos[0] if metodos else None,
        resolucion=resolucion,
        fecha_deposito=data.get("rcsb_accession_info", {}).get("initial_release_date"),
        n_entidades_proteina=info.get("polymer_entity_count_protein"),
        entidades=[],  # se completan con obtener_entidades
    )


def parsear_entidad(data: dict[str, Any], entity_id: str = "1") -> EntidadPolimerica:
    """Convierte la respuesta de ``polymer_entity`` en una entidad.

    ``rcsb_sample_sequence_length`` es la longitud de la cadena cristalizada,
    que es la que corresponde usar para filtrar estructuras (ver la nota en
    :class:`~pdpipe.phase1_data.models.EntidadPolimerica`).
    """
    polimero = data.get("entity_poly", {})
    fuentes = data.get("rcsb_entity_source_organism", []) or []
    primera_fuente = fuentes[0] if fuentes else {}

    # La forma canónica (_can) resuelve los residuos modificados al aminoácido
    # estándar correspondiente, que es lo que se quiere para trabajar después.
    secuencia = polimero.get("pdbx_seq_one_letter_code_can") or polimero.get(
        "pdbx_seq_one_letter_code"
    )
    if secuencia:
        secuencia = "".join(secuencia.split())  # viene con saltos de línea

    return EntidadPolimerica(
        entity_id=str(
            data.get("rcsb_polymer_entity_container_identifiers", {}).get(
                "entity_id", entity_id
            )
        ),
        tipo=polimero.get("rcsb_entity_polymer_type"),
        longitud=polimero.get("rcsb_sample_sequence_length"),
        secuencia=secuencia,
        organismo=primera_fuente.get("scientific_name"),
        tax_id=primera_fuente.get("ncbi_taxonomy_id"),
        uniprot_ids=_uniprot_de_entidad(data),
    )


def _uniprot_de_entidad(entidad: dict[str, Any]) -> list[str]:
    """Accesiones UniProt de una entidad polimérica."""
    accesiones: list[str] = []

    # Camino principal: las referencias alineadas.
    for alineacion in entidad.get("rcsb_polymer_entity_align", []):
        if alineacion.get("reference_database_name") == "UniProt":
            accesion = alineacion.get("reference_database_accession")
            if accesion:
                accesiones.append(accesion)

    # Alternativa: el contenedor de identificadores.
    identificadores = entidad.get("rcsb_polymer_entity_container_identifiers", {})
    for referencia in identificadores.get("reference_sequence_identifiers", []):
        if referencia.get("database_name") == "UniProt":
            accesion = referencia.get("database_accession")
            if accesion:
                accesiones.append(accesion)

    return list(dict.fromkeys(accesiones))


def construir_consulta(
    resolucion_max: float | None = None,
    organismos: list[str] | None = None,
    metodos: list[str] | None = None,
    longitud_min: int | None = None,
    longitud_max: int | None = None,
    limite: int = 100,
    identidad_max: int | None = None,
    inicio: int = 0,
) -> dict[str, Any]:
    """Arma el cuerpo JSON de una consulta a la Search API del RCSB.

    Se construye como un ``and`` de nodos ``terminal``. Si no se pasa ningún
    criterio, se usa uno que siempre matchea, porque la API rechaza una
    consulta vacía.

    Args:
        identidad_max: porcentaje de identidad de secuencia para agrupar
            (30, 50, 70, 90 o 95). Con esto se devuelve un representante por
            grupo en vez de todas las entradas, que es lo que hace falta para
            armar un conjunto diverso y no una pila de mutantes puntuales.
    """
    nodos: list[dict[str, Any]] = []

    if resolucion_max is not None:
        nodos.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": resolucion_max,
                },
            }
        )

    if metodos:
        nodos.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "in",
                    "value": metodos,
                },
            }
        )

    if organismos:
        nodos.append(
            {
                "type": "group",
                "logical_operator": "or",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": (
                                "rcsb_entity_source_organism.scientific_name"
                            ),
                            "operator": "exact_match",
                            "value": organismo,
                        },
                    }
                    for organismo in organismos
                ],
            }
        )

    if longitud_min is not None or longitud_max is not None:
        rango: dict[str, Any] = {}
        if longitud_min is not None:
            rango["from"] = longitud_min
        if longitud_max is not None:
            rango["to"] = longitud_max
        nodos.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "entity_poly.rcsb_sample_sequence_length",
                    "operator": "range",
                    "value": rango,
                },
            }
        )

    if not nodos:
        nodos.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                    "operator": "greater",
                    "value": 0,
                },
            }
        )

    opciones: dict[str, Any] = {
        "paginate": {"start": inicio, "rows": limite},
        "results_content_type": ["experimental"],
    }

    if identidad_max is not None:
        # Sin esto la busqueda devuelve las entradas en orden alfabetico, que
        # para el PDB significa decenas de mutantes puntuales de la misma
        # proteina: 101M, 103M y 104M son todos mioglobina de cachalote, y
        # 102L, 107L y 109L son lisozima T4. Un dataset asi no tiene
        # diversidad aunque tenga muchas filas.
        #
        # Agrupando por identidad de secuencia y pidiendo representantes, el
        # RCSB devuelve una entrada por grupo de secuencias similares.
        opciones["group_by"] = {
            "aggregation_method": "sequence_identity",
            "similarity_cutoff": identidad_max,
        }
        opciones["group_by_return_type"] = "representatives"

    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodos},
        # Al agrupar por identidad hay que pedir entidades y no entradas: la
        # identidad se define sobre la secuencia de una cadena, no sobre el
        # archivo completo, que puede tener varias cadenas distintas.
        "return_type": "polymer_entity" if identidad_max is not None else "entry",
        "request_options": opciones,
    }
