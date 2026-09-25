"""Cliente de la API REST de UniProt (Fase 1).

De cada entrada se extrae lo que alimenta las tablas ``proteinas``,
``funciones``, ``interacciones`` y ``ptms``:

* identidad y secuencia -> ``proteinas``
* términos GO, keywords y el comentario FUNCTION -> ``funciones``
* comentarios INTERACTION y referencias a IntAct/STRING -> ``interacciones``
* features de modificación (fosforilación, glicosilación, puentes
  disulfuro, cross-links) -> ``ptms``

Las formas de los campos están verificadas contra respuestas reales, grabadas
en ``tests/fixtures/uniprot_*.json``.
"""

from __future__ import annotations

from typing import Any

from pdpipe.phase1_data.http_client import ClienteHTTP
from pdpipe.phase1_data.models import Funcion, Interaccion, PTM, RegistroUniProt
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://rest.uniprot.org/uniprotkb"

# Features de UniProt que representan una modificación postraduccional.
# Se deja afuera lo que es estructura secundaria (Helix, Beta strand) o
# anotación de secuencia (Chain, Domain), que no son PTMs.
TIPOS_PTM = frozenset(
    {
        "Modified residue",
        "Glycosylation",
        "Disulfide bond",
        "Cross-link",
        "Lipidation",
        "Peptide",
        "Propeptide",
        "Signal",
        "Transit peptide",
        "Initiator methionine",
    }
)

# Prefijos del término GO según su ontología.
_ASPECTOS_GO = {
    "F": "go_molecular_function",
    "P": "go_biological_process",
    "C": "go_cellular_component",
}


class ClienteUniProt:
    """Descarga y parsea entradas de UniProt."""

    def __init__(self, http: ClienteHTTP) -> None:
        self.http = http

    def obtener_crudo(self, accesion: str) -> dict[str, Any]:
        """JSON crudo de una accesión, tal como lo devuelve UniProt."""
        accesion = accesion.strip().upper()
        return self.http.get_json(
            f"{BASE_URL}/{accesion}.json",
            nombre_cache=f"uniprot_{accesion}.json",
        )

    def obtener(self, accesion: str) -> RegistroUniProt:
        """Descarga y parsea una accesión."""
        return parsear_uniprot(self.obtener_crudo(accesion))


# ---------------------------------------------------------------------------
# Parsers — funciones puras, testeables contra los fixtures sin red
# ---------------------------------------------------------------------------


def parsear_uniprot(data: dict[str, Any]) -> RegistroUniProt:
    """Convierte la respuesta cruda de UniProt en un :class:`RegistroUniProt`."""
    accesion = data.get("primaryAccession", "")
    descripcion = data.get("proteinDescription", {})
    organismo = data.get("organism", {})
    secuencia = data.get("sequence", {})

    genes = data.get("genes", [])
    gen = None
    if genes:
        gen = genes[0].get("geneName", {}).get("value")

    return RegistroUniProt(
        accesion=accesion,
        nombre=_nombre_proteina(descripcion),
        gen=gen,
        organismo=organismo.get("scientificName"),
        tax_id=organismo.get("taxonId"),
        longitud=secuencia.get("length"),
        secuencia=secuencia.get("value"),
        keywords=[k.get("name", "") for k in data.get("keywords", []) if k.get("name")],
        pdb_ids=_referencias_pdb(data),
        funciones=_parsear_funciones(data),
        interacciones=_parsear_interacciones(data),
        ptms=_parsear_ptms(data),
    )


def _nombre_proteina(descripcion: dict[str, Any]) -> str | None:
    """Nombre recomendado; si no hay, el primero de los alternativos."""
    recomendado = descripcion.get("recommendedName", {})
    nombre = recomendado.get("fullName", {}).get("value")
    if nombre:
        return nombre

    enviados = descripcion.get("submissionNames", []) or descripcion.get(
        "alternativeNames", []
    )
    if enviados:
        return enviados[0].get("fullName", {}).get("value")
    return None


def _referencias_pdb(data: dict[str, Any]) -> list[str]:
    """Códigos PDB asociados a la entrada."""
    return [
        ref["id"]
        for ref in data.get("uniProtKBCrossReferences", [])
        if ref.get("database") == "PDB" and ref.get("id")
    ]


def _primera_evidencia(item: dict[str, Any]) -> str | None:
    """Código ECO de la primera evidencia, si la hay."""
    evidencias = item.get("evidences", [])
    if not evidencias:
        return None
    primera = evidencias[0]
    codigo = primera.get("evidenceCode", "")
    fuente = primera.get("source")
    identificador = primera.get("id")
    if fuente and identificador:
        return f"{codigo} ({fuente}:{identificador})"
    return codigo or None


def _parsear_funciones(data: dict[str, Any]) -> list[Funcion]:
    """Anotaciones funcionales: GO, keywords y el comentario FUNCTION."""
    funciones: list[Funcion] = []

    # 1. Términos GO. El valor viene como "F:hydrolase activity": la letra
    #    inicial indica la ontología.
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") != "GO":
            continue
        termino = next(
            (p["value"] for p in ref.get("properties", []) if p.get("key") == "GoTerm"),
            None,
        )
        if not termino:
            continue
        aspecto, _, etiqueta = termino.partition(":")
        evidencia = next(
            (
                p["value"]
                for p in ref.get("properties", [])
                if p.get("key") == "GoEvidenceType"
            ),
            None,
        )
        funciones.append(
            Funcion(
                tipo=_ASPECTOS_GO.get(aspecto, "go_otro"),
                termino_go=ref.get("id"),
                descripcion=etiqueta or termino,
                evidencia=evidencia,
            )
        )

    # 2. Keywords de UniProt.
    for keyword in data.get("keywords", []):
        nombre = keyword.get("name")
        if nombre:
            funciones.append(
                Funcion(
                    tipo="keyword",
                    termino_go=keyword.get("id"),
                    descripcion=nombre,
                )
            )

    # 3. Descripción textual de la función.
    for comentario in data.get("comments", []):
        if comentario.get("commentType") != "FUNCTION":
            continue
        for texto in comentario.get("texts", []):
            valor = texto.get("value")
            if valor:
                funciones.append(
                    Funcion(
                        tipo="descripcion",
                        descripcion=valor,
                        evidencia=_primera_evidencia(texto),
                    )
                )

    return funciones


def _parsear_interacciones(data: dict[str, Any]) -> list[Interaccion]:
    """Interacciones binarias de IntAct más las referencias a bases externas."""
    interacciones: list[Interaccion] = []

    for comentario in data.get("comments", []):
        tipo = comentario.get("commentType")

        if tipo == "INTERACTION":
            for interaccion in comentario.get("interactions", []):
                otro = interaccion.get("interactantTwo", {})
                partner = otro.get("uniProtKBAccession")
                # UniProt lista la homodimerización consigo misma; se conserva
                # porque es información biológica real, no un artefacto.
                interacciones.append(
                    Interaccion(
                        partner_uniprot=partner,
                        partner_gen=otro.get("geneName"),
                        tipo="binaria",
                        fuente="IntAct",
                        n_experimentos=interaccion.get("numberOfExperiments"),
                        evidencia=otro.get("intActId"),
                    )
                )

        elif tipo == "SUBUNIT":
            for texto in comentario.get("texts", []):
                valor = texto.get("value")
                if valor:
                    interacciones.append(
                        Interaccion(
                            partner_uniprot=None,
                            tipo="subunidad",
                            fuente="UniProt",
                            evidencia=_primera_evidencia(texto),
                            partner_gen=None,
                        )
                    )

    return interacciones


def _parsear_ptms(data: dict[str, Any]) -> list[PTM]:
    """Modificaciones postraduccionales a partir de las features."""
    ptms: list[PTM] = []
    for feature in data.get("features", []):
        tipo = feature.get("type")
        if tipo not in TIPOS_PTM:
            continue

        ubicacion = feature.get("location", {})
        inicio = ubicacion.get("start", {}).get("value")
        fin = ubicacion.get("end", {}).get("value")
        if inicio is None:
            continue

        # Un puente disulfuro o un cross-link abarcan dos posiciones; una
        # fosforilación es un solo residuo (start == end).
        ptms.append(
            PTM(
                posicion=inicio,
                posicion_fin=fin if fin is not None and fin != inicio else None,
                tipo=tipo,
                descripcion=feature.get("description") or None,
                evidencia=_primera_evidencia(feature),
            )
        )
    return ptms
