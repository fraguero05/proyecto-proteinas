"""Verificación de bioseguridad de la Fase 1.

El pipeline trabaja solo con proteínas de uso académico estándar (lisozima,
ubiquitina, GFP, hemoglobina, enzimas metabólicas bien caracterizadas). Este
módulo rechaza todo lo que caiga fuera de ese alcance: toxinas, factores de
virulencia y proteínas de organismos de la lista de agentes seleccionados.

Tres capas independientes, porque ninguna sola alcanza:

1. **Organismo** — contra una lista de patógenos y agentes seleccionados.
2. **Keywords de UniProt** — anotación curada (``Toxin``, ``Virulence``...),
   la señal más confiable de las tres.
3. **Texto libre** — nombre de la proteína y título de la estructura, que
   atrapa casos sin anotar en UniProt.

**Falla cerrado.** Si no hay datos suficientes para verificar (UniProt caído,
estructura sin accesión), se rechaza. Un falso positivo cuesta una línea en el
config; un falso negativo mete en el pipeline algo que no debería estar.

El resultado de cada verificación queda registrado en la tabla ``proteinas``
(``motivo_rechazo``) y en el ``run_manifest.json`` de la corrida.
"""

from __future__ import annotations

from typing import Any

import re

from pydantic import BaseModel, ConfigDict, Field

from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Listas de exclusión
# ---------------------------------------------------------------------------

# Géneros y especies cuyas proteínas quedan fuera de alcance. Cubre la lista
# de agentes seleccionados (HHS/USDA) y patógenos de alto riesgo.
ORGANISMOS_EXCLUIDOS: frozenset[str] = frozenset(
    {
        "bacillus anthracis",
        "yersinia pestis",
        "francisella tularensis",
        "burkholderia mallei",
        "burkholderia pseudomallei",
        "brucella",
        "coxiella burnetii",
        "rickettsia prowazekii",
        "clostridium botulinum",
        "clostridium argentinense",
        "clostridium perfringens",
        "clostridioides difficile",
        "corynebacterium diphtheriae",
        "vibrio cholerae",
        "shigella dysenteriae",
        "bordetella pertussis",
        "staphylococcus aureus",
        "streptococcus pyogenes",
        "mycobacterium tuberculosis",
        "ricinus communis",       # ricina
        "abrus precatorius",      # abrina
        "variola",                # viruela
        "ebolavirus",
        "marburgvirus",
        "lassa",
        "nipah",
        "hendra",
        "chapare",
        "junin",
        "machupo",
        "sabia",
        "guanarito",
        "severe acute respiratory syndrome",
        "middle east respiratory syndrome",
        "influenza a virus",
        "conus",                  # conotoxinas
        "androctonus",            # escorpiones
        "leiurus",
        "naja",                   # cobras
        "bungarus",
        "dendroaspis",
        "bothrops",
        "crotalus",
        "oxyuranus",
        "loxosceles",
        "latrodectus",
    }
)

# Keywords de UniProt que marcan una proteína fuera de alcance.
KEYWORDS_EXCLUIDOS: frozenset[str] = frozenset(
    {
        "toxin",
        "enterotoxin",
        "neurotoxin",
        "cytolysis",
        "hemolysis",
        "virulence",
        "bacteriocin",
        "amphibian defense peptide",
        "antibiotic resistance",
        "arthropod defensive protein",
        "dermonecrotic toxin",
        "ion channel impairing toxin",
        "presynaptic neurotoxin",
        "postsynaptic neurotoxin",
    }
)

# Términos en nombres y títulos. Se comparan como palabra completa para que
# "toxin" no matchee "antitoxin" ni "detoxification".
TERMINOS_EXCLUIDOS: frozenset[str] = frozenset(
    {
        "toxin",
        "toxins",
        "enterotoxin",
        "neurotoxin",
        "cytotoxin",
        "exotoxin",
        "endotoxin",
        "hemolysin",
        "leukocidin",
        "aerolysin",
        "anthrax",
        "botulinum",
        "tetanus",
        "diphtheria",
        "ricin",
        "abrin",
        "saporin",
        "shiga",
        "pertussis",
        "conotoxin",
        "bungarotoxin",
        "virulence",
        "invasin",
        "adhesin",
    }
)

# Excepciones: términos que contienen una palabra excluida pero son inocuos o
# directamente lo contrario (una antitoxina neutraliza una toxina).
TERMINOS_PERMITIDOS: frozenset[str] = frozenset(
    {
        "antitoxin",
        "antitoxins",
        "detoxification",
        "detoxifying",
        "toxin-antitoxin",
        "antitoxin-like",
    }
)


class ResultadoBioseguridad(BaseModel):
    """Veredicto de la verificación, con su motivo y la capa que lo detectó."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    permitida: bool
    motivos: list[str] = Field(default_factory=list)
    capa: str | None = None  # organismo | keyword | texto | datos_insuficientes

    @property
    def motivo(self) -> str | None:
        return "; ".join(self.motivos) if self.motivos else None


def _normalizar(texto: str | None) -> str:
    return (texto or "").strip().lower()


def _contiene_palabra(texto: str, termino: str) -> bool:
    """¿Aparece ``termino`` como palabra completa en ``texto``?

    Evita que "toxin" matchee dentro de "antitoxin" o "detoxification".
    """
    return re.search(rf"\b{re.escape(termino)}\b", texto) is not None


def _tiene_termino_permitido(texto: str) -> bool:
    return any(_contiene_palabra(texto, t) for t in TERMINOS_PERMITIDOS)


def verificar_organismo(organismo: str | None) -> list[str]:
    """Capa 1: el organismo de origen contra la lista de exclusión."""
    normalizado = _normalizar(organismo)
    if not normalizado:
        return []
    return [
        f"organismo excluido: '{organismo}' coincide con '{excluido}'"
        for excluido in ORGANISMOS_EXCLUIDOS
        if excluido in normalizado
    ]


def verificar_keywords(keywords: list[str] | None) -> list[str]:
    """Capa 2: las keywords curadas de UniProt."""
    motivos = []
    for keyword in keywords or []:
        if _normalizar(keyword) in KEYWORDS_EXCLUIDOS:
            motivos.append(f"keyword de UniProt excluida: '{keyword}'")
    return motivos


def verificar_texto(*textos: str | None) -> list[str]:
    """Capa 3: nombre de la proteína y título de la estructura."""
    motivos = []
    for texto in textos:
        normalizado = _normalizar(texto)
        if not normalizado or _tiene_termino_permitido(normalizado):
            continue
        for termino in TERMINOS_EXCLUIDOS:
            if _contiene_palabra(normalizado, termino):
                motivos.append(f"término excluido '{termino}' en: '{texto}'")
                break
    return motivos


def verificar(
    organismo: str | None = None,
    keywords: list[str] | None = None,
    nombre: str | None = None,
    titulo: str | None = None,
    exigir_datos: bool = True,
) -> ResultadoBioseguridad:
    """Corre las tres capas y devuelve el veredicto.

    Args:
        organismo: nombre científico del organismo de origen.
        keywords: keywords de UniProt.
        nombre: nombre de la proteína.
        titulo: título de la estructura en el PDB.
        exigir_datos: si es ``True`` (por defecto) y no hay datos suficientes
            para verificar, rechaza. Es el comportamiento de fallar cerrado.

    Returns:
        El veredicto, con los motivos y la capa que disparó el rechazo.
    """
    # Sin nada que mirar, no hay verificación posible.
    if exigir_datos and not any([organismo, keywords, nombre, titulo]):
        return ResultadoBioseguridad(
            permitida=False,
            motivos=[
                "no hay datos suficientes para verificar bioseguridad "
                "(sin organismo, keywords, nombre ni título)"
            ],
            capa="datos_insuficientes",
        )

    if motivos := verificar_organismo(organismo):
        return ResultadoBioseguridad(permitida=False, motivos=motivos, capa="organismo")

    if motivos := verificar_keywords(keywords):
        return ResultadoBioseguridad(permitida=False, motivos=motivos, capa="keyword")

    if motivos := verificar_texto(nombre, titulo):
        return ResultadoBioseguridad(permitida=False, motivos=motivos, capa="texto")

    return ResultadoBioseguridad(permitida=True)


def verificar_proteina(proteina: Any, exigir_datos: bool = True) -> ResultadoBioseguridad:
    """Atajo para verificar un :class:`~pdpipe.phase1_data.models.Proteina`."""
    keywords = [f.descripcion for f in getattr(proteina, "funciones", []) if f.tipo == "keyword"]
    return verificar(
        organismo=getattr(proteina, "organismo", None),
        keywords=keywords,
        nombre=getattr(proteina, "nombre", None),
        titulo=getattr(proteina, "titulo", None),
        exigir_datos=exigir_datos,
    )
