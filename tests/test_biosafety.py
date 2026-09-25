"""Tests de la verificación de bioseguridad (Fase 1).

Dos propiedades que importan más que la cobertura:

* nada de la lista de exclusión pasa por ninguna de las tres capas;
* las proteínas de uso académico estándar no se rechazan por error, porque un
  filtro que rechaza todo es tan inútil como uno que no rechaza nada.
"""

from __future__ import annotations

import pytest

from pdpipe.phase1_data import biosafety


# --------------------------------------------------------- capa 1: organismo


@pytest.mark.parametrize(
    "organismo",
    [
        "Bacillus anthracis",
        "bacillus anthracis",
        "Bacillus anthracis str. Ames",
        "Yersinia pestis",
        "Clostridium botulinum",
        "Ricinus communis",
        "Naja naja",
        "Conus geographus",
        "Variola virus",
    ],
)
def test_organismos_excluidos_se_rechazan(organismo: str):
    resultado = biosafety.verificar(organismo=organismo, nombre="proteína cualquiera")
    assert not resultado.permitida
    assert resultado.capa == "organismo"


@pytest.mark.parametrize(
    "organismo",
    [
        "Homo sapiens",
        "Gallus gallus",
        "Saccharomyces cerevisiae",
        "Escherichia coli",
        "Aequorea victoria",
        "Bos taurus",
        "Arabidopsis thaliana",
    ],
)
def test_organismos_academicos_se_aceptan(organismo: str):
    resultado = biosafety.verificar(organismo=organismo, nombre="lisozima C")
    assert resultado.permitida, resultado.motivo


# ---------------------------------------------------------- capa 2: keywords


@pytest.mark.parametrize(
    "keyword",
    ["Toxin", "toxin", "Enterotoxin", "Neurotoxin", "Virulence", "Hemolysis"],
)
def test_keywords_excluidas_se_rechazan(keyword: str):
    resultado = biosafety.verificar(
        organismo="Homo sapiens", keywords=[keyword], nombre="proteína X"
    )
    assert not resultado.permitida
    assert resultado.capa == "keyword"


def test_keywords_normales_se_aceptan():
    resultado = biosafety.verificar(
        organismo="Gallus gallus",
        keywords=["Hydrolase", "Disulfide bond", "Secreted", "3D-structure"],
        nombre="Lysozyme C",
    )
    assert resultado.permitida, resultado.motivo


def test_una_sola_keyword_mala_alcanza():
    resultado = biosafety.verificar(
        organismo="Homo sapiens",
        keywords=["Hydrolase", "3D-structure", "Toxin"],
        nombre="proteína X",
    )
    assert not resultado.permitida


# ------------------------------------------------------------- capa 3: texto


@pytest.mark.parametrize(
    "nombre",
    [
        "Botulinum neurotoxin type A",
        "Anthrax protective antigen",
        "Ricin A chain",
        "Shiga toxin subunit B",
        "Alpha-hemolysin",
        "Diphtheria toxin",
        "Pertussis toxin subunit 1",
    ],
)
def test_nombres_excluidos_se_rechazan(nombre: str):
    resultado = biosafety.verificar(organismo="Homo sapiens", nombre=nombre)
    assert not resultado.permitida
    assert resultado.capa == "texto"


def test_titulo_del_pdb_tambien_se_revisa():
    resultado = biosafety.verificar(
        organismo="Homo sapiens",
        nombre="proteína sin nombre claro",
        titulo="CRYSTAL STRUCTURE OF BOTULINUM NEUROTOXIN",
    )
    assert not resultado.permitida
    assert resultado.capa == "texto"


@pytest.mark.parametrize(
    "nombre",
    [
        "Antitoxin HigA",
        "Toxin-antitoxin system antitoxin",
        "Glutathione S-transferase, detoxification enzyme",
    ],
)
def test_antitoxinas_y_detoxificacion_no_se_rechazan(nombre: str):
    """Una antitoxina neutraliza una toxina: no es lo mismo.

    Sin comparación por palabra completa y sin la lista de permitidos, el
    substring 'toxin' rechazaría a todas estas.
    """
    resultado = biosafety.verificar(organismo="Escherichia coli", nombre=nombre)
    assert resultado.permitida, resultado.motivo


def test_palabra_completa_no_matchea_substring():
    """'toxin' no debe matchear dentro de otra palabra."""
    assert biosafety.verificar_texto("Cytochrome c oxidase") == []
    assert biosafety.verificar_texto("Intoxination-unrelated protein") == []


# ------------------------------------------------- proteínas reales del Hito 1


def test_ubiquitina_se_acepta():
    resultado = biosafety.verificar(
        organismo="Homo sapiens",
        keywords=["3D-structure", "Cytoplasm", "Isopeptide bond", "Ubl conjugation"],
        nombre="Polyubiquitin-C",
        titulo="STRUCTURE OF UBIQUITIN REFINED AT 1.8 ANGSTROMS RESOLUTION",
    )
    assert resultado.permitida, resultado.motivo


def test_lisozima_se_acepta():
    resultado = biosafety.verificar(
        organismo="Gallus gallus",
        keywords=["Antimicrobial", "Bacteriolytic enzyme", "Hydrolase", "Secreted"],
        nombre="Lysozyme C",
        titulo="REFINEMENT OF TRICLINIC LYSOZYME",
    )
    assert resultado.permitida, resultado.motivo


# ------------------------------------------------------------- fallar cerrado


def test_sin_datos_se_rechaza_por_defecto():
    """Si no hay nada que verificar, no se puede afirmar que sea seguro."""
    resultado = biosafety.verificar()
    assert not resultado.permitida
    assert resultado.capa == "datos_insuficientes"


def test_sin_datos_se_acepta_si_se_pide_explicitamente():
    resultado = biosafety.verificar(exigir_datos=False)
    assert resultado.permitida


def test_solo_titulo_alcanza_para_verificar():
    resultado = biosafety.verificar(titulo="CRYSTAL STRUCTURE OF HUMAN HEMOGLOBIN")
    assert resultado.permitida


# ------------------------------------------------------------------ resultado


def test_el_motivo_explica_el_rechazo():
    resultado = biosafety.verificar(organismo="Bacillus anthracis", nombre="proteína")
    assert resultado.motivo is not None
    assert "anthracis" in resultado.motivo.lower()


def test_aceptada_no_tiene_motivo():
    resultado = biosafety.verificar(organismo="Homo sapiens", nombre="Hemoglobin")
    assert resultado.motivo is None
    assert resultado.capa is None


def test_resultado_es_inmutable():
    resultado = biosafety.verificar(organismo="Homo sapiens", nombre="Hemoglobin")
    with pytest.raises(Exception):
        resultado.permitida = False  # type: ignore[misc]


def test_verificar_proteina_usa_las_keywords_de_las_funciones():
    from pdpipe.phase1_data.models import Funcion, Proteina

    proteina = Proteina(
        pdb_id="9XXX",
        organismo="Homo sapiens",
        nombre="proteína X",
        funciones=[
            Funcion(tipo="keyword", descripcion="Toxin"),
            Funcion(tipo="go_molecular_function", descripcion="hydrolase activity"),
        ],
    )
    resultado = biosafety.verificar_proteina(proteina)
    assert not resultado.permitida
    assert resultado.capa == "keyword"


# ---------------------------------------------------------- las listas mismas


def test_las_listas_estan_normalizadas_en_minusculas():
    """La comparación asume minúsculas; si alguien agrega una mayúscula, falla."""
    for conjunto in (
        biosafety.ORGANISMOS_EXCLUIDOS,
        biosafety.KEYWORDS_EXCLUIDOS,
        biosafety.TERMINOS_EXCLUIDOS,
        biosafety.TERMINOS_PERMITIDOS,
    ):
        for item in conjunto:
            assert item == item.lower(), item
            assert item == item.strip(), item
