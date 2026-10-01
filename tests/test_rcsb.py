"""Tests del cliente RCSB y del constructor de consultas (Fase 1).

Contra fixtures grabadas, sin red.
"""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from pdpipe.phase1_data.rcsb import (
    FILAS_POR_PAGINA,
    ClienteRCSB,
    PDBIDInvalido,
    _uniprot_de_entidad,
    construir_consulta,
    extraer_codigos,
    parsear_entidad,
    parsear_entrada,
    validar_pdb_id,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def entrada_1ubq() -> dict:
    return json.loads((FIXTURES / "rcsb_entry_1UBQ.json").read_text(encoding="utf-8"))


@pytest.fixture
def entrada_1lyz() -> dict:
    return json.loads((FIXTURES / "rcsb_entry_1LYZ.json").read_text(encoding="utf-8"))


@pytest.fixture
def entidad_1ubq() -> dict:
    return json.loads(
        (FIXTURES / "rcsb_entity_1UBQ_1.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def entidad_1lyz() -> dict:
    return json.loads(
        (FIXTURES / "rcsb_entity_1LYZ_1.json").read_text(encoding="utf-8")
    )


# ------------------------------------------------------------ validación de id


@pytest.mark.parametrize("pdb_id", ["1UBQ", "1ubq", " 1ubq ", "4HHB", "1lyz"])
def test_ids_validos_se_normalizan(pdb_id: str):
    assert validar_pdb_id(pdb_id) == pdb_id.strip().upper()


@pytest.mark.parametrize(
    "pdb_id",
    ["UBQ", "1UBQX", "", "   ", "ABCD", "1-BQ", "P0CG48"],
)
def test_ids_invalidos_fallan_antes_de_la_red(pdb_id: str):
    """Un typo tiene que fallar al instante, no como 404 tras tres reintentos."""
    with pytest.raises(PDBIDInvalido):
        validar_pdb_id(pdb_id)


def test_el_error_explica_el_formato():
    with pytest.raises(PDBIDInvalido) as exc:
        validar_pdb_id("XXXX")
    assert "1UBQ" in str(exc.value)


# -------------------------------------------------------------------- parseo


def test_parsea_1ubq(entrada_1ubq: dict):
    entrada = parsear_entrada(entrada_1ubq)
    assert entrada.pdb_id == "1UBQ"
    assert entrada.metodo == "X-RAY DIFFRACTION"
    assert entrada.resolucion == pytest.approx(1.8)
    assert "UBIQUITIN" in (entrada.titulo or "").upper()
    assert entrada.n_entidades_proteina == 1


def test_parsea_1lyz(entrada_1lyz: dict):
    entrada = parsear_entrada(entrada_1lyz)
    assert entrada.pdb_id == "1LYZ"
    assert entrada.metodo == "X-RAY DIFFRACTION"
    assert entrada.resolucion is not None


def test_toma_la_mejor_resolucion_cuando_hay_varias():
    data = {
        "rcsb_id": "9XXX",
        "rcsb_entry_info": {"resolution_combined": [2.5, 1.9, 3.0]},
    }
    assert parsear_entrada(data).resolucion == pytest.approx(1.9)


def test_entrada_sin_resolucion():
    """Las estructuras de RMN no reportan resolución."""
    data = {
        "rcsb_id": "9XXX",
        "exptl": [{"method": "SOLUTION NMR"}],
        "rcsb_entry_info": {},
    }
    entrada = parsear_entrada(data)
    assert entrada.resolucion is None
    assert entrada.metodo == "SOLUTION NMR"


def test_sin_resolucion_no_pasa_un_filtro_de_resolucion():
    """Tratar 'sin resolución' como 0.0 la haría pasar cualquier umbral."""
    data = {"rcsb_id": "9XXX", "rcsb_entry_info": {}}
    entrada = parsear_entrada(data)
    assert entrada.resolucion_o_infinito == float("inf")
    assert not entrada.resolucion_o_infinito <= 2.0


def test_json_vacio_no_rompe():
    entrada = parsear_entrada({})
    assert entrada.pdb_id == ""
    assert entrada.resolucion is None


# ------------------------------------------------------------------ entidades


def test_parsea_la_entidad_de_1ubq(entidad_1ubq: dict):
    entidad = parsear_entidad(entidad_1ubq)
    assert entidad.entity_id == "1"
    assert entidad.es_proteina
    assert entidad.organismo == "Homo sapiens"
    assert entidad.uniprot_ids == ["P0CG48"]


def test_la_longitud_es_la_de_la_cadena_no_la_de_uniprot(entidad_1ubq: dict):
    """El bug que esto previene:

    1UBQ contiene 76 residuos de ubiquitina, pero su UniProt (P0CG48,
    poliubiquitina-C) tiene 685. Tomar la longitud de UniProt hacía que la
    estructura fuera rechazada por 'demasiado larga' con el límite por
    defecto de 600, que es exactamente al revés de la realidad.
    """
    entidad = parsear_entidad(entidad_1ubq)
    assert entidad.longitud == 76
    assert entidad.secuencia is not None
    assert len(entidad.secuencia) == 76


def test_la_secuencia_no_trae_saltos_de_linea(entidad_1lyz: dict):
    entidad = parsear_entidad(entidad_1lyz)
    assert entidad.secuencia is not None
    assert "\n" not in entidad.secuencia
    assert " " not in entidad.secuencia
    assert len(entidad.secuencia) == entidad.longitud


def test_entidad_de_1lyz(entidad_1lyz: dict):
    entidad = parsear_entidad(entidad_1lyz)
    assert entidad.organismo == "Gallus gallus"
    assert entidad.uniprot_ids == ["P00698"]
    assert entidad.longitud == 129  # la cadena madura cristalizada


def test_entidad_vacia_no_rompe():
    entidad = parsear_entidad({}, entity_id="7")
    assert entidad.entity_id == "7"
    assert entidad.longitud is None
    assert not entidad.es_proteina


def test_entrada_expone_los_uniprot_de_sus_entidades(
    entrada_1ubq: dict, entidad_1ubq: dict
):
    entrada = parsear_entrada(entrada_1ubq).model_copy(
        update={"entidades": [parsear_entidad(entidad_1ubq)]}
    )
    assert entrada.uniprot_ids == ["P0CG48"]
    assert entrada.entidad_principal is not None
    assert entrada.entidad_principal.longitud == 76


def test_entidad_principal_ignora_los_acidos_nucleicos():
    from pdpipe.phase1_data.models import EntidadPolimerica, EntradaPDB

    entrada = EntradaPDB(
        pdb_id="9XXX",
        entidades=[
            EntidadPolimerica(entity_id="1", tipo="DNA", longitud=20),
            EntidadPolimerica(entity_id="2", tipo="Protein", longitud=150),
        ],
    )
    assert entrada.entidad_principal.entity_id == "2"


def test_entrada_sin_entidades_no_tiene_principal():
    from pdpipe.phase1_data.models import EntradaPDB

    assert EntradaPDB(pdb_id="9XXX").entidad_principal is None


# ------------------------------------------------------ UniProt de la entidad


def test_extrae_uniprot_de_la_alineacion():
    entidad = {
        "rcsb_polymer_entity_align": [
            {
                "reference_database_name": "UniProt",
                "reference_database_accession": "P00698",
            }
        ]
    }
    assert _uniprot_de_entidad(entidad) == ["P00698"]


def test_extrae_uniprot_del_contenedor():
    entidad = {
        "rcsb_polymer_entity_container_identifiers": {
            "reference_sequence_identifiers": [
                {"database_name": "UniProt", "database_accession": "P0CG48"}
            ]
        }
    }
    assert _uniprot_de_entidad(entidad) == ["P0CG48"]


def test_ignora_referencias_que_no_son_uniprot():
    entidad = {
        "rcsb_polymer_entity_align": [
            {"reference_database_name": "GenBank", "reference_database_accession": "X1"}
        ]
    }
    assert _uniprot_de_entidad(entidad) == []


def test_no_duplica_accesiones():
    entidad = {
        "rcsb_polymer_entity_align": [
            {"reference_database_name": "UniProt", "reference_database_accession": "P1"}
        ],
        "rcsb_polymer_entity_container_identifiers": {
            "reference_sequence_identifiers": [
                {"database_name": "UniProt", "database_accession": "P1"}
            ]
        },
    }
    assert _uniprot_de_entidad(entidad) == ["P1"]


# ------------------------------------------------------------------ consultas


def test_consulta_vacia_es_valida():
    """La Search API rechaza una consulta sin nodos."""
    consulta = construir_consulta()
    assert consulta["query"]["nodes"]
    assert consulta["return_type"] == "entry"


def test_consulta_con_resolucion():
    consulta = construir_consulta(resolucion_max=2.0)
    nodo = consulta["query"]["nodes"][0]
    assert nodo["parameters"]["attribute"] == "rcsb_entry_info.resolution_combined"
    assert nodo["parameters"]["operator"] == "less_or_equal"
    assert nodo["parameters"]["value"] == 2.0


def test_consulta_con_metodos():
    consulta = construir_consulta(metodos=["X-RAY DIFFRACTION"])
    nodo = consulta["query"]["nodes"][0]
    assert nodo["parameters"]["attribute"] == "exptl.method"
    assert nodo["parameters"]["value"] == ["X-RAY DIFFRACTION"]


def test_varios_organismos_van_en_un_or():
    consulta = construir_consulta(organismos=["Homo sapiens", "Mus musculus"])
    grupo = consulta["query"]["nodes"][0]
    assert grupo["logical_operator"] == "or"
    assert len(grupo["nodes"]) == 2


def test_rango_de_longitud():
    consulta = construir_consulta(longitud_min=50, longitud_max=300)
    nodo = consulta["query"]["nodes"][0]
    assert nodo["parameters"]["value"] == {"from": 50, "to": 300}


def test_solo_longitud_minima():
    consulta = construir_consulta(longitud_min=50)
    assert consulta["query"]["nodes"][0]["parameters"]["value"] == {"from": 50}


def test_los_criterios_se_combinan_con_and():
    consulta = construir_consulta(
        resolucion_max=2.0,
        metodos=["X-RAY DIFFRACTION"],
        organismos=["Homo sapiens"],
        longitud_min=50,
    )
    assert consulta["query"]["logical_operator"] == "and"
    assert len(consulta["query"]["nodes"]) == 4


def test_el_limite_va_en_la_paginacion():
    consulta = construir_consulta(limite=25)
    assert consulta["request_options"]["paginate"]["rows"] == 25


def test_sin_agrupar_se_piden_entradas():
    consulta = construir_consulta()
    assert consulta["return_type"] == "entry"
    assert "group_by" not in consulta["request_options"]


def test_agrupar_por_identidad_pide_representantes_de_entidades():
    consulta = construir_consulta(identidad_max=30)
    opciones = consulta["request_options"]
    assert consulta["return_type"] == "polymer_entity"
    assert opciones["group_by"] == {
        "aggregation_method": "sequence_identity",
        "similarity_cutoff": 30,
    }
    assert opciones["group_by_return_type"] == "representatives"


def test_el_inicio_va_en_la_paginacion():
    consulta = construir_consulta(limite=10, inicio=5000)
    assert consulta["request_options"]["paginate"] == {"start": 5000, "rows": 10}


# ------------------------------------------------------------ búsqueda


def test_extrae_codigos_de_entidades_sin_repetir():
    respuesta = {
        "result_set": [
            {"identifier": "1ubq_1"},
            {"identifier": "4HHB_1"},
            {"identifier": "4HHB_2"},
            {"identifier": "1LYZ"},
        ]
    }
    assert extraer_codigos(respuesta) == ["1UBQ", "4HHB", "1LYZ"]


def test_extrae_codigos_de_una_respuesta_vacia():
    assert extraer_codigos({}) == []


class _HTTPFalso:
    """Devuelve páginas consecutivas de una lista de identificadores."""

    def __init__(self, identificadores: list[str]) -> None:
        self.identificadores = identificadores
        self.pedidos: list[dict] = []

    def post_json(self, url: str, cuerpo: dict) -> dict:
        self.pedidos.append(cuerpo)
        pagina = cuerpo["request_options"]["paginate"]
        trozo = self.identificadores[pagina["start"] : pagina["start"] + pagina["rows"]]
        return {"result_set": [{"identifier": i} for i in trozo]}


def test_la_busqueda_recorre_varias_paginas():
    total = FILAS_POR_PAGINA + 3
    http = _HTTPFalso([f"{i:04d}_1" for i in range(total)])
    codigos = ClienteRCSB(http).buscar(limite=10 * FILAS_POR_PAGINA, identidad_max=30)
    assert len(codigos) == total
    # Dos páginas con datos y una tercera vacía que corta el recorrido.
    assert [p["request_options"]["paginate"]["start"] for p in http.pedidos] == [
        0,
        FILAS_POR_PAGINA,
        2 * FILAS_POR_PAGINA,
    ]


def test_la_busqueda_respeta_el_limite():
    http = _HTTPFalso([f"{i:04d}" for i in range(50)])
    codigos = ClienteRCSB(http).buscar(limite=7)
    assert codigos == [f"{i:04d}" for i in range(7)]
    assert len(http.pedidos) == 1
