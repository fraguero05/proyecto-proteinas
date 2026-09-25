"""Tests del cliente y los parsers de UniProt (Fase 1).

Todos corren contra respuestas reales grabadas en ``tests/fixtures/``: nada de
red. Para refrescarlas: ``python tests/fixtures/capturar_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from pdpipe.phase1_data.uniprot import (
    TIPOS_PTM,
    parsear_uniprot,
    _parsear_funciones,
    _parsear_interacciones,
    _parsear_ptms,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def lisozima() -> dict:
    """P00698 — lisozima C de clara de huevo (Gallus gallus)."""
    return json.loads((FIXTURES / "uniprot_P00698.json").read_text(encoding="utf-8"))


@pytest.fixture
def ubiquitina() -> dict:
    """P0CG48 — poliubiquitina C humana."""
    return json.loads((FIXTURES / "uniprot_P0CG48.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ identidad


def test_parsea_la_identidad_de_la_lisozima(lisozima: dict):
    registro = parsear_uniprot(lisozima)
    assert registro.accesion == "P00698"
    assert registro.nombre == "Lysozyme C"
    assert registro.organismo == "Gallus gallus"
    assert registro.tax_id == 9031
    assert registro.gen == "LYZ"


def test_parsea_la_identidad_de_la_ubiquitina(ubiquitina: dict):
    registro = parsear_uniprot(ubiquitina)
    assert registro.accesion == "P0CG48"
    assert registro.organismo == "Homo sapiens"
    assert registro.tax_id == 9606


def test_secuencia_y_longitud_son_consistentes(lisozima: dict):
    registro = parsear_uniprot(lisozima)
    assert registro.longitud == 147
    assert registro.secuencia is not None
    assert len(registro.secuencia) == registro.longitud


def test_extrae_los_codigos_pdb(lisozima: dict):
    registro = parsear_uniprot(lisozima)
    assert registro.pdb_ids
    assert all(len(pid) == 4 for pid in registro.pdb_ids)


def test_extrae_las_keywords(lisozima: dict):
    registro = parsear_uniprot(lisozima)
    assert "Hydrolase" in registro.keywords
    assert "3D-structure" in registro.keywords


# ------------------------------------------------------------------ funciones


def test_parsea_terminos_go(lisozima: dict):
    funciones = _parsear_funciones(lisozima)
    go = [f for f in funciones if f.tipo.startswith("go_")]
    assert go
    assert all(f.termino_go and f.termino_go.startswith("GO:") for f in go)


def test_separa_las_tres_ontologias_go(lisozima: dict):
    tipos = {f.tipo for f in _parsear_funciones(lisozima) if f.tipo.startswith("go_")}
    # La lisozima tiene anotaciones en las tres ontologías.
    assert "go_molecular_function" in tipos
    assert "go_biological_process" in tipos
    assert "go_cellular_component" in tipos


def test_el_prefijo_de_ontologia_no_queda_en_la_descripcion(lisozima: dict):
    """El valor viene como 'F:hydrolase activity'; guardamos solo la etiqueta."""
    go = [f for f in _parsear_funciones(lisozima) if f.tipo.startswith("go_")]
    for funcion in go:
        assert not funcion.descripcion.startswith(("F:", "P:", "C:"))


def test_parsea_keywords_como_funciones(lisozima: dict):
    keywords = [f for f in _parsear_funciones(lisozima) if f.tipo == "keyword"]
    assert keywords
    assert any(f.descripcion == "Hydrolase" for f in keywords)


def test_parsea_la_descripcion_de_funcion(lisozima: dict):
    descripciones = [
        f for f in _parsear_funciones(lisozima) if f.tipo == "descripcion"
    ]
    assert descripciones
    assert len(descripciones[0].descripcion) > 20


def test_registra_la_evidencia_de_los_go(lisozima: dict):
    go = [f for f in _parsear_funciones(lisozima) if f.tipo.startswith("go_")]
    assert any(f.evidencia for f in go)


# -------------------------------------------------------------- interacciones


def test_parsea_interacciones_binarias(lisozima: dict):
    interacciones = _parsear_interacciones(lisozima)
    binarias = [i for i in interacciones if i.tipo == "binaria"]
    assert binarias
    assert all(i.fuente == "IntAct" for i in binarias)


def test_las_interacciones_registran_el_numero_de_experimentos(lisozima: dict):
    binarias = [i for i in _parsear_interacciones(lisozima) if i.tipo == "binaria"]
    assert any(i.n_experimentos and i.n_experimentos > 0 for i in binarias)


def test_parsea_el_comentario_de_subunidad(lisozima: dict):
    subunidades = [
        i for i in _parsear_interacciones(lisozima) if i.tipo == "subunidad"
    ]
    assert subunidades
    assert subunidades[0].fuente == "UniProt"


# ----------------------------------------------------------------------- PTMs


def test_parsea_puentes_disulfuro(lisozima: dict):
    ptms = _parsear_ptms(lisozima)
    disulfuros = [p for p in ptms if p.tipo == "Disulfide bond"]
    # La lisozima de clara de huevo tiene cuatro puentes disulfuro.
    assert len(disulfuros) == 4


def test_los_puentes_disulfuro_guardan_las_dos_posiciones(lisozima: dict):
    disulfuros = [p for p in _parsear_ptms(lisozima) if p.tipo == "Disulfide bond"]
    for puente in disulfuros:
        assert puente.posicion_fin is not None
        assert puente.posicion_fin > puente.posicion


def test_una_ptm_de_un_solo_residuo_no_tiene_posicion_fin(ubiquitina: dict):
    ptms = _parsear_ptms(ubiquitina)
    puntuales = [p for p in ptms if p.tipo == "Modified residue"]
    assert puntuales
    assert all(p.posicion_fin is None for p in puntuales)


def test_no_confunde_estructura_secundaria_con_ptm(lisozima: dict):
    """Helix y Beta strand son features, pero no son modificaciones."""
    tipos = {p.tipo for p in _parsear_ptms(lisozima)}
    assert "Helix" not in tipos
    assert "Beta strand" not in tipos
    assert "Turn" not in tipos
    assert tipos <= TIPOS_PTM


def test_las_posiciones_de_ptm_estan_dentro_de_la_secuencia(lisozima: dict):
    registro = parsear_uniprot(lisozima)
    assert registro.longitud is not None
    for ptm in registro.ptms:
        assert 1 <= ptm.posicion <= registro.longitud
        if ptm.posicion_fin is not None:
            assert ptm.posicion_fin <= registro.longitud


# --------------------------------------------------------------- robustez


def test_json_vacio_no_rompe():
    registro = parsear_uniprot({})
    assert registro.accesion == ""
    assert registro.funciones == []
    assert registro.ptms == []


def test_entrada_sin_nombre_recomendado():
    data = {
        "primaryAccession": "X00001",
        "proteinDescription": {
            "submissionNames": [{"fullName": {"value": "Uncharacterized protein"}}]
        },
    }
    assert parsear_uniprot(data).nombre == "Uncharacterized protein"


def test_feature_sin_posicion_se_ignora():
    data = {
        "features": [
            {"type": "Modified residue", "location": {"start": {}, "end": {}}},
            {
                "type": "Modified residue",
                "location": {"start": {"value": 5}, "end": {"value": 5}},
            },
        ]
    }
    ptms = _parsear_ptms(data)
    assert len(ptms) == 1
    assert ptms[0].posicion == 5


def test_cliente_usa_la_cache(tmp_path: Path, lisozima: dict):
    """El cliente no debe salir a la red si el archivo ya está en caché."""
    from pdpipe.phase1_data.http_client import ClienteHTTP
    from pdpipe.phase1_data.uniprot import ClienteUniProt

    (tmp_path / "uniprot_P00698.json").write_text(
        json.dumps(lisozima), encoding="utf-8"
    )
    http = ClienteHTTP(cache_dir=tmp_path, usar_cache=True)
    # Si intentara conectarse, fallaría: no hay red en los tests.
    http.session = None  # type: ignore[assignment]

    registro = ClienteUniProt(http).obtener("P00698")
    assert registro.nombre == "Lysozyme C"
