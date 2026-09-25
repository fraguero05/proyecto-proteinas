"""Tests de los filtros de curación (Fase 1)."""

from __future__ import annotations

import pytest

from pdpipe.config import DataConfig
from pdpipe.phase1_data.filters import (
    aplicar_filtros,
    filtrar_longitud,
    filtrar_metodo,
    filtrar_organismo,
    filtrar_resolucion,
)
from pdpipe.phase1_data.models import Proteina


@pytest.fixture
def buena() -> Proteina:
    """Una proteína que pasa todos los criterios por defecto."""
    return Proteina(
        pdb_id="1LYZ",
        nombre="Lysozyme C",
        organismo="Gallus gallus",
        metodo="X-RAY DIFFRACTION",
        resolucion=2.0,
        longitud=147,
    )


@pytest.fixture
def criterios() -> DataConfig:
    return DataConfig(
        resolution_max=2.5,
        experimental_methods=["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
        length_min=30,
        length_max=600,
        organisms=[],
    )


# --------------------------------------------------------------- resolución


def test_resolucion_dentro_del_umbral(buena: Proteina):
    assert filtrar_resolucion(buena, 2.5) == []


def test_resolucion_exactamente_en_el_umbral(buena: Proteina):
    """El umbral es inclusivo."""
    assert filtrar_resolucion(buena, 2.0) == []


def test_resolucion_peor_que_el_umbral(buena: Proteina):
    motivos = filtrar_resolucion(buena, 1.5)
    assert len(motivos) == 1
    assert "2.00" in motivos[0]


def test_sin_resolucion_se_rechaza():
    """Dejarla pasar sería asumir que es perfecta."""
    rmn = Proteina(pdb_id="9XXX", metodo="SOLUTION NMR", resolucion=None)
    motivos = filtrar_resolucion(rmn, 2.5)
    assert motivos
    assert "SOLUTION NMR" in motivos[0]


# ------------------------------------------------------------------- método


def test_metodo_aceptado(buena: Proteina):
    assert filtrar_metodo(buena, ["X-RAY DIFFRACTION"]) == []


def test_metodo_es_insensible_a_mayusculas(buena: Proteina):
    assert filtrar_metodo(buena, ["x-ray diffraction"]) == []


def test_metodo_no_aceptado(buena: Proteina):
    motivos = filtrar_metodo(buena, ["SOLUTION NMR"])
    assert motivos
    assert "X-RAY DIFFRACTION" in motivos[0]


def test_lista_de_metodos_vacia_no_filtra(buena: Proteina):
    assert filtrar_metodo(buena, []) == []


def test_sin_metodo_se_rechaza():
    sin_metodo = Proteina(pdb_id="9XXX", metodo=None)
    assert filtrar_metodo(sin_metodo, ["X-RAY DIFFRACTION"])


# ---------------------------------------------------------------- organismo


def test_lista_de_organismos_vacia_no_filtra(buena: Proteina):
    assert filtrar_organismo(buena, []) == []


def test_organismo_coincidente(buena: Proteina):
    assert filtrar_organismo(buena, ["Gallus gallus"]) == []


def test_organismo_por_subcadena():
    """'Homo sapiens' debe matchear 'Homo sapiens (human)'."""
    p = Proteina(pdb_id="9XXX", organismo="Homo sapiens (human)")
    assert filtrar_organismo(p, ["Homo sapiens"]) == []


def test_organismo_es_insensible_a_mayusculas(buena: Proteina):
    assert filtrar_organismo(buena, ["gallus GALLUS"]) == []


def test_organismo_no_coincidente(buena: Proteina):
    motivos = filtrar_organismo(buena, ["Homo sapiens"])
    assert motivos
    assert "Gallus gallus" in motivos[0]


def test_alcanza_con_coincidir_uno(buena: Proteina):
    assert filtrar_organismo(buena, ["Homo sapiens", "Gallus gallus"]) == []


def test_sin_organismo_se_rechaza_si_hay_filtro():
    p = Proteina(pdb_id="9XXX", organismo=None)
    assert filtrar_organismo(p, ["Homo sapiens"])


# ----------------------------------------------------------------- longitud


def test_longitud_dentro_del_rango(buena: Proteina):
    assert filtrar_longitud(buena, 30, 600) == []


@pytest.mark.parametrize("longitud", [30, 600])
def test_los_bordes_del_rango_son_inclusivos(longitud: int):
    p = Proteina(pdb_id="9XXX", longitud=longitud)
    assert filtrar_longitud(p, 30, 600) == []


def test_longitud_muy_corta(buena: Proteina):
    motivos = filtrar_longitud(buena, 200, 600)
    assert motivos
    assert "menor" in motivos[0]


def test_longitud_muy_larga(buena: Proteina):
    motivos = filtrar_longitud(buena, 30, 100)
    assert motivos
    assert "mayor" in motivos[0]


def test_sin_longitud_se_rechaza():
    assert filtrar_longitud(Proteina(pdb_id="9XXX"), 30, 600)


# ----------------------------------------------------------------- conjunto


def test_proteina_buena_pasa_todo(buena: Proteina, criterios: DataConfig):
    resultado = aplicar_filtros(buena, criterios)
    assert resultado.aceptada
    assert resultado.motivo is None


def test_acumula_todos_los_motivos(criterios: DataConfig):
    """No corta en el primer fallo: para ajustar criterios hay que verlos todos."""
    mala = Proteina(
        pdb_id="9XXX",
        organismo="Mus musculus",
        metodo="SOLUTION NMR",
        resolucion=None,
        longitud=5,
    )
    estrictos = criterios.model_copy(update={"organisms": ["Homo sapiens"]})
    resultado = aplicar_filtros(mala, estrictos)

    assert not resultado.aceptada
    # resolución, método, organismo y longitud
    assert len(resultado.motivos) == 4


def test_un_solo_criterio_alcanza_para_rechazar(buena: Proteina, criterios: DataConfig):
    estrictos = criterios.model_copy(update={"resolution_max": 1.0})
    resultado = aplicar_filtros(buena, estrictos)
    assert not resultado.aceptada
    assert len(resultado.motivos) == 1


def test_el_motivo_concatena(buena: Proteina, criterios: DataConfig):
    estrictos = criterios.model_copy(
        update={"resolution_max": 1.0, "length_max": 100}
    )
    resultado = aplicar_filtros(buena, estrictos)
    assert "; " in resultado.motivo


def test_ubiquitina_y_lisozima_pasan_los_criterios_por_defecto(criterios: DataConfig):
    """Las dos proteínas del Hito 1, con sus valores reales."""
    ubiquitina = Proteina(
        pdb_id="1UBQ",
        organismo="Homo sapiens",
        metodo="X-RAY DIFFRACTION",
        resolucion=1.8,
        longitud=76,
    )
    lisozima = Proteina(
        pdb_id="1LYZ",
        organismo="Gallus gallus",
        metodo="X-RAY DIFFRACTION",
        resolucion=2.0,
        longitud=147,
    )
    assert aplicar_filtros(ubiquitina, criterios).aceptada
    assert aplicar_filtros(lisozima, criterios).aceptada
