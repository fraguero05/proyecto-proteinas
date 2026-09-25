"""Tests de la extracción del pLDDT desde el B-factor (Fase 2).

El test que más importa es el de contraste: el pLDDT medio que calculamos
leyendo el B-factor tiene que coincidir con el ``globalMetricValue`` que
reporta la propia API de AlphaFold. Si difieren, el parseo está mal.
"""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from pdpipe.phase2_design.models import BandaPLDDT, ResiduoPLDDT, clasificar_plddt
from pdpipe.phase2_design.plddt import (
    ErrorPLDDT,
    extraer_plddt,
    resumir_plddt,
    secuencia_de_residuos,
)

FIXTURES = Path(__file__).parent / "fixtures"
MODELO_AF = FIXTURES / "AF-P00698-F1-model_v6.pdb"


@pytest.fixture
def residuos() -> list[ResiduoPLDDT]:
    return extraer_plddt(MODELO_AF)


@pytest.fixture
def metadatos_af() -> dict:
    data = json.loads((FIXTURES / "alphafold_P00698.json").read_text(encoding="utf-8"))
    return data[0] if isinstance(data, list) else data


# ------------------------------------------------------------------- bandas


@pytest.mark.parametrize(
    ("valor", "banda"),
    [
        (0.0, BandaPLDDT.MUY_BAJA),
        (49.99, BandaPLDDT.MUY_BAJA),
        (50.0, BandaPLDDT.BAJA),
        (69.99, BandaPLDDT.BAJA),
        (70.0, BandaPLDDT.CONFIABLE),
        (89.99, BandaPLDDT.CONFIABLE),
        (90.0, BandaPLDDT.MUY_ALTA),
        (100.0, BandaPLDDT.MUY_ALTA),
    ],
)
def test_los_cortes_de_banda_son_los_de_deepmind(valor: float, banda: BandaPLDDT):
    assert clasificar_plddt(valor) is banda


def test_cada_banda_tiene_descripcion():
    for banda in BandaPLDDT:
        assert banda.descripcion


# ---------------------------------------------------------------- extracción


def test_extrae_un_residuo_por_posicion(residuos: list[ResiduoPLDDT]):
    # La lisozima C de Gallus gallus tiene 147 aminoácidos con el péptido señal.
    assert len(residuos) == 147


def test_los_residuos_estan_en_orden(residuos: list[ResiduoPLDDT]):
    numeros = [r.numero for r in residuos]
    assert numeros == sorted(numeros)
    assert numeros[0] == 1
    assert numeros[-1] == 147


def test_los_plddt_estan_en_rango(residuos: list[ResiduoPLDDT]):
    """El pLDDT es un porcentaje: fuera de 0-100 el parseo leyó otra columna."""
    for r in residuos:
        assert 0.0 <= r.plddt <= 100.0, f"residuo {r.numero}: {r.plddt}"


def test_cada_residuo_tiene_aminoacido(residuos: list[ResiduoPLDDT]):
    assert all(r.aminoacido for r in residuos)


def test_la_secuencia_reconstruida_coincide_con_la_de_uniprot(
    residuos: list[ResiduoPLDDT], metadatos_af: dict
):
    """Si la secuencia leída del PDB no es la de UniProt, se leyó mal el archivo."""
    assert secuencia_de_residuos(residuos) == metadatos_af["uniprotSequence"]


def test_cada_residuo_sabe_su_banda(residuos: list[ResiduoPLDDT]):
    for r in residuos:
        assert r.banda is clasificar_plddt(r.plddt)


# ------------------------------------------------ contraste contra la API


def test_la_media_calculada_coincide_con_la_de_la_api(
    residuos: list[ResiduoPLDDT], metadatos_af: dict
):
    """El contraste que valida todo el parseo del B-factor.

    Si leyéramos la columna equivocada (la de ocupancia, por ejemplo, que en
    estos archivos vale 1.00), la media daría 1.0 en vez de ~94.

    La tolerancia de 0.05 absorbe un desvío real de redondeo, no un error: el
    formato PDB guarda el B-factor con dos decimales, así que promediamos
    valores ya redondeados, mientras que la API calcula sobre la precisión
    completa. Da 93.89 contra 93.88.
    """
    resumen = resumir_plddt(residuos)
    assert resumen.media == pytest.approx(metadatos_af["globalMetricValue"], abs=0.05)


def test_las_fracciones_por_banda_coinciden_con_las_de_la_api(
    residuos: list[ResiduoPLDDT], metadatos_af: dict
):
    resumen = resumir_plddt(residuos)
    esperadas = {
        BandaPLDDT.MUY_BAJA.value: metadatos_af["fractionPlddtVeryLow"],
        BandaPLDDT.BAJA.value: metadatos_af["fractionPlddtLow"],
        BandaPLDDT.CONFIABLE.value: metadatos_af["fractionPlddtConfident"],
        BandaPLDDT.MUY_ALTA.value: metadatos_af["fractionPlddtVeryHigh"],
    }
    for banda, esperada in esperadas.items():
        assert resumen.fraccion_por_banda[banda] == pytest.approx(esperada, abs=0.005), banda


# ------------------------------------------------------------------ resumen


def test_el_resumen_es_consistente(residuos: list[ResiduoPLDDT]):
    resumen = resumir_plddt(residuos)
    valores = [r.plddt for r in residuos]

    assert resumen.n_residuos == len(residuos)
    assert resumen.minimo == pytest.approx(min(valores), abs=0.01)
    assert resumen.maximo == pytest.approx(max(valores), abs=0.01)
    assert resumen.minimo <= resumen.media <= resumen.maximo


def test_los_conteos_por_banda_suman_el_total(residuos: list[ResiduoPLDDT]):
    resumen = resumir_plddt(residuos)
    assert sum(resumen.conteo_por_banda.values()) == resumen.n_residuos


def test_las_fracciones_suman_uno(residuos: list[ResiduoPLDDT]):
    resumen = resumir_plddt(residuos)
    assert sum(resumen.fraccion_por_banda.values()) == pytest.approx(1.0, abs=0.001)


def test_todas_las_bandas_aparecen_aunque_esten_vacias():
    resumen = resumir_plddt([ResiduoPLDDT(numero=1, plddt=95.0)])
    assert set(resumen.conteo_por_banda) == {b.value for b in BandaPLDDT}
    assert resumen.conteo_por_banda[BandaPLDDT.MUY_BAJA.value] == 0


def test_fraccion_confiable_suma_las_dos_bandas_altas():
    residuos = [
        ResiduoPLDDT(numero=1, plddt=95.0),   # muy alta
        ResiduoPLDDT(numero=2, plddt=75.0),   # confiable
        ResiduoPLDDT(numero=3, plddt=60.0),   # baja
        ResiduoPLDDT(numero=4, plddt=30.0),   # muy baja
    ]
    assert resumir_plddt(residuos).fraccion_confiable == pytest.approx(0.5)


def test_resumir_sin_residuos_falla():
    with pytest.raises(ErrorPLDDT):
        resumir_plddt([])


# ------------------------------------------------------------------ errores


def test_archivo_inexistente_falla(tmp_path: Path):
    with pytest.raises(ErrorPLDDT, match="No existe"):
        extraer_plddt(tmp_path / "fantasma.pdb")


def test_formato_no_soportado_falla(tmp_path: Path):
    archivo = tmp_path / "modelo.xyz"
    archivo.write_text("no soy un pdb", encoding="utf-8")
    with pytest.raises(ErrorPLDDT, match="Formato no soportado"):
        extraer_plddt(archivo)


def test_pdb_sin_carbonos_alfa_falla(tmp_path: Path):
    """Un archivo con solo aguas no es un modelo de proteína."""
    archivo = tmp_path / "aguas.pdb"
    archivo.write_text(
        "HETATM    1  O   HOH A   1      11.104   6.134  -6.504  1.00 20.00           O\nEND\n",
        encoding="utf-8",
    )
    with pytest.raises(ErrorPLDDT, match="carbonos alfa"):
        extraer_plddt(archivo)


def test_residuo_no_estandar_no_rompe_la_secuencia():
    residuos = [
        ResiduoPLDDT(numero=1, aminoacido="M", plddt=90.0),
        ResiduoPLDDT(numero=2, aminoacido=None, plddt=90.0),
        ResiduoPLDDT(numero=3, aminoacido="K", plddt=90.0),
    ]
    assert secuencia_de_residuos(residuos) == "MXK"


# ------------------------------------------------------------ residuos bajos


def test_residuos_bajo_filtra_por_umbral():
    from pdpipe.phase2_design.models import ModeloPredicho

    modelo = ModeloPredicho(
        uniprot_id="P00001",
        residuos=[
            ResiduoPLDDT(numero=1, plddt=95.0),
            ResiduoPLDDT(numero=2, plddt=65.0),
            ResiduoPLDDT(numero=3, plddt=40.0),
        ],
    )
    bajos = modelo.residuos_bajo(70.0)
    assert [r.numero for r in bajos] == [2, 3]


def test_la_lisozima_tiene_baja_confianza_en_el_peptido_senal(
    residuos: list[ResiduoPLDDT],
):
    """Los primeros residuos son el péptido señal, que es flexible.

    No es un detalle cosmético: es la comprobación de que el pLDDT por residuo
    refleja biología real y no un número uniforme mal leído.
    """
    primeros = [r.plddt for r in residuos[:10]]
    cuerpo = [r.plddt for r in residuos[30:120]]
    assert min(primeros) < min(cuerpo)
