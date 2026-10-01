"""Tests del dataset de estructura secundaria (Fase 4, parte A).

Ningún test toca la red. Las estructuras salen de las fixtures del repo y las
familias de proteínas para probar el agrupamiento se generan sintéticamente,
con mutaciones controladas sobre una secuencia base, para que la respuesta
correcta se conozca de antemano.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from pdpipe.phase4_ml import dataset as ds
from pdpipe.phase4_ml import splits as sp
from pdpipe.phase4_ml.dssp import CLASES_Q3, ErrorDSSP, a_tres_estados, asignar
from pdpipe.phase4_ml.models import Metricas, Proteina, Split

FIXTURES = Path(__file__).parent / "fixtures"
UBQ = FIXTURES / "1UBQ.pdb"
LYZ = FIXTURES / "1LYZ.pdb"

AA = "ACDEFGHIKLMNPQRSTVWY"


def familias(n_familias: int = 10, por_familia: int = 3, largo: int = 120, seed: int = 1):
    """Familias de proteínas: dentro de cada una, secuencias casi idénticas.

    Es el escenario que el agrupamiento tiene que detectar. La respuesta
    correcta se conoce por construcción: tantos grupos como familias.
    """
    rng = random.Random(seed)
    proteinas = []
    for familia in range(n_familias):
        base = "".join(rng.choice(AA) for _ in range(largo))
        for miembro in range(por_familia):
            letras = list(base)
            for _ in range(5):
                letras[rng.randrange(largo)] = rng.choice(AA)
            proteinas.append(
                Proteina(
                    identificador=f"F{familia}M{miembro}",
                    secuencia="".join(letras),
                    estructura="C" * largo,
                )
            )
    return proteinas


# ------------------------------------------------------------------- DSSP


def test_asigna_estructura_a_la_ubiquitina():
    """La ubiquitina es rica en hebras beta: es el plegamiento beta-grasp."""
    secuencia, q3, numeros = asignar(UBQ)

    assert len(secuencia) == len(q3) == len(numeros) == 76
    assert secuencia.startswith("MQIFVKTLTGK")
    assert q3.count("E") > q3.count("H"), "la ubiquitina tiene más hebra que hélice"


def test_asigna_estructura_a_la_lisozima():
    """La lisozima es lo contrario: predominantemente helicoidal."""
    _, q3, _ = asignar(LYZ)

    assert q3.count("H") > q3.count("E"), "la lisozima tiene más hélice que hebra"


def test_solo_usa_las_tres_clases():
    _, q3, _ = asignar(UBQ)

    assert set(q3) <= set(CLASES_Q3)


@pytest.mark.parametrize(
    ("ocho", "tres"),
    [("H", "H"), ("G", "H"), ("I", "H"), ("E", "E"), ("B", "E"),
     ("T", "C"), ("S", "C"), (" ", "C"), ("NA", "C")],
)
def test_el_mapeo_de_ocho_a_tres_estados(ocho, tres):
    """Agrupar G e I con H, y B con E, es lo que hace comparable el Q3."""
    assert a_tres_estados(ocho) == tres


def test_una_estructura_inexistente_falla(tmp_path: Path):
    with pytest.raises(ErrorDSSP, match="No existe"):
        asignar(tmp_path / "fantasma.pdb")


def test_un_archivo_ilegible_falla(tmp_path: Path):
    basura = tmp_path / "basura.pdb"
    basura.write_text("no soy un PDB\n", encoding="utf-8")

    with pytest.raises(ErrorDSSP):
        asignar(basura)


# -------------------------------------------------------------- codificación


def test_la_ventana_se_centra_en_el_residuo():
    assert ds.una_ventana("MQIFVKT", 3, 7) == "MQIFVKT"
    assert ds.una_ventana("MQIFVKT", 3, 3) == "IFV"


def test_los_bordes_se_rellenan():
    """El relleno es un símbolo propio, distinto de X.

    "no hay residuo acá" y "hay un residuo que no reconozco" son cosas
    distintas: mezclarlas le enseñaría al modelo que los extremos son raros
    en vez de que son extremos.
    """
    # Centro 0: faltan los tres de la izquierda. Centro 6 (el último): faltan
    # los tres de la derecha.
    assert ds.una_ventana("MQIFVKT", 0, 7) == "---MQIF"
    assert ds.una_ventana("MQIFVKT", 6, 7) == "FVKT---"
    assert ds.RELLENO != ds.DESCONOCIDO


def test_una_ventana_par_falla():
    """Sin centro no hay residuo que etiquetar."""
    with pytest.raises(ValueError, match="impar"):
        ds.una_ventana("MQIFVKT", 3, 16)


def test_los_residuos_no_estandar_caen_en_desconocido():
    assert ds.codificar_residuo("U") == ds.INDICE_AA[ds.DESCONOCIDO]
    assert ds.codificar_residuo("A") == ds.INDICE_AA["A"]


def test_el_one_hot_tiene_la_forma_esperada():
    proteinas = [Proteina(identificador="X", secuencia="ACDEF", estructura="HHEEC")]

    X, y, grupos = ds.a_ventanas(proteinas, ventana=5)

    assert X.shape == (5, 5 * len(ds.ALFABETO))
    assert y.shape == (5,)
    # Cada posición de la ventana prende exactamente un símbolo.
    assert np.allclose(X.sum(axis=1), 5.0)
    assert set(grupos) == {"X"}


def test_las_secuencias_completas_conservan_el_largo():
    proteinas = [
        Proteina(identificador="A", secuencia="ACDEF", estructura="HHEEC"),
        Proteina(identificador="B", secuencia="MQIFVKT", estructura="CCCHHHE"),
    ]

    entradas, salidas, ids = ds.a_secuencias(proteinas)

    assert [e.shape for e in entradas] == [(5, len(ds.ALFABETO)), (7, len(ds.ALFABETO))]
    assert [len(s) for s in salidas] == [5, 7]
    assert ids == ["A", "B"]


def test_un_dataset_vacio_no_rompe():
    X, y, grupos = ds.a_ventanas([], ventana=17)

    assert X.shape == (0, 17 * len(ds.ALFABETO))
    assert len(y) == 0 and len(grupos) == 0


def test_la_composicion_suma_uno():
    proteinas, _ = ds.construir({"1UBQ": UBQ, "1LYZ": LYZ})
    composicion = ds.composicion(proteinas)

    assert abs(sum(composicion.values()) - 1.0) < 0.01
    assert set(composicion) == set(CLASES_Q3)


def test_construir_descarta_las_cadenas_cortas():
    proteinas, descartadas = ds.construir({"1UBQ": UBQ}, longitud_min=200)

    assert proteinas == []
    assert any("76 residuos" in d for d in descartadas)


def test_construir_reporta_lo_que_no_pudo_procesar(tmp_path: Path):
    """Un dataset que silencia sus descartes no se puede auditar."""
    basura = tmp_path / "roto.pdb"
    basura.write_text("no soy un PDB\n", encoding="utf-8")

    proteinas, descartadas = ds.construir({"1UBQ": UBQ, "ROTO": basura})

    assert [p.identificador for p in proteinas] == ["1UBQ"]
    assert len(descartadas) == 1 and "ROTO" in descartadas[0]


# ------------------------------------------------------- identidad y grupos


@pytest.mark.parametrize(
    ("nombre", "a", "b", "esperado"),
    [
        ("idénticas", "MQIFVKTLTGKTITLEV", "MQIFVKTLTGKTITLEV", 1.0),
        ("una mutación", "MQIFVKTLTGKTITLEV", "MQIFVKTLTGKTITLEA", 16 / 17),
        ("no relacionadas", "MQIFVKTLTGKTITLEV", "WWWWWCCCCCPPPPPGG", 0.0),
        ("mitad distinta", "AAAAAAAAAA", "AAAAACCCCC", 0.5),
    ],
)
def test_la_identidad_de_secuencia(nombre, a, b, esperado):
    assert abs(sp.identidad(a, b) - esperado) < 0.02, nombre


def test_una_secuencia_contenida_en_otra_es_identica():
    """Regresión: con huecos de extremo penalizados daba 0.14 en vez de 1.0.

    Una proteína corta contenida en una larga es el caso de redundancia más
    claro que hay, y el alineamiento global lo castigaba por los huecos de los
    bordes. Con los extremos libres sale lo correcto.
    """
    assert sp.identidad("TLTGKTI", "MQIFVKTLTGKTITLEV") == 1.0


def test_el_filtro_de_kmeros_descarta_lo_que_no_se_parece():
    assert sp.solapamiento_kmeros(sp.kmeros("MQIFVKTLTG"), sp.kmeros("MQIFVKTLTG")) == 1.0
    assert sp.solapamiento_kmeros(sp.kmeros("MQIFVKTLTG"), sp.kmeros("WWCCPPGGAA")) == 0.0


def test_el_agrupamiento_encuentra_las_familias():
    """Diez familias de tres miembros tienen que dar diez grupos de tres."""
    grupos = sp.agrupar(familias(), identidad_max=0.3)

    assert len(grupos) == 10
    assert sorted(len(g) for g in grupos) == [3] * 10


def test_proteinas_distintas_quedan_en_grupos_separados():
    proteinas = familias(n_familias=5, por_familia=1)

    grupos = sp.agrupar(proteinas, identidad_max=0.3)

    assert len(grupos) == 5


# ----------------------------------------------------------------- división


def test_la_division_respeta_las_proporciones():
    proteinas = familias(n_familias=20, por_familia=1)

    division = sp.dividir(proteinas, val_size=0.2, test_size=0.2, seed=42)

    assert division.total == 20
    assert len(division.val) == pytest.approx(4, abs=2)
    assert len(division.test) == pytest.approx(4, abs=2)


def test_ninguna_familia_se_parte_entre_conjuntos():
    """El test que le da sentido al Q3.

    Si dos proteínas parecidas caen una en train y otra en test, el modelo
    puede acertar la segunda recordando la primera, y el Q3 mide memorización.
    """
    proteinas = familias()
    division = sp.dividir(proteinas, val_size=0.2, test_size=0.2, seed=42)

    for proteina in proteinas:
        familia = proteina.identificador.split("M")[0]
        hermanas = [p for p in proteinas if p.identificador.startswith(familia + "M")]
        conjuntos = {division.conjunto_de(h.identificador) for h in hermanas}
        assert len(conjuntos) == 1, f"{familia} quedó repartida en {conjuntos}"


def test_la_verificacion_no_encuentra_redundancia():
    proteinas = familias()
    division = sp.dividir(proteinas, val_size=0.2, test_size=0.2, seed=42)

    assert sp.verificar(proteinas, division, 0.3) == []


def test_la_verificacion_detecta_una_division_mal_hecha():
    """Si el control no detectara una división mala, no serviría de nada."""
    proteinas = familias(n_familias=1, por_familia=2)
    mala = Split(
        train=[proteinas[0].identificador],
        test=[proteinas[1].identificador],
        identidad_max=0.3,
    )

    assert len(sp.verificar(proteinas, mala, 0.3)) == 1


def test_la_division_es_reproducible():
    proteinas = familias()

    una = sp.dividir(proteinas, val_size=0.2, test_size=0.2, seed=7)
    otra = sp.dividir(proteinas, val_size=0.2, test_size=0.2, seed=7)

    assert una.train == otra.train and una.test == otra.test


def test_sin_proteinas_falla():
    with pytest.raises(sp.ErrorDeSplit, match="No hay proteínas"):
        sp.dividir([])


def test_proporciones_que_no_dejan_entrenamiento_fallan():
    with pytest.raises(sp.ErrorDeSplit, match="no deja nada"):
        sp.dividir(familias(), val_size=0.6, test_size=0.5)


# ------------------------------------------------------------------ modelos


def test_una_proteina_desalineada_no_se_puede_construir():
    """El error que no avisa: el modelo aprendería la etiqueta corrida.

    Nada falla si secuencia y estructura tienen largos distintos: el
    entrenamiento corre y el Q3 sale apenas peor, indistinguible de un modelo
    mediocre.
    """
    with pytest.raises(ValueError, match="76|residuos"):
        Proteina(identificador="X", secuencia="M" * 76, estructura="C" * 70)


def test_los_numeros_de_residuo_tambien_se_validan():
    with pytest.raises(ValueError, match="números de residuo"):
        Proteina(identificador="X", secuencia="MQI", estructura="CCC", numeros=[1, 2])


def test_la_mejora_sobre_la_base_se_calcula():
    """Un Q3 que no supera a predecir la clase mayoritaria no aprendió nada."""
    metricas = Metricas(modelo="bilstm", q3=0.76, q3_base=0.47)

    assert metricas.mejora_sobre_base == 0.29


# ------------------------------------------------------------- bioseguridad


@pytest.mark.parametrize("solo_curadas", [True, False])
def test_el_dataset_nunca_usa_rechazos_por_bioseguridad(config_file, monkeypatch, solo_curadas):
    """Ni con ``--all``, aunque las coordenadas estén en disco.

    Es el caso de una estructura bajada antes de que la lista la incluyera:
    tiene archivo, así que no basta con saltear lo que falta en disco.
    """
    from pdpipe.config import load_config
    from pdpipe.phase1_data.database import abrir_base
    from pdpipe.phase1_data.models import Proteina as Registro
    from pdpipe.phase4_ml import pipeline

    config = load_config(config_file)
    with abrir_base(config.resolved_paths()["database"]) as base:
        base.guardar_proteina(Registro(pdb_id="1UBQ", archivo_path=str(UBQ), curada=True))
        base.guardar_proteina(
            Registro(
                pdb_id="1LYZ",
                archivo_path=str(LYZ),
                curada=False,
                motivo_rechazo="[bioseguridad/keyword] keyword de UniProt excluida: 'Toxin'",
            )
        )

    vistas = {}

    def capturar(estructuras, longitud_min):
        vistas.update(estructuras)
        raise RuntimeError("alcanza con saber qué llegó")

    monkeypatch.setattr(pipeline.ds, "construir", capturar)
    with pytest.raises(RuntimeError, match="alcanza"):
        pipeline.construir_dataset(config=config, solo_curadas=solo_curadas)
    assert set(vistas) == {"1UBQ"}
