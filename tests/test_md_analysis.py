"""Tests del análisis de trayectorias de MD (Fase 3, parte C).

Ninguno necesita GROMACS: trabajan sobre trayectorias incluidas en el repo.

Las dos fixtures son sintéticas y están construidas para que la respuesta
correcta se conozca de antemano:

* ``traj_1UBQ.pdb`` — 11 cuadros de la ubiquitina con ruido creciente
  proporcional a la distancia al centro **y una rotación rígida de 4° por
  cuadro**. La rotación está puesta a propósito: el RMSD tiene que ignorarla
  y el RMSF tiene que verse afectado si no se alinea antes.
* ``aguas_hbond.pdb`` — dos aguas con hidrógenos explícitos; en el primer
  cuadro forman un puente lineal a 2.80 Å y en el segundo están a 6.00 Å.
  La respuesta correcta es exactamente ``[1, 0]``.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from pdpipe.config import load_config
from pdpipe.phase3_md import analysis as an
from pdpipe.phase3_md import pipeline as fase3
from pdpipe.phase3_md.figures import generar_figuras
from pdpipe.phase3_md.models import (
    PerfilPorResiduo,
    ResultadoAnalisisMD,
    SerieTemporal,
)

FIXTURES = Path(__file__).parent / "fixtures"
TRAYECTORIA = FIXTURES / "traj_1UBQ.pdb"
AGUAS = FIXTURES / "aguas_hbond.pdb"
ESTATICA = FIXTURES / "1UBQ.pdb"


@pytest.fixture
def universo():
    return an.cargar(TRAYECTORIA)


@pytest.fixture
def cfg(config_file: Path):
    return load_config(config_file)


# ------------------------------------------------------------------- carga


def test_carga_los_cuadros_del_pdb_multimodelo(universo):
    assert len(universo.trajectory) == 11
    assert len(universo.residues) == 76


def test_topologia_inexistente_falla(tmp_path: Path):
    with pytest.raises(an.ErrorDeAnalisis, match="No existe la topología"):
        an.cargar(tmp_path / "fantasma.pdb")


def test_trayectoria_inexistente_falla(tmp_path: Path):
    with pytest.raises(an.ErrorDeAnalisis, match="No existe la trayectoria"):
        an.cargar(ESTATICA, tmp_path / "fantasma.xtc")


def test_archivo_ilegible_falla(tmp_path: Path):
    basura = tmp_path / "basura.pdb"
    basura.write_text("esto no es un PDB\n", encoding="utf-8")

    with pytest.raises(an.ErrorDeAnalisis):
        an.cargar(basura)


# -------------------------------------------------------------------- RMSD


def test_el_primer_cuadro_tiene_rmsd_cero(universo):
    """Contra sí mismo, por definición."""
    assert an.calcular_rmsd(universo).inicial == 0.0


def test_el_rmsd_crece_a_lo_largo_de_la_trayectoria(universo):
    rmsd = an.calcular_rmsd(universo)

    assert rmsd.final > rmsd.inicial
    assert rmsd.deriva > 0


def test_el_rmsd_ignora_una_rotacion_de_cuerpo_rigido(tmp_path: Path):
    """El test que justifica usar RMSD con superposición.

    Se arma una trayectoria donde la proteína **solo rota**, sin cambiar de
    forma. El RMSD tiene que dar prácticamente cero: rotar no es cambiar. Sin
    superposición daría varios ångström y se leería como un cambio estructural
    que no ocurrió.
    """
    import MDAnalysis as mda

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(ESTATICA))
        prot = u.select_atoms("protein")
        base = prot.positions.copy()
        centro = base.mean(axis=0)

        destino = tmp_path / "solo_rotacion.pdb"
        with mda.Writer(str(destino), n_atoms=prot.n_atoms, multiframe=True) as w:
            for i in range(5):
                ang = np.deg2rad(30.0 * i)
                R = np.array([
                    [np.cos(ang), -np.sin(ang), 0],
                    [np.sin(ang), np.cos(ang), 0],
                    [0, 0, 1],
                ])
                prot.positions = (base - centro) @ R.T + centro
                w.write(prot)

    rmsd = an.calcular_rmsd(an.cargar(destino))

    assert rmsd.maximo < 0.1, f"la rotación no se eliminó: RMSD máximo {rmsd.maximo} Å"


def test_una_seleccion_vacia_falla(universo):
    with pytest.raises(an.ErrorDeAnalisis, match="no tiene átomos"):
        an.calcular_rmsd(universo, seleccion="resname NOEXISTE")


# -------------------------------------------------------------------- RMSF


def test_alinear_baja_el_rmsf(universo):
    """El RMSF sin alinear mide también la rotación del conjunto.

    La fixture rota 4° por cuadro. Si no se alinea, ese giro aparece como
    flexibilidad en todos los residuos por igual, y el número deja de
    significar "qué tan móvil es esta región".
    """
    sin_alinear = an.calcular_rmsf(universo)
    alineado = an.calcular_rmsf(an.alinear(an.cargar(TRAYECTORIA)))

    assert sin_alinear.media > alineado.media * 1.5


def test_el_rmsf_da_un_valor_por_residuo(universo):
    perfil = an.calcular_rmsf(an.alinear(universo))

    assert len(perfil.valores) == len(perfil.residuos) == 76
    assert all(v >= 0 for v in perfil.valores)


def test_los_extremos_son_los_mas_moviles(universo):
    """La cola C-terminal de la ubiquitina (74-76) es la región flexible."""
    perfil = an.calcular_rmsf(an.alinear(universo))
    moviles = [r for r, _ in perfil.mas_moviles(3)]

    assert 76 in moviles


@pytest.mark.parametrize("seleccion", ["protein and name CA", "backbone", "protein"])
def test_el_perfil_tiene_un_valor_por_residuo_con_cualquier_seleccion(seleccion):
    """Regresión: MDAnalysis devuelve el RMSF por átomo, no por residuo.

    Con la selección por defecto hay un carbono alfa por residuo y coinciden,
    pero ``backbone`` son cuatro átomos por residuo y ``protein`` muchos más.
    Sin promediar, el perfil repetía el mismo número de residuo y
    ``mas_moviles`` devolvía tres veces el 76.
    """
    perfil = an.calcular_rmsf(
        an.alinear(an.cargar(TRAYECTORIA), seleccion), seleccion
    )

    assert len(perfil.residuos) == 76
    assert len(set(perfil.residuos)) == 76, "hay residuos repetidos en el perfil"
    assert perfil.residuos == sorted(perfil.residuos)

    moviles = [r for r, _ in perfil.mas_moviles(5)]
    assert len(set(moviles)) == 5, f"mas_moviles repite residuos: {moviles}"


# ----------------------------------------------------------- radio de giro


def test_el_radio_de_giro_es_positivo_y_por_cuadro(universo):
    rg = an.calcular_radio_de_giro(universo)

    assert len(rg.valores) == 11
    assert rg.minimo > 0


# -------------------------------------------------------------------- SASA


def test_la_sasa_es_positiva(universo):
    sasa = an.calcular_sasa(universo, paso=5)

    assert sasa.minimo > 0
    assert sasa.unidad == "Å²"


def test_el_paso_reduce_los_cuadros_analizados(universo):
    """11 cuadros con paso 5 son los índices 0, 5 y 10."""
    assert len(an.calcular_sasa(universo, paso=5).valores) == 3


def test_un_paso_invalido_falla(universo):
    with pytest.raises(an.ErrorDeAnalisis, match="paso"):
        an.calcular_sasa(universo, paso=0)


# ------------------------------------------------------ puentes de hidrógeno


def test_cuenta_los_puentes_de_la_fixture_de_aguas():
    """Geometría calculada a mano: 1 puente en el primer cuadro, 0 en el segundo."""
    puentes = an.calcular_puentes_de_hidrogeno(
        an.cargar(AGUAS),
        donantes="name OW",
        hidrogenos="name HW1 HW2",
        aceptores="name OW",
    )

    assert puentes.valores == [1.0, 0.0]


def test_sin_hidrogenos_avisa_en_vez_de_contar_cero(universo):
    """Cero puentes y "no se pueden contar" son cosas muy distintas.

    Una estructura de rayos X no trae hidrógenos. Devolver cero se leería
    como que la proteína no tiene puentes, que es falso.
    """
    with pytest.raises(an.ErrorDeAnalisis, match="hidrógenos"):
        an.calcular_puentes_de_hidrogeno(universo)


def test_detecta_si_hay_hidrogenos(universo):
    assert an.tiene_hidrogenos(universo) is False
    assert an.tiene_hidrogenos(an.cargar(AGUAS)) is True


# ----------------------------------------------------------------- modelos


def test_estadisticos_de_una_serie():
    serie = SerieTemporal(
        nombre="X", unidad="Å", tiempos_ps=[0, 1, 2], valores=[1.0, 2.0, 6.0]
    )

    assert serie.media == 3.0
    assert serie.inicial == 1.0
    assert serie.final == 6.0
    assert serie.deriva == 5.0
    assert serie.maximo == 6.0


def test_una_serie_vacia_no_rompe():
    serie = SerieTemporal(nombre="X", unidad="Å")

    assert serie.media == 0.0
    assert serie.deriva == 0.0


def test_mas_moviles_ordena_de_mayor_a_menor():
    perfil = PerfilPorResiduo(
        nombre="RMSF", unidad="Å", residuos=[1, 2, 3], valores=[0.5, 2.0, 1.0]
    )

    assert perfil.mas_moviles(2) == [(2, 2.0), (3, 1.0)]


def test_series_omite_las_que_no_se_calcularon():
    resultado = ResultadoAnalisisMD(
        topologia=Path("x.pdb"),
        rmsd=SerieTemporal(nombre="RMSD", unidad="Å", tiempos_ps=[0], valores=[0.0]),
    )

    assert list(resultado.series()) == ["rmsd"]


# ---------------------------------------------------------------- pipeline


def test_analizar_deja_tabla_y_resumen(cfg):
    resultado = fase3.analizar(cfg, TRAYECTORIA, con_figuras=False)

    assert resultado.tabla.is_file()
    assert resultado.resumen_json.is_file()
    assert resultado.n_frames == 11

    datos = json.loads(resultado.resumen_json.read_text(encoding="utf-8"))
    assert {"rmsd", "radio_giro", "sasa", "rmsf"} <= set(datos["medidas"])


def test_la_tabla_trae_todas_las_magnitudes(cfg):
    resultado = fase3.analizar(cfg, TRAYECTORIA, con_figuras=False)

    with resultado.tabla.open(encoding="utf-8") as handle:
        magnitudes = {f["magnitud"] for f in csv.DictReader(handle)}

    assert {"RMSD", "Radio de giro", "SASA", "RMSF"} <= magnitudes


def test_una_medida_que_falla_no_corta_el_analisis(cfg):
    """Sin hidrógenos no hay puentes, pero el resto tiene que salir igual."""
    resultado = fase3.analizar(cfg, TRAYECTORIA, con_figuras=False)

    assert resultado.puentes is None
    assert resultado.rmsd is not None
    assert any("puentes" in a for a in resultado.advertencias)


def test_avisa_cuando_no_hay_paso_de_tiempo(cfg):
    """Un PDB multi-modelo no declara dt y MDAnalysis asume 1 ps.

    Sin el aviso, el eje de tiempo de las figuras diría picosegundos cuando
    en realidad son índices de cuadro.
    """
    resultado = fase3.analizar(cfg, TRAYECTORIA, con_figuras=False)

    assert resultado.dt_ps is None
    assert any("paso de tiempo" in a for a in resultado.advertencias)


def test_genera_las_figuras(cfg):
    resultado = fase3.analizar(cfg, TRAYECTORIA, con_figuras=True)

    assert len(resultado.figuras) == 5
    assert all(f.is_file() and f.stat().st_size > 0 for f in resultado.figuras)
    assert any(f.name.endswith("_resumen.png") for f in resultado.figuras)


def test_sin_figuras_no_genera_png(cfg):
    resultado = fase3.analizar(cfg, TRAYECTORIA, con_figuras=False)

    assert resultado.figuras == []


def test_registra_entradas_y_salidas_en_el_manifiesto(cfg):
    from pdpipe.utils.manifest import RunManifest

    manifest = RunManifest.start(command="md-analyze", seed=cfg.seed, config=cfg.to_dict())
    resultado = fase3.analizar(cfg, TRAYECTORIA, con_figuras=False, manifest=manifest)

    datos = manifest.to_dict()
    assert TRAYECTORIA.name in datos["inputs"]
    assert resultado.tabla.name in datos["outputs"]


def test_el_paso_de_sasa_llega_desde_la_cli(cfg):
    completo = fase3.analizar(cfg, TRAYECTORIA, con_figuras=False)
    salteado = fase3.analizar(cfg, TRAYECTORIA, paso_sasa=5, con_figuras=False)

    assert len(completo.sasa.valores) == 11
    assert len(salteado.sasa.valores) == 3


# -------------------------------------------------------------------- figuras


def test_las_figuras_se_nombran_por_la_topologia(cfg, tmp_path: Path):
    resultado = fase3.analizar(cfg, TRAYECTORIA, con_figuras=False)
    figuras = generar_figuras(resultado, tmp_path / "figs")

    nombres = {f.name for f in figuras}
    assert "traj_1UBQ_rmsd.png" in nombres
    assert "traj_1UBQ_rmsf.png" in nombres


def test_el_error_de_cargas_explica_que_hay_que_usar_el_tpr():
    """El mensaje tiene que decir cómo salir del paso, no solo qué falló.

    MDAnalysis dice "This Universe does not contain charge information", que
    no le sugiere a nadie que el problema es haber pasado el `.gro` en vez del
    `.tpr`.
    """
    universo = an.cargar(AGUAS)

    with pytest.raises(an.ErrorDeAnalisis, match=r"\.tpr"):
        # Sin selecciones explícitas MDAnalysis intenta deducir los donantes
        # por las cargas parciales, que un PDB no trae.
        an.calcular_puentes_de_hidrogeno(universo)
