"""Tests de la preparación del sistema y la simulación (Fase 3, partes A y B).

**Ningún test ejecuta GROMACS.** No está instalado en el entorno de
desarrollo, así que se prueba todo lo que sí se puede verificar sin el
binario, que es más de lo que parece:

* la limpieza de la estructura, que corre con Biopython;
* la generación de los ``.mdp`` desde el config, incluida la cuenta de pasos;
* el parseo de los errores de GROMACS, contra texto real de sus mensajes;
* el **encadenado de las cuatro etapas**, con un cliente falso que anota los
  comandos en vez de correrlos. Ahí es donde se esconde el error grave: si el
  NPT no recibe el checkpoint del NVT, arranca con velocidades nuevas y se
  pierde la equilibración térmica sin que nada falle.

Lo que no se puede verificar acá es que GROMACS acepte estos comandos. Eso se
comprueba corriendo el notebook.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pdpipe.config import load_config
from pdpipe.phase3_md import mdp as mdp_mod
from pdpipe.phase3_md.gromacs import (
    ClienteGromacs,
    ErrorDeGromacs,
    GromacsNoDisponible,
    extraer_error,
)
from pdpipe.phase3_md.preparation import (
    _iones_agregados,
    _resumen_gro,
    limpiar_estructura,
    preparar_sistema,
)
from pdpipe.phase3_md.simulation import correr_etapa, simular

FIXTURES = Path(__file__).parent / "fixtures"
UBQ = FIXTURES / "1UBQ.pdb"
LYZ = FIXTURES / "1LYZ.pdb"


@pytest.fixture
def cfg(config_file: Path):
    return load_config(config_file)


# ------------------------------------------------- envoltorio de GROMACS


def test_un_binario_inexistente_no_esta_disponible():
    gmx = ClienteGromacs("gmx_que_no_existe_12345")

    assert gmx.disponible() is False
    assert gmx.ruta() is None


def test_verificar_disponible_explica_como_instalarlo():
    gmx = ClienteGromacs("gmx_que_no_existe_12345")

    with pytest.raises(GromacsNoDisponible, match="apt install gromacs"):
        gmx.verificar_disponible()


def test_correr_sin_binario_falla_antes_de_intentarlo():
    gmx = ClienteGromacs("gmx_que_no_existe_12345")

    with pytest.raises(GromacsNoDisponible):
        gmx.correr("pdb2gmx", "-f", "x.pdb")


def test_la_version_es_none_si_no_esta_el_binario():
    """Mejor no registrar versión que registrar una inventada."""
    assert ClienteGromacs("gmx_que_no_existe_12345").version() is None


def test_extrae_el_bloque_de_error_fatal():
    """GROMACS enmarca el error entre guiones, tras banners y citas.

    Sin este parseo el usuario recibe el banner completo con las referencias
    bibliográficas y tiene que buscar la causa a mano.
    """
    salida = """
                      :-) GROMACS - gmx pdb2gmx, 2023.1 (-:

Executable:   /usr/bin/gmx
Data prefix:  /usr
Command line:
  gmx pdb2gmx -f limpio.pdb -o procesado.gro

Using the Amber99sb-ildn force field

-------------------------------------------------------
Program:     gmx pdb2gmx, version 2023.1
Source file: src/gromacs/gmxpreprocess/resall.cpp (line 660)

Fatal error:
Residue 'LIG' not found in residue topology database
Looked for LIG in the .rtp file
-------------------------------------------------------
"""
    mensaje = extraer_error(salida)

    assert "Residue 'LIG' not found" in mensaje
    assert "GROMACS - gmx pdb2gmx" not in mensaje, "quedó el banner"
    assert "Data prefix" not in mensaje


def test_sin_bloque_de_error_devuelve_el_final():
    mensaje = extraer_error("linea 1\nlinea 2\nalgo salio mal\n")

    assert "algo salio mal" in mensaje


def test_sin_salida_lo_dice():
    assert extraer_error("", "") == "(sin salida)"


def test_usa_stdout_si_stderr_esta_vacio():
    assert "en stdout" in extraer_error("", "el error vino en stdout")


# -------------------------------------------------------------- archivos .mdp


def test_la_cuenta_de_pasos_es_la_esperada():
    """100 ps con un paso de 2 fs son exactamente 50 000 pasos."""
    assert mdp_mod.pasos(100.0, 2.0) == 50_000
    assert mdp_mod.pasos(2000.0, 2.0) == 1_000_000
    assert mdp_mod.pasos(10.0, 1.0) == 10_000


def test_un_paso_de_tiempo_invalido_falla():
    with pytest.raises(ValueError, match="positivo"):
        mdp_mod.pasos(100.0, 0.0)


def test_escribe_los_cuatro_mdp(cfg, tmp_path: Path):
    rutas = mdp_mod.escribir_todos(cfg.md, tmp_path, seed=cfg.seed)

    assert set(rutas) == {"minim", "nvt", "npt", "prod"}
    assert all(r.is_file() for r in rutas.values())


def test_una_etapa_desconocida_falla(cfg, tmp_path: Path):
    with pytest.raises(ValueError, match="desconocida"):
        mdp_mod.escribir_mdp("equilibracion", cfg.md, tmp_path)


def test_la_minimizacion_usa_descenso_pronunciado(cfg):
    contenido = mdp_mod.contenido_minim(cfg.md)

    assert "integrator               = steep" in contenido
    assert str(cfg.md.minimization_steps) in contenido


def test_el_nvt_genera_velocidades_con_la_semilla_del_config(cfg):
    """La semilla es lo que hace reproducible la simulación."""
    contenido = mdp_mod.contenido_nvt(cfg.md, seed=cfg.seed)

    assert "gen_vel                  = yes" in contenido
    assert f"gen_seed                 = {cfg.seed}" in contenido
    assert "pcoupl                   = no" in contenido


def test_el_npt_continua_del_nvt_y_acopla_presion(cfg):
    contenido = mdp_mod.contenido_npt(cfg.md)

    assert "continuation             = yes" in contenido
    assert "gen_vel                  = no" in contenido, "regeneraría velocidades"
    assert "pcoupl                   = C-rescale" in contenido
    assert f"ref_p                    = {cfg.md.pressure_bar}" in contenido
    # Sin esto las restricciones de posición pelean contra el barostato.
    assert "refcoord_scaling         = com" in contenido


@pytest.mark.parametrize("etapa", ["nvt", "npt"])
def test_las_etapas_de_equilibracion_sujetan_la_proteina(cfg, etapa):
    generador = {"nvt": lambda: mdp_mod.contenido_nvt(cfg.md, 42),
                 "npt": lambda: mdp_mod.contenido_npt(cfg.md)}[etapa]

    assert "define                   = -DPOSRES" in generador()


def test_la_produccion_suelta_la_proteina(cfg):
    """Con POSRES en producción la proteína no se movería: el punto del hito."""
    contenido = mdp_mod.contenido_prod(cfg.md)

    assert "-DPOSRES" not in contenido
    assert "pcoupl                   = Parrinello-Rahman" in contenido


def test_la_temperatura_del_config_llega_a_los_mdp(cfg, tmp_path: Path):
    rutas = mdp_mod.escribir_todos(cfg.md, tmp_path)
    nvt = rutas["nvt"].read_text(encoding="utf-8")

    assert str(cfg.md.temperature_k) in nvt


def test_la_duracion_de_produccion_sale_del_config(cfg, tmp_path: Path):
    esperados = mdp_mod.pasos(cfg.md.production_ns * 1000, cfg.md.timestep_fs)
    contenido = mdp_mod.escribir_mdp("prod", cfg.md, tmp_path).read_text(encoding="utf-8")

    assert f"nsteps                   = {esperados}" in contenido


# ------------------------------------------------------ limpieza de estructura


def test_quita_las_aguas_cristalograficas(tmp_path: Path):
    """1UBQ trae 58 aguas del cristal; el solvente lo pone solvate después."""
    resultado = limpiar_estructura(UBQ, tmp_path / "limpio.pdb")

    assert resultado.aguas_quitadas == 58
    assert resultado.atomos_iniciales == 660
    assert resultado.atomos_finales == 602
    assert resultado.salida.is_file()


def test_conserva_las_aguas_si_se_pide(tmp_path: Path):
    resultado = limpiar_estructura(UBQ, tmp_path / "con_agua.pdb", quitar_aguas=False)

    assert resultado.aguas_quitadas == 0
    assert resultado.atomos_finales == 660


def test_limpia_tambien_la_lisozima(tmp_path: Path):
    resultado = limpiar_estructura(LYZ, tmp_path / "limpio.pdb")

    assert resultado.aguas_quitadas == 101
    assert resultado.atomos_finales == 1001


def test_una_estructura_sin_aguas_queda_igual(tmp_path: Path):
    """El modelo de AlphaFold no trae solvente ni heteroátomos."""
    modelo = FIXTURES / "AF-P00698-F1-model_v6.pdb"
    resultado = limpiar_estructura(modelo, tmp_path / "limpio.pdb")

    assert resultado.aguas_quitadas == 0
    assert resultado.heteroatomos_quitados == 0
    assert resultado.atomos_iniciales == resultado.atomos_finales


def test_una_estructura_inexistente_falla(tmp_path: Path):
    with pytest.raises(ValueError, match="No existe"):
        limpiar_estructura(tmp_path / "fantasma.pdb", tmp_path / "salida.pdb")


def test_un_archivo_ilegible_falla(tmp_path: Path):
    basura = tmp_path / "basura.pdb"
    basura.write_text("no soy un PDB\n", encoding="utf-8")

    with pytest.raises(ValueError):
        limpiar_estructura(basura, tmp_path / "salida.pdb")


# ------------------------------------------------------- lectura de salidas


def test_cuenta_atomos_y_aguas_de_un_gro(tmp_path: Path):
    """El .gro pone el total en la línea 2 y el residuo en las columnas 6-10."""
    gro = tmp_path / "sistema.gro"
    gro.write_text(
        "Sistema de prueba\n"
        "    7\n"
        "    1MET      N    1   1.000   1.000   1.000\n"
        "    1MET     CA    2   1.100   1.000   1.000\n"
        "    2SOL     OW    3   2.000   2.000   2.000\n"
        "    2SOL    HW1    4   2.100   2.000   2.000\n"
        "    2SOL    HW2    5   2.000   2.100   2.000\n"
        "    3SOL     OW    6   3.000   3.000   3.000\n"
        "    3SOL    HW1    7   3.100   3.000   3.000\n"
        "   4.00000   4.00000   4.00000\n",
        encoding="utf-8",
    )

    total, aguas = _resumen_gro(gro)

    assert total == 7
    assert aguas == 1  # 5 átomos SOL // 3


def test_un_gro_inexistente_no_rompe(tmp_path: Path):
    assert _resumen_gro(tmp_path / "fantasma.gro") == (0, 0)


def test_lee_los_iones_que_reporta_genion():
    """Texto tal como lo escribe genion."""
    salida = (
        "Reading file iones.tpr, VERSION 2023.1\n"
        "Replacing 8 solute molecules in topology file (topol.top)\n"
        " with 4 NA ions and 4 CL ions.\n"
    )

    assert _iones_agregados(salida, "") == {"NA": 4, "CL": 4}


def test_sin_linea_de_iones_devuelve_vacio():
    """Antes que inventar un número, ninguno."""
    assert _iones_agregados("otra cosa cualquiera", "") == {}


# -------------------------------------------- encadenado de las cuatro etapas


class _ClienteFalso(ClienteGromacs):
    """Anota los comandos en vez de ejecutarlos.

    Permite verificar el encadenado sin GROMACS: qué archivo entra en cada
    etapa, cuál es la referencia de las restricciones y de qué checkpoint se
    continúa.
    """

    def __init__(self) -> None:
        super().__init__("gmx_falso")
        self.llamadas: list[tuple[str, list[str]]] = []

    def disponible(self) -> bool:
        return True

    def version(self) -> str:
        return "2023.1-falso"

    def correr(self, herramienta, *argumentos, **kwargs):
        self.llamadas.append((herramienta, [str(a) for a in argumentos]))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    def argumentos_de(self, herramienta: str, indice: int) -> list[str]:
        """Los argumentos de la n-ésima llamada a una herramienta."""
        propias = [a for h, a in self.llamadas if h == herramienta]
        return propias[indice]


@pytest.fixture
def sistema_falso(cfg, tmp_path: Path):
    from pdpipe.phase3_md.models import SistemaPreparado

    mdps = mdp_mod.escribir_todos(cfg.md, tmp_path, seed=cfg.seed)
    estructura = tmp_path / "sistema.gro"
    estructura.write_text("falso\n    0\n   1 1 1\n", encoding="utf-8")
    topologia = tmp_path / "topol.top"
    topologia.write_text("; falso\n", encoding="utf-8")

    return SistemaPreparado(
        directorio=tmp_path,
        estructura=estructura,
        topologia=topologia,
        mdp=mdps,
    )


def test_corre_las_cuatro_etapas_en_orden(cfg, sistema_falso):
    gmx = _ClienteFalso()

    resultado = simular(cfg, sistema_falso, cliente=gmx)

    assert [e.nombre for e in resultado.etapas] == ["minim", "nvt", "npt", "prod"]
    # Dos llamadas por etapa: grompp y mdrun.
    assert [h for h, _ in gmx.llamadas] == ["grompp", "mdrun"] * 4


def test_el_npt_continua_del_checkpoint_del_nvt(cfg, sistema_falso):
    """El test que evita perder la equilibración térmica en silencio.

    Sin ``-t nvt.cpt`` el NPT arranca con velocidades nuevas: no falla, no
    avisa, y los 100 ps de NVT se tiran a la basura.
    """
    gmx = _ClienteFalso()
    simular(cfg, sistema_falso, cliente=gmx)

    npt = gmx.argumentos_de("grompp", 2)

    assert "-t" in npt
    assert npt[npt.index("-t") + 1] == "nvt.cpt"
    assert npt[npt.index("-c") + 1] == "nvt.gro"


def test_las_etapas_con_restricciones_pasan_la_referencia(cfg, sistema_falso):
    """POSRES sin ``-r`` hace fallar a grompp: necesita contra qué sujetar."""
    gmx = _ClienteFalso()
    simular(cfg, sistema_falso, cliente=gmx)

    nvt = gmx.argumentos_de("grompp", 1)
    npt = gmx.argumentos_de("grompp", 2)

    assert nvt[nvt.index("-r") + 1] == "em.gro"
    assert npt[npt.index("-r") + 1] == "nvt.gro"


def test_la_produccion_no_pasa_referencia_pero_si_checkpoint(cfg, sistema_falso):
    gmx = _ClienteFalso()
    simular(cfg, sistema_falso, cliente=gmx)

    prod = gmx.argumentos_de("grompp", 3)

    assert "-r" not in prod, "producción no lleva restricciones de posición"
    assert prod[prod.index("-t") + 1] == "npt.cpt"
    assert prod[prod.index("-c") + 1] == "npt.gro"


def test_la_minimizacion_arranca_del_sistema_solvatado(cfg, sistema_falso):
    gmx = _ClienteFalso()
    simular(cfg, sistema_falso, cliente=gmx)

    minim = gmx.argumentos_de("grompp", 0)

    assert minim[minim.index("-c") + 1] == "sistema.gro"
    assert "-t" not in minim, "la minimización no continúa de nada"


def test_el_override_de_ns_reescribe_el_mdp_de_produccion(cfg, sistema_falso):
    """La duración vive en el .mdp como pasos, no es un flag de mdrun."""
    gmx = _ClienteFalso()

    resultado = simular(cfg, sistema_falso, ns=0.5, cliente=gmx)

    contenido = sistema_falso.mdp["prod"].read_text(encoding="utf-8")
    assert f"nsteps                   = {mdp_mod.pasos(500.0, cfg.md.timestep_fs)}" in contenido
    assert resultado.ns_simulados == 0.5


def test_anota_pasos_y_picosegundos_de_cada_etapa(cfg, sistema_falso):
    gmx = _ClienteFalso()

    resultado = simular(cfg, sistema_falso, cliente=gmx)
    por_nombre = {e.nombre: e for e in resultado.etapas}

    assert por_nombre["minim"].ps_simulados == 0.0
    assert por_nombre["nvt"].ps_simulados == cfg.md.nvt_ps
    assert por_nombre["prod"].pasos == mdp_mod.pasos(
        cfg.md.production_ns * 1000, cfg.md.timestep_fs
    )


def test_registra_la_version_de_gromacs(cfg, sistema_falso):
    resultado = simular(cfg, sistema_falso, cliente=_ClienteFalso())

    assert resultado.version_gromacs == "2023.1-falso"


def test_una_etapa_sin_mdp_falla(cfg, sistema_falso):
    del sistema_falso.mdp["npt"]

    with pytest.raises(ErrorDeGromacs, match="npt"):
        correr_etapa(
            _ClienteFalso(), sistema_falso, "npt",
            entrada_gro="nvt.gro", prefijo="npt",
        )


# ------------------------------------------------ preparación sin GROMACS


def test_preparar_sistema_sin_gromacs_explica_como_instalarlo(cfg, tmp_path: Path):
    gmx = ClienteGromacs("gmx_que_no_existe_12345")

    with pytest.raises(GromacsNoDisponible, match="apt install gromacs"):
        preparar_sistema(cfg, UBQ, tmp_path / "md", cliente=gmx)


def test_simular_sin_gromacs_explica_como_instalarlo(cfg, sistema_falso):
    gmx = ClienteGromacs("gmx_que_no_existe_12345")

    with pytest.raises(GromacsNoDisponible):
        simular(cfg, sistema_falso, cliente=gmx)
