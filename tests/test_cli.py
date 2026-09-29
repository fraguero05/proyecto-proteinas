"""Tests de la CLI.

En el Hito 0 los comandos de fase validan argumentos y config, pero terminan
con EXIT_PENDING. Estos tests fijan ese contrato para que, cuando cada hito
los implemente, quede claro qué cambió.
"""

from __future__ import annotations

from pathlib import Path

import json

import pytest
from typer.testing import CliRunner

from pdpipe import __version__
from pdpipe.cli import EXIT_ERROR, EXIT_PENDING, app

FIXTURES_DIR = Path(__file__).parent / "fixtures"

runner = CliRunner()


@pytest.fixture
def cfg(config_file: Path) -> list[str]:
    """Prefijo de argumentos con el config temporal."""
    return ["--config", str(config_file)]


# ------------------------------------------------------------------ básicos


def test_help():
    resultado = runner.invoke(app, ["--help"])
    assert resultado.exit_code == 0
    assert "pdpipe" in resultado.output


def test_version():
    resultado = runner.invoke(app, ["--version"])
    assert resultado.exit_code == 0
    assert __version__ in resultado.output


def test_sin_argumentos_muestra_ayuda():
    resultado = runner.invoke(app, [])
    assert "Usage" in resultado.output or "Uso" in resultado.output


@pytest.mark.parametrize(
    "comando",
    [
        "fetch",
        "curate",
        "predict",
        "design",
        "simulate",
        "md-analyze",
        "analyze",
        "report",
        "info",
        "db-stats",
    ],
)
def test_cada_comando_tiene_help(comando: str):
    resultado = runner.invoke(app, [comando, "--help"])
    assert resultado.exit_code == 0, resultado.output


def test_config_inexistente_es_error(tmp_path: Path):
    resultado = runner.invoke(app, ["--config", str(tmp_path / "nope.yaml"), "info"])
    assert resultado.exit_code == EXIT_ERROR
    assert "configuración" in resultado.output.lower()


def test_config_invalido_es_error(write_config, config_dict):
    config_dict["ml"]["window_size"] = 10  # par: inválido
    ruta = write_config(config_dict)
    resultado = runner.invoke(app, ["--config", str(ruta), "info"])
    assert resultado.exit_code == EXIT_ERROR


# ------------------------------------------------------------------ info


def test_info_muestra_el_entorno(cfg):
    resultado = runner.invoke(app, [*cfg, "info"])
    assert resultado.exit_code == 0
    assert "Entorno" in resultado.output
    assert "GROMACS" in resultado.output


def test_info_save_escribe_el_manifiesto(config_file: Path):
    resultado = runner.invoke(app, ["--config", str(config_file), "info", "--save"])
    assert resultado.exit_code == 0

    runs = config_file.parent / "runs"
    manifiestos = list(runs.glob("*/run_manifest.json"))
    assert len(manifiestos) == 1

    data = json.loads(manifiestos[0].read_text(encoding="utf-8"))
    assert data["command"] == "info"
    assert data["status"] == "ok"
    assert data["seeds"]["value"] == 123  # la semilla del config de prueba
    assert data["config"]["md"]["force_field"] == "amber99sb-ildn"


# ------------------------------------------------- comandos pendientes (fases)


@pytest.mark.parametrize(
    ("argumentos", "hito"),
    [
        (["analyze", "--run-id", "20260101T000000Z-abcdef"], 5),
        (["report", "--run-id", "20260101T000000Z-abcdef"], 5),
    ],
)
def test_comandos_pendientes_informan_su_hito(cfg, argumentos, hito):
    resultado = runner.invoke(app, [*cfg, *argumentos])
    assert resultado.exit_code == EXIT_PENDING, resultado.output
    assert f"Hito {hito}" in resultado.output


def test_design_sin_proteinmpnn_explica_como_instalarlo(cfg, pdb_fixture: Path):
    """Implementado en el Hito 2B, pero ProteinMPNN se clona aparte.

    Sigue saliendo con EXIT_PENDING porque falta una herramienta del entorno,
    no porque falten datos: el mensaje tiene que traer el git clone.
    """
    resultado = runner.invoke(app, [*cfg, "design", "--input", str(pdb_fixture)])
    assert resultado.exit_code == EXIT_PENDING
    assert "git clone" in resultado.output


def test_simulate_sin_gromacs_explica_como_instalarlo(cfg, pdb_fixture: Path):
    """Implementado en el Hito 3, pero GROMACS es un binario externo.

    Sigue saliendo con EXIT_PENDING porque falta una herramienta del entorno,
    no porque falten datos: el mensaje tiene que traer la instalación.
    """
    resultado = runner.invoke(app, [*cfg, "simulate", "--input", str(pdb_fixture)])
    assert resultado.exit_code == EXIT_PENDING
    assert "apt install gromacs" in resultado.output


def test_simulate_clean_only_no_necesita_gromacs(cfg):
    """La limpieza de la estructura corre con Biopython, sin GROMACS."""
    resultado = runner.invoke(
        app,
        [*cfg, "simulate", "--input", str(FIXTURES_DIR / "1UBQ.pdb"), "--clean-only"],
    )

    assert resultado.exit_code == 0, resultado.output
    assert "58" in resultado.output, "no reportó las aguas quitadas"


# ------------------------------------------------------- validación de argumentos


def test_fetch_sin_identificadores_es_error(cfg):
    resultado = runner.invoke(app, [*cfg, "fetch"])
    assert resultado.exit_code == EXIT_ERROR
    assert "--pdb-id" in resultado.output


def test_predict_sin_uniprot_es_error(cfg):
    resultado = runner.invoke(app, [*cfg, "predict"])
    assert resultado.exit_code == EXIT_ERROR


def test_design_sin_input_es_error(cfg):
    resultado = runner.invoke(app, [*cfg, "design"])
    assert resultado.exit_code == EXIT_ERROR


def test_design_con_input_inexistente_es_error(cfg, tmp_path: Path):
    resultado = runner.invoke(
        app, [*cfg, "design", "--input", str(tmp_path / "fantasma.pdb")]
    )
    assert resultado.exit_code == EXIT_ERROR
    assert "no existe" in resultado.output.lower()


def test_simulate_con_input_inexistente_es_error(cfg, tmp_path: Path):
    resultado = runner.invoke(
        app, [*cfg, "simulate", "--input", str(tmp_path / "fantasma.pdb")]
    )
    assert resultado.exit_code == EXIT_ERROR


def test_analyze_sin_run_id_es_error(cfg):
    resultado = runner.invoke(app, [*cfg, "analyze"])
    assert resultado.exit_code == EXIT_ERROR


# ------------------------------------------------------------ Fase 1 (Hito 1)


def test_db_stats_sin_base_avisa(cfg):
    resultado = runner.invoke(app, [*cfg, "db-stats"])
    assert resultado.exit_code == EXIT_ERROR
    assert "fetch" in resultado.output


def test_db_stats_muestra_el_contenido(config_file: Path):
    """Con una base poblada a mano, sin red."""
    from pdpipe.config import load_config
    from pdpipe.phase1_data import BaseDatos
    from pdpipe.phase1_data.models import Proteina

    cfg_obj = load_config(config_file)
    with BaseDatos(cfg_obj.resolved_paths()["database"]) as db:
        db.guardar_proteina(
            Proteina(
                pdb_id="1UBQ",
                uniprot_id="P0CG48",
                organismo="Homo sapiens",
                metodo="X-RAY DIFFRACTION",
                resolucion=1.8,
                longitud=76,
                curada=True,
            )
        )

    resultado = runner.invoke(app, ["--config", str(config_file), "db-stats"])
    assert resultado.exit_code == 0
    assert "1UBQ" in resultado.output
    assert "P0CG48" in resultado.output
    assert "curada" in resultado.output


# ------------------------------------------------------------------ overrides


def _params_del_manifiesto(tmp_path: Path) -> dict:
    """Lee los params del único run_manifest.json que dejó la corrida."""
    manifiestos = list((tmp_path / "runs").glob("*/run_manifest.json"))
    assert len(manifiestos) == 1, manifiestos
    return json.loads(manifiestos[0].read_text(encoding="utf-8"))["params"]


def test_design_usa_el_n_del_config(cfg, pdb_fixture: Path, tmp_path: Path):
    """Sin --n-sequences toma el valor del config (4 en la fixture)."""
    runner.invoke(app, [*cfg, "design", "--input", str(pdb_fixture)])

    assert _params_del_manifiesto(tmp_path)["n_sequences"] == 4


def test_design_respeta_el_override_de_cli(cfg, pdb_fixture: Path, tmp_path: Path):
    runner.invoke(
        app, [*cfg, "design", "--input", str(pdb_fixture), "--n-sequences", "16"]
    )

    assert _params_del_manifiesto(tmp_path)["n_sequences"] == 16


def test_predict_respeta_el_override_de_fuente(cfg):
    """ESMFold todavía no está: el override se respeta y lo informa."""
    resultado = runner.invoke(
        app, [*cfg, "predict", "--uniprot", "P0CG48", "--source", "esmfold"]
    )
    assert resultado.exit_code == EXIT_PENDING
    assert "ESMFold" in resultado.output


def test_predict_con_colabfold_manda_al_notebook(cfg):
    resultado = runner.invoke(
        app, [*cfg, "predict", "--uniprot", "P0CG48", "--source", "colabfold"]
    )
    assert resultado.exit_code == EXIT_PENDING
    assert "notebook" in resultado.output


# -------------------------------------------------------------- md-analyze


def test_md_analyze_sin_topology_es_error(cfg):
    resultado = runner.invoke(app, [*cfg, "md-analyze"])
    assert resultado.exit_code == EXIT_ERROR


def test_md_analyze_con_topology_inexistente_es_error(cfg, tmp_path: Path):
    resultado = runner.invoke(
        app, [*cfg, "md-analyze", "--topology", str(tmp_path / "fantasma.pdb")]
    )
    assert resultado.exit_code == EXIT_ERROR
    assert "no existe" in resultado.output.lower()


def test_md_analyze_con_trayectoria_inexistente_es_error(cfg, tmp_path: Path):
    traj = FIXTURES_DIR / "traj_1UBQ.pdb"
    resultado = runner.invoke(
        app,
        [*cfg, "md-analyze", "--topology", str(traj),
         "--trajectory", str(tmp_path / "fantasma.xtc")],
    )
    assert resultado.exit_code == EXIT_ERROR


def test_md_analyze_analiza_y_deja_manifiesto(cfg, tmp_path: Path):
    traj = FIXTURES_DIR / "traj_1UBQ.pdb"
    resultado = runner.invoke(
        app, [*cfg, "md-analyze", "--topology", str(traj), "--no-figures"]
    )

    assert resultado.exit_code == 0, resultado.output
    assert "RMSD" in resultado.output
    manifiestos = list((tmp_path / "runs").glob("*/run_manifest.json"))
    assert len(manifiestos) == 1


def test_md_analyze_avisa_de_los_puentes_sin_hidrogenos(cfg):
    """La advertencia tiene que ser visible, no quedar solo en el JSON."""
    traj = FIXTURES_DIR / "traj_1UBQ.pdb"
    resultado = runner.invoke(
        app, [*cfg, "md-analyze", "--topology", str(traj), "--no-figures"]
    )

    assert "Advertencia" in resultado.output
