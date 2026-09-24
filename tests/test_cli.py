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
    ["fetch", "curate", "predict", "design", "simulate", "analyze", "report", "info"],
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
        (["fetch", "--pdb-id", "1UBQ"], 1),
        (["curate"], 1),
        (["predict", "--uniprot", "P0CG48"], 2),
        (["analyze", "--run-id", "20260101T000000Z-abcdef"], 5),
        (["report", "--run-id", "20260101T000000Z-abcdef"], 5),
    ],
)
def test_comandos_pendientes_informan_su_hito(cfg, argumentos, hito):
    resultado = runner.invoke(app, [*cfg, *argumentos])
    assert resultado.exit_code == EXIT_PENDING, resultado.output
    assert f"Hito {hito}" in resultado.output


def test_design_pendiente(cfg, pdb_fixture: Path):
    resultado = runner.invoke(app, [*cfg, "design", "--input", str(pdb_fixture)])
    assert resultado.exit_code == EXIT_PENDING
    assert "Hito 2" in resultado.output


def test_simulate_pendiente(cfg, pdb_fixture: Path):
    resultado = runner.invoke(app, [*cfg, "simulate", "--input", str(pdb_fixture)])
    assert resultado.exit_code == EXIT_PENDING
    assert "Hito 3" in resultado.output


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


# ------------------------------------------------------------------ overrides


def test_design_usa_el_n_del_config(cfg, pdb_fixture: Path):
    """Sin --n-sequences toma el valor del config (4 en la fixture)."""
    resultado = runner.invoke(app, [*cfg, "design", "--input", str(pdb_fixture)])
    assert "4 variantes" in resultado.output


def test_design_respeta_el_override_de_cli(cfg, pdb_fixture: Path):
    resultado = runner.invoke(
        app, [*cfg, "design", "--input", str(pdb_fixture), "--n-sequences", "16"]
    )
    assert "16 variantes" in resultado.output


def test_predict_respeta_el_override_de_fuente(cfg):
    resultado = runner.invoke(
        app, [*cfg, "predict", "--uniprot", "P0CG48", "--source", "esmfold"]
    )
    assert "esmfold" in resultado.output
