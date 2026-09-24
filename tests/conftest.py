"""Fixtures compartidas.

Regla del proyecto: ningún test toca la red. Todo lo que necesite datos usa
archivos chicos incluidos en ``tests/fixtures/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Config mínimo pero completo. Los tests que necesiten variar un valor parten
# de acá y sobrescriben solo lo suyo.
CONFIG_VALIDO: dict = {
    "seed": 123,
    "paths": {
        "data_raw": "data/raw",
        "data_interim": "data/interim",
        "data_processed": "data/processed",
        "runs": "runs",
        "database": "data/pdpipe.sqlite",
    },
    "logging": {"level": "DEBUG", "to_file": False},
    "data": {
        "resolution_max": 2.0,
        "experimental_methods": ["X-RAY DIFFRACTION"],
        "length_min": 50,
        "length_max": 300,
        "organisms": ["Homo sapiens"],
        "biosafety_check": True,
        "http": {"timeout_s": 10, "max_retries": 2, "cache": True},
    },
    "design": {
        "structure_source": "alphafold_db",
        "plddt_min": 70.0,
        "designer": "proteinmpnn",
        "n_sequences": 4,
        "temperature": 0.1,
        "fixed_positions": [],
    },
    "md": {
        "gromacs_bin": "gmx",
        "force_field": "amber99sb-ildn",
        "water_model": "tip3p",
        "box_shape": "cubic",
        "box_padding_nm": 1.0,
        "ion_concentration_m": 0.15,
        "minimization_steps": 1000,
        "nvt_ps": 10,
        "npt_ps": 10,
        "production_ns": 0.1,
        "temperature_k": 300.0,
        "pressure_bar": 1.0,
        "timestep_fs": 2.0,
        "threads": 0,
    },
    "ml": {
        "window_size": 13,
        "test_size": 0.2,
        "val_size": 0.1,
        "identity_threshold": 0.3,
        "model": "bilstm",
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.2,
        "batch_size": 8,
        "epochs": 2,
        "learning_rate": 0.001,
        "device": "cpu",
    },
    "analysis": {
        "rmsd_selection": "name CA",
        "compute_tm_score": True,
        "report_formats": ["html"],
        "figure_dpi": 100,
    },
}


@pytest.fixture
def config_dict() -> dict:
    """Copia mutable del config válido de referencia."""
    import copy

    return copy.deepcopy(CONFIG_VALIDO)


@pytest.fixture
def config_file(tmp_path: Path, config_dict: dict) -> Path:
    """Escribe el config válido en un archivo temporal y devuelve su ruta."""
    destino = tmp_path / "config.yaml"
    destino.write_text(yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")
    return destino


@pytest.fixture
def write_config(tmp_path: Path):
    """Fábrica: escribe un dict arbitrario como config.yaml temporal."""

    def _write(data: dict, nombre: str = "config.yaml") -> Path:
        destino = tmp_path / nombre
        destino.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return destino

    return _write


@pytest.fixture
def pdb_fixture() -> Path:
    """Estructura PDB mínima incluida en el repo (no requiere red)."""
    return FIXTURES_DIR / "mini.pdb"
