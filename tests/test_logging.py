"""Tests del logging del pipeline."""

from __future__ import annotations

from pathlib import Path

import logging

from pdpipe.utils.logging import get_logger, setup_logging


def test_setup_devuelve_el_logger_raiz():
    logger = setup_logging(level="INFO")
    assert logger.name == "pdpipe"
    assert logger.handlers


def test_get_logger_cuelga_del_raiz():
    assert get_logger("pdpipe.phase1_data").name == "pdpipe.phase1_data"
    assert get_logger("phase1_data").name == "pdpipe.phase1_data"
    assert get_logger(None).name == "pdpipe"
    assert get_logger("pdpipe").name == "pdpipe"


def test_escribe_al_archivo(tmp_path: Path):
    log_file = tmp_path / "logs" / "run.log"
    setup_logging(level="INFO", log_file=log_file)
    get_logger("test").info("estructura 1UBQ descargada")

    logging.shutdown()
    assert log_file.is_file()
    assert "1UBQ descargada" in log_file.read_text(encoding="utf-8")


def test_el_archivo_guarda_debug_aunque_la_consola_este_en_info(tmp_path: Path):
    """El detalle tiene que quedar en disco aunque no se muestre en pantalla."""
    log_file = tmp_path / "run.log"
    setup_logging(level="INFO", log_file=log_file)
    get_logger("test").debug("comando gmx completo: gmx mdrun -deffnm md")

    logging.shutdown()
    assert "gmx mdrun" in log_file.read_text(encoding="utf-8")


def test_crea_los_directorios_del_log(tmp_path: Path):
    log_file = tmp_path / "a" / "b" / "run.log"
    setup_logging(log_file=log_file)
    assert log_file.parent.is_dir()


def test_llamadas_repetidas_no_duplican_handlers(tmp_path: Path):
    for _ in range(3):
        logger = setup_logging(level="INFO", log_file=tmp_path / "run.log")
    assert len(logger.handlers) == 2  # consola + archivo


def test_no_propaga_al_root_logger():
    logger = setup_logging()
    assert logger.propagate is False
