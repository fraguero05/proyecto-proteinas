"""Logging del pipeline (transversal a las cinco fases).

Dos destinos: consola (con formato de ``rich``) y, si la corrida tiene una
carpeta asignada, un ``run.log`` dentro de ella. El archivo guarda siempre a
nivel DEBUG aunque la consola esté en INFO: cuando una simulación de GROMACS
falla a la hora y media, el detalle tiene que estar en algún lado.
"""

from __future__ import annotations

from pathlib import Path

import logging

from rich.logging import RichHandler

_ROOT_LOGGER_NAME = "pdpipe"
_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    force: bool = True,
) -> logging.Logger:
    """Configura el logger raíz de ``pdpipe``.

    Args:
        level: nivel para la consola (DEBUG/INFO/WARNING/ERROR).
        log_file: si se pasa, además escribe ahí a nivel DEBUG. Se crean los
            directorios padre que falten.
        force: descarta los handlers previos. Evita duplicar líneas cuando se
            llama más de una vez en el mismo proceso (típico en los tests).

    Returns:
        El logger ``pdpipe`` ya configurado.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    # El logger deja pasar todo; cada handler filtra por su cuenta.
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        omit_repeated_times=False,
    )
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Devuelve un logger hijo de ``pdpipe``.

    Se usa como ``get_logger(__name__)`` desde cada módulo; el prefijo
    ``pdpipe.`` se normaliza para que la jerarquía quede bien armada.
    """
    if not name or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
