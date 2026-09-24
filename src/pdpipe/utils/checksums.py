"""Hashes SHA-256 de archivos de entrada (trazabilidad de las cinco fases).

El ``run_manifest.json`` registra el hash de cada archivo que entró a una
corrida. Eso permite demostrar, meses después y para el documento de tesis,
que dos corridas partieron exactamente del mismo PDB y no de una versión
re-descargada del RCSB que cambió en el medio.
"""

from __future__ import annotations

from pathlib import Path

import hashlib

# Leer en bloques en vez de cargar el archivo entero: una trayectoria de MD
# puede pesar varios GB.
_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def file_sha256(path: str | Path) -> str:
    """SHA-256 hexadecimal del contenido de un archivo.

    Raises:
        FileNotFoundError: si la ruta no existe o no es un archivo.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No se puede hashear, no es un archivo: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory(
    directory: str | Path,
    pattern: str = "*",
    recursive: bool = True,
) -> dict[str, str]:
    """Hashea los archivos de un directorio.

    Args:
        directory: carpeta a recorrer.
        pattern: glob de filtrado (por ejemplo ``"*.pdb"``).
        recursive: si recorre subdirectorios.

    Returns:
        Mapa ``ruta relativa POSIX -> sha256``, ordenado por ruta para que el
        resultado sea estable entre sistemas de archivos.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"No es un directorio: {directory}")

    globber = directory.rglob if recursive else directory.glob
    files = sorted(p for p in globber(pattern) if p.is_file())
    return {p.relative_to(directory).as_posix(): file_sha256(p) for p in files}


def short_hash(value: str, length: int = 8) -> str:
    """Prefijo corto de un hash, para nombres de archivo y logs."""
    return value[:length]
