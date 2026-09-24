"""Utilidades transversales a las cinco fases del pipeline.

Nada de ciencia acá: logging, hashes de archivos y el ``run_manifest.json``
que hace reproducible cada corrida.
"""

from pdpipe.utils.checksums import file_sha256, hash_directory
from pdpipe.utils.logging import get_logger, setup_logging
from pdpipe.utils.manifest import RunManifest, new_run_id, set_global_seed

__all__ = [
    "RunManifest",
    "file_sha256",
    "get_logger",
    "hash_directory",
    "new_run_id",
    "set_global_seed",
    "setup_logging",
]
