"""``run_manifest.json`` — el registro reproducible de cada corrida.

Cada ejecución del pipeline escribe un manifiesto con: identificador y
timestamp, versiones del software (intérprete, paquetes científicos, GROMACS),
semillas efectivamente aplicadas, parámetros usados y hashes SHA-256 de los
archivos de entrada y salida.

Es el artefacto que hace verificable lo que se reporta en el capítulo de
resultados de la tesis: quien quiera repetir una corrida tiene acá todo lo que
necesita para saber si partió del mismo estado.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import importlib.metadata
import json
import os
import platform
import random
import secrets
import shutil
import subprocess

from pydantic import BaseModel, ConfigDict, Field

from pdpipe.utils.checksums import file_sha256

MANIFEST_NAME = "run_manifest.json"

# Paquetes cuya versión se registra si están instalados. La lista cubre las
# dependencias que pueden cambiar un resultado numérico entre corridas.
_TRACKED_PACKAGES = (
    "pdpipe",
    "numpy",
    "scipy",
    "pandas",
    "biopython",
    "mdanalysis",
    "torch",
    "scikit-learn",
    "matplotlib",
    "pydantic",
    "typer",
)


def utc_now() -> datetime:
    """Momento actual en UTC, con tzinfo. Nunca hora local: las corridas se
    comparan entre máquinas."""
    return datetime.now(timezone.utc)


def new_run_id() -> str:
    """Identificador de corrida ordenable cronológicamente.

    Formato ``AAAAMMDDTHHMMSSZ-xxxxxx``. El sufijo aleatorio evita colisiones
    entre corridas lanzadas en el mismo segundo.
    """
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def set_global_seed(seed: int) -> dict[str, Any]:
    """Fija la semilla en todas las fuentes de aleatoriedad disponibles.

    Toca ``random`` siempre, y ``numpy``/``torch`` solo si están instalados
    (el Hito 0 corre sin dependencias científicas).

    Returns:
        Qué se fijó realmente, para dejarlo en el manifiesto. Los generadores
        no disponibles quedan registrados como no aplicados, que es
        información tan relevante como la semilla misma.
    """
    applied: dict[str, Any] = {"value": seed, "python_random": True}
    random.seed(seed)

    # PYTHONHASHSEED solo tiene efecto si se define antes de arrancar el
    # intérprete; se registra su valor real para no dar una falsa garantía.
    applied["pythonhashseed"] = os.environ.get("PYTHONHASHSEED")

    try:
        import numpy as np

        np.random.seed(seed)
        applied["numpy"] = True
    except ImportError:
        applied["numpy"] = False

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            applied["torch_cuda"] = True
        applied["torch"] = True
    except ImportError:
        applied["torch"] = False

    return applied


def collect_software_versions(gromacs_bin: str | None = None) -> dict[str, Any]:
    """Versiones del entorno: intérprete, sistema operativo y paquetes."""
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    packages: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    versions["packages"] = packages

    if gromacs_bin:
        versions["gromacs"] = _gromacs_version(gromacs_bin)

    return versions


def _gromacs_version(gromacs_bin: str) -> str | None:
    """Versión de GROMACS, o ``None`` si no está instalado o no responde."""
    executable = shutil.which(gromacs_bin)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    for line in result.stdout.splitlines():
        if "GROMACS version" in line:
            return line.split(":", 1)[-1].strip()
    return None


class RunManifest(BaseModel):
    """Manifiesto de una corrida. Se construye durante la ejecución y se
    serializa al final con :meth:`save`."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=new_run_id)
    command: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    status: str = "running"  # running | ok | error | interrumpida
    error: str | None = None

    seeds: dict[str, Any] = Field(default_factory=dict)
    software: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    # ---------------------------------------------------------------- build

    @classmethod
    def start(
        cls,
        command: str,
        seed: int,
        config: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        gromacs_bin: str | None = None,
        run_id: str | None = None,
    ) -> "RunManifest":
        """Crea un manifiesto y aplica la semilla global de una sola vez."""
        return cls(
            run_id=run_id or new_run_id(),
            command=command,
            seeds=set_global_seed(seed),
            software=collect_software_versions(gromacs_bin),
            config=config or {},
            params=params or {},
        )

    def add_input(self, path: str | Path, key: str | None = None) -> str:
        """Registra el hash de un archivo de entrada. Devuelve el hash."""
        digest = file_sha256(path)
        self.inputs[key or Path(path).name] = digest
        return digest

    def add_output(self, path: str | Path, key: str | None = None) -> str:
        """Registra el hash de un archivo producido por la corrida."""
        digest = file_sha256(path)
        self.outputs[key or Path(path).name] = digest
        return digest

    def add_note(self, message: str) -> None:
        """Anota algo relevante para interpretar los resultados después."""
        self.notes.append(message)

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        """Cierra la corrida y sella el timestamp de fin."""
        self.status = status
        self.error = error
        self.finished_at = utc_now()

    @property
    def duration_s(self) -> float | None:
        """Duración en segundos, o ``None`` si todavía no terminó."""
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    # ----------------------------------------------------------------- I/O

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["duration_s"] = self.duration_s
        return data

    def save(self, directory: str | Path) -> Path:
        """Escribe ``run_manifest.json`` en ``directory``, creándolo si falta."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / MANIFEST_NAME
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=False),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        """Relee un manifiesto guardado (para ``analyze`` y ``report``)."""
        path = Path(path)
        if path.is_dir():
            path = path / MANIFEST_NAME
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("duration_s", None)  # derivado, no es un campo del modelo
        return cls(**data)
