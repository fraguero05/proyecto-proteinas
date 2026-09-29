"""Envoltorio de GROMACS por subprocess (Fase 3).

GROMACS es un binario externo, no una biblioteca de Python. Todo el pipeline
lo usa a través de :class:`ClienteGromacs`, que resuelve tres cosas que en
crudo son incómodas:

* **Errores legibles.** GROMACS escribe sus fallas en stderr mezcladas con
  banners, notas y citas bibliográficas. Acá se extrae la parte que importa.
* **Entrada interactiva.** Varias herramientas (``pdb2gmx``, ``genion``)
  preguntan por consola qué grupo usar. Se contesta por stdin en vez de
  quedar esperando para siempre.
* **Ausencia del binario.** En una máquina sin GROMACS el comando tiene que
  decir cómo instalarlo, no tirar un ``FileNotFoundError``.

No hay fallback silencioso: si ``gmx`` no está, el pipeline lo dice y corta.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

# Una etapa de producción puede tardar horas; el techo es para que un cuelgue
# no deje el proceso tomado para siempre, no una expectativa de duración.
TIMEOUT_S = 48 * 3600

INSTALACION = """GROMACS no está instalado o no está en el PATH.

En Ubuntu o WSL:
    sudo apt update && sudo apt install gromacs

En macOS:
    brew install gromacs

En Google Colab:
    !apt-get install -qq gromacs

Si lo instalaste en otra ruta, ajustá md.gromacs_bin en el config.yaml."""


class ErrorDeGromacs(Exception):
    """Una herramienta de GROMACS terminó con error."""


class GromacsNoDisponible(ErrorDeGromacs):
    """No se encontró el ejecutable de GROMACS.

    Se distingue del error de ejecución porque no es un problema de los datos
    sino del entorno: el mensaje tiene que explicar cómo instalarlo.
    """


class ClienteGromacs:
    """Corre herramientas de GROMACS y traduce sus fallas."""

    def __init__(self, binario: str = "gmx", timeout_s: int = TIMEOUT_S) -> None:
        self.binario = binario
        self.timeout_s = timeout_s

    # -- disponibilidad ----------------------------------------------------

    def ruta(self) -> str | None:
        """Ruta del ejecutable, o ``None`` si no está en el PATH."""
        return shutil.which(self.binario)

    def disponible(self) -> bool:
        return self.ruta() is not None

    def verificar_disponible(self) -> None:
        if not self.disponible():
            raise GromacsNoDisponible(
                f"No se encontró el ejecutable '{self.binario}'.\n\n{INSTALACION}"
            )

    def version(self) -> str | None:
        """Versión que reporta ``gmx --version``, para el manifiesto.

        Devuelve ``None`` si no se puede determinar: registrar una versión
        equivocada en el manifiesto sería peor que no registrar ninguna.
        """
        if not self.disponible():
            return None
        try:
            proceso = subprocess.run(  # noqa: S603 - binario del config, sin shell
                [self.binario, "--version"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        for linea in (proceso.stdout or "").splitlines():
            if "GROMACS version" in linea:
                return linea.split(":", 1)[-1].strip()
        return None

    # -- ejecución ---------------------------------------------------------

    def correr(
        self,
        herramienta: str,
        *argumentos: str,
        entrada: str | None = None,
        directorio: str | Path | None = None,
        etiqueta: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Ejecuta ``gmx <herramienta> <argumentos>``.

        Args:
            herramienta: ``pdb2gmx``, ``editconf``, ``grompp``, ``mdrun``, ...
            argumentos: se pasan tal cual, ya convertidos a ``str``.
            entrada: texto que se manda por stdin. Necesario para las
                herramientas que preguntan qué grupo usar; sin esto quedan
                esperando una respuesta que nunca llega.
            directorio: directorio de trabajo. GROMACS deja muchos archivos
                intermedios, así que conviene acotarlos a una carpeta.
            etiqueta: nombre del paso para el log.

        Raises:
            GromacsNoDisponible: si falta el ejecutable.
            ErrorDeGromacs: si la herramienta devuelve un código distinto de 0.
        """
        self.verificar_disponible()

        comando = [self.binario, herramienta, *[str(a) for a in argumentos]]
        nombre = etiqueta or herramienta
        logger.info("GROMACS [%s]: %s", nombre, " ".join(comando[1:]))

        try:
            proceso = subprocess.run(  # noqa: S603 - comando armado acá, sin shell
                comando,
                input=entrada,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=str(directorio) if directorio else None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ErrorDeGromacs(
                f"'{nombre}' no terminó en {self.timeout_s / 3600:.0f} h y se canceló."
            ) from exc
        except OSError as exc:
            raise ErrorDeGromacs(f"No se pudo ejecutar '{nombre}': {exc}") from exc

        if proceso.returncode != 0:
            raise ErrorDeGromacs(
                f"'{nombre}' falló (código {proceso.returncode}):\n\n"
                + extraer_error(proceso.stderr, proceso.stdout)
            )

        return proceso


def extraer_error(stderr: str | None, stdout: str | None = None) -> str:
    """Saca el mensaje de error útil de la salida de GROMACS.

    GROMACS enmarca sus errores entre líneas de guiones, con la forma::

        -------------------------------------------------------
        Program:     gmx pdb2gmx, version 2023.1
        Source file: src/gromacs/.../resall.cpp (line 660)

        Fatal error:
        Residue 'LIG' not found in residue topology database
        -------------------------------------------------------

    Interesa el bloque de ``Fatal error`` en adelante. Si no aparece, se
    devuelven las últimas líneas, que es mejor que volcar el banner completo
    con las citas bibliográficas.
    """
    texto = (stderr or "").strip() or (stdout or "").strip()
    if not texto:
        return "(sin salida)"

    marcador = re.search(r"^\s*(Fatal error|Error in user input)\s*:?\s*$", texto, re.M)
    if marcador:
        resto = texto[marcador.start():]
        # Corta en la línea de guiones que cierra el bloque.
        cierre = re.search(r"^-{10,}\s*$", resto, re.M)
        bloque = resto[: cierre.start()] if cierre else resto
        lineas = [ln.rstrip() for ln in bloque.splitlines() if ln.strip()]
        if lineas:
            return "\n".join(lineas)

    lineas = [ln.rstrip() for ln in texto.splitlines() if ln.strip()]
    return "\n".join(lineas[-15:])


__all__ = [
    "INSTALACION",
    "ClienteGromacs",
    "ErrorDeGromacs",
    "GromacsNoDisponible",
    "extraer_error",
]
