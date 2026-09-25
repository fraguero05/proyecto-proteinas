"""Interfaz de diseño de secuencias (Fase 2, parte B).

La tesis usa ProteinMPNN, pero el diseño de secuencia sobre un esqueleto fijo
es un problema con varias herramientas posibles (Rosetta con ``fixbb``,
ESM-IF, LigandMPNN). Para no atar el resto del pipeline a una de ellas, todas
entran por esta interfaz: la CLI y el orquestador hablan con
:class:`SequenceDesigner` y no saben cuál está atrás.

Enchufar una implementación nueva es escribir una subclase y registrarla en
:data:`DISENADORES`; no hay que tocar ni la CLI ni el pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from pdpipe.phase2_design.models import ResultadoDiseno


class ErrorDeDiseno(Exception):
    """Falló la generación de variantes."""


class DisenadorNoDisponible(ErrorDeDiseno):
    """La herramienta de diseño no está instalada o no se puede ejecutar.

    Se distingue de :class:`ErrorDeDiseno` porque no es un problema de los
    datos: el mensaje tiene que decir cómo instalarla, no qué salió mal.
    """


class SequenceDesigner(ABC):
    """Genera variantes de secuencia para un esqueleto dado.

    Implementa el paso de diseño de la Fase 2 de la tesis: se mantiene la
    estructura tridimensional y se busca qué secuencias podrían plegarse en
    ella, que es el problema inverso al de la predicción de estructura.
    """

    #: Nombre con el que se elige desde ``design.designer`` en el config.
    nombre: ClassVar[str] = "abstracto"

    @abstractmethod
    def disponible(self) -> bool:
        """Indica si la herramienta puede ejecutarse en este entorno.

        Se consulta antes de correr para poder fallar con un mensaje útil en
        vez de con una excepción de subprocess a mitad de camino.
        """

    @abstractmethod
    def motivo_no_disponible(self) -> str:
        """Explica qué falta y cómo instalarlo, para mostrárselo al usuario."""

    @abstractmethod
    def disenar(
        self,
        estructura: Path,
        n_secuencias: int,
        temperatura: float,
        posiciones_fijas: list[int] | None = None,
        seed: int | None = None,
    ) -> ResultadoDiseno:
        """Genera ``n_secuencias`` variantes sobre el esqueleto de ``estructura``.

        Args:
            estructura: PDB de partida (experimental o predicho).
            n_secuencias: cuántas variantes generar.
            temperatura: muestreo; más alta, más diversidad y menos fidelidad.
            posiciones_fijas: residuos a no mutar, en **numeración del PDB**.
            seed: semilla del muestreo, para que la corrida sea reproducible.

        Raises:
            DisenadorNoDisponible: la herramienta no está instalada.
            ErrorDeDiseno: la corrida falló o su salida no se pudo interpretar.
        """

    def verificar_disponible(self) -> None:
        """Lanza :class:`DisenadorNoDisponible` si la herramienta no está."""
        if not self.disponible():
            raise DisenadorNoDisponible(self.motivo_no_disponible())


def obtener_disenador(nombre: str, **kwargs: object) -> SequenceDesigner:
    """Construye el diseñador registrado bajo ``nombre``.

    La importación es perezosa para que pedir ``proteinmpnn`` no arrastre a
    PyTorch cuando lo único que se quiere es otro backend.
    """
    clave = nombre.strip().lower()

    if clave == "proteinmpnn":
        from pdpipe.phase2_design.proteinmpnn import DisenadorProteinMPNN

        return DisenadorProteinMPNN(**kwargs)  # type: ignore[arg-type]

    if clave == "rosetta":
        raise DisenadorNoDisponible(
            "Rosetta no está implementado: requiere licencia y compilación "
            "larga, y quedó fuera de la primera versión a propósito. La "
            "interfaz SequenceDesigner existe justamente para enchufarlo "
            "después sin tocar el resto del pipeline."
        )

    raise DisenadorNoDisponible(
        f"Diseñador desconocido: '{nombre}'. Opciones: {sorted(DISENADORES)}"
    )


#: Backends conocidos. ``rosetta`` figura como pendiente a propósito.
DISENADORES = {"proteinmpnn", "rosetta"}


__all__ = [
    "DISENADORES",
    "DisenadorNoDisponible",
    "ErrorDeDiseno",
    "SequenceDesigner",
    "obtener_disenador",
]
