"""Modelos de datos de la Fase 2.

Representan una estructura predicha y su confianza por residuo (pLDDT), que es
lo que decide qué partes del modelo sirven para diseñar sobre ellas y cuáles
hay que mirar con desconfianza.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BandaPLDDT(str, Enum):
    """Bandas de confianza definidas por DeepMind para el pLDDT.

    No son un invento del pipeline: son los cortes que usa AlphaFold DB para
    colorear sus modelos y los que reporta su API en los campos
    ``fractionPlddt*``. Se respetan tal cual para que los números del pipeline
    sean comparables con los del sitio.
    """

    MUY_BAJA = "muy_baja"      # < 50   — probablemente desordenado
    BAJA = "baja"              # 50-70  — poco confiable
    CONFIABLE = "confiable"    # 70-90  — buena confianza en el plegamiento
    MUY_ALTA = "muy_alta"      # >= 90  — confianza de calidad experimental

    @property
    def descripcion(self) -> str:
        return {
            BandaPLDDT.MUY_BAJA: "muy baja (<50): probablemente desordenado",
            BandaPLDDT.BAJA: "baja (50-70): poco confiable",
            BandaPLDDT.CONFIABLE: "confiable (70-90)",
            BandaPLDDT.MUY_ALTA: "muy alta (>=90): calidad experimental",
        }[self]


def clasificar_plddt(valor: float) -> BandaPLDDT:
    """Ubica un pLDDT en su banda de confianza."""
    if valor < 50:
        return BandaPLDDT.MUY_BAJA
    if valor < 70:
        return BandaPLDDT.BAJA
    if valor < 90:
        return BandaPLDDT.CONFIABLE
    return BandaPLDDT.MUY_ALTA


class ResiduoPLDDT(_Base):
    """Confianza del modelo en un residuo concreto."""

    numero: int
    aminoacido: str | None = None
    plddt: float

    @property
    def banda(self) -> BandaPLDDT:
        return clasificar_plddt(self.plddt)


class ResumenPLDDT(_Base):
    """Estadísticos del pLDDT sobre todo el modelo."""

    n_residuos: int
    media: float
    mediana: float
    minimo: float
    maximo: float
    conteo_por_banda: dict[str, int] = Field(default_factory=dict)
    fraccion_por_banda: dict[str, float] = Field(default_factory=dict)

    @property
    def fraccion_confiable(self) -> float:
        """Proporción de residuos con pLDDT >= 70.

        Es el número que suele decidir si vale la pena seguir adelante con un
        modelo predicho o buscar una estructura experimental.
        """
        return self.fraccion_por_banda.get(
            BandaPLDDT.CONFIABLE.value, 0.0
        ) + self.fraccion_por_banda.get(BandaPLDDT.MUY_ALTA.value, 0.0)


class ModeloPredicho(_Base):
    """Una estructura predicha, con su procedencia y su confianza."""

    uniprot_id: str
    entry_id: str | None = None
    version: int | None = None
    fuente: str = "alphafold_db"
    archivo: Path | None = None
    sha256: str | None = None

    nombre: str | None = None
    organismo: str | None = None
    tax_id: int | None = None
    secuencia: str | None = None

    residuos: list[ResiduoPLDDT] = Field(default_factory=list)
    resumen: ResumenPLDDT | None = None

    # pLDDT medio según la propia API, para contrastarlo con el que calculamos
    # del B-factor. Si difieren, algo se parseó mal.
    metrica_global_api: float | None = None

    def residuos_bajo(self, umbral: float) -> list[ResiduoPLDDT]:
        """Residuos por debajo de un pLDDT dado, en orden de secuencia."""
        return [r for r in self.residuos if r.plddt < umbral]
