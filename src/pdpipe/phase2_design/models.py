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


# ---------------------------------------------------------------------------
# Parte B — diseño de secuencias
# ---------------------------------------------------------------------------


class Mutacion(_Base):
    """Un cambio de aminoácido respecto de la secuencia original.

    ``posicion`` va en **numeración del PDB**, no en índice de la secuencia,
    para que coincida con el CSV de pLDDT de la parte A y con lo que se ve en
    un visualizador. Las dos numeraciones difieren en cuanto la estructura no
    empieza en 1 o le falta un tramo.
    """

    posicion: int
    original: str
    nueva: str

    def __str__(self) -> str:
        """Notación estándar de mutación puntual, p. ej. ``K48R``."""
        return f"{self.original}{self.posicion}{self.nueva}"


class VarianteSecuencia(_Base):
    """Una secuencia generada por el diseñador sobre el esqueleto de entrada."""

    id: str
    secuencia: str
    mutaciones: list[Mutacion] = Field(default_factory=list)

    # Métricas que reporta ProteinMPNN. Son log-verosimilitudes negativas:
    # **más bajo es mejor**, al revés de lo que sugiere la palabra "score".
    score: float | None = None
    global_score: float | None = None
    # Fracción de la secuencia original que el modelo reprodujo por su cuenta.
    recuperacion: float | None = None
    temperatura: float | None = None

    @property
    def n_mutaciones(self) -> int:
        return len(self.mutaciones)

    @property
    def identidad(self) -> float:
        """Identidad con la secuencia original, entre 0 y 1."""
        if not self.secuencia:
            return 0.0
        return round(1 - self.n_mutaciones / len(self.secuencia), 4)

    def notacion_mutaciones(self) -> str:
        """Las mutaciones en una sola cadena, p. ej. ``K48R,T22S``."""
        return ",".join(str(m) for m in self.mutaciones)


class ResultadoDiseno(_Base):
    """Salida de una corrida de diseño sobre una estructura."""

    estructura: Path
    secuencia_original: str
    designer: str
    variantes: list[VarianteSecuencia] = Field(default_factory=list)

    modelo: str | None = None
    seed: int | None = None
    posiciones_fijas: list[int] = Field(default_factory=list)

    fasta: Path | None = None
    tabla: Path | None = None

    @property
    def n_variantes(self) -> int:
        return len(self.variantes)

    def mejor(self) -> VarianteSecuencia | None:
        """La variante de menor ``global_score`` (más baja = más probable).

        Devuelve ``None`` si no hay variantes o si ninguna trae score, en vez
        de inventar un orden arbitrario.
        """
        con_score = [v for v in self.variantes if v.global_score is not None]
        if not con_score:
            return None
        return min(con_score, key=lambda v: v.global_score)  # type: ignore[arg-type,return-value]
