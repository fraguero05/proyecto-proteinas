"""Modelos de datos del análisis de trayectorias (Fase 3).

Dos formas de mirar una simulación: cómo evoluciona una magnitud en el tiempo
(:class:`SerieTemporal`) y cómo se reparte a lo largo de la cadena
(:class:`PerfilPorResiduo`). El RMSD, el radio de giro, la SASA y los puentes
de hidrógeno son lo primero; el RMSF es lo segundo.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SerieTemporal(_Base):
    """Una magnitud medida cuadro a cuadro de la trayectoria."""

    nombre: str
    unidad: str
    tiempos_ps: list[float] = Field(default_factory=list)
    valores: list[float] = Field(default_factory=list)

    @property
    def media(self) -> float:
        return round(mean(self.valores), 4) if self.valores else 0.0

    @property
    def desvio(self) -> float:
        """Desviación estándar poblacional (no muestral).

        La serie no es una muestra de algo mayor: son todos los cuadros que
        se simularon, así que se divide por N y no por N-1.
        """
        return round(pstdev(self.valores), 4) if len(self.valores) > 1 else 0.0

    @property
    def minimo(self) -> float:
        return round(min(self.valores), 4) if self.valores else 0.0

    @property
    def maximo(self) -> float:
        return round(max(self.valores), 4) if self.valores else 0.0

    @property
    def inicial(self) -> float:
        return round(self.valores[0], 4) if self.valores else 0.0

    @property
    def final(self) -> float:
        return round(self.valores[-1], 4) if self.valores else 0.0

    @property
    def deriva(self) -> float:
        """Cuánto cambió entre el primer y el último cuadro.

        En el RMSD es la señal más directa de si el sistema se estabilizó o
        todavía se está alejando de la estructura de partida.
        """
        return round(self.final - self.inicial, 4)


class PerfilPorResiduo(_Base):
    """Una magnitud medida por residuo, a lo largo de la cadena."""

    nombre: str
    unidad: str
    residuos: list[int] = Field(default_factory=list)
    valores: list[float] = Field(default_factory=list)

    @property
    def media(self) -> float:
        return round(mean(self.valores), 4) if self.valores else 0.0

    def mas_moviles(self, n: int = 10) -> list[tuple[int, float]]:
        """Los ``n`` residuos de mayor valor, de mayor a menor.

        En el RMSF son las regiones flexibles: extremos, loops y todo lo que
        no esté sujeto por el núcleo.
        """
        pares = sorted(
            zip(self.residuos, self.valores), key=lambda p: p[1], reverse=True
        )
        return [(r, round(v, 4)) for r, v in pares[:n]]


class ResultadoAnalisisMD(_Base):
    """Todo lo que se midió sobre una trayectoria."""

    topologia: Path
    trayectoria: Path | None = None

    n_frames: int = 0
    n_atomos: int = 0
    n_residuos: int = 0
    dt_ps: float | None = None

    rmsd: SerieTemporal | None = None
    radio_giro: SerieTemporal | None = None
    sasa: SerieTemporal | None = None
    puentes: SerieTemporal | None = None
    rmsf: PerfilPorResiduo | None = None

    figuras: list[Path] = Field(default_factory=list)
    tabla: Path | None = None
    resumen_json: Path | None = None

    # Lo que el análisis no pudo hacer y por qué. Va al JSON y al manifiesto:
    # un resultado incompleto tiene que decir que lo está.
    advertencias: list[str] = Field(default_factory=list)

    @property
    def duracion_ps(self) -> float | None:
        """Tiempo simulado que cubre la trayectoria analizada."""
        if self.rmsd and self.rmsd.tiempos_ps:
            return round(self.rmsd.tiempos_ps[-1], 3)
        return None

    def series(self) -> dict[str, SerieTemporal]:
        """Las series temporales que sí se calcularon, por nombre corto."""
        candidatas = {
            "rmsd": self.rmsd,
            "radio_giro": self.radio_giro,
            "sasa": self.sasa,
            "puentes": self.puentes,
        }
        return {k: v for k, v in candidatas.items() if v is not None}


__all__ = ["PerfilPorResiduo", "ResultadoAnalisisMD", "SerieTemporal"]
