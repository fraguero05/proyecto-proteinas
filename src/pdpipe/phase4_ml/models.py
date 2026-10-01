"""Modelos de datos de la Fase 4.

Una :class:`Proteina` es una cadena con su secuencia y su asignación de DSSP
alineadas residuo a residuo. Un :class:`Split` es un reparto de proteínas en
train, validación y test. Un :class:`Metricas` es lo que se reporta de una
evaluación.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Proteina(_Base):
    """Una cadena con su estructura secundaria asignada."""

    identificador: str
    secuencia: str
    estructura: str
    numeros: list[int] = Field(default_factory=list)
    archivo: Path | None = None

    @model_validator(mode="after")
    def _mismo_largo(self) -> "Proteina":
        """Secuencia y estructura tienen que estar alineadas residuo a residuo.

        Si se desalinean, el modelo aprende a predecir la etiqueta del residuo
        equivocado y nada falla: el entrenamiento corre igual y el Q3 sale
        apenas peor, que es indistinguible de un modelo mediocre.
        """
        if len(self.secuencia) != len(self.estructura):
            raise ValueError(
                f"{self.identificador}: la secuencia tiene {len(self.secuencia)} "
                f"residuos y la estructura {len(self.estructura)}"
            )
        if self.numeros and len(self.numeros) != len(self.secuencia):
            raise ValueError(
                f"{self.identificador}: {len(self.numeros)} números de residuo "
                f"para {len(self.secuencia)} residuos"
            )
        return self

    def __len__(self) -> int:
        return len(self.secuencia)


class Split(_Base):
    """Reparto de proteínas en entrenamiento, validación y prueba."""

    train: list[str] = Field(default_factory=list)
    val: list[str] = Field(default_factory=list)
    test: list[str] = Field(default_factory=list)

    #: Umbral de identidad de secuencia con el que se agruparon las proteínas
    #: antes de repartir. Queda registrado porque es lo que hace creíble el
    #: resultado del test.
    identidad_max: float = 0.3
    #: Cuántos grupos de secuencias similares se formaron.
    n_grupos: int = 0

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def conjunto_de(self, identificador: str) -> str | None:
        for nombre in ("train", "val", "test"):
            if identificador in getattr(self, nombre):
                return nombre
        return None


class Metricas(_Base):
    """Resultado de evaluar un modelo sobre un conjunto."""

    modelo: str
    conjunto: str = "test"
    n_residuos: int = 0

    #: Fracción de residuos bien clasificados. Es el Q3 de la literatura.
    q3: float = 0.0
    #: Q3 de predecir siempre la clase mayoritaria. Es el piso contra el que
    #: hay que comparar: un modelo que no lo supera no aprendió nada.
    q3_base: float = 0.0

    precision: dict[str, float] = Field(default_factory=dict)
    recall: dict[str, float] = Field(default_factory=dict)
    f1: dict[str, float] = Field(default_factory=dict)

    #: Matriz de confusión en orden CLASES_Q3, filas = real, columnas = predicho.
    confusion: list[list[int]] = Field(default_factory=list)

    @property
    def mejora_sobre_base(self) -> float:
        """Cuántos puntos de Q3 por encima de predecir la clase mayoritaria."""
        return round(self.q3 - self.q3_base, 4)


__all__ = ["Metricas", "Proteina", "Split"]
