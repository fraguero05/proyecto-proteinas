"""Modelos de datos de la Fase 1.

Representan lo que se persiste en las cuatro tablas de la tesis
(``proteinas``, ``funciones``, ``interacciones``, ``ptms``), desacoplados de la
forma cruda de las respuestas de RCSB y UniProt: si esas APIs cambian, se
actualizan los parsers y estos modelos quedan igual.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntidadPolimerica(_Base):
    """Una cadena polipeptídica dentro de una entrada del PDB.

    Importa distinguirla de la entrada y de la proteína de UniProt: la
    longitud y la secuencia que valen para filtrar una *estructura* son las de
    la cadena efectivamente cristalizada, no las de la proteína completa. 1UBQ
    contiene 76 residuos de ubiquitina; su UniProt (P0CG48, poliubiquitina-C)
    tiene 685. Filtrar por 685 descartaría la estructura por larga cuando en
    realidad es corta.
    """

    entity_id: str
    tipo: str | None = None
    longitud: int | None = None
    secuencia: str | None = None
    organismo: str | None = None
    tax_id: int | None = None
    uniprot_ids: list[str] = Field(default_factory=list)

    @property
    def es_proteina(self) -> bool:
        return (self.tipo or "").lower() == "protein"


class EntradaPDB(_Base):
    """Metadatos de una entrada del RCSB PDB."""

    pdb_id: str
    titulo: str | None = None
    metodo: str | None = None
    resolucion: float | None = None
    fecha_deposito: str | None = None
    n_entidades_proteina: int | None = None
    entidades: list[EntidadPolimerica] = Field(default_factory=list)

    @property
    def uniprot_ids(self) -> list[str]:
        """Accesiones UniProt de todas sus entidades, sin duplicados."""
        todas = [acc for e in self.entidades for acc in e.uniprot_ids]
        return list(dict.fromkeys(todas))

    @property
    def entidad_principal(self) -> EntidadPolimerica | None:
        """La primera entidad proteica, que es la que representa la estructura.

        Para un complejo multi-cadena esto es una simplificación consciente:
        el pipeline trabaja sobre proteínas monoméricas en esta versión.
        """
        return next((e for e in self.entidades if e.es_proteina), None)

    @property
    def resolucion_o_infinito(self) -> float:
        """Resolución, o infinito si no tiene.

        Las estructuras de RMN no reportan resolución; tratarlas como 0.0 las
        haría pasar cualquier filtro ``resolution_max``, que es exactamente lo
        contrario de lo que se quiere.
        """
        return self.resolucion if self.resolucion is not None else float("inf")


class Funcion(_Base):
    """Anotación funcional (tabla ``funciones``)."""

    tipo: str  # go_molecular_function | go_biological_process | keyword | descripcion
    termino_go: str | None = None
    descripcion: str
    evidencia: str | None = None


class Interaccion(_Base):
    """Interacción proteína-proteína (tabla ``interacciones``)."""

    partner_uniprot: str | None = None
    partner_gen: str | None = None
    tipo: str  # binaria | subunidad
    fuente: str  # IntAct | STRING | UniProt
    n_experimentos: int | None = None
    evidencia: str | None = None


class PTM(_Base):
    """Modificación postraduccional (tabla ``ptms``)."""

    posicion: int
    posicion_fin: int | None = None  # puentes disulfuro y cross-links abarcan dos
    tipo: str
    descripcion: str | None = None
    evidencia: str | None = None


class RegistroUniProt(_Base):
    """Entrada de UniProt ya parseada."""

    accesion: str
    nombre: str | None = None
    gen: str | None = None
    organismo: str | None = None
    tax_id: int | None = None
    longitud: int | None = None
    secuencia: str | None = None
    keywords: list[str] = Field(default_factory=list)
    pdb_ids: list[str] = Field(default_factory=list)
    funciones: list[Funcion] = Field(default_factory=list)
    interacciones: list[Interaccion] = Field(default_factory=list)
    ptms: list[PTM] = Field(default_factory=list)


class Proteina(_Base):
    """Fila de la tabla ``proteinas``: el cruce de RCSB con UniProt."""

    pdb_id: str
    uniprot_id: str | None = None
    nombre: str | None = None
    organismo: str | None = None
    tax_id: int | None = None
    metodo: str | None = None
    resolucion: float | None = None
    longitud: int | None = None
    secuencia: str | None = None
    archivo_path: str | None = None
    sha256: str | None = None
    fecha_descarga: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    curada: bool = False
    motivo_rechazo: str | None = None

    # Anotaciones asociadas, que van a las otras tres tablas.
    funciones: list[Funcion] = Field(default_factory=list)
    interacciones: list[Interaccion] = Field(default_factory=list)
    ptms: list[PTM] = Field(default_factory=list)

    @property
    def rechazada_por_bioseguridad(self) -> bool:
        """Si la verificación de bioseguridad la rechazó.

        Es un rechazo distinto de los de curación: no se revierte aflojando
        criterios, y ninguna fase posterior debe usar la proteína.
        """
        return bool(self.motivo_rechazo and self.motivo_rechazo.startswith("[bioseguridad"))


class ResultadoFiltro(_Base):
    """Veredicto de la curación sobre una proteína."""

    aceptada: bool
    motivos: list[str] = Field(default_factory=list)

    @property
    def motivo(self) -> Optional[str]:
        """Los motivos concatenados, o ``None`` si fue aceptada."""
        return "; ".join(self.motivos) if self.motivos else None
