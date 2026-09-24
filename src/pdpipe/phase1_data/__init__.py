"""Fase 1 — Recolección de datos.

Descarga y curación de estructuras desde el RCSB PDB y de anotaciones
funcionales desde UniProt, con filtros por resolución, organismo, método
experimental y longitud, y persistencia en una base SQLite local
(tablas ``proteinas``, ``funciones``, ``interacciones``, ``ptms``).

Incluye la verificación de bioseguridad que rechaza toxinas y proteínas de
patógenos de la lista de agentes seleccionados (ver README).

Estado: pendiente — se implementa en el Hito 1.
"""

__all__: list[str] = []
