"""Fase 2 — Diseño computacional.

Dos pasos encadenados:

* Predicción de estructura, por orden de preferencia: descarga desde la
  AlphaFold Protein Structure Database vía UniProt ID (con pLDDT leído del
  campo B-factor), ColabFold en notebook, o ESMFold vía API.
* Generación de variantes de secuencia sobre esa estructura, detrás de la
  interfaz abstracta ``SequenceDesigner``. La primera implementación es
  ProteinMPNN sobre CPU; Rosetta queda para una segunda iteración.

Estado: pendiente — se implementa en el Hito 2.
"""

__all__: list[str] = []
