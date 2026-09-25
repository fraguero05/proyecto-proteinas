"""Fase 2 — Diseño computacional.

Dos pasos encadenados:

* Predicción de estructura, por orden de preferencia: descarga desde la
  AlphaFold Protein Structure Database vía UniProt ID (con pLDDT leído del
  campo B-factor), ColabFold en notebook, o ESMFold vía API.
* Generación de variantes de secuencia sobre esa estructura, detrás de la
  interfaz abstracta ``SequenceDesigner``. La primera implementación es
  ProteinMPNN sobre CPU; Rosetta queda para una segunda iteración.

Estado: parte A (AlphaFold DB + pLDDT) implementada en el Hito 2.
La parte B (ProteinMPNN) está pendiente.
"""

from pdpipe.phase2_design.alphafold import ClienteAlphaFold, SinModeloPredicho
from pdpipe.phase2_design.models import (
    BandaPLDDT,
    ModeloPredicho,
    ResiduoPLDDT,
    ResumenPLDDT,
    clasificar_plddt,
)
from pdpipe.phase2_design.pipeline import FuenteNoDisponible, predecir
from pdpipe.phase2_design.plddt import (
    ErrorPLDDT,
    anotar_modelo,
    extraer_plddt,
    resumir_plddt,
    secuencia_de_residuos,
)

__all__ = [
    "BandaPLDDT",
    "ClienteAlphaFold",
    "ErrorPLDDT",
    "FuenteNoDisponible",
    "ModeloPredicho",
    "ResiduoPLDDT",
    "ResumenPLDDT",
    "SinModeloPredicho",
    "anotar_modelo",
    "clasificar_plddt",
    "extraer_plddt",
    "predecir",
    "resumir_plddt",
    "secuencia_de_residuos",
]
