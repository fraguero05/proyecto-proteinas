"""Fase 2 — Diseño computacional.

Dos pasos encadenados:

* Predicción de estructura, por orden de preferencia: descarga desde la
  AlphaFold Protein Structure Database vía UniProt ID (con pLDDT leído del
  campo B-factor), ColabFold en notebook, o ESMFold vía API.
* Generación de variantes de secuencia sobre esa estructura, detrás de la
  interfaz abstracta ``SequenceDesigner``. La primera implementación es
  ProteinMPNN sobre CPU; Rosetta queda para una segunda iteración.

Estado: partes A (AlphaFold DB + pLDDT) y B (ProteinMPNN) implementadas en
el Hito 2. ColabFold y ESMFold quedan pendientes.
"""

from pdpipe.phase2_design.alphafold import ClienteAlphaFold, SinModeloPredicho
from pdpipe.phase2_design.designer import (
    DisenadorNoDisponible,
    ErrorDeDiseno,
    SequenceDesigner,
    obtener_disenador,
)
from pdpipe.phase2_design.models import (
    BandaPLDDT,
    ModeloPredicho,
    Mutacion,
    ResiduoPLDDT,
    ResultadoDiseno,
    ResumenPLDDT,
    VarianteSecuencia,
    clasificar_plddt,
)
from pdpipe.phase2_design.pipeline import FuenteNoDisponible, disenar, predecir
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
    "DisenadorNoDisponible",
    "ErrorDeDiseno",
    "ErrorPLDDT",
    "FuenteNoDisponible",
    "ModeloPredicho",
    "Mutacion",
    "ResiduoPLDDT",
    "ResultadoDiseno",
    "ResumenPLDDT",
    "SequenceDesigner",
    "SinModeloPredicho",
    "VarianteSecuencia",
    "anotar_modelo",
    "clasificar_plddt",
    "disenar",
    "extraer_plddt",
    "obtener_disenador",
    "predecir",
    "resumir_plddt",
    "secuencia_de_residuos",
]
