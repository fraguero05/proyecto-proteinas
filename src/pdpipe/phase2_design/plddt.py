"""Extracción del pLDDT desde el campo B-factor (Fase 2).

AlphaFold no escribe el pLDDT en un campo propio: lo guarda en la columna del
B-factor de cada átomo, reutilizando un campo que en una estructura
experimental significa otra cosa (el factor de temperatura). Por eso un
archivo de AlphaFold **no** se puede interpretar como uno del PDB sin saber
esto: valores de "B-factor" de 90 serían pésimos en un cristal y son
excelentes en una predicción.

El valor es el mismo para todos los átomos de un residuo, así que se lee del
carbono alfa, que existe en todos los aminoácidos.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median

import warnings

from pdpipe.phase2_design.models import (
    BandaPLDDT,
    ModeloPredicho,
    ResiduoPLDDT,
    ResumenPLDDT,
    clasificar_plddt,
)
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

# Código de tres letras -> una letra. Se incluyen solo los 20 estándar:
# AlphaFold no predice residuos modificados.
_TRES_A_UNA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


class ErrorPLDDT(Exception):
    """No se pudo extraer el pLDDT del archivo."""


def extraer_plddt(archivo: str | Path) -> list[ResiduoPLDDT]:
    """Lee el pLDDT por residuo de un modelo de AlphaFold.

    Args:
        archivo: ruta a un ``.pdb`` o ``.cif`` de AlphaFold.

    Returns:
        Un :class:`ResiduoPLDDT` por residuo, en orden de secuencia.

    Raises:
        ErrorPLDDT: si el archivo no se puede leer o no tiene carbonos alfa.
    """
    archivo = Path(archivo)
    if not archivo.is_file():
        raise ErrorPLDDT(f"No existe el archivo: {archivo}")

    try:
        from Bio.PDB import MMCIFParser, PDBParser
    except ImportError as exc:  # pragma: no cover - depende del extra `data`
        raise ErrorPLDDT(
            "Biopython no está instalado. Instalá el extra: "
            'uv pip install -e ".[data]"'
        ) from exc

    sufijo = archivo.suffix.lower()
    if sufijo == ".cif":
        parser = MMCIFParser(QUIET=True)
    elif sufijo in {".pdb", ".ent"}:
        parser = PDBParser(QUIET=True)
    else:
        raise ErrorPLDDT(
            f"Formato no soportado: '{sufijo}'. Se esperaba .pdb o .cif."
        )

    try:
        with warnings.catch_warnings():
            # Biopython avisa de cosas normales en modelos predichos.
            warnings.simplefilter("ignore")
            estructura = parser.get_structure("modelo", str(archivo))
    except Exception as exc:  # noqa: BLE001 - Biopython lanza de todo
        raise ErrorPLDDT(f"No se pudo parsear {archivo.name}: {exc}") from exc

    residuos: list[ResiduoPLDDT] = []
    # Un modelo de AlphaFold tiene un solo modelo y una sola cadena; se recorre
    # el primero de cada uno para no depender de eso implícitamente.
    for modelo in estructura:
        for cadena in modelo:
            for residuo in cadena:
                if "CA" not in residuo:
                    continue  # aguas y heteroátomos
                carbono_alfa = residuo["CA"]
                nombre = residuo.get_resname().strip().upper()
                residuos.append(
                    ResiduoPLDDT(
                        numero=residuo.get_id()[1],
                        aminoacido=_TRES_A_UNA.get(nombre),
                        plddt=float(carbono_alfa.get_bfactor()),
                    )
                )
            break  # solo la primera cadena
        break  # solo el primer modelo

    if not residuos:
        raise ErrorPLDDT(
            f"{archivo.name} no tiene carbonos alfa: ¿es un modelo de proteína?"
        )

    return residuos


def resumir_plddt(residuos: list[ResiduoPLDDT]) -> ResumenPLDDT:
    """Estadísticos y distribución por banda de confianza."""
    if not residuos:
        raise ErrorPLDDT("No hay residuos que resumir")

    valores = [r.plddt for r in residuos]
    total = len(valores)

    conteos = {banda.value: 0 for banda in BandaPLDDT}
    for valor in valores:
        conteos[clasificar_plddt(valor).value] += 1

    return ResumenPLDDT(
        n_residuos=total,
        media=round(mean(valores), 2),
        mediana=round(median(valores), 2),
        minimo=round(min(valores), 2),
        maximo=round(max(valores), 2),
        conteo_por_banda=conteos,
        fraccion_por_banda={
            banda: round(cantidad / total, 4) for banda, cantidad in conteos.items()
        },
    )


def secuencia_de_residuos(residuos: list[ResiduoPLDDT]) -> str:
    """Secuencia en una letra a partir de los residuos leídos.

    Un residuo no estándar (que AlphaFold no debería producir) aparece como
    ``X`` en vez de romper: es preferible una secuencia con un hueco marcado a
    no tener secuencia.
    """
    return "".join(r.aminoacido or "X" for r in residuos)


def anotar_modelo(modelo: ModeloPredicho, archivo: str | Path) -> ModeloPredicho:
    """Completa un modelo con su pLDDT por residuo y su resumen."""
    residuos = extraer_plddt(archivo)
    resumen = resumir_plddt(residuos)

    # Contraste contra el pLDDT medio que reporta la propia API: si difieren,
    # algo se parseó mal y es mejor enterarse acá que tres fases después.
    if modelo.metrica_global_api is not None:
        diferencia = abs(resumen.media - modelo.metrica_global_api)
        if diferencia > 1.0:
            logger.warning(
                "El pLDDT medio calculado (%.2f) difiere del que reporta la API "
                "(%.2f) en %.2f puntos. Revisá el parseo del B-factor.",
                resumen.media,
                modelo.metrica_global_api,
                diferencia,
            )

    return modelo.model_copy(
        update={
            "residuos": residuos,
            "resumen": resumen,
            "secuencia": modelo.secuencia or secuencia_de_residuos(residuos),
        }
    )
