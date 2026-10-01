"""Asignación de estructura secundaria con DSSP (Fase 4).

DSSP (Kabsch y Sander, 1983) asigna la estructura secundaria de cada residuo
a partir de la geometría de los puentes de hidrógeno del esqueleto, no de la
secuencia. Es el estándar con el que se etiquetan los datasets de predicción
de estructura secundaria, y acá genera las etiquetas que el modelo va a
aprender a predecir.

**No se usa el binario externo.** ``Bio.PDB.DSSP`` es un envoltorio de
``mkdssp``, que no se puede instalar sin permisos de administrador en el
entorno de desarrollo. Se usa la implementación de MDTraj, que trae el
algoritmo completo y se instala con pip.

Los ocho estados de DSSP se colapsan a tres (Q3), que es lo que predice el
modelo y la convención de la literatura:

====  ==========================================  ===
DSSP  Qué es                                      Q3
====  ==========================================  ===
H     hélice alfa                                 H
G     hélice 3-10                                 H
I     hélice pi                                   H
E     hebra beta extendida                        E
B     puente beta aislado                         E
T     giro con puente de hidrógeno                C
S     codo sin puente de hidrógeno                C
' '   nada de lo anterior                         C
====  ==========================================  ===
"""

from __future__ import annotations

import warnings
from pathlib import Path

from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

#: Las tres clases que predice el modelo, en orden fijo. El orden importa:
#: define las columnas de la matriz de confusión y del one-hot.
CLASES_Q3 = ("H", "E", "C")

#: De los ocho estados de DSSP a las tres clases. Es la convención habitual;
#: agrupar G e I con H, y B con E, es lo que hace comparable un Q3 con los de
#: la literatura.
OCHO_A_TRES = {
    "H": "H", "G": "H", "I": "H",
    "E": "E", "B": "E",
    "T": "C", "S": "C", " ": "C", "-": "C", "": "C",
    # MDTraj marca con NA los residuos a los que no puede asignar estado,
    # típicamente por falta de átomos del esqueleto.
    "NA": "C",
}


class ErrorDSSP(Exception):
    """No se pudo asignar estructura secundaria."""


def disponible() -> bool:
    """Indica si la implementación de DSSP se puede usar."""
    try:
        import mdtraj  # noqa: F401
    except ImportError:
        return False
    return True


def motivo_no_disponible() -> str:
    return (
        "MDTraj no está instalado y es lo que provee DSSP.\n\n"
        "    uv pip install mdtraj\n\n"
        "Se usa MDTraj en vez del binario `mkdssp` porque este último no se "
        "puede instalar sin permisos de administrador."
    )


def asignar(
    estructura: str | Path, cadena: str | None = None
) -> tuple[str, str, list[int]]:
    """Calcula secuencia y estructura secundaria de una estructura.

    Args:
        estructura: archivo ``.pdb`` o ``.cif``.
        cadena: identificador de cadena a usar. Si no se pasa, se toma la
            primera cadena de proteína, que es lo habitual en las estructuras
            de una sola cadena del dataset.

    Returns:
        Una tupla ``(secuencia, estructura_secundaria, numeros)``, las tres del
        mismo largo y alineadas residuo a residuo. La estructura secundaria ya
        viene colapsada a las tres clases de :data:`CLASES_Q3`.

    Raises:
        ErrorDSSP: si falta MDTraj, el archivo no se puede leer, o la cadena
            no tiene residuos de proteína.
    """
    if not disponible():
        raise ErrorDSSP(motivo_no_disponible())

    import mdtraj as md

    estructura = Path(estructura)
    if not estructura.is_file():
        raise ErrorDSSP(f"No existe la estructura: {estructura}")

    try:
        with warnings.catch_warnings():
            # MDTraj avisa de residuos no estándar y de cadenas sin nombre,
            # que son normales en archivos del PDB.
            warnings.simplefilter("ignore")
            traza = md.load(str(estructura))
    except Exception as exc:  # noqa: BLE001 - MDTraj lanza de todo
        raise ErrorDSSP(f"No se pudo leer {estructura.name}: {exc}") from exc

    seleccion = "protein"
    if cadena:
        seleccion = f"protein and chainid {_indice_de_cadena(traza, cadena)}"

    try:
        indices = traza.topology.select(seleccion)
    except Exception as exc:  # noqa: BLE001
        raise ErrorDSSP(f"Selección inválida en {estructura.name}: {exc}") from exc

    if len(indices) == 0:
        raise ErrorDSSP(
            f"{estructura.name} no tiene residuos de proteína"
            + (f" en la cadena {cadena}" if cadena else "")
        )

    proteina = traza.atom_slice(indices)
    if cadena is None:
        proteina = _primera_cadena(proteina)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # simplified=False da los ocho estados; el colapso a tres se hace
            # acá para dejarlo explícito y testeable, en vez de delegarlo.
            estados = md.compute_dssp(proteina, simplified=False)[0]
    except Exception as exc:  # noqa: BLE001
        raise ErrorDSSP(f"DSSP falló sobre {estructura.name}: {exc}") from exc

    residuos = list(proteina.topology.residues)
    if len(residuos) != len(estados):
        raise ErrorDSSP(
            f"{estructura.name}: DSSP devolvió {len(estados)} estados para "
            f"{len(residuos)} residuos. No se pueden alinear."
        )

    secuencia = "".join(_una_letra(r) for r in residuos)
    q3 = "".join(a_tres_estados(e) for e in estados)
    numeros = [int(r.resSeq) for r in residuos]

    logger.debug(
        "%s: %d residuos, %s", estructura.name, len(secuencia), _resumen(q3)
    )
    return secuencia, q3, numeros


def a_tres_estados(estado: str) -> str:
    """Colapsa un estado de DSSP a una de las tres clases de Q3."""
    return OCHO_A_TRES.get(str(estado).strip().upper() or " ", "C")


def _indice_de_cadena(traza, cadena: str) -> int:
    """Traduce un identificador de cadena del PDB al índice de MDTraj."""
    for c in traza.topology.chains:
        residuos = list(c.residues)
        if residuos and getattr(c, "chain_id", None) == cadena:
            return c.index
    raise ErrorDSSP(f"No se encontró la cadena '{cadena}'")


def _primera_cadena(traza):
    """Se queda con la primera cadena que tenga residuos.

    Las estructuras del PDB suelen traer varias copias de la misma proteína
    en la unidad asimétrica. Quedarse con una evita contar la misma secuencia
    dos veces en el dataset, que sería redundancia encubierta.
    """
    for c in traza.topology.chains:
        if len(list(c.residues)) > 0:
            indices = traza.topology.select(f"chainid {c.index}")
            return traza.atom_slice(indices)
    return traza


def _una_letra(residuo) -> str:
    """Código de una letra de un residuo, o ``X`` si no es estándar."""
    codigo = getattr(residuo, "code", None)
    return codigo if codigo else "X"


def _resumen(q3: str) -> str:
    total = len(q3) or 1
    return " ".join(f"{c}={q3.count(c) / total:.0%}" for c in CLASES_Q3)


__all__ = [
    "CLASES_Q3",
    "ErrorDSSP",
    "OCHO_A_TRES",
    "a_tres_estados",
    "asignar",
    "disponible",
    "motivo_no_disponible",
]
