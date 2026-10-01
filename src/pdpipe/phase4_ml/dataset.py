"""Dataset de estructura secundaria (Fase 4).

Convierte estructuras del PDB en ejemplos de entrenamiento: la secuencia es la
entrada, la asignación de DSSP es la etiqueta.

Dos representaciones de los mismos datos, porque los dos modelos las necesitan
distintas:

* **Ventanas deslizantes** (:func:`a_ventanas`) — cada residuo se representa
  por su vecindario de ``ventana`` posiciones, aplanado en un vector. Es lo
  que necesitan la regresión logística y el random forest, que esperan
  entradas de tamaño fijo.
* **Secuencias completas** (:func:`a_secuencias`) — cada proteína entera, de
  largo variable. Es lo que aprovecha el BiLSTM, que puede mirar toda la
  cadena y no solo una ventana.

Que los dos modelos partan del mismo conjunto de proteínas es lo que hace la
comparación honesta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pdpipe.phase4_ml.dssp import CLASES_Q3, ErrorDSSP, asignar
from pdpipe.phase4_ml.models import Proteina
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

#: Los veinte aminoácidos estándar, en orden fijo. El orden define las
#: columnas del one-hot, así que no puede cambiar entre entrenar y predecir.
AMINOACIDOS = "ACDEFGHIKLMNPQRSTVWY"

#: Símbolo para un residuo no estándar (selenometionina, modificados, etc.).
DESCONOCIDO = "X"

#: Relleno de los bordes. Es un símbolo propio y no `X` a propósito: "no hay
#: residuo acá" y "hay un residuo que no reconozco" son cosas distintas, y
#: mezclarlas le enseñaría al modelo que los extremos son raros en vez de que
#: son extremos.
RELLENO = "-"

#: Alfabeto completo: 20 estándar + desconocido + relleno = 22 símbolos.
ALFABETO = AMINOACIDOS + DESCONOCIDO + RELLENO

INDICE_AA = {letra: i for i, letra in enumerate(ALFABETO)}
INDICE_Q3 = {clase: i for i, clase in enumerate(CLASES_Q3)}


def construir(
    estructuras: dict[str, Path], longitud_min: int = 30
) -> tuple[list[Proteina], list[str]]:
    """Corre DSSP sobre cada estructura y arma la lista de proteínas.

    Args:
        estructuras: mapa de identificador a archivo de coordenadas.
        longitud_min: descarta cadenas más cortas. Una cadena de 10 residuos
            aporta ruido y casi ningún ejemplo útil.

    Returns:
        Las proteínas procesadas y la lista de las que se descartaron, con el
        motivo. Lo segundo importa: un dataset que silencia sus descartes no
        se puede auditar.
    """
    proteinas: list[Proteina] = []
    descartadas: list[str] = []

    for identificador, archivo in sorted(estructuras.items()):
        try:
            secuencia, q3, numeros = asignar(archivo)
        except ErrorDSSP as exc:
            descartadas.append(f"{identificador}: {exc}")
            continue

        if len(secuencia) < longitud_min:
            descartadas.append(
                f"{identificador}: {len(secuencia)} residuos, menos que el "
                f"mínimo de {longitud_min}"
            )
            continue

        proteinas.append(
            Proteina(
                identificador=identificador,
                secuencia=secuencia,
                estructura=q3,
                numeros=numeros,
                archivo=Path(archivo),
            )
        )

    logger.info(
        "Dataset: %d proteínas, %d residuos, %d descartadas",
        len(proteinas),
        sum(len(p.secuencia) for p in proteinas),
        len(descartadas),
    )
    return proteinas, descartadas


def codificar_residuo(letra: str) -> int:
    """Índice en el alfabeto; los no estándar caen en ``X``."""
    return INDICE_AA.get(letra.upper(), INDICE_AA[DESCONOCIDO])


def una_ventana(secuencia: str, centro: int, ventana: int) -> str:
    """Vecindario de ``ventana`` residuos centrado en ``centro``.

    Los bordes se rellenan con :data:`RELLENO`. La ventana tiene que ser impar
    para que exista un centro: con 16 posiciones no hay residuo del medio.
    """
    if ventana % 2 == 0:
        raise ValueError(f"La ventana tiene que ser impar, no {ventana}")

    mitad = ventana // 2
    piezas = []
    for offset in range(-mitad, mitad + 1):
        posicion = centro + offset
        if 0 <= posicion < len(secuencia):
            piezas.append(secuencia[posicion])
        else:
            piezas.append(RELLENO)
    return "".join(piezas)


def a_ventanas(
    proteinas: list[Proteina], ventana: int = 17
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Representación de ventanas deslizantes, para los modelos clásicos.

    Returns:
        ``(X, y, grupos)``. ``X`` tiene forma ``(n_residuos, ventana * 22)``
        con el one-hot aplanado; ``y`` es el índice de la clase Q3; y
        ``grupos`` dice de qué proteína viene cada fila.

        ``grupos`` es lo que permite partir en train/val/test **por proteína**
        y no por residuo: dos residuos vecinos de la misma cadena son casi el
        mismo ejemplo, y repartirlos entre train y test inflaría el Q3.
    """
    if not proteinas:
        return (
            np.zeros((0, ventana * len(ALFABETO)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=object),
        )

    filas: list[np.ndarray] = []
    etiquetas: list[int] = []
    grupos: list[str] = []

    for proteina in proteinas:
        for centro in range(len(proteina.secuencia)):
            trozo = una_ventana(proteina.secuencia, centro, ventana)
            codificada = np.zeros((ventana, len(ALFABETO)), dtype=np.float32)
            for i, letra in enumerate(trozo):
                codificada[i, codificar_residuo(letra)] = 1.0
            filas.append(codificada.reshape(-1))
            etiquetas.append(INDICE_Q3[proteina.estructura[centro]])
            grupos.append(proteina.identificador)

    return (
        np.stack(filas),
        np.array(etiquetas, dtype=np.int64),
        np.array(grupos, dtype=object),
    )


def a_secuencias(
    proteinas: list[Proteina],
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Representación de secuencias completas, para el BiLSTM.

    Returns:
        ``(X, y, identificadores)``, donde cada ``X[i]`` tiene forma
        ``(largo_i, 22)`` y cada ``y[i]`` forma ``(largo_i,)``. Los largos
        varían, así que el agrupado en lotes necesita relleno y máscara.
    """
    entradas: list[np.ndarray] = []
    salidas: list[np.ndarray] = []
    identificadores: list[str] = []

    for proteina in proteinas:
        largo = len(proteina.secuencia)
        codificada = np.zeros((largo, len(ALFABETO)), dtype=np.float32)
        for i, letra in enumerate(proteina.secuencia):
            codificada[i, codificar_residuo(letra)] = 1.0
        entradas.append(codificada)
        salidas.append(
            np.array([INDICE_Q3[c] for c in proteina.estructura], dtype=np.int64)
        )
        identificadores.append(proteina.identificador)

    return entradas, salidas, identificadores


def composicion(proteinas: list[Proteina]) -> dict[str, float]:
    """Fracción de residuos de cada clase Q3 en el conjunto.

    Es el número contra el que hay que comparar cualquier Q3: un modelo que
    predice siempre la clase mayoritaria ya acierta esa fracción sin aprender
    nada. Si el Q3 del modelo no la supera con holgura, no aprendió.
    """
    total = sum(len(p.estructura) for p in proteinas)
    if total == 0:
        return {clase: 0.0 for clase in CLASES_Q3}
    conteos = {
        clase: sum(p.estructura.count(clase) for p in proteinas)
        for clase in CLASES_Q3
    }
    return {clase: round(n / total, 4) for clase, n in conteos.items()}


__all__ = [
    "ALFABETO",
    "AMINOACIDOS",
    "DESCONOCIDO",
    "INDICE_AA",
    "INDICE_Q3",
    "RELLENO",
    "a_secuencias",
    "a_ventanas",
    "codificar_residuo",
    "composicion",
    "construir",
    "una_ventana",
]
