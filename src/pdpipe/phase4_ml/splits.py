"""División train/val/test controlando redundancia de secuencia (Fase 4).

Es el paso que decide si el Q3 que se reporta significa algo.

Si dos proteínas con secuencias parecidas caen una en entrenamiento y otra en
prueba, el modelo puede acertar la segunda recordando la primera. El Q3 sale
alto y mide memorización, no capacidad de generalizar. Por eso no se reparten
proteínas sueltas sino **grupos de proteínas similares**: todo un grupo cae
entero en el mismo conjunto.

**CD-HIT no está instalado**, así que el agrupamiento se hace acá, como ya
preveía el plan del proyecto. El método tiene dos etapas:

1. **Filtro por k-meros.** Comparar todos los pares de 500 secuencias son
   125.000 alineamientos, demasiado lento. Dos proteínas que no comparten
   casi ningún tripéptido no pueden ser 30% idénticas, así que primero se
   descartan esos pares con una medida barata de solapamiento de k-meros.
2. **Alineamiento de los candidatos.** A los pares que pasan el filtro se les
   calcula la identidad real con un alineamiento global de Biopython.

Es la misma idea que usa CD-HIT: filtrar barato y confirmar caro.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pdpipe.phase4_ml.models import Proteina, Split
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

#: Tamaño del k-mero del filtro previo. Con 3 el alfabeto da 8.000 tripéptidos
#: posibles, suficiente para que dos proteínas no relacionadas compartan pocos.
K = 3

#: Solapamiento mínimo de k-meros para molestarse en alinear un par. Se deja
#: holgado a propósito: alinear de más cuesta tiempo, pero filtrar de más
#: dejaría pasar redundancia al test, que es el error que importa evitar.
UMBRAL_KMEROS = 0.10


class ErrorDeSplit(Exception):
    """No se pudo armar la división."""


def kmeros(secuencia: str, k: int = K) -> set[str]:
    """Conjunto de k-meros de una secuencia."""
    if len(secuencia) < k:
        return {secuencia} if secuencia else set()
    return {secuencia[i : i + k] for i in range(len(secuencia) - k + 1)}


def solapamiento_kmeros(a: set[str], b: set[str]) -> float:
    """Fracción de k-meros del conjunto más chico que están en el otro.

    Se divide por el más chico y no por la unión porque una proteína corta
    contenida en una larga es redundancia aunque la unión sea grande.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def identidad(secuencia_a: str, secuencia_b: str) -> float:
    """Identidad de secuencia por alineamiento global, entre 0 y 1.

    Se divide por el largo de la secuencia más corta, que es el criterio de
    CD-HIT: así una proteína chica contenida en otra grande cuenta como
    redundante, que es lo que corresponde.
    """
    from Bio import Align

    if not secuencia_a or not secuencia_b:
        return 0.0

    alineador = Align.PairwiseAligner()
    alineador.mode = "global"
    alineador.match_score = 1.0
    alineador.mismatch_score = -1.0
    alineador.open_gap_score = -2.0
    alineador.extend_gap_score = -0.5
    # Huecos de los extremos gratis. Sin esto, una secuencia corta contenida
    # en una larga paga un hueco por cada residuo que le falta y da identidad
    # baja, cuando en realidad es el caso de redundancia más claro que hay.
    # Biopython 1.86 renombró estos atributos; se soportan los dos nombres.
    # La consulta va sobre la clase y no sobre la instancia a propósito:
    # `hasattr` sobre la instancia ejecuta el getter, que lanza ValueError
    # cuando los puntajes de hueco de apertura y extensión difieren.
    clase = type(alineador)
    for nuevo, viejo in (
        ("end_insertion_score", "target_end_gap_score"),
        ("end_deletion_score", "query_end_gap_score"),
    ):
        setattr(alineador, nuevo if hasattr(clase, nuevo) else viejo, 0.0)

    try:
        mejor = alineador.align(secuencia_a, secuencia_b)[0]
    except Exception as exc:  # noqa: BLE001 - Biopython lanza de todo
        raise ErrorDeSplit(f"No se pudo alinear: {exc}") from exc

    # Se cuentan las identidades del alineamiento y no se usa el puntaje como
    # proxy: el puntaje mezcla coincidencias, diferencias y huecos en un solo
    # número, y dividirlo por el largo no da una fracción interpretable.
    fila_a, fila_b = str(mejor[0]), str(mejor[1])
    iguales = sum(1 for x, y in zip(fila_a, fila_b) if x == y and x != "-")
    return round(iguales / min(len(secuencia_a), len(secuencia_b)), 6)


def agrupar(
    proteinas: Sequence[Proteina], identidad_max: float = 0.3
) -> list[list[str]]:
    """Agrupa proteínas con identidad mayor al umbral.

    Devuelve una lista de grupos, cada uno con los identificadores de las
    proteínas que quedaron juntas. Una proteína sin parientes forma un grupo
    de uno.
    """
    identificadores = [p.identificador for p in proteinas]
    secuencias = {p.identificador: p.secuencia for p in proteinas}
    conjuntos = {p.identificador: kmeros(p.secuencia) for p in proteinas}

    padre = {i: i for i in identificadores}

    def raiz(x: str) -> str:
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(a: str, b: str) -> None:
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            padre[rb] = ra

    alineados = 0
    for i, uno in enumerate(identificadores):
        for otro in identificadores[i + 1 :]:
            if raiz(uno) == raiz(otro):
                continue  # ya están juntos, alinear no cambiaría nada
            if solapamiento_kmeros(conjuntos[uno], conjuntos[otro]) < UMBRAL_KMEROS:
                continue
            alineados += 1
            if identidad(secuencias[uno], secuencias[otro]) >= identidad_max:
                unir(uno, otro)

    grupos: dict[str, list[str]] = {}
    for identificador in identificadores:
        grupos.setdefault(raiz(identificador), []).append(identificador)

    resultado = sorted(grupos.values(), key=len, reverse=True)
    logger.info(
        "Agrupamiento: %d proteínas -> %d grupos (%d pares alineados, "
        "%d descartados por k-meros)",
        len(identificadores),
        len(resultado),
        alineados,
        len(identificadores) * (len(identificadores) - 1) // 2 - alineados,
    )
    return resultado


def dividir(
    proteinas: Sequence[Proteina],
    val_size: float = 0.15,
    test_size: float = 0.15,
    identidad_max: float = 0.3,
    seed: int = 42,
) -> Split:
    """Reparte las proteínas en train/val/test sin partir grupos.

    Los grupos se asignan de mayor a menor al conjunto que esté más lejos de
    su cupo. Hacerlo al revés —de menor a mayor— dejaría los grupos grandes
    para el final, cuando ya no entran en ningún lado sin desbalancear todo.

    Raises:
        ErrorDeSplit: si las proporciones no son válidas o no hay proteínas.
    """
    if not proteinas:
        raise ErrorDeSplit("No hay proteínas para dividir")
    if not 0 <= val_size < 1 or not 0 <= test_size < 1:
        raise ErrorDeSplit(
            f"Las proporciones tienen que estar entre 0 y 1: "
            f"val={val_size}, test={test_size}"
        )
    if val_size + test_size >= 1:
        raise ErrorDeSplit(
            f"val ({val_size}) + test ({test_size}) no deja nada para entrenar"
        )

    grupos = agrupar(proteinas, identidad_max)

    # El desempate se sortea para que dos grupos del mismo tamaño no vayan
    # siempre al mismo conjunto por el orden en que llegaron.
    sorteo = random.Random(seed)
    sorteo.shuffle(grupos)
    grupos.sort(key=len, reverse=True)

    total = len(proteinas)
    objetivo = {
        "train": total * (1 - val_size - test_size),
        "val": total * val_size,
        "test": total * test_size,
    }
    asignados: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for grupo in grupos:
        faltante = {
            nombre: objetivo[nombre] - len(asignados[nombre])
            for nombre in asignados
        }
        destino = max(faltante, key=lambda n: faltante[n])
        asignados[destino].extend(grupo)

    split = Split(
        train=sorted(asignados["train"]),
        val=sorted(asignados["val"]),
        test=sorted(asignados["test"]),
        identidad_max=identidad_max,
        n_grupos=len(grupos),
    )

    logger.info(
        "División: train=%d val=%d test=%d (de %d grupos)",
        len(split.train),
        len(split.val),
        len(split.test),
        len(grupos),
    )
    return split


def verificar(
    proteinas: Sequence[Proteina], split: Split, identidad_max: float = 0.3
) -> list[tuple[str, str, float]]:
    """Busca pares redundantes que hayan quedado en conjuntos distintos.

    Es la comprobación que le da sentido al Q3: si devuelve algo, el test
    comparte secuencias con el entrenamiento y el número está inflado.

    Returns:
        Los pares problemáticos con su identidad. Vacío es lo que se espera.
    """
    por_id = {p.identificador: p for p in proteinas}
    problemas: list[tuple[str, str, float]] = []

    for conjunto_a, conjunto_b in (("train", "test"), ("train", "val"), ("val", "test")):
        for uno in getattr(split, conjunto_a):
            for otro in getattr(split, conjunto_b):
                if uno not in por_id or otro not in por_id:
                    continue
                a, b = por_id[uno], por_id[otro]
                if solapamiento_kmeros(kmeros(a.secuencia), kmeros(b.secuencia)) < UMBRAL_KMEROS:
                    continue
                valor = identidad(a.secuencia, b.secuencia)
                if valor >= identidad_max:
                    problemas.append((uno, otro, round(valor, 4)))

    if problemas:
        logger.warning(
            "%d pares redundantes entre conjuntos: el Q3 del test va a estar inflado",
            len(problemas),
        )
    return problemas


__all__ = [
    "ErrorDeSplit",
    "K",
    "UMBRAL_KMEROS",
    "agrupar",
    "dividir",
    "identidad",
    "kmeros",
    "solapamiento_kmeros",
    "verificar",
]
