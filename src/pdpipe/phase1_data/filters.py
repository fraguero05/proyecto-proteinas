"""Filtros de curación de la Fase 1.

Criterios de calidad sobre lo descargado: resolución, método experimental,
organismo y longitud de cadena. Son funciones puras sobre un
:class:`~pdpipe.phase1_data.models.Proteina`, para poder testearlas sin red ni
base de datos.

Cada filtro acumula *todos* los motivos de rechazo en vez de cortar en el
primero: para decidir si conviene aflojar un criterio hay que ver todo lo que
falló, no solo lo primero que falló.
"""

from __future__ import annotations

from pdpipe.config import DataConfig
from pdpipe.phase1_data.models import Proteina, ResultadoFiltro
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)


def filtrar_resolucion(proteina: Proteina, resolucion_max: float) -> list[str]:
    """Rechaza estructuras de resolución peor que el umbral.

    Una estructura sin resolución (RMN, predicciones) se rechaza: no se la
    puede comparar contra un umbral, y dejarla pasar sería asumir que es
    perfecta.
    """
    if proteina.resolucion is None:
        return [
            f"sin resolución reportada (método: {proteina.metodo or 'desconocido'})"
        ]
    if proteina.resolucion > resolucion_max:
        return [
            f"resolución {proteina.resolucion:.2f} Å peor que el máximo "
            f"{resolucion_max:.2f} Å"
        ]
    return []


def filtrar_metodo(proteina: Proteina, metodos_aceptados: list[str]) -> list[str]:
    """Rechaza métodos experimentales fuera de la lista aceptada."""
    if not metodos_aceptados:
        return []
    if proteina.metodo is None:
        return ["sin método experimental reportado"]
    # Se compara en mayúsculas: el PDB usa "X-RAY DIFFRACTION".
    aceptados = {m.strip().upper() for m in metodos_aceptados}
    if proteina.metodo.strip().upper() not in aceptados:
        return [
            f"método '{proteina.metodo}' fuera de los aceptados "
            f"({', '.join(sorted(aceptados))})"
        ]
    return []


def filtrar_organismo(proteina: Proteina, organismos: list[str]) -> list[str]:
    """Rechaza organismos fuera de la lista. Lista vacía = sin filtro."""
    if not organismos:
        return []
    if proteina.organismo is None:
        return ["sin organismo reportado"]
    # Coincidencia por subcadena: "Homo sapiens" matchea "Homo sapiens (human)".
    objetivo = proteina.organismo.strip().lower()
    if not any(o.strip().lower() in objetivo for o in organismos):
        return [
            f"organismo '{proteina.organismo}' fuera de los buscados "
            f"({', '.join(organismos)})"
        ]
    return []


def filtrar_longitud(
    proteina: Proteina, longitud_min: int, longitud_max: int
) -> list[str]:
    """Rechaza cadenas fuera del rango de longitud."""
    if proteina.longitud is None:
        return ["sin longitud de secuencia"]
    if proteina.longitud < longitud_min:
        return [f"longitud {proteina.longitud} menor al mínimo {longitud_min}"]
    if proteina.longitud > longitud_max:
        return [f"longitud {proteina.longitud} mayor al máximo {longitud_max}"]
    return []


def aplicar_filtros(proteina: Proteina, config: DataConfig) -> ResultadoFiltro:
    """Aplica todos los filtros de calidad y devuelve el veredicto conjunto.

    No incluye la verificación de bioseguridad, que vive en
    :mod:`pdpipe.phase1_data.biosafety` y se aplica antes: son cosas
    distintas, y mezclarlas haría que un rechazo por bioseguridad se lea como
    un simple problema de calidad.
    """
    motivos: list[str] = []
    motivos += filtrar_resolucion(proteina, config.resolution_max)
    motivos += filtrar_metodo(proteina, config.experimental_methods)
    motivos += filtrar_organismo(proteina, config.organisms)
    motivos += filtrar_longitud(proteina, config.length_min, config.length_max)

    return ResultadoFiltro(aceptada=not motivos, motivos=motivos)
