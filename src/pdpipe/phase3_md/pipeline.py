"""Orquestación del análisis de trayectorias (Fase 3, parte C).

Toma una trayectoria ya simulada y produce las cinco medidas, sus figuras y
una tabla. No necesita GROMACS: la simulación es la parte B y puede haber
corrido en otra máquina o en un notebook de Colab.

Criterio de fallo: si una medida no se puede calcular, se anota como
advertencia y el análisis sigue. Perder los puentes de hidrógeno por una
estructura sin hidrógenos no debería costar también el RMSD.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pdpipe.config import Config
from pdpipe.phase3_md import analysis as an
from pdpipe.phase3_md.figures import generar_figuras
from pdpipe.phase3_md.models import ResultadoAnalisisMD
from pdpipe.utils.logging import get_logger
from pdpipe.utils.manifest import RunManifest

logger = get_logger(__name__)


def analizar(
    config: Config,
    topologia: str | Path,
    trayectoria: str | Path | None = None,
    seleccion: str = an.SELECCION_ESQUELETO,
    paso_sasa: int | None = None,
    con_figuras: bool = True,
    manifest: RunManifest | None = None,
) -> ResultadoAnalisisMD:
    """Analiza una trayectoria y deja tablas y figuras en ``data/processed``.

    Args:
        config: configuración del pipeline.
        topologia: ``.gro``, ``.pdb`` o ``.tpr`` con los átomos.
        trayectoria: ``.xtc``/``.trr``; si falta, se usan los cuadros del
            propio archivo de topología.
        seleccion: átomos para RMSD y RMSF; por defecto los carbonos alfa.
        paso_sasa: analizar uno de cada N cuadros para la SASA. Si no se pasa,
            sale de ``md.sasa_stride``.
        con_figuras: generar los PNG.
        manifest: si se pasa, se le registran entradas y salidas.

    Raises:
        ErrorDeAnalisis: si la trayectoria no se puede abrir. Los fallos de
            una medida concreta van a ``advertencias``, no acá.
    """
    rutas = config.resolved_paths()
    topologia = Path(topologia)
    paso = paso_sasa if paso_sasa is not None else config.md.sasa_stride

    universo = an.cargar(topologia, trayectoria)
    advertencias: list[str] = []

    dt = _paso_de_tiempo(universo)
    if dt is None:
        advertencias.append(
            "La trayectoria no declara paso de tiempo; se asume 1 ps por "
            "cuadro. Los tiempos del eje X son índices, no picosegundos reales."
        )

    resultado = ResultadoAnalisisMD(
        topologia=topologia,
        trayectoria=Path(trayectoria) if trayectoria else None,
        n_frames=len(universo.trajectory),
        n_atomos=len(universo.atoms),
        n_residuos=len(universo.residues),
        dt_ps=dt,
    )

    # El RMSD va antes de alinear: superpone cada cuadro por su cuenta y no
    # le afecta, pero así queda explícito que no depende del alineado.
    rmsd = _intentar("RMSD", advertencias, an.calcular_rmsd, universo, seleccion)
    radio = _intentar(
        "radio de giro", advertencias, an.calcular_radio_de_giro, universo, an.SELECCION_PROTEINA
    )
    sasa = _intentar(
        "SASA", advertencias, an.calcular_sasa, universo, an.SELECCION_PROTEINA, paso
    )
    puentes = _intentar(
        "puentes de hidrógeno", advertencias, an.calcular_puentes_de_hidrogeno, universo
    )

    # El RMSF sí lo necesita: sin alinear mide también la rotación y la
    # traslación del conjunto, no la flexibilidad interna.
    rmsf = None
    try:
        an.alinear(universo, seleccion)
        rmsf = _intentar("RMSF", advertencias, an.calcular_rmsf, universo, seleccion)
    except Exception as exc:  # noqa: BLE001 - alinear puede fallar de varias formas
        advertencias.append(f"No se pudo alinear la trayectoria, se omite el RMSF: {exc}")

    resultado = resultado.model_copy(
        update={
            "rmsd": rmsd,
            "radio_giro": radio,
            "sasa": sasa,
            "puentes": puentes,
            "rmsf": rmsf,
            "advertencias": advertencias,
        }
    )

    destino = rutas["data_processed"]
    tabla, resumen = _escribir_salidas(resultado, destino)
    figuras = generar_figuras(resultado, destino / "figuras") if con_figuras else []
    resultado = resultado.model_copy(
        update={"tabla": tabla, "resumen_json": resumen, "figuras": figuras}
    )

    for aviso in advertencias:
        logger.warning("%s", aviso)

    if manifest:
        manifest.add_input(topologia, key=topologia.name)
        if trayectoria:
            manifest.add_input(trayectoria, key=Path(trayectoria).name)
        manifest.add_output(tabla, key=tabla.name)
        manifest.add_output(resumen, key=resumen.name)
        for figura in figuras:
            manifest.add_output(figura, key=figura.name)
        if rmsd:
            manifest.add_note(
                f"{topologia.stem}: RMSD final {rmsd.final} Å "
                f"(deriva {rmsd.deriva} Å en {resultado.n_frames} cuadros)"
            )
        for aviso in advertencias:
            manifest.add_note(f"advertencia: {aviso}")

    return resultado


def _intentar(etiqueta: str, advertencias: list[str], funcion, *args):
    """Corre una medida y, si falla, la anota en vez de cortar el análisis."""
    try:
        return funcion(*args)
    except an.ErrorDeAnalisis as exc:
        advertencias.append(f"No se pudo calcular {etiqueta}: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001 - MDAnalysis y Biopython lanzan de todo
        advertencias.append(f"Falló el cálculo de {etiqueta}: {exc}")
        return None


def _paso_de_tiempo(universo) -> float | None:
    """Paso de tiempo real entre cuadros, o ``None`` si no está declarado.

    Un PDB multi-modelo no lo trae y MDAnalysis asume 1 ps. Distinguirlo
    importa: si no, el eje de tiempo de las figuras miente sin avisar.
    """
    import warnings

    try:
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter("always")
            dt = float(universo.trajectory.dt)
        if any("no dt information" in str(c.message) for c in capturados):
            return None
    except Exception:  # noqa: BLE001
        return None
    return dt if dt > 0 else None


def _escribir_salidas(
    resultado: ResultadoAnalisisMD, destino: Path
) -> tuple[Path, Path]:
    """Escribe la tabla de series y el JSON de resumen."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    base = resultado.topologia.stem

    # Formato largo: la SASA puede tener menos puntos que el resto si se
    # analizó con paso, y una tabla ancha obligaría a rellenar con huecos.
    tabla = destino / f"{base}_md_series.csv"
    with tabla.open("w", encoding="utf-8", newline="") as handle:
        escritor = csv.writer(handle)
        escritor.writerow(["magnitud", "unidad", "tiempo_ps", "valor"])
        for serie in resultado.series().values():
            for t, v in zip(serie.tiempos_ps, serie.valores):
                escritor.writerow([serie.nombre, serie.unidad, f"{t:.3f}", f"{v:.4f}"])
        if resultado.rmsf:
            for r, v in zip(resultado.rmsf.residuos, resultado.rmsf.valores):
                escritor.writerow(["RMSF", resultado.rmsf.unidad, f"residuo {r}", f"{v:.4f}"])

    resumen = destino / f"{base}_md_resumen.json"
    datos: dict = {
        "topologia": str(resultado.topologia),
        "trayectoria": str(resultado.trayectoria) if resultado.trayectoria else None,
        "n_frames": resultado.n_frames,
        "n_atomos": resultado.n_atomos,
        "n_residuos": resultado.n_residuos,
        "dt_ps": resultado.dt_ps,
        "duracion_ps": resultado.duracion_ps,
        "advertencias": resultado.advertencias,
        "medidas": {},
    }
    for clave, serie in resultado.series().items():
        datos["medidas"][clave] = {
            "nombre": serie.nombre,
            "unidad": serie.unidad,
            "media": serie.media,
            "desvio": serie.desvio,
            "minimo": serie.minimo,
            "maximo": serie.maximo,
            "inicial": serie.inicial,
            "final": serie.final,
            "deriva": serie.deriva,
            "n_puntos": len(serie.valores),
        }
    if resultado.rmsf:
        datos["medidas"]["rmsf"] = {
            "nombre": resultado.rmsf.nombre,
            "unidad": resultado.rmsf.unidad,
            "media": resultado.rmsf.media,
            "mas_moviles": resultado.rmsf.mas_moviles(10),
        }
    resumen.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Series: %s", tabla)
    logger.info("Resumen: %s", resumen)
    return tabla, resumen


__all__ = ["analizar"]
