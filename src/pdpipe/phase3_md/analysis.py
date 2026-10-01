"""Análisis de trayectorias de dinámica molecular (Fase 3).

Cinco medidas sobre la trayectoria, que responden preguntas distintas:

* **RMSD** — cuánto se alejó la estructura de la de partida. Dice si el
  sistema se estabilizó o sigue derivando.
* **RMSF** — cuánto se mueve cada residuo alrededor de su posición media.
  Dice *dónde* está la flexibilidad: extremos, loops, sitios de unión.
* **Radio de giro** — qué tan compacta es. Si crece sostenidamente, la
  proteína se está desplegando.
* **SASA** — cuánta superficie queda expuesta al solvente. Se mueve junto con
  el radio de giro cuando algo se abre.
* **Puentes de hidrógeno** — cuántos hay cuadro a cuadro. Es lo que sostiene
  hélices y hojas: si caen, la estructura secundaria se está perdiendo.

Todo se calcula con MDAnalysis salvo la SASA, que MDAnalysis no implementa
(ver :func:`calcular_sasa`).
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from statistics import mean

from pdpipe.phase3_md.models import PerfilPorResiduo, SerieTemporal
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

# Selección por defecto para RMSD y RMSF. Los carbonos alfa alcanzan para
# seguir el movimiento del esqueleto y son mucho menos ruidosos que incluir
# las cadenas laterales, que rotan todo el tiempo sin que la proteína cambie.
SELECCION_ESQUELETO = "protein and name CA"
SELECCION_PROTEINA = "protein"


class ErrorDeAnalisis(Exception):
    """No se pudo analizar la trayectoria."""


def cargar(topologia: str | Path, trayectoria: str | Path | None = None):
    """Abre la trayectoria con MDAnalysis.

    Args:
        topologia: archivo con los átomos y sus conexiones (``.gro``, ``.pdb``,
            ``.tpr``).
        trayectoria: coordenadas en el tiempo (``.xtc``, ``.trr``, ``.dcd``).
            Si no se pasa, se usan las del propio archivo de topología, que
            para un PDB multi-modelo son todos sus modelos.

    Raises:
        ErrorDeAnalisis: si falta un archivo o MDAnalysis no puede leerlo.
    """
    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover - depende del extra `md`
        raise ErrorDeAnalisis(
            "MDAnalysis no está instalado. Instalá el extra: "
            'uv pip install -e ".[md]"'
        ) from exc

    topologia = Path(topologia)
    if not topologia.is_file():
        raise ErrorDeAnalisis(f"No existe la topología: {topologia}")
    if trayectoria is not None:
        trayectoria = Path(trayectoria)
        if not trayectoria.is_file():
            raise ErrorDeAnalisis(f"No existe la trayectoria: {trayectoria}")

    try:
        with warnings.catch_warnings():
            # MDAnalysis avisa de cosas normales en archivos de GROMACS
            # (falta de masas, de elementos, de información de caja).
            warnings.simplefilter("ignore")
            universo = (
                mda.Universe(str(topologia), str(trayectoria))
                if trayectoria
                else mda.Universe(str(topologia))
            )
    except Exception as exc:  # noqa: BLE001 - MDAnalysis lanza de todo
        raise ErrorDeAnalisis(
            f"No se pudo leer {topologia.name}"
            + (f" con {trayectoria.name}" if trayectoria else "")
            + f": {exc}"
        ) from exc

    if len(universo.atoms) == 0:
        raise ErrorDeAnalisis(f"{topologia.name} no tiene átomos")

    logger.info(
        "Trayectoria: %d cuadros, %d átomos, %d residuos",
        len(universo.trajectory),
        len(universo.atoms),
        len(universo.residues),
    )
    return universo


def alinear(universo, seleccion: str = SELECCION_ESQUELETO):
    """Superpone todos los cuadros contra el primero, en memoria.

    Es un paso obligatorio antes del RMSF, no una optimización: una proteína
    que rota o se traslada dentro de la caja produce un RMSF alto en todos
    los residuos por igual, aunque internamente no se mueva nada. Alinear
    saca ese movimiento de cuerpo rígido y deja solo la flexibilidad interna.

    El RMSD no necesita esto porque superpone cada cuadro por su cuenta.
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    referencia = mda.Universe(universo.filename)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        align.AlignTraj(
            universo, referencia, select=seleccion, in_memory=True
        ).run()
    return universo


def calcular_rmsd(
    universo, seleccion: str = SELECCION_ESQUELETO
) -> SerieTemporal:
    """RMSD contra el primer cuadro, con superposición óptima.

    La superposición es lo que hace que el número signifique "cambió de
    forma" y no "se movió dentro de la caja".
    """
    from MDAnalysis.analysis import rms

    if len(universo.select_atoms(seleccion)) == 0:
        raise ErrorDeAnalisis(
            f"La selección '{seleccion}' no tiene átomos. "
            "¿La topología incluye la proteína?"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analisis = rms.RMSD(universo, universo, select=seleccion).run()

    # Columnas: cuadro, tiempo (ps), RMSD (Å).
    datos = analisis.results.rmsd
    return SerieTemporal(
        nombre="RMSD",
        unidad="Å",
        tiempos_ps=[float(f[1]) for f in datos],
        valores=[float(f[2]) for f in datos],
    )


def calcular_rmsf(
    universo, seleccion: str = SELECCION_ESQUELETO
) -> PerfilPorResiduo:
    """RMSF por residuo sobre la trayectoria **ya alineada**.

    Se asume que :func:`alinear` corrió antes. Si no, el resultado incluye el
    movimiento de cuerpo rígido y no significa lo que parece.

    MDAnalysis devuelve un RMSF **por átomo**. Con la selección por defecto
    hay un carbono alfa por residuo y las dos cosas coinciden, pero con
    cualquier otra (``backbone`` son cuatro átomos por residuo) habría varios
    valores para el mismo número de residuo: el perfil repetiría residuos y
    ``mas_moviles`` devolvería el mismo tres veces. Se promedia por residuo
    para que el resultado sea de verdad lo que dice ser.
    """
    from MDAnalysis.analysis import rms

    atomos = universo.select_atoms(seleccion)
    if len(atomos) == 0:
        raise ErrorDeAnalisis(f"La selección '{seleccion}' no tiene átomos")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analisis = rms.RMSF(atomos).run()

    por_residuo: dict[int, list[float]] = {}
    for atomo, valor in zip(atomos, analisis.results.rmsf):
        por_residuo.setdefault(int(atomo.resid), []).append(float(valor))

    residuos = sorted(por_residuo)
    return PerfilPorResiduo(
        nombre="RMSF",
        unidad="Å",
        residuos=residuos,
        valores=[mean(por_residuo[r]) for r in residuos],
    )


def calcular_radio_de_giro(
    universo, seleccion: str = SELECCION_PROTEINA
) -> SerieTemporal:
    """Radio de giro cuadro a cuadro: qué tan compacta está la proteína."""
    atomos = universo.select_atoms(seleccion)
    if len(atomos) == 0:
        raise ErrorDeAnalisis(f"La selección '{seleccion}' no tiene átomos")

    tiempos: list[float] = []
    valores: list[float] = []
    with warnings.catch_warnings():
        # Un PDB multi-modelo no declara paso de tiempo; MDAnalysis asume
        # 1 ps y avisa en cada cuadro. El aviso se da una sola vez, arriba.
        warnings.simplefilter("ignore")
        for cuadro in universo.trajectory:
            tiempos.append(float(cuadro.time))
            valores.append(float(atomos.radius_of_gyration()))

    return SerieTemporal(
        nombre="Radio de giro", unidad="Å", tiempos_ps=tiempos, valores=valores
    )


def calcular_sasa(
    universo, seleccion: str = SELECCION_PROTEINA, paso: int = 1
) -> SerieTemporal:
    """Superficie accesible al solvente, por el algoritmo de Shrake-Rupley.

    **MDAnalysis no calcula SASA**: no tiene módulo de superficie. Se usa la
    implementación de Biopython (``Bio.PDB.SASA.ShrakeRupley``), que ya es
    dependencia del proyecto por la Fase 1.

    Es la medida más cara de las cinco —rueda una esfera de prueba sobre cada
    átomo—, así que ``paso`` permite analizar uno de cada N cuadros. Para una
    trayectoria de producción con miles de cuadros es la diferencia entre
    minutos y horas.

    Args:
        universo: trayectoria abierta.
        seleccion: qué átomos; por defecto la proteína sola, porque incluir el
            agua haría que el número no signifique nada.
        paso: analizar uno de cada ``paso`` cuadros.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    atomos = universo.select_atoms(seleccion)
    if len(atomos) == 0:
        raise ErrorDeAnalisis(f"La selección '{seleccion}' no tiene átomos")
    if paso < 1:
        raise ErrorDeAnalisis(f"El paso tiene que ser >= 1, no {paso}")

    calculador = ShrakeRupley()
    parser = PDBParser(QUIET=True)
    tiempos: list[float] = []
    valores: list[float] = []

    with tempfile.TemporaryDirectory(prefix="pdpipe_sasa_") as tmp:
        # Se escribe cada cuadro y se lo vuelve a leer con Biopython en vez de
        # traducir coordenadas a mano: el orden de átomos entre las dos
        # bibliotecas tiene que coincidir exactamente, y un desfase silencioso
        # daría una SASA plausible pero falsa.
        cuadro_pdb = Path(tmp) / "cuadro.pdb"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for cuadro in universo.trajectory[::paso]:
                atomos.write(str(cuadro_pdb))
                estructura = parser.get_structure("cuadro", str(cuadro_pdb))
                calculador.compute(estructura, level="S")
                tiempos.append(float(cuadro.time))
                valores.append(float(estructura.sasa))

    return SerieTemporal(
        nombre="SASA", unidad="Å²", tiempos_ps=tiempos, valores=valores
    )


def tiene_hidrogenos(universo) -> bool:
    """Indica si la estructura trae hidrógenos explícitos.

    Las estructuras de rayos X casi nunca los tienen: el cristal no los
    resuelve. Los agrega ``pdb2gmx`` al preparar el sistema, así que una
    trayectoria de GROMACS sí debería traerlos.
    """
    try:
        if len(universo.select_atoms("element H")) > 0:
            return True
    except Exception:  # noqa: BLE001 - sin columna de elemento, se prueba por nombre
        pass
    return len(universo.select_atoms("name H*")) > 0


def calcular_puentes_de_hidrogeno(
    universo,
    seleccion: str = SELECCION_PROTEINA,
    donantes: str | None = None,
    hidrogenos: str | None = None,
    aceptores: str | None = None,
) -> SerieTemporal:
    """Cuenta puentes de hidrógeno **dentro de la proteína**, cuadro a cuadro.

    Criterios geométricos por defecto de MDAnalysis: donante y aceptor a
    menos de 3.0 Å y ángulo D-H···A mayor a 150°.

    La restricción a la proteína no es un detalle: en un sistema solvatado el
    agua aporta órdenes de magnitud más puentes que la proteína. Contando
    todo, la ubiquitina con sus ~8.500 aguas da unos 8.700 puentes, de los
    cuales apenas medio centenar son suyos. Ese número no sirve para lo que
    se lo quiere: si la proteína perdiera toda su estructura secundaria, la
    caída de ~50 puentes sería indistinguible del ruido del solvente.

    Args:
        universo: trayectoria abierta.
        seleccion: a qué átomos limitar el conteo. Por defecto la proteína.
            Con ``"all"`` cuenta todo el sistema, agua incluida.
        donantes, hidrogenos, aceptores: selecciones explícitas, para los
            casos en que no haya cargas o se quiera otro criterio. Si se
            pasan, ``seleccion`` se ignora para ese rol.

    Raises:
        ErrorDeAnalisis: si la estructura no tiene hidrógenos explícitos. Sin
            ellos el resultado sería cero puentes en todos los cuadros, que
            se leería como "esta proteína no tiene puentes" en vez de "este
            archivo no permite contarlos".
    """
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
        HydrogenBondAnalysis,
    )

    if not tiene_hidrogenos(universo):
        raise ErrorDeAnalisis(
            "La estructura no tiene hidrógenos explícitos, así que no se "
            "pueden contar puentes de hidrógeno. Es normal en estructuras de "
            "rayos X; los agrega pdb2gmx al preparar el sistema, y una "
            "trayectoria de GROMACS sí los trae."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if hidrogenos and aceptores:
                # Con las selecciones dadas a mano no hace falta adivinar
                # nada, y por lo tanto tampoco hacen falta las cargas. Es la
                # única vía para una topología que no las trae.
                analisis = HydrogenBondAnalysis(
                    universo,
                    donors_sel=donantes,
                    hydrogens_sel=hidrogenos,
                    acceptors_sel=aceptores,
                )
            else:
                # El constructor sin selecciones adivina sobre todo el
                # sistema, agua incluida. Se lo deja construir así —necesita
                # las cargas— y recién después se acotan las selecciones a
                # `seleccion`, antes de correr.
                analisis = HydrogenBondAnalysis(universo)
                analisis.hydrogens_sel = hidrogenos or analisis.guess_hydrogens(seleccion)
                analisis.acceptors_sel = aceptores or analisis.guess_acceptors(seleccion)
                if donantes:
                    analisis.donors_sel = donantes

            analisis.run()
    except Exception as exc:  # noqa: BLE001 - MDAnalysis lanza NoDataError y otros
        if "charge" in str(exc).lower():
            raise ErrorDeAnalisis(
                "La topología no trae cargas parciales, y sin ellas MDAnalysis "
                "no puede deducir qué átomos son donantes.\n\n"
                "Si la trayectoria viene de GROMACS, pasá el archivo .tpr como "
                "topología en vez del .gro: el .gro tiene solo coordenadas, "
                "mientras que el .tpr compila también el campo de fuerza y sus "
                "cargas.\n\n"
                "    pdpipe md-analyze --topology md/prod.tpr --trajectory md/prod.xtc\n\n"
                "La otra salida es indicar las selecciones a mano con los "
                "parámetros donantes, hidrogenos y aceptores."
            ) from exc
        raise ErrorDeAnalisis(
            f"No se pudieron contar los puentes de hidrógeno: {exc}"
        ) from exc

    conteos = analisis.count_by_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tiempos = [float(c.time) for c in universo.trajectory]

    return SerieTemporal(
        nombre="Puentes de hidrógeno",
        unidad="conteo",
        tiempos_ps=tiempos[: len(conteos)],
        valores=[float(v) for v in conteos],
    )


__all__ = [
    "SELECCION_ESQUELETO",
    "SELECCION_PROTEINA",
    "ErrorDeAnalisis",
    "alinear",
    "calcular_puentes_de_hidrogeno",
    "calcular_radio_de_giro",
    "calcular_rmsd",
    "calcular_rmsf",
    "calcular_sasa",
    "cargar",
    "tiene_hidrogenos",
]
