"""Las cuatro etapas de simulación (Fase 3, parte B).

Cada etapa son dos llamadas a GROMACS: ``grompp`` compila los parámetros, la
topología y las coordenadas en un ``.tpr``, y ``mdrun`` lo ejecuta.

El encadenado importa y no es arbitrario:

* La **minimización** arranca del sistema recién solvatado.
* El **NVT** arranca de la estructura minimizada y **genera** velocidades.
* El **NPT** continúa del NVT y hereda sus velocidades por el checkpoint. Sin
  el ``-t``, empezaría de cero y se perdería la equilibración térmica.
* La **producción** continúa del NPT, ya sin restricciones de posición.

Las dos etapas con restricciones (NVT y NPT) necesitan además ``-r``, la
estructura de referencia contra la que se sujetan los átomos.
"""

from __future__ import annotations

import time
from pathlib import Path

from pdpipe.config import Config
from pdpipe.phase3_md.gromacs import ClienteGromacs, ErrorDeGromacs
from pdpipe.phase3_md.mdp import pasos
from pdpipe.phase3_md.models import (
    EtapaSimulacion,
    ResultadoSimulacion,
    SistemaPreparado,
)
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)


def correr_etapa(
    gmx: ClienteGromacs,
    sistema: SistemaPreparado,
    etapa: str,
    entrada_gro: str,
    prefijo: str,
    referencia: str | None = None,
    checkpoint: str | None = None,
    threads: int = 0,
    reanudar: bool = True,
) -> EtapaSimulacion:
    """Compila y ejecuta una etapa.

    Con ``reanudar``, una etapa que ya terminó no se vuelve a correr y una que
    quedó a medias se retoma desde su checkpoint. Importa más de lo que parece:
    una producción de 2 ns en CPU son horas, y sin esto una sesión de Colab
    cortada obliga a rehacer también la minimización y las dos equilibraciones,
    que ya estaban bien.

    Args:
        gmx: cliente de GROMACS.
        sistema: sistema preparado en la parte A.
        etapa: clave del ``.mdp`` (``minim``, ``nvt``, ``npt``, ``prod``).
        entrada_gro: coordenadas de partida, relativas al directorio.
        prefijo: nombre base de las salidas (``-deffnm``).
        referencia: estructura de referencia para las restricciones de
            posición. Obligatoria en NVT y NPT.
        checkpoint: ``.cpt`` del que continuar, para heredar velocidades.
        threads: hilos de CPU; 0 deja que GROMACS decida.

    Raises:
        ErrorDeGromacs: si grompp o mdrun fallan.
    """
    directorio = sistema.directorio
    mdp = sistema.mdp.get(etapa)
    if mdp is None:
        raise ErrorDeGromacs(f"No hay archivo .mdp para la etapa '{etapa}'")

    def _si_existe(nombre: str) -> Path | None:
        ruta = directorio / nombre
        return ruta if ruta.is_file() else None

    # Una etapa terminada deja su .gro final. Si está, no hay nada que rehacer.
    terminada = _si_existe(f"{prefijo}.gro")
    if reanudar and terminada:
        logger.info(
            "Etapa %s ya estaba hecha, se reutiliza %s", etapa, terminada.name
        )
        return EtapaSimulacion(
            nombre=etapa,
            estructura=terminada,
            trayectoria=_si_existe(f"{prefijo}.xtc"),
            tpr=_si_existe(f"{prefijo}.tpr"),
            energia=_si_existe(f"{prefijo}.edr"),
            log=_si_existe(f"{prefijo}.log"),
            reutilizada=True,
        )

    # Un .cpt sin .gro significa que mdrun se cortó a mitad de camino.
    a_medias = _si_existe(f"{prefijo}.cpt") if reanudar else None

    argumentos = [
        "-f", mdp.name,
        "-c", entrada_gro,
        "-p", sistema.topologia.name,
        "-o", f"{prefijo}.tpr",
    ]
    if referencia:
        argumentos += ["-r", referencia]
    if checkpoint:
        argumentos += ["-t", checkpoint]

    gmx.correr(
        "grompp", *argumentos, directorio=directorio, etiqueta=f"grompp ({etapa})"
    )

    comando_mdrun = ["-deffnm", prefijo]
    if threads > 0:
        comando_mdrun += ["-nt", str(threads)]
    if a_medias:
        # GROMACS retoma desde el paso que quedó registrado en el checkpoint,
        # no desde cero, y continúa escribiendo sobre la misma trayectoria.
        comando_mdrun += ["-cpi", a_medias.name]
        logger.info("Retomando %s desde %s", etapa, a_medias.name)

    arranque = time.monotonic()
    gmx.correr(
        "mdrun", *comando_mdrun, directorio=directorio, etiqueta=f"mdrun ({etapa})"
    )
    segundos = time.monotonic() - arranque

    logger.info("Etapa %s terminada en %.1f s", etapa, segundos)
    return EtapaSimulacion(
        nombre=etapa,
        estructura=_si_existe(f"{prefijo}.gro"),
        trayectoria=_si_existe(f"{prefijo}.xtc"),
        tpr=_si_existe(f"{prefijo}.tpr"),
        energia=_si_existe(f"{prefijo}.edr"),
        log=_si_existe(f"{prefijo}.log"),
        segundos=round(segundos, 2),
    )


def simular(
    config: Config,
    sistema: SistemaPreparado,
    ns: float | None = None,
    cliente: ClienteGromacs | None = None,
    reanudar: bool = True,
) -> ResultadoSimulacion:
    """Corre minimización, NVT, NPT y producción sobre un sistema preparado.

    Args:
        config: configuración del pipeline.
        sistema: salida de :func:`~pdpipe.phase3_md.preparation.preparar_sistema`.
        ns: sobrescribe ``md.production_ns``. Cambiar la duración obliga a
            reescribir el ``.mdp`` de producción, así que se hace acá.
        cliente: cliente de GROMACS; se construye del config si no se pasa.
        reanudar: reutiliza las etapas ya terminadas y retoma las que quedaron
            a medias. Poner ``False`` fuerza a rehacer todo desde cero.

    Raises:
        GromacsNoDisponible: si falta ``gmx``.
        ErrorDeGromacs: si alguna etapa falla.
    """
    md = config.md
    gmx = cliente or ClienteGromacs(md.gromacs_bin)
    gmx.verificar_disponible()

    if ns is not None and ns != md.production_ns:
        _reescribir_produccion(config, sistema, ns)
    duracion_ns = ns if ns is not None else md.production_ns

    etapas: list[EtapaSimulacion] = []

    # 1. Minimización: sin velocidades, sin restricciones.
    etapas.append(
        correr_etapa(
            gmx, sistema, "minim",
            entrada_gro=sistema.estructura.name,
            prefijo="em",
            threads=md.threads,
            reanudar=reanudar,
        )
    )

    # 2. NVT: genera velocidades; la referencia de las restricciones es la
    # estructura minimizada.
    etapas.append(
        correr_etapa(
            gmx, sistema, "nvt",
            entrada_gro="em.gro",
            prefijo="nvt",
            referencia="em.gro",
            threads=md.threads,
            reanudar=reanudar,
        )
    )

    # 3. NPT: continúa del checkpoint del NVT para no perder la temperatura
    # ya equilibrada.
    etapas.append(
        correr_etapa(
            gmx, sistema, "npt",
            entrada_gro="nvt.gro",
            prefijo="npt",
            referencia="nvt.gro",
            checkpoint="nvt.cpt",
            threads=md.threads,
            reanudar=reanudar,
        )
    )

    # 4. Producción: sin -r, porque el .mdp ya no define POSRES.
    etapas.append(
        correr_etapa(
            gmx, sistema, "prod",
            entrada_gro="npt.gro",
            prefijo="prod",
            checkpoint="npt.cpt",
            threads=md.threads,
            reanudar=reanudar,
        )
    )

    reutilizadas = [e.nombre for e in etapas if e.reutilizada]
    if reutilizadas:
        logger.info(
            "Etapas reutilizadas de una corrida anterior: %s", ", ".join(reutilizadas)
        )

    etapas = _anotar_duraciones(etapas, config, duracion_ns)
    produccion = etapas[-1]

    resultado = ResultadoSimulacion(
        sistema=sistema,
        etapas=etapas,
        trayectoria=produccion.trayectoria,
        estructura_final=produccion.estructura,
        ns_simulados=duracion_ns,
        version_gromacs=gmx.version(),
    )

    logger.info(
        "Simulación completa: %.2f ns en %.0f s (%s)",
        duracion_ns,
        resultado.duracion_total_s,
        produccion.trayectoria.name if produccion.trayectoria else "sin trayectoria",
    )
    return resultado


def _reescribir_produccion(
    config: Config, sistema: SistemaPreparado, ns: float
) -> None:
    """Regenera ``prod.mdp`` con otra duración.

    La duración vive dentro del ``.mdp`` como número de pasos, así que pedir
    otra cantidad de nanosegundos no es un argumento de ``mdrun``: hay que
    reescribir el archivo antes de compilarlo.
    """
    from pdpipe.phase3_md.mdp import escribir_mdp

    md = config.md.model_copy(update={"production_ns": ns})
    sistema.mdp["prod"] = escribir_mdp("prod", md, sistema.directorio, seed=config.seed)
    logger.info("Producción ajustada a %.3f ns", ns)


def _anotar_duraciones(
    etapas: list[EtapaSimulacion], config: Config, ns: float
) -> list[EtapaSimulacion]:
    """Completa pasos y picosegundos de cada etapa, para el manifiesto."""
    md = config.md
    ps_por_etapa = {
        "minim": 0.0,  # la minimización no avanza en el tiempo
        "nvt": md.nvt_ps,
        "npt": md.npt_ps,
        "prod": ns * 1000.0,
    }
    resultado = []
    for etapa in etapas:
        ps = ps_por_etapa.get(etapa.nombre, 0.0)
        n = md.minimization_steps if etapa.nombre == "minim" else pasos(ps, md.timestep_fs)
        resultado.append(etapa.model_copy(update={"pasos": n, "ps_simulados": ps}))
    return resultado


__all__ = ["correr_etapa", "simular"]
