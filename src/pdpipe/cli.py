"""Interfaz de línea de comandos del pipeline (transversal a las cinco fases).

Un comando por fase, más ``info`` para diagnosticar el entorno. En el Hito 0
los comandos de las fases validan sus argumentos y el ``config.yaml``, pero
todavía no ejecutan ciencia: informan en qué hito se implementan y terminan
con código de salida 2.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Iterator, NoReturn, Optional

import shutil

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pdpipe import __version__
from pdpipe.config import Config, ConfigError, load_config
from pdpipe.utils.logging import get_logger, setup_logging
from pdpipe.utils.manifest import RunManifest, collect_software_versions

# Código de salida para un comando que todavía no está implementado.
# Se distingue del 1 (error real) para que un script pueda diferenciarlos.
EXIT_PENDING = 2
EXIT_ERROR = 1

console = Console()
logger = get_logger(__name__)


@contextmanager
def _manifiesto_ante_fallas(manifest: RunManifest, directorio: Path) -> Iterator[None]:
    """Garantiza que una corrida cortada deje su ``run_manifest.json``.

    Los errores previstos los maneja cada comando con su propio mensaje; esto
    cubre el resto: un Ctrl+C o una excepción inesperada. Sin esto quedaba la
    carpeta de la corrida con un ``run.log`` vacío y sin rastro de qué pasó.
    La excepción se vuelve a lanzar, así que el traceback se sigue viendo.
    """
    try:
        yield
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        logger.warning("Corrida interrumpida por el usuario")
        manifest.finish("interrumpida", error="Cortada con Ctrl+C")
        manifest.save(directorio)
        raise
    except Exception as exc:
        logger.exception("Error inesperado")
        manifest.finish("error", error=f"{type(exc).__name__}: {exc}")
        manifest.save(directorio)
        raise

app = typer.Typer(
    name="pdpipe",
    help="Pipeline reproducible de diseño computacional de proteínas (FP-UNA).",
    add_completion=False,
    no_args_is_help=True,
)


class State:
    """Opciones globales, compartidas por todos los subcomandos."""

    def __init__(self) -> None:
        self.config: Config | None = None
        self.config_path: Path | None = None
        self.log_level: str = "INFO"


state = State()


def _pending(comando: str, hito: int, detalle: str = "") -> NoReturn:
    """Informa que un comando todavía no está implementado y corta la ejecución."""
    cuerpo = (
        f"El comando [bold]{comando}[/bold] se implementa en el "
        f"[bold]Hito {hito}[/bold]."
    )
    if detalle:
        cuerpo += f"\n\n{detalle}"
    console.print(
        Panel(
            cuerpo,
            title="[yellow]Pendiente[/yellow]",
            border_style="yellow",
            expand=False,
        )
    )
    raise typer.Exit(code=EXIT_PENDING)


def _require_config() -> Config:
    """Devuelve el config ya cargado, o corta con un error legible."""
    if state.config is None:
        console.print(
            "[red]Error:[/red] no hay configuración cargada. "
            "Pasá --config con la ruta a tu config.yaml."
        )
        raise typer.Exit(code=EXIT_ERROR)
    return state.config


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"pdpipe {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Ruta al config.yaml. Por defecto lo busca hacia arriba desde el cwd.",
            exists=False,
            dir_okay=False,
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="DEBUG, INFO, WARNING o ERROR."),
    ] = "INFO",
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True,
                     help="Muestra la versión y sale."),
    ] = False,
) -> None:
    """Opciones comunes a todos los comandos."""
    state.log_level = log_level.upper()
    setup_logging(level=state.log_level)

    try:
        state.config = load_config(config)
        state.config_path = state.config.source_path
        logger.debug("Configuración cargada desde %s", state.config_path)
    except ConfigError as exc:
        # No es fatal todavía: `info` y `--help` funcionan igual sin config.
        state.config = None
        logger.debug("No se pudo cargar la configuración: %s", exc)
        if config is not None:
            # Si el usuario pidió un archivo explícito, no lo ignoramos.
            console.print(f"[red]Error de configuración:[/red] {exc}")
            raise typer.Exit(code=EXIT_ERROR) from exc


# ---------------------------------------------------------------------------
# Diagnóstico del entorno
# ---------------------------------------------------------------------------


@app.command()
def info(
    save: Annotated[
        bool,
        typer.Option("--save", help="Escribe un run_manifest.json en runs/<run_id>/."),
    ] = False,
) -> None:
    """Muestra el estado del entorno: versiones, config y herramientas externas."""
    cfg = state.config
    gromacs_bin = cfg.md.gromacs_bin if cfg else "gmx"
    versiones = collect_software_versions(gromacs_bin=gromacs_bin)

    entorno = Table(title="Entorno", show_header=False, expand=False)
    entorno.add_column("clave", style="cyan")
    entorno.add_column("valor")
    entorno.add_row("pdpipe", __version__)
    entorno.add_row("Python", versiones["python"])
    entorno.add_row("Plataforma", versiones["platform"])
    entorno.add_row(
        "config.yaml",
        str(state.config_path) if state.config_path else "[yellow]no encontrado[/yellow]",
    )
    entorno.add_row("Semilla", str(cfg.seed) if cfg else "-")
    console.print(entorno)

    paquetes = Table(title="Dependencias", expand=False)
    paquetes.add_column("Paquete", style="cyan")
    paquetes.add_column("Versión")
    for nombre, version in versiones["packages"].items():
        estado = version or "[yellow]no instalado[/yellow]"
        paquetes.add_row(nombre, estado)
    console.print(paquetes)

    externas = Table(title="Herramientas externas", expand=False)
    externas.add_column("Herramienta", style="cyan")
    externas.add_column("Estado")
    gmx_version = versiones.get("gromacs")
    externas.add_row(
        f"GROMACS ({gromacs_bin})",
        gmx_version or "[yellow]no disponible — necesario desde el Hito 3[/yellow]",
    )
    for herramienta, hito in (("dssp", 4), ("mkdssp", 4), ("cd-hit", 4)):
        ruta = shutil.which(herramienta)
        externas.add_row(
            herramienta,
            ruta or f"[dim]no disponible — opcional, Hito {hito}[/dim]",
        )
    console.print(externas)

    if save:
        cfg = _require_config()
        manifest = RunManifest.start(
            command="info",
            seed=cfg.seed,
            config=cfg.to_dict(),
            gromacs_bin=cfg.md.gromacs_bin,
        )
        manifest.add_note("Diagnóstico de entorno, sin cómputo científico.")
        manifest.finish("ok")
        destino = Path(cfg.resolved_paths()["runs"]) / manifest.run_id
        ruta = manifest.save(destino)
        console.print(f"\nManifiesto escrito en [green]{ruta}[/green]")


# ---------------------------------------------------------------------------
# Fase 1 — Recolección de datos
# ---------------------------------------------------------------------------


@app.command()
def fetch(
    pdb_id: Annotated[
        Optional[list[str]],
        typer.Option("--pdb-id", help="Código PDB a descargar. Repetible."),
    ] = None,
    uniprot: Annotated[
        Optional[list[str]],
        typer.Option("--uniprot", help="Accesión UniProt a descargar. Repetible."),
    ] = None,
    formato: Annotated[
        str,
        typer.Option("--formato", help="Formato de coordenadas: pdb o cif."),
    ] = "pdb",
    force: Annotated[
        bool,
        typer.Option("--force", help="Vuelve a descargar aunque ya esté en caché."),
    ] = False,
    search: Annotated[
        Optional[int],
        typer.Option(
            "--search",
            help="Busca en el RCSB con los criterios del config y baja hasta N entradas.",
        ),
    ] = None,
) -> None:
    """Descarga estructuras del RCSB PDB y anotaciones de UniProt (Fase 1).

    Con ``--search N`` consulta la Search API con los criterios de ``data``
    del config (resolución, organismos, método, longitud) y baja las primeras
    N entradas que los cumplan. Es la forma de armar un conjunto grande, como
    el que necesita el dataset de la Fase 4.
    """
    from pdpipe.phase1_data import PDBIDInvalido
    from pdpipe.phase1_data import pipeline as fase1

    cfg = _require_config()
    if not pdb_id and not uniprot and not search:
        console.print(
            "[red]Error:[/red] indicá al menos un --pdb-id, un --uniprot o --search N."
        )
        raise typer.Exit(code=EXIT_ERROR)

    if search:
        encontrados = _buscar_en_rcsb(cfg, search)
        if not encontrados:
            console.print(
                "[yellow]La búsqueda no devolvió entradas.[/yellow] "
                "Revisá los criterios de `data` en el config.yaml."
            )
            raise typer.Exit(code=EXIT_ERROR)
        console.print(f"La búsqueda devolvió [green]{len(encontrados)}[/green] entradas.")
        pdb_id = list(dict.fromkeys([*(pdb_id or []), *encontrados]))

    manifest = RunManifest.start(
        command="fetch",
        seed=cfg.seed,
        config=cfg.to_dict(),
        params={
            "pdb_id": list(pdb_id or []),
            "uniprot": list(uniprot or []),
            "formato": formato,
            "force": force,
        },
    )
    directorio = Path(cfg.resolved_paths()["runs"]) / manifest.run_id
    if cfg.logging.to_file:
        setup_logging(level=state.log_level, log_file=directorio / "run.log")

    with _manifiesto_ante_fallas(manifest, directorio):
        try:
            resultado = fase1.fetch(
                config=cfg,
                pdb_ids=list(pdb_id or []),
                uniprot_ids=list(uniprot or []),
                formato=formato,
                forzar=force,
                manifest=manifest,
            )
        except PDBIDInvalido as exc:
            console.print(f"[red]Error:[/red] {exc}")
            manifest.finish("error", error=str(exc))
            manifest.save(directorio)
            raise typer.Exit(code=EXIT_ERROR) from exc

    manifest.finish("ok")
    manifest.save(directorio)

    tabla = Table(title="fetch", expand=False)
    tabla.add_column("Resultado", style="cyan")
    tabla.add_column("Cantidad", justify="right")
    tabla.add_row("Descargadas", str(len(resultado.descargadas)))
    tabla.add_row("Rechazadas (bioseguridad)", str(len(resultado.rechazadas)))
    tabla.add_row("Fallidas", str(len(resultado.fallidas)))
    console.print(tabla)

    for pid, motivo in resultado.rechazadas:
        console.print(f"  [yellow]{pid}[/yellow] rechazada: {motivo}")
    for pid, error in resultado.fallidas:
        console.print(f"  [red]{pid}[/red] falló: {error}")

    console.print(f"Manifiesto: [green]{directorio / 'run_manifest.json'}[/green]")

    if resultado.fallidas:
        raise typer.Exit(code=EXIT_ERROR)


#: Identidad de secuencia a la que se agrupa la búsqueda. 30% es el corte
#: habitual para considerar dos proteínas no redundantes.
IDENTIDAD_BUSQUEDA = 30

#: Tope de representantes a listar antes de muestrear. La Search API devuelve
#: en orden alfabético y los códigos PDB son casi cronológicos, así que tomar
#: los primeros N daría puras estructuras de los años noventa. Se listan todos
#: los grupos (~21.000, cinco páginas, unos 15 s) y se muestrea de ahí.
TOPE_CANDIDATOS = 30_000


def _buscar_en_rcsb(cfg, limite: int) -> list[str]:
    """Consulta la Search API del RCSB con los criterios del config.

    Dos cosas que no son obvias y sin las cuales el conjunto no sirve para
    entrenar nada:

    * **Se agrupa por identidad de secuencia.** Sin eso la búsqueda devuelve
      decenas de mutantes puntuales de la misma proteína —101M, 103M y 104M
      son todos mioglobina de cachalote— y el conjunto tiene muchas filas
      pero casi ninguna diversidad.
    * **Se muestrea al azar sobre el listado completo.** El orden de la API es
      alfabético y los códigos PDB son casi cronológicos, así que tomar los
      primeros N sesga hacia las estructuras más viejas: 500 entradas serían
      todas de los años noventa. Se listan los ~21.000 grupos del PDB entero y
      se sortean los que hagan falta. El muestreo usa la semilla del config,
      así que la selección es reproducible.
    """
    import random

    from pdpipe.phase1_data.http_client import ClienteHTTP
    from pdpipe.phase1_data.rcsb import ClienteRCSB

    rutas = cfg.resolved_paths()
    http = ClienteHTTP(
        timeout_s=cfg.data.http.timeout_s,
        max_retries=cfg.data.http.max_retries,
        cache_dir=rutas["data_raw"],
        usar_cache=cfg.data.http.cache,
    )
    try:
        candidatos = ClienteRCSB(http).buscar(
            resolucion_max=cfg.data.resolution_max,
            organismos=list(cfg.data.organisms) or None,
            metodos=[m.value if hasattr(m, "value") else str(m)
                     for m in cfg.data.experimental_methods] or None,
            longitud_min=cfg.data.length_min,
            longitud_max=cfg.data.length_max,
            limite=TOPE_CANDIDATOS,
            identidad_max=IDENTIDAD_BUSQUEDA,
        )
    finally:
        http.close()

    if len(candidatos) <= limite:
        return candidatos

    sorteo = random.Random(cfg.seed)
    return sorted(sorteo.sample(candidatos, limite))


@app.command()
def curate(
    resolution_max: Annotated[
        Optional[float],
        typer.Option("--resolution-max", help="Resolución máxima en angstroms."),
    ] = None,
    organism: Annotated[
        Optional[list[str]],
        typer.Option("--organism", help="Organismo de origen. Repetible."),
    ] = None,
    method: Annotated[
        Optional[list[str]],
        typer.Option("--method", help="Método experimental aceptado. Repetible."),
    ] = None,
    length_min: Annotated[
        Optional[int], typer.Option("--length-min", help="Longitud mínima en residuos.")
    ] = None,
    length_max: Annotated[
        Optional[int], typer.Option("--length-max", help="Longitud máxima en residuos.")
    ] = None,
) -> None:
    """Filtra lo descargado y marca qué queda curado en la base (Fase 1).

    No usa la red: trabaja sobre lo que ya bajó `fetch`, así que se puede
    repetir con distintos criterios sin volver a descargar nada.
    """
    from pdpipe.phase1_data import pipeline as fase1

    cfg = _require_config()

    manifest = RunManifest.start(
        command="curate",
        seed=cfg.seed,
        config=cfg.to_dict(),
    )
    directorio = Path(cfg.resolved_paths()["runs"]) / manifest.run_id
    if cfg.logging.to_file:
        setup_logging(level=state.log_level, log_file=directorio / "run.log")

    resultado = fase1.curate(
        config=cfg,
        resolucion_max=resolution_max,
        organismos=list(organism) if organism else None,
        metodos=list(method) if method else None,
        longitud_min=length_min,
        longitud_max=length_max,
        manifest=manifest,
    )
    manifest.finish("ok")
    manifest.save(directorio)

    tabla = Table(title="curate", expand=False)
    tabla.add_column("Resultado", style="cyan")
    tabla.add_column("Cantidad", justify="right")
    tabla.add_row("Aceptadas", str(len(resultado.aceptadas)))
    tabla.add_row("Rechazadas", str(len(resultado.rechazadas)))
    console.print(tabla)

    for pid, motivo in resultado.rechazadas:
        console.print(f"  [yellow]{pid}[/yellow]: {motivo}")

    console.print(f"Manifiesto: [green]{directorio / 'run_manifest.json'}[/green]")


@app.command(name="db-stats")
def db_stats() -> None:
    """Muestra el contenido de la base SQLite local (Fase 1)."""
    from pdpipe.phase1_data import BaseDatos

    cfg = _require_config()
    ruta = Path(cfg.resolved_paths()["database"])
    if not ruta.is_file():
        console.print(
            f"[yellow]La base todavía no existe[/yellow] ({ruta}).\n"
            f"Corré [cyan]pdpipe fetch --pdb-id 1UBQ[/cyan] para crearla."
        )
        raise typer.Exit(code=EXIT_ERROR)

    with BaseDatos(ruta) as db:
        conteos = db.contar()
        proteinas = db.listar_proteinas()

    resumen = Table(title=f"Base: {ruta.name}", expand=False)
    resumen.add_column("Tabla", style="cyan")
    resumen.add_column("Filas", justify="right")
    for clave in ("proteinas", "funciones", "interacciones", "ptms"):
        resumen.add_row(clave, str(conteos[clave]))
    resumen.add_row("[green]curadas[/green]", str(conteos["curadas"]))
    resumen.add_row("[yellow]rechazadas[/yellow]", str(conteos["rechazadas"]))
    resumen.add_row("[dim]sin curar[/dim]", str(conteos["sin_curar"]))
    console.print(resumen)

    if conteos["sin_curar"]:
        console.print(
            f"[dim]{conteos['sin_curar']} proteína(s) sin evaluar. "
            f"Corré [cyan]pdpipe curate[/cyan] para aplicar los criterios.[/dim]"
        )

    if not proteinas:
        return

    detalle = Table(title="Proteínas", expand=False)
    for columna in ("PDB", "UniProt", "Organismo", "Método", "Res. (Å)", "Long.", "Estado"):
        detalle.add_column(columna)
    for p in proteinas:
        estado = (
            "[green]curada[/green]"
            if p.curada
            else f"[yellow]{(p.motivo_rechazo or 'sin curar')[:40]}[/yellow]"
        )
        detalle.add_row(
            p.pdb_id,
            p.uniprot_id or "-",
            (p.organismo or "-")[:25],
            (p.metodo or "-")[:18],
            f"{p.resolucion:.2f}" if p.resolucion is not None else "-",
            str(p.longitud or "-"),
            estado,
        )
    console.print(detalle)


# ---------------------------------------------------------------------------
# Fase 2 — Diseño computacional
# ---------------------------------------------------------------------------


@app.command()
def predict(
    uniprot: Annotated[
        Optional[str], typer.Option("--uniprot", help="Accesión UniProt a modelar.")
    ] = None,
    source: Annotated[
        Optional[str],
        typer.Option("--source", help="alphafold_db, colabfold o esmfold."),
    ] = None,
    formato: Annotated[
        str, typer.Option("--formato", help="Formato del modelo: pdb o cif.")
    ] = "pdb",
    force: Annotated[
        bool, typer.Option("--force", help="Vuelve a descargar aunque esté en caché.")
    ] = False,
) -> None:
    """Obtiene la estructura predicha y reporta el pLDDT por residuo (Fase 2)."""
    from pdpipe.phase2_design import SinModeloPredicho
    from pdpipe.phase2_design import pipeline as fase2
    from pdpipe.phase2_design.pipeline import FuenteNoDisponible
    from pdpipe.phase2_design.plddt import ErrorPLDDT

    cfg = _require_config()
    if uniprot is None:
        console.print("[red]Error:[/red] indicá --uniprot.")
        raise typer.Exit(code=EXIT_ERROR)

    manifest = RunManifest.start(
        command="predict",
        seed=cfg.seed,
        config=cfg.to_dict(),
        params={
            "uniprot": uniprot,
            "source": source or cfg.design.structure_source.value,
            "formato": formato,
        },
    )
    directorio = Path(cfg.resolved_paths()["runs"]) / manifest.run_id
    if cfg.logging.to_file:
        setup_logging(level=state.log_level, log_file=directorio / "run.log")

    try:
        modelo = fase2.predecir(
            config=cfg,
            uniprot_id=uniprot,
            fuente=source,
            formato=formato,
            forzar=force,
            manifest=manifest,
        )
    except FuenteNoDisponible as exc:
        console.print(
            Panel(str(exc), title="[yellow]Fuente no disponible[/yellow]",
                  border_style="yellow", expand=False)
        )
        manifest.finish("error", error=str(exc))
        manifest.save(directorio)
        raise typer.Exit(code=EXIT_PENDING) from exc
    except (SinModeloPredicho, ErrorPLDDT) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        manifest.finish("error", error=str(exc))
        manifest.save(directorio)
        raise typer.Exit(code=EXIT_ERROR) from exc

    manifest.finish("ok")
    manifest.save(directorio)
    _mostrar_modelo(modelo, cfg.design.plddt_min)
    console.print(f"Manifiesto: [green]{directorio / 'run_manifest.json'}[/green]")


def _mostrar_modelo(modelo, plddt_min: float) -> None:
    """Imprime los metadatos del modelo y la distribución de su pLDDT."""
    from pdpipe.phase2_design.models import BandaPLDDT

    ficha = Table(title=f"Modelo predicho — {modelo.uniprot_id}", show_header=False)
    ficha.add_column("clave", style="cyan")
    ficha.add_column("valor")
    ficha.add_row("Entrada", modelo.entry_id or "-")
    ficha.add_row("Fuente", f"{modelo.fuente} v{modelo.version}")
    ficha.add_row("Proteína", modelo.nombre or "-")
    ficha.add_row("Organismo", modelo.organismo or "-")
    ficha.add_row("Archivo", str(modelo.archivo))
    console.print(ficha)

    resumen = modelo.resumen
    if resumen is None:
        return

    color = "green" if resumen.media >= plddt_min else "yellow"
    estadisticos = Table(title="pLDDT", show_header=False)
    estadisticos.add_column("clave", style="cyan")
    estadisticos.add_column("valor")
    estadisticos.add_row("Residuos", str(resumen.n_residuos))
    estadisticos.add_row("Media", f"[{color}]{resumen.media}[/{color}]")
    estadisticos.add_row("Mediana", str(resumen.mediana))
    estadisticos.add_row("Rango", f"{resumen.minimo} – {resumen.maximo}")
    estadisticos.add_row(
        "Confiable (>=70)", f"{resumen.fraccion_confiable:.1%}"
    )
    if modelo.metrica_global_api is not None:
        estadisticos.add_row(
            "[dim]Media según la API[/dim]", f"[dim]{modelo.metrica_global_api}[/dim]"
        )
    console.print(estadisticos)

    bandas = Table(title="Distribución por banda de confianza")
    bandas.add_column("Banda", style="cyan")
    bandas.add_column("Residuos", justify="right")
    bandas.add_column("Fracción", justify="right")
    for banda in BandaPLDDT:
        cantidad = resumen.conteo_por_banda.get(banda.value, 0)
        fraccion = resumen.fraccion_por_banda.get(banda.value, 0.0)
        bandas.add_row(banda.descripcion, str(cantidad), f"{fraccion:.1%}")
    console.print(bandas)

    if resumen.media < plddt_min:
        console.print(
            f"[yellow]Aviso:[/yellow] el pLDDT medio ({resumen.media}) está por "
            f"debajo del mínimo configurado ({plddt_min}). El modelo puede no ser "
            f"confiable para diseñar sobre él."
        )


@app.command()
def design(
    input: Annotated[
        Optional[Path],
        typer.Option("--input", "-i", help="Estructura de partida (.pdb o .cif)."),
    ] = None,
    n_sequences: Annotated[
        Optional[int],
        typer.Option("--n-sequences", "-n", help="Cantidad de variantes a generar."),
    ] = None,
    temperature: Annotated[
        Optional[float],
        typer.Option("--temperature", help="Temperatura de muestreo de ProteinMPNN."),
    ] = None,
) -> None:
    """Genera variantes de secuencia con ProteinMPNN sobre CPU (Fase 2)."""
    from pdpipe.phase2_design import pipeline as fase2
    from pdpipe.phase2_design.designer import DisenadorNoDisponible, ErrorDeDiseno

    cfg = _require_config()
    if input is None:
        console.print("[red]Error:[/red] indicá --input con la estructura de partida.")
        raise typer.Exit(code=EXIT_ERROR)
    if not input.exists():
        console.print(f"[red]Error:[/red] no existe el archivo de entrada: {input}")
        raise typer.Exit(code=EXIT_ERROR)

    n = n_sequences if n_sequences is not None else cfg.design.n_sequences
    t = temperature if temperature is not None else cfg.design.temperature

    manifest = RunManifest.start(
        command="design",
        seed=cfg.seed,
        config=cfg.to_dict(),
        params={
            "input": str(input),
            "n_sequences": n,
            "temperature": t,
            "designer": cfg.design.designer.value,
        },
    )
    directorio = Path(cfg.resolved_paths()["runs"]) / manifest.run_id
    if cfg.logging.to_file:
        setup_logging(level=state.log_level, log_file=directorio / "run.log")

    try:
        resultado = fase2.disenar(
            config=cfg,
            estructura=input,
            n_secuencias=n,
            temperatura=t,
            manifest=manifest,
        )
    except DisenadorNoDisponible as exc:
        # No es un error de los datos sino del entorno: el mensaje dice cómo
        # instalar la herramienta, así que va en panel y con el código de
        # "pendiente", igual que las fuentes de estructura no implementadas.
        console.print(
            Panel(str(exc), title="[yellow]Diseñador no disponible[/yellow]",
                  border_style="yellow", expand=False)
        )
        manifest.finish("error", error=str(exc))
        manifest.save(directorio)
        raise typer.Exit(code=EXIT_PENDING) from exc
    except ErrorDeDiseno as exc:
        console.print(f"[red]Error:[/red] {exc}")
        manifest.finish("error", error=str(exc))
        manifest.save(directorio)
        raise typer.Exit(code=EXIT_ERROR) from exc

    manifest.finish("ok")
    manifest.save(directorio)
    _mostrar_diseno(resultado)
    console.print(f"Manifiesto: [green]{directorio / 'run_manifest.json'}[/green]")


def _mostrar_diseno(resultado) -> None:
    """Imprime la tabla de variantes con su score y sus mutaciones."""
    tabla = Table(
        title=f"Variantes de {resultado.estructura.stem} — {resultado.designer}"
    )
    tabla.add_column("variante", style="cyan")
    tabla.add_column("global_score", justify="right")
    tabla.add_column("recuperación", justify="right")
    tabla.add_column("mut.", justify="right")
    tabla.add_column("identidad", justify="right")
    tabla.add_column("mutaciones")

    mejor = resultado.mejor()
    for v in resultado.variantes:
        # global_score es una log-verosimilitud negativa: más bajo es mejor.
        destacar = mejor is not None and v.id == mejor.id
        estilo = "bold green" if destacar else ""
        mutaciones = v.notacion_mutaciones() or "(ninguna)"
        if len(mutaciones) > 60:
            mutaciones = mutaciones[:57] + "…"
        tabla.add_row(
            v.id,
            f"{v.global_score:.4f}" if v.global_score is not None else "-",
            f"{v.recuperacion:.3f}" if v.recuperacion is not None else "-",
            str(v.n_mutaciones),
            f"{v.identidad:.1%}",
            mutaciones,
            style=estilo,
        )

    console.print(tabla)
    console.print(
        "[dim]global_score es una log-verosimilitud negativa: "
        "más bajo es mejor. En verde, la mejor variante.[/dim]"
    )
    if resultado.posiciones_fijas:
        console.print(
            f"Posiciones fijas (sin mutar): {resultado.posiciones_fijas}"
        )
    console.print(f"Variantes: [green]{resultado.fasta}[/green]")
    console.print(f"Tabla:     [green]{resultado.tabla}[/green]")


# ---------------------------------------------------------------------------
# Fase 3 — Validación estructural
# ---------------------------------------------------------------------------


@app.command()
def simulate(
    input: Annotated[
        Optional[Path],
        typer.Option("--input", "-i", help="Estructura a simular (.pdb)."),
    ] = None,
    ns: Annotated[
        Optional[float],
        typer.Option("--ns", help="Nanosegundos de producción."),
    ] = None,
    outdir: Annotated[
        Optional[Path],
        typer.Option("--outdir", "-o", help="Carpeta de trabajo de la simulación."),
    ] = None,
    clean_only: Annotated[
        bool,
        typer.Option(
            "--clean-only",
            help="Solo limpiar la estructura (aguas y heteroátomos). No requiere GROMACS.",
        ),
    ] = False,
    no_resume: Annotated[
        bool,
        typer.Option(
            "--no-resume",
            help="Rehacer todas las etapas aunque ya estén hechas.",
        ),
    ] = False,
) -> None:
    """Corre minimización, equilibración y MD corta con GROMACS (Fase 3).

    Con ``--clean-only`` hace únicamente la limpieza de la estructura, que es
    la parte que no necesita GROMACS y sirve para revisar qué se va a sacar
    antes de simular.
    """
    from pdpipe.phase3_md.gromacs import ErrorDeGromacs, GromacsNoDisponible
    from pdpipe.phase3_md.preparation import limpiar_estructura, preparar_sistema
    from pdpipe.phase3_md.simulation import simular

    cfg = _require_config()
    if input is None:
        console.print("[red]Error:[/red] indicá --input con la estructura a simular.")
        raise typer.Exit(code=EXIT_ERROR)
    if not input.exists():
        console.print(f"[red]Error:[/red] no existe el archivo de entrada: {input}")
        raise typer.Exit(code=EXIT_ERROR)

    duracion = ns if ns is not None else cfg.md.production_ns
    destino = outdir or (Path(cfg.resolved_paths()["data_interim"]) / f"md_{input.stem}")

    if clean_only:
        try:
            limpieza = limpiar_estructura(input, Path(destino) / "limpio.pdb")
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=EXIT_ERROR) from exc
        _mostrar_limpieza(limpieza)
        return

    manifest = RunManifest.start(
        command="simulate",
        seed=cfg.seed,
        config=cfg.to_dict(),
        params={
            "input": str(input),
            "ns": duracion,
            "outdir": str(destino),
            "force_field": cfg.md.force_field,
            "water_model": cfg.md.water_model,
        },
    )
    directorio = Path(cfg.resolved_paths()["runs"]) / manifest.run_id
    if cfg.logging.to_file:
        setup_logging(level=state.log_level, log_file=directorio / "run.log")

    try:
        sistema = preparar_sistema(cfg, input, destino)
        resultado = simular(cfg, sistema, ns=ns, reanudar=not no_resume)
    except GromacsNoDisponible as exc:
        # Falta una herramienta del entorno, no un dato: el mensaje trae la
        # instalación y el código de salida es el de "pendiente".
        console.print(
            Panel(str(exc), title="[yellow]GROMACS no disponible[/yellow]",
                  border_style="yellow", expand=False)
        )
        manifest.finish("error", error=str(exc))
        manifest.save(directorio)
        raise typer.Exit(code=EXIT_PENDING) from exc
    except (ErrorDeGromacs, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        manifest.finish("error", error=str(exc))
        manifest.save(directorio)
        raise typer.Exit(code=EXIT_ERROR) from exc

    if sistema.limpieza:
        manifest.add_input(input, key=input.name)
    for etapa in resultado.etapas:
        if etapa.trayectoria:
            manifest.add_output(etapa.trayectoria, key=etapa.trayectoria.name)
        if etapa.estructura:
            manifest.add_output(etapa.estructura, key=etapa.estructura.name)
    manifest.add_note(
        f"GROMACS {resultado.version_gromacs or '?'}: {resultado.ns_simulados} ns "
        f"en {resultado.duracion_total_s} s"
    )

    manifest.finish("ok")
    manifest.save(directorio)
    _mostrar_simulacion(resultado)
    console.print(f"Manifiesto: [green]{directorio / 'run_manifest.json'}[/green]")


def _mostrar_limpieza(limpieza) -> None:
    """Imprime qué se sacó de la estructura."""
    tabla = Table(title=f"Limpieza — {limpieza.entrada.name}", show_header=False)
    tabla.add_column("clave", style="cyan")
    tabla.add_column("valor")
    # Flecha ASCII a propósito: la consola de Windows usa cp1252 y U+2192 no
    # existe en esa codepage, así que imprimirlo revienta con UnicodeEncodeError.
    tabla.add_row("Átomos", f"{limpieza.atomos_iniciales} -> {limpieza.atomos_finales}")
    tabla.add_row("Aguas quitadas", str(limpieza.aguas_quitadas))
    tabla.add_row("Heteroátomos quitados", str(limpieza.heteroatomos_quitados))
    tabla.add_row("Hidrógenos quitados", str(limpieza.hidrogenos_quitados))
    if limpieza.altloc_descartadas:
        tabla.add_row("Conformaciones alternativas", str(limpieza.altloc_descartadas))
    if limpieza.modelos_descartados:
        tabla.add_row("Modelos descartados", str(limpieza.modelos_descartados))
    console.print(tabla)

    if limpieza.heteroatomos:
        detalle = ", ".join(f"{k} ×{v}" for k, v in sorted(limpieza.heteroatomos.items()))
        console.print(f"Heteroátomos: {detalle}")
    for aviso in limpieza.advertencias:
        console.print(f"[yellow]Advertencia:[/yellow] {aviso}")
    console.print(f"Estructura limpia: [green]{limpieza.salida}[/green]")


def _mostrar_simulacion(resultado) -> None:
    """Imprime el sistema armado y el resumen de las cuatro etapas."""
    sistema = resultado.sistema
    ficha = Table(title="Sistema simulado", show_header=False)
    ficha.add_column("clave", style="cyan")
    ficha.add_column("valor")
    ficha.add_row("Campo de fuerza", sistema.campo_de_fuerza)
    ficha.add_row("Modelo de agua", sistema.modelo_de_agua)
    ficha.add_row("Caja", sistema.forma_de_caja)
    ficha.add_row("Átomos", str(sistema.n_atomos))
    ficha.add_row("Aguas", str(sistema.n_aguas))
    if sistema.iones:
        ficha.add_row("Iones", ", ".join(f"{k} ×{v}" for k, v in sorted(sistema.iones.items())))
    if resultado.version_gromacs:
        ficha.add_row("GROMACS", resultado.version_gromacs)
    console.print(ficha)

    etapas = Table(title="Etapas")
    etapas.add_column("etapa", style="cyan")
    etapas.add_column("pasos", justify="right")
    etapas.add_column("ps", justify="right")
    etapas.add_column("tiempo real", justify="right")
    for etapa in resultado.etapas:
        etapas.add_row(
            etapa.nombre,
            f"{etapa.pasos:,}",
            f"{etapa.ps_simulados:g}" if etapa.ps_simulados else "-",
            "reutilizada" if etapa.reutilizada
            else (f"{etapa.segundos:.1f} s" if etapa.segundos else "-"),
        )
    console.print(etapas)

    if sistema.limpieza and sistema.limpieza.advertencias:
        for aviso in sistema.limpieza.advertencias:
            console.print(f"[yellow]Advertencia:[/yellow] {aviso}")

    if resultado.trayectoria:
        console.print(f"Trayectoria: [green]{resultado.trayectoria}[/green]")
        # Se sugiere el .tpr y no el .gro: es el unico que trae las cargas
        # del campo de fuerza, sin las cuales no se pueden contar puentes.
        topologia = resultado.topologia_para_analisis or resultado.estructura_final
        console.print(
            "[dim]Analizala con: pdpipe md-analyze --topology "
            f"{topologia} --trajectory {resultado.trayectoria}[/dim]"
        )


@app.command(name="md-analyze")
def md_analyze(
    topology: Annotated[
        Optional[Path],
        typer.Option("--topology", "-t", help="Topología: .gro, .pdb o .tpr."),
    ] = None,
    trajectory: Annotated[
        Optional[Path],
        typer.Option("--trajectory", "-x", help="Trayectoria: .xtc, .trr o .dcd."),
    ] = None,
    selection: Annotated[
        Optional[str],
        typer.Option("--selection", help="Selección de átomos para RMSD y RMSF."),
    ] = None,
    sasa_stride: Annotated[
        Optional[int],
        typer.Option("--sasa-stride", help="Analizar 1 de cada N cuadros para la SASA."),
    ] = None,
    no_figures: Annotated[
        bool, typer.Option("--no-figures", help="No generar los PNG.")
    ] = False,
) -> None:
    """Analiza una trayectoria de MD: RMSD, RMSF, Rg, SASA y puentes (Fase 3).

    No requiere GROMACS: trabaja sobre una trayectoria ya simulada, venga de
    donde venga. Si la simulación corrió en Colab, este es el paso que la
    trae de vuelta al pipeline.
    """
    from pdpipe.phase3_md import pipeline as fase3
    from pdpipe.phase3_md.analysis import ErrorDeAnalisis

    cfg = _require_config()
    if topology is None:
        console.print("[red]Error:[/red] indicá --topology con la estructura.")
        raise typer.Exit(code=EXIT_ERROR)
    if not topology.exists():
        console.print(f"[red]Error:[/red] no existe la topología: {topology}")
        raise typer.Exit(code=EXIT_ERROR)
    if trajectory is not None and not trajectory.exists():
        console.print(f"[red]Error:[/red] no existe la trayectoria: {trajectory}")
        raise typer.Exit(code=EXIT_ERROR)

    manifest = RunManifest.start(
        command="md-analyze",
        seed=cfg.seed,
        config=cfg.to_dict(),
        params={
            "topology": str(topology),
            "trajectory": str(trajectory) if trajectory else None,
            "sasa_stride": sasa_stride if sasa_stride is not None else cfg.md.sasa_stride,
        },
    )
    directorio = Path(cfg.resolved_paths()["runs"]) / manifest.run_id
    if cfg.logging.to_file:
        setup_logging(level=state.log_level, log_file=directorio / "run.log")

    try:
        resultado = fase3.analizar(
            config=cfg,
            topologia=topology,
            trayectoria=trajectory,
            paso_sasa=sasa_stride,
            con_figuras=not no_figures,
            manifest=manifest,
            **({"seleccion": selection} if selection else {}),
        )
    except ErrorDeAnalisis as exc:
        console.print(f"[red]Error:[/red] {exc}")
        manifest.finish("error", error=str(exc))
        manifest.save(directorio)
        raise typer.Exit(code=EXIT_ERROR) from exc

    manifest.finish("ok")
    manifest.save(directorio)
    _mostrar_analisis_md(resultado)
    console.print(f"Manifiesto: [green]{directorio / 'run_manifest.json'}[/green]")


def _mostrar_analisis_md(resultado) -> None:
    """Imprime el resumen de las medidas y dónde quedaron las salidas."""
    ficha = Table(title=f"Trayectoria — {resultado.topologia.stem}", show_header=False)
    ficha.add_column("clave", style="cyan")
    ficha.add_column("valor")
    ficha.add_row("Cuadros", str(resultado.n_frames))
    ficha.add_row("Átomos", str(resultado.n_atomos))
    ficha.add_row("Residuos", str(resultado.n_residuos))
    if resultado.dt_ps:
        ficha.add_row("Paso de tiempo", f"{resultado.dt_ps} ps")
    if resultado.duracion_ps is not None:
        ficha.add_row("Duración analizada", f"{resultado.duracion_ps} ps")
    console.print(ficha)

    medidas = Table(title="Medidas")
    medidas.add_column("magnitud", style="cyan")
    medidas.add_column("media", justify="right")
    medidas.add_column("desvío", justify="right")
    medidas.add_column("inicial", justify="right")
    medidas.add_column("final", justify="right")
    medidas.add_column("deriva", justify="right")
    for serie in resultado.series().values():
        medidas.add_row(
            f"{serie.nombre} ({serie.unidad})",
            f"{serie.media:g}",
            f"{serie.desvio:g}",
            f"{serie.inicial:g}",
            f"{serie.final:g}",
            f"{serie.deriva:+g}",
        )
    if resultado.rmsf:
        medidas.add_row(
            f"RMSF ({resultado.rmsf.unidad})",
            f"{resultado.rmsf.media:g}", "-", "-", "-", "-",
        )
    console.print(medidas)

    if resultado.rmsf:
        moviles = ", ".join(f"{r} ({v:g} Å)" for r, v in resultado.rmsf.mas_moviles(5))
        console.print(f"Residuos más móviles: {moviles}")

    for aviso in resultado.advertencias:
        console.print(f"[yellow]Advertencia:[/yellow] {aviso}")

    console.print(f"Series:  [green]{resultado.tabla}[/green]")
    console.print(f"Resumen: [green]{resultado.resumen_json}[/green]")
    if resultado.figuras:
        console.print(
            f"Figuras: [green]{len(resultado.figuras)} PNG en "
            f"{resultado.figuras[0].parent}[/green]"
        )


# ---------------------------------------------------------------------------
# Fase 4 — Modelos de IA
# ---------------------------------------------------------------------------


@app.command()
def dataset(
    all_structures: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Incluir también las estructuras que no pasaron la curación.",
        ),
    ] = False,
    min_length: Annotated[
        Optional[int],
        typer.Option("--min-length", help="Descartar cadenas más cortas que N residuos."),
    ] = None,
) -> None:
    """Arma el dataset de estructura secundaria con DSSP (Fase 4).

    Corre DSSP sobre las estructuras curadas en la Fase 1, colapsa los ocho
    estados a tres, y reparte las proteínas en train/val/test agrupando por
    identidad de secuencia para que el test no mida memorización.
    """
    from pdpipe.phase4_ml.pipeline import ErrorDeDataset, construir_dataset

    cfg = _require_config()

    manifest = RunManifest.start(
        command="dataset",
        seed=cfg.seed,
        config=cfg.to_dict(),
        params={
            "solo_curadas": not all_structures,
            "min_length": min_length if min_length is not None else cfg.data.length_min,
            "identidad_max": cfg.ml.identity_threshold,
        },
    )
    directorio = Path(cfg.resolved_paths()["runs"]) / manifest.run_id
    if cfg.logging.to_file:
        setup_logging(level=state.log_level, log_file=directorio / "run.log")

    with _manifiesto_ante_fallas(manifest, directorio):
        try:
            _, division, resumen = construir_dataset(
                config=cfg,
                solo_curadas=not all_structures,
                longitud_min=min_length,
                manifest=manifest,
            )
        except ErrorDeDataset as exc:
            console.print(f"[red]Error:[/red] {exc}")
            manifest.finish("error", error=str(exc))
            manifest.save(directorio)
            raise typer.Exit(code=EXIT_ERROR) from exc

    manifest.finish("ok")
    manifest.save(directorio)
    _mostrar_dataset(division, resumen)
    console.print(f"Manifiesto: [green]{directorio / 'run_manifest.json'}[/green]")


def _mostrar_dataset(division, resumen: dict) -> None:
    """Imprime el tamaño del dataset, su composición y los controles."""
    ficha = Table(title="Dataset de estructura secundaria", show_header=False)
    ficha.add_column("clave", style="cyan")
    ficha.add_column("valor")
    ficha.add_row("Proteínas", str(resumen["n_proteinas"]))
    ficha.add_row("Residuos", f"{resumen['n_residuos']:,}")
    ficha.add_row("Grupos de secuencia", str(resumen["n_grupos"]))
    ficha.add_row("Identidad máxima", f"{resumen['identidad_max']:.0%}")
    ficha.add_row("Ventana", str(resumen["ventana"]))
    console.print(ficha)

    tabla = Table(title="Conjuntos")
    tabla.add_column("conjunto", style="cyan")
    tabla.add_column("proteínas", justify="right")
    tabla.add_column("residuos", justify="right")
    for clase in resumen["clases"]:
        tabla.add_column(clase, justify="right")
    for nombre in ("train", "val", "test"):
        datos = resumen["conjuntos"][nombre]
        tabla.add_row(
            nombre,
            str(datos["n_proteinas"]),
            f"{datos['n_residuos']:,}",
            *[f"{datos['composicion'][c]:.1%}" for c in resumen["clases"]],
        )
    console.print(tabla)

    mayoritaria = max(resumen["composicion"], key=lambda c: resumen["composicion"][c])
    console.print(
        f"[dim]Clase mayoritaria: {mayoritaria} "
        f"({resumen['composicion'][mayoritaria]:.1%}). Un modelo que prediga "
        "siempre esa ya acierta esa fracción: es el piso a superar.[/dim]"
    )

    if resumen["descartadas"]:
        console.print(
            f"[yellow]{len(resumen['descartadas'])} estructuras descartadas.[/yellow] "
            "El detalle está en el JSON del resumen."
        )
    if resumen["pares_redundantes"]:
        console.print(
            f"[red]Atención:[/red] {len(resumen['pares_redundantes'])} pares "
            "redundantes entre conjuntos. El Q3 del test va a estar inflado."
        )
    else:
        console.print(
            "[green]Sin redundancia entre conjuntos[/green] al umbral configurado."
        )


@app.command()
def train(
    model: Annotated[
        Optional[list[str]],
        typer.Option(
            "--model",
            "-m",
            help="bilstm, logistic o random_forest. Repetible; 'all' entrena los tres. "
            "Por defecto, ml.model del config.",
        ),
    ] = None,
    epochs: Annotated[
        Optional[int],
        typer.Option("--epochs", min=1, help="Pisa ml.epochs del config."),
    ] = None,
    no_figures: Annotated[
        bool,
        typer.Option("--no-figures", help="No generar las figuras."),
    ] = False,
) -> None:
    """Entrena y evalúa los predictores de estructura secundaria (Fase 4).

    Usa el dataset que dejó `pdpipe dataset`. Todos los modelos pedidos se
    entrenan sobre el mismo reparto train/val/test, así que sus Q3 son
    directamente comparables.
    """
    from pdpipe.phase4_ml.classifiers import MODELOS, ErrorDeModelo
    from pdpipe.phase4_ml.pipeline import ErrorDeDataset
    from pdpipe.phase4_ml.training import entrenar_modelos

    cfg = _require_config()

    pedidos = model or [cfg.ml.model.value]
    modelos = list(MODELOS) if "all" in pedidos else list(dict.fromkeys(pedidos))
    desconocidos = [m for m in modelos if m not in MODELOS]
    if desconocidos:
        console.print(
            f"[red]Error:[/red] modelo desconocido: {', '.join(desconocidos)}. "
            f"Opciones: {', '.join(MODELOS)} o all."
        )
        raise typer.Exit(code=EXIT_ERROR)

    manifest = RunManifest.start(
        command="train",
        seed=cfg.seed,
        config=cfg.to_dict(),
        params={
            "modelos": modelos,
            "epochs": epochs if epochs is not None else cfg.ml.epochs,
        },
    )
    directorio = Path(cfg.resolved_paths()["runs"]) / manifest.run_id
    if cfg.logging.to_file:
        setup_logging(level=state.log_level, log_file=directorio / "run.log")

    with _manifiesto_ante_fallas(manifest, directorio):
        try:
            resultados = entrenar_modelos(
                config=cfg,
                modelos=modelos,
                destino=directorio,
                epocas=epochs,
                con_figuras=not no_figures,
                manifest=manifest,
            )
        except (ImportError, ErrorDeDataset, ErrorDeModelo) as exc:
            console.print(f"[red]Error:[/red] {exc}")
            manifest.finish("error", error=str(exc))
            manifest.save(directorio)
            raise typer.Exit(code=EXIT_ERROR) from exc

    manifest.finish("ok")
    manifest.save(directorio)
    _mostrar_entrenamiento(resultados)
    console.print(f"Modelos, métricas y figuras: [green]{directorio}[/green]")


def _mostrar_entrenamiento(resultados) -> None:
    """Tabla comparativa de los modelos sobre el test."""
    from pdpipe.phase4_ml.dssp import CLASES_Q3
    from pdpipe.phase4_ml.metrics import f1_macro

    tabla = Table(title="Estructura secundaria — conjunto de test")
    tabla.add_column("modelo", style="cyan")
    tabla.add_column("Q3", justify="right")
    tabla.add_column("Q3 val", justify="right")
    for clase in CLASES_Q3:
        tabla.add_column(f"F1 {clase}", justify="right")
    tabla.add_column("F1 macro", justify="right")
    tabla.add_column("época", justify="right")
    tabla.add_column("tiempo", justify="right")

    for r in resultados:
        tabla.add_row(
            r.nombre,
            f"{r.test.q3:.3f}",
            f"{r.val.q3:.3f}",
            *[f"{r.test.f1[c]:.3f}" for c in CLASES_Q3],
            f"{f1_macro(r.test):.3f}",
            str(r.historial["mejor_epoca"]) if len(r.historial["epocas"]) > 1 else "—",
            f"{r.segundos:.0f} s",
        )
    console.print(tabla)

    piso = resultados[0].test.q3_base
    console.print(
        f"[dim]Piso: predecir siempre la clase mayoritaria del entrenamiento "
        f"da Q3 = {piso:.3f} en el test.[/dim]"
    )
    for r in resultados:
        if r.test.mejora_sobre_base <= 0.05:
            console.print(
                f"[yellow]{r.nombre} apenas supera el piso "
                f"(+{r.test.mejora_sobre_base:.3f}).[/yellow] No aprendió mucho."
            )


# ---------------------------------------------------------------------------
# Fase 5 — Análisis y reportes
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    run_id: Annotated[
        Optional[str], typer.Option("--run-id", help="Identificador de la corrida.")
    ] = None,
) -> None:
    """Compara las variantes contra la referencia experimental (Fase 5)."""
    _require_config()
    if run_id is None:
        console.print("[red]Error:[/red] indicá --run-id.")
        raise typer.Exit(code=EXIT_ERROR)
    _pending("analyze", hito=5)


@app.command()
def report(
    run_id: Annotated[
        Optional[str], typer.Option("--run-id", help="Identificador de la corrida.")
    ] = None,
    format: Annotated[
        Optional[list[str]],
        typer.Option("--format", help="html y/o markdown. Repetible."),
    ] = None,
) -> None:
    """Genera el reporte final en HTML y Markdown (Fase 5)."""
    _require_config()
    if run_id is None:
        console.print("[red]Error:[/red] indicá --run-id.")
        raise typer.Exit(code=EXIT_ERROR)
    _pending("report", hito=5)


if __name__ == "__main__":
    app()
