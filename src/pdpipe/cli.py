"""Interfaz de línea de comandos del pipeline (transversal a las cinco fases).

Un comando por fase, más ``info`` para diagnosticar el entorno. En el Hito 0
los comandos de las fases validan sus argumentos y el ``config.yaml``, pero
todavía no ejecutan ciencia: informan en qué hito se implementan y terminan
con código de salida 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn, Optional

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
) -> None:
    """Descarga estructuras del RCSB PDB y anotaciones de UniProt (Fase 1)."""
    from pdpipe.phase1_data import PDBIDInvalido
    from pdpipe.phase1_data import pipeline as fase1

    cfg = _require_config()
    if not pdb_id and not uniprot:
        console.print("[red]Error:[/red] indicá al menos un --pdb-id o un --uniprot.")
        raise typer.Exit(code=EXIT_ERROR)

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
) -> None:
    """Corre minimización, equilibración y MD corta con GROMACS (Fase 3)."""
    cfg = _require_config()
    if input is None:
        console.print("[red]Error:[/red] indicá --input con la estructura a simular.")
        raise typer.Exit(code=EXIT_ERROR)
    if not input.exists():
        console.print(f"[red]Error:[/red] no existe el archivo de entrada: {input}")
        raise typer.Exit(code=EXIT_ERROR)

    duracion = ns if ns is not None else cfg.md.production_ns
    if shutil.which(cfg.md.gromacs_bin) is None:
        console.print(
            Panel(
                f"No se encontró el ejecutable [bold]{cfg.md.gromacs_bin}[/bold] en el PATH.\n"
                f"GROMACS es necesario desde el Hito 3. En Ubuntu/WSL:\n\n"
                f"    [cyan]sudo apt install gromacs[/cyan]\n\n"
                f"Si lo instalaste en otra ruta, ajustá [cyan]md.gromacs_bin[/cyan] "
                f"en el config.yaml.",
                title="[yellow]GROMACS no disponible[/yellow]",
                border_style="yellow",
                expand=False,
            )
        )
    _pending("simulate", hito=3, detalle=f"Producción configurada: {duracion} ns.")


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
