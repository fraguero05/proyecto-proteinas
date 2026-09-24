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
    force: Annotated[
        bool,
        typer.Option("--force", help="Vuelve a descargar aunque ya esté en caché."),
    ] = False,
) -> None:
    """Descarga estructuras del RCSB PDB y anotaciones de UniProt (Fase 1)."""
    _require_config()
    if not pdb_id and not uniprot:
        console.print("[red]Error:[/red] indicá al menos un --pdb-id o un --uniprot.")
        raise typer.Exit(code=EXIT_ERROR)
    _pending("fetch", hito=1)


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
    """Filtra lo descargado y lo carga en la base SQLite local (Fase 1)."""
    _require_config()
    _pending(
        "curate",
        hito=1,
        detalle="Los filtros pasados por CLI sobrescriben los de config.yaml > data.",
    )


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
) -> None:
    """Obtiene la estructura predicha y reporta el pLDDT por residuo (Fase 2)."""
    cfg = _require_config()
    if uniprot is None:
        console.print("[red]Error:[/red] indicá --uniprot.")
        raise typer.Exit(code=EXIT_ERROR)
    elegida = source or cfg.design.structure_source.value
    _pending("predict", hito=2, detalle=f"Fuente de estructura seleccionada: {elegida}.")


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
    cfg = _require_config()
    if input is None:
        console.print("[red]Error:[/red] indicá --input con la estructura de partida.")
        raise typer.Exit(code=EXIT_ERROR)
    if not input.exists():
        console.print(f"[red]Error:[/red] no existe el archivo de entrada: {input}")
        raise typer.Exit(code=EXIT_ERROR)
    n = n_sequences if n_sequences is not None else cfg.design.n_sequences
    _pending("design", hito=2, detalle=f"Se generarían {n} variantes.")


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
