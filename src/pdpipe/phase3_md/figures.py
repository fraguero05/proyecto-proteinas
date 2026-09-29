"""Figuras del análisis de trayectorias (Fase 3).

Un PNG por magnitud más un panel resumen con las cinco juntas, que es la que
suele ir al documento. Todo con matplotlib en modo sin pantalla, porque esto
corre desde la CLI y también tiene que andar en un servidor o en Colab.
"""

from __future__ import annotations

from pathlib import Path

from pdpipe.phase3_md.models import PerfilPorResiduo, ResultadoAnalisisMD, SerieTemporal
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

DPI = 150


def _plt():
    """Importa matplotlib forzando el backend sin pantalla.

    El backend se fija antes de importar ``pyplot``: si se deja el que elige
    solo, en una máquina sin entorno gráfico falla al crear la figura.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depende del extra `md`
        raise ImportError(
            'matplotlib no está instalado. Instalá el extra: uv pip install -e ".[md]"'
        ) from exc

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _serie(ax, serie: SerieTemporal, color: str) -> None:
    """Dibuja una serie temporal con su media marcada."""
    ax.plot(serie.tiempos_ps, serie.valores, color=color, linewidth=1.2)
    ax.axhline(
        serie.media,
        color=color,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label=f"media {serie.media:g} {serie.unidad}",
    )
    ax.set_xlabel("tiempo (ps)")
    ax.set_ylabel(f"{serie.nombre} ({serie.unidad})")
    ax.set_title(serie.nombre)
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)


def figura_serie(serie: SerieTemporal, destino: Path, color: str = "#1f77b4") -> Path:
    """Guarda una serie temporal como PNG."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    _serie(ax, serie, color)
    fig.tight_layout()
    fig.savefig(destino, dpi=DPI)
    plt.close(fig)
    return destino


def figura_rmsf(perfil: PerfilPorResiduo, destino: Path, n_etiquetas: int = 5) -> Path:
    """Guarda el perfil de RMSF, marcando los residuos más móviles.

    Las etiquetas son lo que hace útil el gráfico: sin ellas hay que ir a
    contar picos a mano para saber qué región es la flexible.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(perfil.residuos, perfil.valores, color="#d62728", linewidth=1.2)
    ax.axhline(
        perfil.media,
        color="#d62728",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label=f"media {perfil.media:g} {perfil.unidad}",
    )

    for resid, valor in perfil.mas_moviles(n_etiquetas):
        ax.annotate(
            str(resid),
            xy=(resid, valor),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize="small",
            color="#d62728",
        )

    ax.set_xlabel("residuo")
    ax.set_ylabel(f"{perfil.nombre} ({perfil.unidad})")
    ax.set_title(f"{perfil.nombre} por residuo")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destino, dpi=DPI)
    plt.close(fig)
    return destino


def figura_resumen(resultado: ResultadoAnalisisMD, destino: Path) -> Path:
    """Panel con todo lo que se pudo medir, en una sola imagen."""
    plt = _plt()

    colores = {
        "rmsd": "#1f77b4",
        "radio_giro": "#2ca02c",
        "sasa": "#ff7f0e",
        "puentes": "#9467bd",
    }
    series = resultado.series()
    paneles = len(series) + (1 if resultado.rmsf else 0)
    if paneles == 0:
        raise ValueError("No hay nada medido para graficar")

    filas = (paneles + 1) // 2
    fig, ejes = plt.subplots(filas, 2, figsize=(12, 3.6 * filas), squeeze=False)
    planos = [e for fila in ejes for e in fila]

    for eje, (clave, serie) in zip(planos, series.items()):
        _serie(eje, serie, colores.get(clave, "#1f77b4"))

    if resultado.rmsf:
        eje = planos[len(series)]
        eje.plot(resultado.rmsf.residuos, resultado.rmsf.valores, color="#d62728", linewidth=1.2)
        eje.set_xlabel("residuo")
        eje.set_ylabel(f"RMSF ({resultado.rmsf.unidad})")
        eje.set_title("RMSF por residuo")
        eje.grid(alpha=0.3)

    # Los paneles que sobran se apagan, si no quedan ejes vacíos con marcas.
    for eje in planos[paneles:]:
        eje.axis("off")

    fig.suptitle(
        f"Análisis de {resultado.topologia.stem} — {resultado.n_frames} cuadros",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(destino, dpi=DPI)
    plt.close(fig)
    return destino


def generar_figuras(resultado: ResultadoAnalisisMD, destino: Path) -> list[Path]:
    """Genera todos los PNG del análisis y devuelve sus rutas."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    base = resultado.topologia.stem

    colores = {
        "rmsd": "#1f77b4",
        "radio_giro": "#2ca02c",
        "sasa": "#ff7f0e",
        "puentes": "#9467bd",
    }

    figuras: list[Path] = []
    for clave, serie in resultado.series().items():
        figuras.append(
            figura_serie(serie, destino / f"{base}_{clave}.png", colores.get(clave, "#1f77b4"))
        )
    if resultado.rmsf:
        figuras.append(figura_rmsf(resultado.rmsf, destino / f"{base}_rmsf.png"))
    if figuras:
        figuras.append(figura_resumen(resultado, destino / f"{base}_resumen.png"))

    logger.info("Figuras: %d PNG en %s", len(figuras), destino)
    return figuras


__all__ = ["figura_resumen", "figura_rmsf", "figura_serie", "generar_figuras"]
