"""Carga y validación del ``config.yaml`` (transversal a las cinco fases).

Un único archivo YAML parametriza todo el pipeline. Acá se lo valida con
pydantic v2 antes de que cualquier fase lo use, con ``extra="forbid"``: una
clave mal escrita corta la ejecución en vez de convertirse en un valor por
defecto silencioso que después aparece como un resultado raro a mitad de una
simulación de dos horas.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


class ConfigError(Exception):
    """El ``config.yaml`` no existe, no es YAML válido o no pasa la validación."""


class _Base(BaseModel):
    """Base común: prohíbe claves desconocidas y congela la instancia."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# Enums — restringen los valores válidos a nivel de esquema
# --------------------------------------------------------------------------


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class StructureSource(str, Enum):
    ALPHAFOLD_DB = "alphafold_db"
    COLABFOLD = "colabfold"
    ESMFOLD = "esmfold"


class DesignerBackend(str, Enum):
    PROTEINMPNN = "proteinmpnn"
    ROSETTA = "rosetta"  # reservado para la segunda iteración


class BoxShape(str, Enum):
    CUBIC = "cubic"
    DODECAHEDRON = "dodecahedron"
    OCTAHEDRON = "octahedron"


class MLModel(str, Enum):
    BILSTM = "bilstm"
    LOGISTIC = "logistic"
    RANDOM_FOREST = "random_forest"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class ReportFormat(str, Enum):
    HTML = "html"
    MARKDOWN = "markdown"


# --------------------------------------------------------------------------
# Secciones
# --------------------------------------------------------------------------


class PathsConfig(_Base):
    """Rutas del proyecto. Relativas se resuelven contra la raíz del repo."""

    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    runs: Path = Path("runs")
    database: Path = Path("data/pdpipe.sqlite")

    @field_serializer("*", when_used="json")
    def _serializar_como_posix(self, value: Path) -> str:
        """Serializa siempre con barras POSIX.

        Sin esto, el mismo config.yaml produce ``data/raw`` en Linux y
        ``data\\\\raw`` en Windows dentro del run_manifest.json, y dos corridas
        idénticas en máquinas distintas parecen diferentes al compararlas.
        """
        return value.as_posix()


class LoggingConfig(_Base):
    level: LogLevel = LogLevel.INFO
    to_file: bool = True


class HTTPConfig(_Base):
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    cache: bool = True


class DataConfig(_Base):
    """Fase 1 — criterios de curación y acceso a las APIs públicas."""

    resolution_max: float = Field(default=2.5, gt=0, le=20)
    experimental_methods: list[str] = Field(
        default_factory=lambda: ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
    )
    length_min: int = Field(default=30, ge=1)
    length_max: int = Field(default=600, ge=1)
    organisms: list[str] = Field(default_factory=list)
    biosafety_check: bool = True
    http: HTTPConfig = Field(default_factory=HTTPConfig)

    @model_validator(mode="after")
    def _check_length_range(self) -> "DataConfig":
        if self.length_min > self.length_max:
            raise ValueError(
                f"length_min ({self.length_min}) no puede ser mayor que "
                f"length_max ({self.length_max})"
            )
        return self


class DesignConfig(_Base):
    """Fase 2 — predicción de estructura y generación de variantes."""

    structure_source: StructureSource = StructureSource.ALPHAFOLD_DB
    plddt_min: float = Field(default=70.0, ge=0, le=100)
    designer: DesignerBackend = DesignerBackend.PROTEINMPNN
    n_sequences: int = Field(default=8, ge=1, le=1000)
    temperature: float = Field(default=0.1, gt=0, le=10)
    fixed_positions: list[int] = Field(default_factory=list)

    # ProteinMPNN no se instala con pip: es un repositorio con los pesos
    # adentro, así que hace falta saber dónde quedó clonado. Relativa se
    # resuelve contra la raíz del repo.
    proteinmpnn_home: Path = Path("tools/ProteinMPNN")
    proteinmpnn_model: str = "v_48_020"

    @field_serializer("proteinmpnn_home", when_used="json")
    def _serializar_como_posix(self, value: Path) -> str:
        """Mismo criterio que en :class:`PathsConfig`: barras POSIX siempre.

        Si no, dos corridas idénticas en Windows y en Linux dejan rutas
        distintas en el ``run_manifest.json`` y parecen no serlo.
        """
        return value.as_posix()


class MDConfig(_Base):
    """Fase 3 — dinámica molecular con GROMACS."""

    gromacs_bin: str = "gmx"
    force_field: str = "amber99sb-ildn"
    water_model: str = "tip3p"
    box_shape: BoxShape = BoxShape.CUBIC
    box_padding_nm: float = Field(default=1.0, gt=0, le=10)
    ion_concentration_m: float = Field(default=0.15, ge=0, le=5)

    minimization_steps: int = Field(default=50_000, ge=1)
    nvt_ps: float = Field(default=100.0, gt=0)
    npt_ps: float = Field(default=100.0, gt=0)
    production_ns: float = Field(default=2.0, gt=0)
    temperature_k: float = Field(default=300.0, gt=0)
    pressure_bar: float = Field(default=1.0, gt=0)
    timestep_fs: float = Field(default=2.0, gt=0, le=5)
    threads: int = Field(default=0, ge=0, description="0 = automático")

    # Cada cuántos picosegundos se guarda un cuadro. Guardar cada paso
    # llenaría el disco sin aportar nada: los cuadros consecutivos están
    # correlacionados. 10 ps sobre 2 ns dan ~200 cuadros.
    output_every_ps: float = Field(default=10.0, gt=0)

    # La SASA es la medida más cara del análisis: rueda una esfera de prueba
    # sobre cada átomo, cuadro por cuadro. En una trayectoria de producción
    # conviene analizar uno de cada N.
    sasa_stride: int = Field(default=1, ge=1)


class MLConfig(_Base):
    """Fase 4 — predicción de estructura secundaria a partir de secuencia."""

    window_size: int = Field(default=17, ge=3, le=51)
    test_size: float = Field(default=0.15, gt=0, lt=1)
    val_size: float = Field(default=0.15, gt=0, lt=1)
    identity_threshold: float = Field(default=0.3, gt=0, le=1)

    model: MLModel = MLModel.BILSTM
    hidden_size: int = Field(default=128, ge=1)
    num_layers: int = Field(default=2, ge=1, le=10)
    dropout: float = Field(default=0.3, ge=0, lt=1)
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=30, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    device: Device = Device.CPU

    @model_validator(mode="after")
    def _check_window_is_odd(self) -> "MLConfig":
        # La ventana se centra en el residuo a predecir, así que necesita un
        # número impar de posiciones para tener el mismo contexto a cada lado.
        if self.window_size % 2 == 0:
            raise ValueError(
                f"window_size debe ser impar para centrarse en un residuo "
                f"(recibido: {self.window_size})"
            )
        return self

    @model_validator(mode="after")
    def _check_splits(self) -> "MLConfig":
        if self.test_size + self.val_size >= 1:
            raise ValueError(
                f"test_size + val_size debe ser menor a 1 para dejar datos de "
                f"entrenamiento (recibido: {self.test_size + self.val_size:.2f})"
            )
        return self


class AnalysisConfig(_Base):
    """Fase 5 — comparación contra la referencia experimental y reportes."""

    rmsd_selection: str = "name CA"
    compute_tm_score: bool = True
    report_formats: list[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.HTML, ReportFormat.MARKDOWN]
    )
    figure_dpi: int = Field(default=150, ge=50, le=600)


# --------------------------------------------------------------------------
# Raíz
# --------------------------------------------------------------------------


class Config(_Base):
    """Configuración completa del pipeline."""

    seed: int = Field(default=42, ge=0)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    design: DesignConfig = Field(default_factory=DesignConfig)
    md: MDConfig = Field(default_factory=MDConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    # Ruta del archivo del que se cargó, para dejarla en el manifiesto.
    # Excluida del dump para que no ensucie la comparación entre corridas.
    source_path: Path | None = Field(default=None, exclude=True)

    def resolved_paths(self, root: Path | None = None) -> dict[str, Path]:
        """Devuelve las rutas de ``paths`` resueltas como absolutas.

        Las relativas se resuelven contra ``root`` (por defecto, el directorio
        que contiene el ``config.yaml``, o el cwd si se construyó en memoria).
        """
        if root is None:
            root = self.source_path.parent if self.source_path else Path.cwd()
        root = root.resolve()
        resueltas: dict[str, Path] = {}
        for name in type(self.paths).model_fields:
            value: Path = getattr(self.paths, name)
            resueltas[name] = value if value.is_absolute() else root / value
        return resueltas

    def to_dict(self) -> dict[str, Any]:
        """Config como dict serializable (para el ``run_manifest.json``)."""
        return self.model_dump(mode="json")


DEFAULT_CONFIG_NAME = "config.yaml"


def load_config(path: str | Path | None = None) -> Config:
    """Carga y valida el ``config.yaml``.

    Args:
        path: ruta al archivo. Si es ``None``, busca ``config.yaml`` en el cwd
            y luego hacia arriba en los directorios padre.

    Raises:
        ConfigError: si no se encuentra, no es YAML válido, o no valida.
    """
    if path is None:
        found = find_config()
        if found is None:
            raise ConfigError(
                f"No se encontró {DEFAULT_CONFIG_NAME} en {Path.cwd()} ni en sus "
                f"directorios padre. Pasá la ruta con --config."
            )
        path = found

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"No existe el archivo de configuración: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} no es YAML válido: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} debe contener un mapeo en la raíz, no {type(raw).__name__}"
        )

    try:
        config = Config(**raw, source_path=path.resolve())
    except Exception as exc:  # pydantic.ValidationError y errores de los validadores
        raise ConfigError(f"Configuración inválida en {path}:\n{exc}") from exc

    return config


def find_config(start: Path | None = None) -> Path | None:
    """Busca ``config.yaml`` desde ``start`` hacia arriba. ``None`` si no hay."""
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        candidate = directory / DEFAULT_CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None
