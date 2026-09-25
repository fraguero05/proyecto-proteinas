"""Diseño de secuencias con ProteinMPNN sobre CPU (Fase 2, parte B).

ProteinMPNN resuelve el problema inverso al de la predicción de estructura:
dado un esqueleto, propone secuencias que podrían plegarse en él. Corre en CPU
en segundos para proteínas chicas, que es lo que lo hace viable en una laptop
sin GPU.

**No es un paquete de PyPI.** Es un repositorio con los pesos adentro, así que
se clona y se invoca por subprocess, igual que se hará con GROMACS en la Fase
3. Eso mantiene código de terceros fuera de este repo y deja que sus pesos
``.pt`` (excluidos por el ``.gitignore``) vivan en el clon.

    git clone https://github.com/dauparas/ProteinMPNN tools/ProteinMPNN
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import ClassVar

from pdpipe.phase2_design.designer import (
    DisenadorNoDisponible,
    ErrorDeDiseno,
    SequenceDesigner,
)
from pdpipe.phase2_design.models import (
    Mutacion,
    ResultadoDiseno,
    VarianteSecuencia,
)
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

SCRIPT = "protein_mpnn_run.py"
MODELO_POR_DEFECTO = "v_48_020"

# Una corrida en CPU de una proteína chica tarda segundos; el techo alto es
# para no cortar un caso grande por las dudas, no una expectativa.
TIMEOUT_S = 1800

# Captura `clave=valor` en los encabezados FASTA de ProteinMPNN tolerando los
# valores entre corchetes (`designed_chains=['A', 'B']`), que traen comas
# adentro y romperían un split(",") ingenuo.
_CAMPO = re.compile(r"(\w+)=(\[[^\]]*\]|[^,\s]+)")


class DisenadorProteinMPNN(SequenceDesigner):
    """Envoltorio de ``protein_mpnn_run.py``."""

    nombre: ClassVar[str] = "proteinmpnn"

    def __init__(
        self,
        home: str | Path = Path("tools/ProteinMPNN"),
        modelo: str = MODELO_POR_DEFECTO,
        timeout_s: int = TIMEOUT_S,
    ) -> None:
        self.home = Path(home)
        self.modelo = modelo
        self.timeout_s = timeout_s

    # -- disponibilidad ----------------------------------------------------

    @property
    def script(self) -> Path:
        return self.home / SCRIPT

    def disponible(self) -> bool:
        return self.script.is_file() and find_spec("torch") is not None

    def motivo_no_disponible(self) -> str:
        if not self.script.is_file():
            return (
                f"No se encontró {SCRIPT} en '{self.home}'. ProteinMPNN no se "
                "instala con pip: es un repositorio con los pesos adentro.\n\n"
                "    git clone https://github.com/dauparas/ProteinMPNN "
                f"{self.home}\n\n"
                "Si lo clonaste en otro lado, apuntá design.proteinmpnn_home "
                "del config.yaml a esa ruta."
            )
        return (
            "PyTorch no está instalado y ProteinMPNN lo necesita. La versión "
            "de CPU alcanza:\n\n"
            '    uv pip install -e ".[ml]"\n'
        )

    # -- diseño ------------------------------------------------------------

    def disenar(
        self,
        estructura: Path,
        n_secuencias: int,
        temperatura: float,
        posiciones_fijas: list[int] | None = None,
        seed: int | None = None,
    ) -> ResultadoDiseno:
        self.verificar_disponible()

        estructura = Path(estructura).resolve()
        if not estructura.is_file():
            raise ErrorDeDiseno(f"No existe la estructura de entrada: {estructura}")

        residuos = _residuos_de(estructura)
        referencia = "".join(aa for _, aa in residuos)

        with tempfile.TemporaryDirectory(prefix="pdpipe_mpnn_") as tmp:
            salida = Path(tmp)
            comando = [
                sys.executable,
                str(self.script.resolve()),
                "--pdb_path", str(estructura),
                "--out_folder", str(salida),
                "--num_seq_per_target", str(n_secuencias),
                "--sampling_temp", str(temperatura),
                "--model_name", self.modelo,
                # Lote de 1: en CPU no hay nada que ganar paralelizando y el
                # pico de memoria queda acotado en una laptop.
                "--batch_size", "1",
            ]
            if seed is not None:
                comando += ["--seed", str(seed)]

            if posiciones_fijas:
                jsonl = salida / "fixed_positions.jsonl"
                _escribir_posiciones_fijas(
                    jsonl, estructura.stem, residuos, posiciones_fijas
                )
                comando += ["--fixed_positions_jsonl", str(jsonl)]

            _ejecutar(comando, cwd=self.home.resolve(), timeout_s=self.timeout_s)

            fasta = salida / "seqs" / f"{estructura.stem}.fa"
            if not fasta.is_file():
                raise ErrorDeDiseno(
                    f"ProteinMPNN terminó sin errores pero no dejó {fasta.name} "
                    f"en {salida / 'seqs'}. Revisá la salida del comando."
                )
            texto = fasta.read_text(encoding="utf-8")

        original, variantes = parsear_fasta_mpnn(texto)

        # La secuencia que reporta ProteinMPNN tiene que ser la misma que
        # leímos del PDB. Si no coinciden, estamos comparando contra otra cosa
        # y todas las mutaciones que sigan serían inventadas.
        if original and referencia and original != referencia:
            raise ErrorDeDiseno(
                "La secuencia original que reporta ProteinMPNN no coincide con "
                f"la del PDB ({len(original)} vs {len(referencia)} residuos). "
                "No se pueden calcular las mutaciones con confianza."
            )

        numeracion = [numero for numero, _ in residuos]
        variantes = [
            v.model_copy(
                update={"mutaciones": calcular_mutaciones(referencia, v.secuencia, numeracion)}
            )
            for v in variantes
        ]

        logger.info(
            "ProteinMPNN generó %d variantes (modelo %s, T=%s)",
            len(variantes),
            self.modelo,
            temperatura,
        )

        return ResultadoDiseno(
            estructura=estructura,
            secuencia_original=referencia,
            designer=self.nombre,
            variantes=variantes,
            modelo=self.modelo,
            seed=seed,
            posiciones_fijas=sorted(posiciones_fijas or []),
        )


# ---------------------------------------------------------------------------
# Piezas sueltas — separadas para poder testearlas sin ProteinMPNN instalado
# ---------------------------------------------------------------------------


def _ejecutar(comando: list[str], cwd: Path, timeout_s: int) -> None:
    """Corre ProteinMPNN y traduce sus fallas a un mensaje legible."""
    logger.info("Ejecutando ProteinMPNN: %s", " ".join(comando[1:]))
    try:
        proceso = subprocess.run(  # noqa: S603 - comando armado acá, sin shell
            comando,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ErrorDeDiseno(
            f"ProteinMPNN no terminó en {timeout_s} s. ¿La estructura es muy "
            "grande o se pidieron demasiadas secuencias?"
        ) from exc
    except OSError as exc:
        raise ErrorDeDiseno(f"No se pudo ejecutar ProteinMPNN: {exc}") from exc

    if proceso.returncode != 0:
        detalle = (proceso.stderr or proceso.stdout or "").strip()
        if "ModuleNotFoundError" in detalle:
            raise DisenadorNoDisponible(
                "A ProteinMPNN le falta una dependencia de Python:\n\n"
                f"{_ultimas_lineas(detalle)}\n\n"
                'Probá con: uv pip install -e ".[ml]"'
            )
        raise ErrorDeDiseno(
            f"ProteinMPNN falló (código {proceso.returncode}):\n\n"
            f"{_ultimas_lineas(detalle)}"
        )


def _ultimas_lineas(texto: str, n: int = 15) -> str:
    """Cola de una salida de error, para no volcar un traceback entero."""
    lineas = [linea for linea in texto.splitlines() if linea.strip()]
    return "\n".join(lineas[-n:]) if lineas else "(sin salida)"


def _residuos_de(estructura: Path) -> list[tuple[int, str]]:
    """Numeración del PDB y secuencia de la primera cadena.

    Reusa el lector de la parte A, que ya resuelve el recorrido de la
    estructura y el mapeo de tres letras a una; acá se descarta el B-factor,
    que en una estructura experimental no es pLDDT.
    """
    from pdpipe.phase2_design.plddt import ErrorPLDDT, extraer_plddt

    try:
        residuos = extraer_plddt(estructura)
    except ErrorPLDDT as exc:
        raise ErrorDeDiseno(f"No se pudo leer la estructura de entrada: {exc}") from exc

    return [(r.numero, r.aminoacido or "X") for r in residuos]


def _escribir_posiciones_fijas(
    destino: Path,
    nombre_pdb: str,
    residuos: list[tuple[int, str]],
    posiciones_pdb: list[int],
) -> dict[str, dict[str, list[int]]]:
    """Traduce posiciones fijas de numeración PDB al índice que espera ProteinMPNN.

    ProteinMPNN numera las posiciones fijas **1..N sobre la cadena diseñada**,
    no por el número de residuo del PDB. Confundir las dos numeraciones no da
    error: congela el residuo equivocado y el resultado parece correcto, así
    que una posición que no exista en la estructura se trata como error y no
    se ignora en silencio.
    """
    indice_por_numero = {numero: i + 1 for i, (numero, _) in enumerate(residuos)}

    faltantes = [p for p in posiciones_pdb if p not in indice_por_numero]
    if faltantes:
        raise ErrorDeDiseno(
            "Estas posiciones de design.fixed_positions no existen en la "
            f"estructura: {faltantes}. La numeración tiene que ser la del PDB "
            f"(va de {residuos[0][0]} a {residuos[-1][0]})."
        )

    indices = sorted(indice_por_numero[p] for p in posiciones_pdb)
    # ProteinMPNN asume una sola cadena cuando el PDB tiene una sola; se usa
    # "A", que es la que escribe AlphaFold.
    datos = {nombre_pdb: {"A": indices}}
    destino.write_text(json.dumps(datos) + "\n", encoding="utf-8")

    logger.info(
        "Posiciones fijas (PDB -> índice de cadena): %s",
        ", ".join(f"{p}->{indice_por_numero[p]}" for p in sorted(posiciones_pdb)),
    )
    return datos


def parsear_fasta_mpnn(texto: str) -> tuple[str, list[VarianteSecuencia]]:
    """Interpreta el FASTA que deja ProteinMPNN en ``seqs/``.

    El formato pone los datos en el encabezado como ``clave=valor``. El primer
    registro es la secuencia original del PDB; los siguientes son las
    variantes muestreadas::

        >1UBQ, score=1.23, global_score=1.34, designed_chains=['A'], seed=42
        MQIFVKTLTGK...
        >T=0.1, sample=1, score=0.87, global_score=0.99, seq_recovery=0.65
        MQIFVKTLTGR...

    Returns:
        La secuencia original y la lista de variantes, **sin** las mutaciones
        calculadas todavía (eso necesita la numeración del PDB).

    Raises:
        ErrorDeDiseno: si el archivo está vacío o trae varias cadenas.
    """
    registros = _registros_fasta(texto)
    if not registros:
        raise ErrorDeDiseno("El FASTA de ProteinMPNN está vacío")

    _, original = registros[0]
    if "/" in original:
        raise ErrorDeDiseno(
            "La estructura tiene más de una cadena y el diseño multicadena "
            "todavía no está implementado. Usá un PDB de una sola cadena."
        )

    variantes: list[VarianteSecuencia] = []
    for orden, (encabezado, secuencia) in enumerate(registros[1:], start=1):
        campos = dict(_CAMPO.findall(encabezado))
        variantes.append(
            VarianteSecuencia(
                id=f"var{orden:02d}",
                secuencia=secuencia,
                score=_a_float(campos.get("score")),
                global_score=_a_float(campos.get("global_score")),
                recuperacion=_a_float(campos.get("seq_recovery")),
                temperatura=_a_float(campos.get("T")),
            )
        )

    return original, variantes


def _registros_fasta(texto: str) -> list[tuple[str, str]]:
    """Parte un FASTA en pares (encabezado, secuencia)."""
    registros: list[tuple[str, str]] = []
    encabezado: str | None = None
    piezas: list[str] = []

    for linea in texto.splitlines():
        linea = linea.strip()
        if not linea:
            continue
        if linea.startswith(">"):
            if encabezado is not None:
                registros.append((encabezado, "".join(piezas)))
            encabezado = linea[1:]
            piezas = []
        elif encabezado is not None:
            piezas.append(linea)

    if encabezado is not None:
        registros.append((encabezado, "".join(piezas)))
    return registros


def _a_float(valor: str | None) -> float | None:
    if valor is None:
        return None
    try:
        return float(valor)
    except ValueError:
        return None


def calcular_mutaciones(
    original: str, variante: str, numeracion: list[int]
) -> list[Mutacion]:
    """Diferencias entre la secuencia original y una variante.

    Las posiciones salen de ``numeracion`` (la del PDB), no del índice, para
    que coincidan con el CSV de pLDDT y con lo que muestra un visualizador.

    Raises:
        ErrorDeDiseno: si las longitudes no coinciden. Alinear secuencias de
            distinto largo es otro problema; acá tienen que ser el mismo
            esqueleto, y si no lo son es un error a mostrar, no a resolver.
    """
    if len(original) != len(variante):
        raise ErrorDeDiseno(
            f"La variante tiene {len(variante)} residuos y la original "
            f"{len(original)}: no son la misma estructura."
        )
    if len(numeracion) != len(original):
        raise ErrorDeDiseno(
            f"La numeración tiene {len(numeracion)} posiciones para una "
            f"secuencia de {len(original)} residuos."
        )

    return [
        Mutacion(posicion=numero, original=antes, nueva=despues)
        for numero, antes, despues in zip(numeracion, original, variante)
        if antes != despues
    ]


__all__ = [
    "DisenadorProteinMPNN",
    "calcular_mutaciones",
    "parsear_fasta_mpnn",
]
