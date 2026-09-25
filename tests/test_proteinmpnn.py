"""Tests del diseño de secuencias con ProteinMPNN (Fase 2, parte B).

Ningún test corre ProteinMPNN: se prueba el parseo de su salida (contra una
fixture con el formato real), el cálculo de mutaciones y la traducción de
numeraciones, que es donde están los errores silenciosos. La corrida de verdad
está al final, marcada con ``skipif`` para que la suite siga pasando sin el
clon ni PyTorch instalados.
"""

from __future__ import annotations

from pathlib import Path

import csv
import json

import pytest

from pdpipe.config import load_config
from pdpipe.phase2_design import pipeline as fase2
from pdpipe.phase2_design.designer import (
    DisenadorNoDisponible,
    ErrorDeDiseno,
    SequenceDesigner,
    obtener_disenador,
)
from pdpipe.phase2_design.models import (
    Mutacion,
    ResultadoDiseno,
    VarianteSecuencia,
)
from pdpipe.phase2_design.plddt import extraer_plddt, secuencia_de_residuos
from pdpipe.phase2_design.proteinmpnn import (
    DisenadorProteinMPNN,
    _escribir_posiciones_fijas,
    calcular_mutaciones,
    parsear_fasta_mpnn,
)

FIXTURES = Path(__file__).parent / "fixtures"
SALIDA_MPNN = FIXTURES / "proteinmpnn_1UBQ.fa"
PDB_1UBQ = FIXTURES / "1UBQ.pdb"


@pytest.fixture
def salida_mpnn() -> str:
    return SALIDA_MPNN.read_text(encoding="utf-8")


@pytest.fixture
def secuencia_1ubq() -> str:
    return secuencia_de_residuos(extraer_plddt(PDB_1UBQ))


# --------------------------------------------------------------- parseo FASTA


def test_parsea_original_y_variantes(salida_mpnn, secuencia_1ubq):
    original, variantes = parsear_fasta_mpnn(salida_mpnn)

    assert original == secuencia_1ubq
    assert len(variantes) == 3
    assert [v.id for v in variantes] == ["var01", "var02", "var03"]


def test_lee_los_scores_del_encabezado(salida_mpnn):
    _, variantes = parsear_fasta_mpnn(salida_mpnn)
    primera = variantes[0]

    assert primera.score == pytest.approx(0.8234)
    assert primera.global_score == pytest.approx(0.8901)
    assert primera.recuperacion == pytest.approx(0.9737)
    assert primera.temperatura == pytest.approx(0.1)


def test_une_las_secuencias_partidas_en_varias_lineas(salida_mpnn, secuencia_1ubq):
    """var03 viene envuelta a 60 columnas: tiene que quedar entera."""
    _, variantes = parsear_fasta_mpnn(salida_mpnn)

    assert len(variantes[2].secuencia) == len(secuencia_1ubq)


def test_los_corchetes_del_encabezado_no_rompen_el_parseo():
    """``designed_chains=['A', 'B']`` trae una coma adentro del valor.

    Partir el encabezado por comas dejaría ``global_score`` sin leer y las
    variantes saldrían sin score, en silencio.
    """
    texto = (
        ">1UBQ, score=1.05, global_score=1.06, designed_chains=['A', 'B'], seed=42\n"
        "MQIF\n"
        ">T=0.2, sample=1, score=0.81, global_score=0.92, seq_recovery=0.75\n"
        "MQIF\n"
    )
    _, variantes = parsear_fasta_mpnn(texto)

    assert variantes[0].global_score == pytest.approx(0.92)
    assert variantes[0].temperatura == pytest.approx(0.2)


def test_fasta_vacio_falla():
    with pytest.raises(ErrorDeDiseno, match="vac"):
        parsear_fasta_mpnn("")


def test_multicadena_falla_explicitamente():
    texto = ">1ABC, score=1.0\nMQIF/GGSK\n>T=0.1, sample=1, score=0.8\nMQIF/GGSK\n"

    with pytest.raises(ErrorDeDiseno, match="multicadena"):
        parsear_fasta_mpnn(texto)


# ------------------------------------------------------------------ mutaciones


def test_detecta_las_mutaciones_esperadas(salida_mpnn, secuencia_1ubq):
    """La fixture trae var01 con K11R y K48R puestas a mano."""
    _, variantes = parsear_fasta_mpnn(salida_mpnn)
    numeracion = list(range(1, len(secuencia_1ubq) + 1))

    mutaciones = calcular_mutaciones(secuencia_1ubq, variantes[0].secuencia, numeracion)

    assert [str(m) for m in mutaciones] == ["K11R", "K48R"]


def test_una_variante_identica_no_tiene_mutaciones(salida_mpnn, secuencia_1ubq):
    _, variantes = parsear_fasta_mpnn(salida_mpnn)
    numeracion = list(range(1, len(secuencia_1ubq) + 1))

    assert calcular_mutaciones(secuencia_1ubq, variantes[1].secuencia, numeracion) == []


def test_usa_la_numeracion_del_pdb_no_el_indice():
    """Una estructura que arranca en 101 tiene que reportar 102, no 2.

    Es el error que no avisa: con numeración de índice el pipeline informaría
    una mutación en un residuo que no es el que cambió.
    """
    mutaciones = calcular_mutaciones("AKG", "ARG", numeracion=[101, 102, 103])

    assert [str(m) for m in mutaciones] == ["K102R"]


def test_longitudes_distintas_fallan():
    with pytest.raises(ErrorDeDiseno, match="no son la misma estructura"):
        calcular_mutaciones("AKG", "AK", numeracion=[1, 2, 3])


def test_numeracion_incompleta_falla():
    with pytest.raises(ErrorDeDiseno, match="numeración"):
        calcular_mutaciones("AKG", "ARG", numeracion=[1, 2])


# ------------------------------------------------------------ posiciones fijas


def test_traduce_posiciones_fijas_a_indice_de_cadena(tmp_path: Path):
    """ProteinMPNN numera 1..N sobre la cadena, no por el número del PDB.

    Con la estructura arrancando en 101, fijar el residuo 103 del PDB tiene
    que escribir el índice 3.
    """
    residuos = [(101, "A"), (102, "K"), (103, "G"), (104, "S")]

    datos = _escribir_posiciones_fijas(
        tmp_path / "fixed.jsonl", "1ABC", residuos, [101, 103]
    )

    assert datos == {"1ABC": {"A": [1, 3]}}
    escrito = json.loads((tmp_path / "fixed.jsonl").read_text(encoding="utf-8"))
    assert escrito == datos


def test_una_posicion_fija_inexistente_falla(tmp_path: Path):
    """Ignorarla en silencio mutaría un residuo que se pidió congelar."""
    residuos = [(1, "A"), (2, "K")]

    with pytest.raises(ErrorDeDiseno, match="999"):
        _escribir_posiciones_fijas(tmp_path / "fixed.jsonl", "X", residuos, [999])


# ------------------------------------------------------------- disponibilidad


def test_sin_clon_no_esta_disponible(tmp_path: Path):
    disenador = DisenadorProteinMPNN(home=tmp_path / "no_existe")

    assert disenador.disponible() is False
    assert "git clone" in disenador.motivo_no_disponible()


def test_verificar_disponible_lanza_con_instrucciones(tmp_path: Path):
    disenador = DisenadorProteinMPNN(home=tmp_path / "no_existe")

    with pytest.raises(DisenadorNoDisponible, match="git clone"):
        disenador.verificar_disponible()


def test_rosetta_avisa_que_es_para_la_segunda_iteracion():
    with pytest.raises(DisenadorNoDisponible, match="Rosetta"):
        obtener_disenador("rosetta")


def test_disenador_desconocido_falla():
    with pytest.raises(DisenadorNoDisponible, match="desconocido"):
        obtener_disenador("alphadesign")


def test_proteinmpnn_implementa_la_interfaz(tmp_path: Path):
    disenador = obtener_disenador("proteinmpnn", home=tmp_path)

    assert isinstance(disenador, SequenceDesigner)
    assert disenador.nombre == "proteinmpnn"


# ------------------------------------------------------------------- modelos


def test_la_notacion_de_mutacion_es_la_estandar():
    assert str(Mutacion(posicion=48, original="K", nueva="R")) == "K48R"


def test_identidad_y_conteo_de_mutaciones():
    variante = VarianteSecuencia(
        id="var01",
        secuencia="A" * 100,
        mutaciones=[Mutacion(posicion=i, original="K", nueva="R") for i in (1, 2, 3)],
    )

    assert variante.n_mutaciones == 3
    assert variante.identidad == pytest.approx(0.97)
    assert variante.notacion_mutaciones() == "K1R,K2R,K3R"


def test_la_mejor_variante_es_la_de_global_score_mas_bajo():
    """Es una log-verosimilitud negativa: más bajo es mejor, no más alto."""
    resultado = ResultadoDiseno(
        estructura=Path("x.pdb"),
        secuencia_original="AKG",
        designer="proteinmpnn",
        variantes=[
            VarianteSecuencia(id="var01", secuencia="AKG", global_score=1.5),
            VarianteSecuencia(id="var02", secuencia="ARG", global_score=0.7),
            VarianteSecuencia(id="var03", secuencia="AKS", global_score=1.1),
        ],
    )

    assert resultado.mejor().id == "var02"


def test_sin_scores_no_inventa_una_mejor():
    resultado = ResultadoDiseno(
        estructura=Path("x.pdb"),
        secuencia_original="AKG",
        designer="proteinmpnn",
        variantes=[VarianteSecuencia(id="var01", secuencia="AKG")],
    )

    assert resultado.mejor() is None


# -------------------------------------------------------------------- salidas


class _DisenadorFalso(SequenceDesigner):
    """Diseñador de mentira: devuelve variantes fijas, sin correr nada.

    Existe para probar el orquestador y las salidas sin ProteinMPNN instalado;
    que alcance con esto es justamente lo que demuestra que la interfaz
    SequenceDesigner desacopla el pipeline de la herramienta.
    """

    nombre = "falso"

    def disponible(self) -> bool:
        return True

    def motivo_no_disponible(self) -> str:  # pragma: no cover - siempre disponible
        return ""

    def disenar(self, estructura, n_secuencias, temperatura, posiciones_fijas=None, seed=None):
        return ResultadoDiseno(
            estructura=Path(estructura),
            secuencia_original="MQIFV",
            designer=self.nombre,
            seed=seed,
            posiciones_fijas=sorted(posiciones_fijas or []),
            variantes=[
                VarianteSecuencia(
                    id="var01",
                    secuencia="MQRFV",
                    global_score=0.88,
                    score=0.80,
                    recuperacion=0.8,
                    mutaciones=[Mutacion(posicion=3, original="I", nueva="R")],
                ),
                VarianteSecuencia(
                    id="var02", secuencia="MQIFV", global_score=1.20, recuperacion=1.0
                ),
            ],
        )


@pytest.fixture
def cfg(config_file: Path):
    return load_config(config_file)


def test_escribe_fasta_con_la_original_primero(cfg, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(fase2, "obtener_disenador", lambda *a, **k: _DisenadorFalso())

    resultado = fase2.disenar(cfg, estructura=PDB_1UBQ)

    lineas = resultado.fasta.read_text(encoding="utf-8").splitlines()
    assert lineas[0] == ">1UBQ_original"
    assert lineas[1] == "MQIFV"
    assert ">1UBQ_var01" in lineas[2]
    assert "mutaciones=I3R" in lineas[2]


def test_la_tabla_trae_score_mutaciones_e_identidad(cfg, monkeypatch):
    monkeypatch.setattr(fase2, "obtener_disenador", lambda *a, **k: _DisenadorFalso())

    resultado = fase2.disenar(cfg, estructura=PDB_1UBQ)

    with resultado.tabla.open(encoding="utf-8") as handle:
        filas = list(csv.DictReader(handle))

    assert [f["variante"] for f in filas] == ["var01", "var02"]
    assert filas[0]["mutaciones"] == "I3R"
    assert filas[0]["n_mutaciones"] == "1"
    assert filas[0]["identidad"] == "0.8"
    assert filas[1]["mutaciones"] == ""


def test_el_override_de_n_secuencias_llega_al_disenador(cfg, monkeypatch):
    recibido: dict = {}

    class _Espia(_DisenadorFalso):
        def disenar(self, estructura, n_secuencias, temperatura, posiciones_fijas=None, seed=None):
            recibido.update(n=n_secuencias, t=temperatura, seed=seed)
            return super().disenar(estructura, n_secuencias, temperatura, posiciones_fijas, seed)

    monkeypatch.setattr(fase2, "obtener_disenador", lambda *a, **k: _Espia())

    fase2.disenar(cfg, estructura=PDB_1UBQ, n_secuencias=16, temperatura=0.3)

    assert recibido == {"n": 16, "t": 0.3, "seed": cfg.seed}


def test_sin_override_toma_los_valores_del_config(cfg, monkeypatch):
    recibido: dict = {}

    class _Espia(_DisenadorFalso):
        def disenar(self, estructura, n_secuencias, temperatura, posiciones_fijas=None, seed=None):
            recibido.update(n=n_secuencias, t=temperatura)
            return super().disenar(estructura, n_secuencias, temperatura, posiciones_fijas, seed)

    monkeypatch.setattr(fase2, "obtener_disenador", lambda *a, **k: _Espia())

    fase2.disenar(cfg, estructura=PDB_1UBQ)

    assert recibido == {"n": cfg.design.n_sequences, "t": cfg.design.temperature}


def test_registra_las_salidas_en_el_manifiesto(cfg, tmp_path: Path, monkeypatch):
    from pdpipe.utils.manifest import RunManifest

    monkeypatch.setattr(fase2, "obtener_disenador", lambda *a, **k: _DisenadorFalso())
    manifest = RunManifest.start(command="design", seed=cfg.seed, config=cfg.to_dict())

    resultado = fase2.disenar(cfg, estructura=PDB_1UBQ, manifest=manifest)

    manifest.finish("ok")
    datos = json.loads(json.dumps(manifest.to_dict()))
    assert resultado.fasta.name in datos["outputs"]
    assert resultado.tabla.name in datos["outputs"]


# ------------------------------------- salidas reales del repo de ProteinMPNN

# El clon trae en outputs/ las salidas de sus propios ejemplos. Parsearlas es
# la prueba más fuerte de que el formato que espera el pipeline es el real, y
# no cuesta una corrida: son archivos ya grabados.
_OUTPUTS_MPNN = Path(__file__).parent.parent / "tools" / "ProteinMPNN" / "outputs"
_SALIDAS_REALES = sorted(_OUTPUTS_MPNN.rglob("*.fa")) if _OUTPUTS_MPNN.is_dir() else []

sin_clon = pytest.mark.skipif(
    not _SALIDAS_REALES,
    reason="ProteinMPNN no está clonado (ver README, Hito 2 parte B)",
)


@sin_clon
@pytest.mark.parametrize("fasta", _SALIDAS_REALES, ids=lambda p: p.parent.parent.name + "/" + p.name)
def test_parsea_las_salidas_reales_del_repo(fasta: Path):
    """Cada salida de ejemplo se parsea entera, o se rechaza por multicadena.

    Lo que no puede pasar es que una monocadena se parsee a medias: si el
    formato del encabezado cambiara, las variantes saldrían sin score y sin
    recuperación, en silencio.
    """
    try:
        original, variantes = parsear_fasta_mpnn(fasta.read_text(encoding="utf-8"))
    except ErrorDeDiseno as exc:
        assert "multicadena" in str(exc), exc
        return

    assert original and variantes
    for v in variantes:
        assert v.global_score is not None
        assert v.score is not None
        assert v.recuperacion is not None
        assert v.temperatura is not None
        assert len(v.secuencia) == len(original)


# ------------------------------------------------------ corrida real (opcional)


_disenador_real = DisenadorProteinMPNN()

pytestmark_real = pytest.mark.skipif(
    not _disenador_real.disponible(),
    reason="ProteinMPNN no está clonado o falta PyTorch (ver README, Hito 2 parte B)",
)


@pytestmark_real
def test_corrida_real_de_proteinmpnn(tmp_path: Path):
    """Única prueba que ejecuta ProteinMPNN de verdad. Requiere el clon y torch."""
    resultado = _disenador_real.disenar(
        estructura=PDB_1UBQ, n_secuencias=2, temperatura=0.1, seed=42
    )

    assert resultado.n_variantes == 2
    assert len(resultado.secuencia_original) == 76
    for variante in resultado.variantes:
        assert len(variante.secuencia) == 76
        assert variante.global_score is not None
