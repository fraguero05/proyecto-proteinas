"""Tests del cliente de AlphaFold DB y del flujo `predict` (Fase 2).

Contra fixtures grabadas, sin red.
"""

from __future__ import annotations

from pathlib import Path

import json
import shutil

import pytest

from pdpipe.config import Config
from pdpipe.phase1_data.http_client import ClienteHTTP, RecursoNoEncontrado
from pdpipe.phase2_design.alphafold import (
    ClienteAlphaFold,
    SinModeloPredicho,
    parsear_metadatos,
)

FIXTURES = Path(__file__).parent / "fixtures"
MODELO_AF = "AF-P00698-F1-model_v6.pdb"


@pytest.fixture
def metadatos() -> dict:
    data = json.loads((FIXTURES / "alphafold_P00698.json").read_text(encoding="utf-8"))
    return data[0] if isinstance(data, list) else data


# ----------------------------------------------------------------- metadatos


def test_parsea_los_metadatos(metadatos: dict):
    modelo = parsear_metadatos(metadatos)
    assert modelo.uniprot_id == "P00698"
    assert modelo.entry_id == "AF-P00698-F1"
    assert modelo.nombre == "Lysozyme C"
    assert modelo.organismo == "Gallus gallus"
    assert modelo.tax_id == 9031
    assert modelo.fuente == "alphafold_db"
    assert modelo.version is not None


def test_guarda_la_metrica_global_para_contrastar(metadatos: dict):
    modelo = parsear_metadatos(metadatos)
    assert modelo.metrica_global_api == pytest.approx(93.88, abs=0.01)


def test_metadatos_vacios_no_rompen():
    modelo = parsear_metadatos({})
    assert modelo.uniprot_id == ""
    assert modelo.resumen is None


# ------------------------------------------------------------------ cliente


@pytest.fixture
def http_con_cache(tmp_path: Path) -> ClienteHTTP:
    """Cliente con la respuesta de la API ya en caché: no sale a la red."""
    shutil.copy(FIXTURES / "alphafold_P00698.json", tmp_path / "alphafold_P00698.json")
    return ClienteHTTP(cache_dir=tmp_path, usar_cache=True)


def test_obtiene_metadatos_desde_la_cache(http_con_cache: ClienteHTTP):
    http_con_cache.session = None  # type: ignore[assignment]
    data = ClienteAlphaFold(http_con_cache).obtener_metadatos("P00698")
    assert data["uniprotAccession"] == "P00698"


def test_normaliza_la_accesion(http_con_cache: ClienteHTTP):
    http_con_cache.session = None  # type: ignore[assignment]
    data = ClienteAlphaFold(http_con_cache).obtener_metadatos("  p00698  ")
    assert data["entryId"] == "AF-P00698-F1"


def test_accesion_sin_modelo_da_error_util(tmp_path: Path, monkeypatch):
    """Un 404 tiene que explicar la alternativa, no solo fallar."""
    http = ClienteHTTP(cache_dir=tmp_path, usar_cache=False)
    monkeypatch.setattr(
        ClienteHTTP,
        "get_json",
        lambda self, url, nombre_cache=None: (_ for _ in ()).throw(
            RecursoNoEncontrado("404")
        ),
    )
    with pytest.raises(SinModeloPredicho) as exc:
        ClienteAlphaFold(http).obtener_metadatos("X99999")
    assert "ColabFold" in str(exc.value)


def test_formato_invalido_falla(http_con_cache: ClienteHTTP, tmp_path: Path):
    with pytest.raises(ValueError, match="Formato"):
        ClienteAlphaFold(http_con_cache).descargar("P00698", tmp_path, formato="xyz")


def test_descarga_usa_el_archivo_en_cache(tmp_path: Path):
    """Si el modelo ya está en destino, no se vuelve a bajar."""
    cache = tmp_path / "raw"
    cache.mkdir()
    shutil.copy(FIXTURES / "alphafold_P00698.json", cache / "alphafold_P00698.json")

    destino = tmp_path / "processed"
    destino.mkdir()
    shutil.copy(FIXTURES / MODELO_AF, destino / MODELO_AF)

    http = ClienteHTTP(cache_dir=cache, usar_cache=True)
    http.session = None  # type: ignore[assignment]

    modelo = ClienteAlphaFold(http).descargar("P00698", destino)
    assert modelo.archivo.name == MODELO_AF
    assert modelo.sha256 is not None


# -------------------------------------------------------- predict end-to-end


@pytest.fixture
def entorno(tmp_path: Path, monkeypatch) -> Config:
    """Proyecto temporal con el modelo y los metadatos ya en disco."""
    raw = tmp_path / "data" / "raw"
    procesados = tmp_path / "data" / "processed"
    raw.mkdir(parents=True)
    procesados.mkdir(parents=True)

    shutil.copy(FIXTURES / "alphafold_P00698.json", raw / "alphafold_P00698.json")
    shutil.copy(FIXTURES / MODELO_AF, procesados / MODELO_AF)

    def sin_red(*args, **kwargs):
        raise AssertionError("un test intentó salir a la red")

    monkeypatch.setattr("requests.Session.get", sin_red)
    monkeypatch.setattr("requests.Session.post", sin_red)

    return Config(
        seed=1,
        paths={
            "data_raw": raw,
            "data_interim": tmp_path / "data" / "interim",
            "data_processed": procesados,
            "runs": tmp_path / "runs",
            "database": tmp_path / "data" / "test.sqlite",
        },
    )


def test_predict_devuelve_el_modelo_anotado(entorno: Config):
    from pdpipe.phase2_design import predecir

    modelo = predecir(entorno, uniprot_id="P00698")

    assert modelo.uniprot_id == "P00698"
    assert len(modelo.residuos) == 147
    assert modelo.resumen is not None
    # 93.89 calculado contra 93.88 de la API: redondeo del B-factor a dos
    # decimales en el formato PDB (ver test_plddt.py).
    assert modelo.resumen.media == pytest.approx(93.88, abs=0.05)


def test_predict_escribe_el_csv_por_residuo(entorno: Config):
    import csv

    from pdpipe.phase2_design import predecir

    predecir(entorno, uniprot_id="P00698")

    csv_path = entorno.resolved_paths()["data_processed"] / "AF-P00698-F1_plddt.csv"
    assert csv_path.is_file()

    with csv_path.open(encoding="utf-8") as handle:
        filas = list(csv.DictReader(handle))

    assert len(filas) == 147
    assert filas[0]["residuo"] == "1"
    assert set(filas[0]) == {"residuo", "aminoacido", "plddt", "banda"}
    assert all(0 <= float(f["plddt"]) <= 100 for f in filas)


def test_predict_escribe_el_resumen_json(entorno: Config):
    from pdpipe.phase2_design import predecir

    predecir(entorno, uniprot_id="P00698")

    json_path = entorno.resolved_paths()["data_processed"] / "AF-P00698-F1_resumen.json"
    datos = json.loads(json_path.read_text(encoding="utf-8"))

    assert datos["uniprot_id"] == "P00698"
    assert datos["resumen"]["n_residuos"] == 147
    assert "residuos" not in datos  # el detalle va al CSV, no acá
    assert datos["plddt_csv"].endswith("_plddt.csv")


def test_predict_registra_las_salidas_en_el_manifiesto(entorno: Config):
    from pdpipe.phase2_design import predecir
    from pdpipe.utils.manifest import RunManifest

    manifest = RunManifest.start(command="predict", seed=1)
    predecir(entorno, uniprot_id="P00698", manifest=manifest)

    assert MODELO_AF in manifest.outputs
    assert "AF-P00698-F1_plddt.csv" in manifest.outputs
    assert any("pLDDT medio" in nota for nota in manifest.notes)


def test_predict_avisa_si_el_plddt_es_bajo(entorno: Config, caplog):
    from pdpipe.phase2_design import predecir

    exigente = entorno.model_copy(
        update={"design": entorno.design.model_copy(update={"plddt_min": 99.0})}
    )
    with caplog.at_level("WARNING"):
        predecir(exigente, uniprot_id="P00698")

    assert any("por debajo del mínimo" in r.message for r in caplog.records)


def test_predict_no_avisa_si_el_plddt_alcanza(entorno: Config, caplog):
    from pdpipe.phase2_design import predecir

    with caplog.at_level("WARNING"):
        predecir(entorno, uniprot_id="P00698")  # plddt_min por defecto: 70

    assert not any("por debajo del mínimo" in r.message for r in caplog.records)


# ------------------------------------------------------------- otras fuentes


def test_colabfold_explica_que_va_en_notebook(entorno: Config):
    from pdpipe.phase2_design import predecir
    from pdpipe.phase2_design.pipeline import FuenteNoDisponible

    with pytest.raises(FuenteNoDisponible, match="notebook"):
        predecir(entorno, uniprot_id="P00698", fuente="colabfold")


def test_esmfold_todavia_no_esta(entorno: Config):
    from pdpipe.phase2_design import predecir
    from pdpipe.phase2_design.pipeline import FuenteNoDisponible

    with pytest.raises(FuenteNoDisponible, match="ESMFold"):
        predecir(entorno, uniprot_id="P00698", fuente="esmfold")


def test_fuente_desconocida_falla(entorno: Config):
    from pdpipe.phase2_design import predecir
    from pdpipe.phase2_design.pipeline import FuenteNoDisponible

    with pytest.raises(FuenteNoDisponible, match="desconocida"):
        predecir(entorno, uniprot_id="P00698", fuente="rosettafold")
