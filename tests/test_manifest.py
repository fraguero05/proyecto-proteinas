"""Tests del run_manifest.json — el registro reproducible de cada corrida."""

from __future__ import annotations

from pathlib import Path

import json
import random
import re

import pytest

from pdpipe.utils.manifest import (
    MANIFEST_NAME,
    RunManifest,
    collect_software_versions,
    new_run_id,
    set_global_seed,
)


# ------------------------------------------------------------------ run_id


def test_run_id_tiene_el_formato_esperado():
    run_id = new_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", run_id), run_id


def test_run_ids_no_colisionan():
    """Mismo segundo, distintos ids: el sufijo aleatorio hace su trabajo."""
    ids = {new_run_id() for _ in range(200)}
    assert len(ids) == 200


def test_run_ids_son_ordenables_cronologicamente():
    primero = new_run_id()
    segundo = new_run_id()
    # El prefijo temporal domina el orden lexicográfico.
    assert primero[:16] <= segundo[:16]


# ------------------------------------------------------------------ semillas


def test_set_global_seed_hace_reproducible_a_random():
    set_global_seed(42)
    primera = [random.random() for _ in range(5)]
    set_global_seed(42)
    segunda = [random.random() for _ in range(5)]
    assert primera == segunda


def test_semillas_distintas_dan_secuencias_distintas():
    set_global_seed(1)
    a = [random.random() for _ in range(5)]
    set_global_seed(2)
    b = [random.random() for _ in range(5)]
    assert a != b


def test_set_global_seed_reporta_lo_que_aplico():
    aplicado = set_global_seed(7)
    assert aplicado["value"] == 7
    assert aplicado["python_random"] is True
    # numpy y torch pueden no estar instalados; lo importante es que el estado
    # quede registrado explícitamente y no se asuma.
    assert isinstance(aplicado["numpy"], bool)
    assert isinstance(aplicado["torch"], bool)


# ------------------------------------------------------------------ versiones


def test_collect_software_versions_incluye_python():
    versiones = collect_software_versions()
    assert versiones["python"].startswith("3.")
    assert "platform" in versiones
    assert versiones["packages"]["pdpipe"] is not None


def test_gromacs_ausente_no_rompe():
    versiones = collect_software_versions(gromacs_bin="gmx_que_no_existe_12345")
    assert versiones["gromacs"] is None


# ------------------------------------------------------------------ manifiesto


def test_start_arma_un_manifiesto_completo():
    manifest = RunManifest.start(command="fetch", seed=99, params={"pdb_id": "1UBQ"})
    assert manifest.command == "fetch"
    assert manifest.seeds["value"] == 99
    assert manifest.params["pdb_id"] == "1UBQ"
    assert manifest.status == "running"
    assert manifest.finished_at is None
    assert manifest.software["python"]


def test_add_input_registra_el_hash(pdb_fixture: Path):
    from pdpipe.utils.checksums import file_sha256

    manifest = RunManifest.start(command="design", seed=1)
    digest = manifest.add_input(pdb_fixture)
    assert digest == file_sha256(pdb_fixture)
    assert manifest.inputs["mini.pdb"] == digest


def test_add_input_con_clave_propia(pdb_fixture: Path):
    manifest = RunManifest.start(command="design", seed=1)
    manifest.add_input(pdb_fixture, key="referencia")
    assert "referencia" in manifest.inputs


def test_add_output_registra_el_hash(tmp_path: Path):
    salida = tmp_path / "variantes.fasta"
    salida.write_text(">var01\nMAGS\n", encoding="utf-8")
    manifest = RunManifest.start(command="design", seed=1)
    manifest.add_output(salida)
    assert "variantes.fasta" in manifest.outputs


def test_finish_sella_el_estado():
    manifest = RunManifest.start(command="info", seed=1)
    manifest.finish("ok")
    assert manifest.status == "ok"
    assert manifest.finished_at is not None
    assert manifest.duration_s is not None
    assert manifest.duration_s >= 0


def test_finish_con_error_guarda_el_motivo():
    manifest = RunManifest.start(command="simulate", seed=1)
    manifest.finish("error", error="GROMACS no encontrado")
    assert manifest.status == "error"
    assert "GROMACS" in manifest.error


def test_duracion_es_none_mientras_corre():
    manifest = RunManifest.start(command="info", seed=1)
    assert manifest.duration_s is None


def test_notas_se_acumulan():
    manifest = RunManifest.start(command="curate", seed=1)
    manifest.add_note("1LYZ descartada por resolución")
    manifest.add_note("1UBQ aceptada")
    assert len(manifest.notes) == 2


# ------------------------------------------------------------------ persistencia


def test_save_escribe_json_valido(tmp_path: Path):
    manifest = RunManifest.start(command="info", seed=5)
    manifest.finish("ok")
    ruta = manifest.save(tmp_path / "runs" / manifest.run_id)

    assert ruta.name == MANIFEST_NAME
    assert ruta.is_file()
    data = json.loads(ruta.read_text(encoding="utf-8"))
    assert data["run_id"] == manifest.run_id
    assert data["status"] == "ok"
    assert data["seeds"]["value"] == 5
    assert data["duration_s"] is not None


def test_save_crea_los_directorios_que_falten(tmp_path: Path):
    manifest = RunManifest.start(command="info", seed=1)
    destino = tmp_path / "a" / "b" / "c"
    ruta = manifest.save(destino)
    assert ruta.parent == destino


def test_round_trip_save_load(tmp_path: Path, pdb_fixture: Path):
    original = RunManifest.start(
        command="design", seed=13, params={"n_sequences": 8}
    )
    original.add_input(pdb_fixture)
    original.add_note("corrida de prueba")
    original.finish("ok")
    original.save(tmp_path)

    recuperado = RunManifest.load(tmp_path)
    assert recuperado.run_id == original.run_id
    assert recuperado.command == "design"
    assert recuperado.params["n_sequences"] == 8
    assert recuperado.inputs == original.inputs
    assert recuperado.notes == ["corrida de prueba"]
    assert recuperado.status == "ok"


def test_load_acepta_ruta_al_archivo(tmp_path: Path):
    manifest = RunManifest.start(command="info", seed=1)
    ruta = manifest.save(tmp_path)
    recuperado = RunManifest.load(ruta)
    assert recuperado.run_id == manifest.run_id


def test_json_es_utf8_legible(tmp_path: Path):
    manifest = RunManifest.start(command="curate", seed=1)
    manifest.add_note("descartada por resolución baja")
    ruta = manifest.save(tmp_path)
    crudo = ruta.read_text(encoding="utf-8")
    assert "resolución" in crudo  # ensure_ascii=False


def test_timestamps_en_utc():
    manifest = RunManifest.start(command="info", seed=1)
    assert manifest.started_at.tzinfo is not None
    assert manifest.started_at.utcoffset().total_seconds() == 0


def test_manifiesto_rechaza_campos_desconocidos():
    with pytest.raises(Exception):
        RunManifest(run_id="x", campo_inventado=1)  # type: ignore[call-arg]
