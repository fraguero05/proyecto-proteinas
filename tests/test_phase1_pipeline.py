"""Tests de integración de la Fase 1: fetch y curate end-to-end.

Sin red: se precarga ``data_raw`` con las respuestas grabadas, y el cliente
HTTP las lee de su caché. Solo se simula la descarga del archivo de
coordenadas, que no pasa por la caché de JSON.
"""

from __future__ import annotations

from pathlib import Path

import json
import shutil

import pytest

from pdpipe.config import Config
from pdpipe.phase1_data import BaseDatos
from pdpipe.phase1_data import pipeline as fase1

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def entorno(tmp_path: Path, monkeypatch) -> Config:
    """Proyecto temporal con la caché precargada y la red bloqueada."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)

    # Metadatos de RCSB y UniProt, como si ya se hubieran descargado.
    for nombre in (
        "rcsb_entry_1UBQ.json",
        "rcsb_entry_1LYZ.json",
        "rcsb_entity_1UBQ_1.json",
        "rcsb_entity_1LYZ_1.json",
        "uniprot_P0CG48.json",
        "uniprot_P00698.json",
    ):
        shutil.copy(FIXTURES / nombre, raw / nombre)

    # Descargar coordenadas se resuelve copiando el PDB de la fixture.
    def descarga_falsa(self, pdb_id, destino, formato="pdb"):
        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        final = destino / f"{pdb_id}.pdb"
        if not final.is_file():
            origen = FIXTURES / f"{pdb_id}.pdb"
            shutil.copy(origen if origen.is_file() else FIXTURES / "mini.pdb", final)
        return final

    monkeypatch.setattr(
        "pdpipe.phase1_data.rcsb.ClienteRCSB.descargar_estructura", descarga_falsa
    )

    # Cualquier petición real es un error: los tests no tocan la red.
    def sin_red(*args, **kwargs):
        raise AssertionError("un test intentó salir a la red")

    monkeypatch.setattr("requests.Session.get", sin_red)
    monkeypatch.setattr("requests.Session.post", sin_red)

    return Config(
        seed=1,
        paths={
            "data_raw": raw,
            "data_interim": tmp_path / "data" / "interim",
            "data_processed": tmp_path / "data" / "processed",
            "runs": tmp_path / "runs",
            "database": tmp_path / "data" / "test.sqlite",
        },
    )


# -------------------------------------------------------------------- fetch


def test_fetch_descarga_y_persiste(entorno: Config):
    resultado = fase1.fetch(entorno, pdb_ids=["1UBQ"])

    assert resultado.descargadas == ["1UBQ"]
    assert not resultado.fallidas
    assert not resultado.rechazadas

    with BaseDatos(entorno.resolved_paths()["database"]) as db:
        proteina = db.obtener_proteina("1UBQ")

    assert proteina is not None
    assert proteina.uniprot_id == "P0CG48"
    assert proteina.organismo == "Homo sapiens"
    assert proteina.metodo == "X-RAY DIFFRACTION"
    assert proteina.resolucion == pytest.approx(1.8)
    assert proteina.sha256 is not None


def test_la_longitud_guardada_es_la_de_la_estructura(entorno: Config):
    """Regresión: no confundir la cadena cristalizada con la proteína entera.

    1UBQ tiene 76 residuos; su UniProt P0CG48 (poliubiquitina-C) tiene 685.
    Guardar 685 hacía que `curate` la rechazara por superar el largo máximo
    de 600, cuando es una de las proteínas más chicas del PDB.
    """
    fase1.fetch(entorno, pdb_ids=["1UBQ"])

    with BaseDatos(entorno.resolved_paths()["database"]) as db:
        proteina = db.obtener_proteina("1UBQ")

    assert proteina.longitud == 76
    assert proteina.secuencia is not None
    assert len(proteina.secuencia) == 76
    assert proteina.secuencia.startswith("MQIFVKTLTGK")


def test_fetch_guarda_las_anotaciones(entorno: Config):
    fase1.fetch(entorno, pdb_ids=["1LYZ"])

    with BaseDatos(entorno.resolved_paths()["database"]) as db:
        proteina = db.obtener_proteina("1LYZ")

    assert proteina.funciones
    assert proteina.ptms
    assert any(p.tipo == "Disulfide bond" for p in proteina.ptms)


def test_fetch_de_varias(entorno: Config):
    resultado = fase1.fetch(entorno, pdb_ids=["1UBQ", "1LYZ"])
    assert set(resultado.descargadas) == {"1UBQ", "1LYZ"}

    with BaseDatos(entorno.resolved_paths()["database"]) as db:
        assert db.contar()["proteinas"] == 2


def test_fetch_no_duplica_ids_repetidos(entorno: Config):
    resultado = fase1.fetch(entorno, pdb_ids=["1UBQ", "1ubq", "1UBQ"])
    assert resultado.descargadas == ["1UBQ"]


def test_fetch_es_idempotente(entorno: Config):
    fase1.fetch(entorno, pdb_ids=["1UBQ"])
    fase1.fetch(entorno, pdb_ids=["1UBQ"])

    with BaseDatos(entorno.resolved_paths()["database"]) as db:
        conteos = db.contar()
    assert conteos["proteinas"] == 1


def test_fetch_registra_los_archivos_en_el_manifiesto(entorno: Config):
    from pdpipe.utils.manifest import RunManifest

    manifest = RunManifest.start(command="fetch", seed=1)
    fase1.fetch(entorno, pdb_ids=["1UBQ"], manifest=manifest)
    assert "1UBQ.pdb" in manifest.outputs


def test_fetch_con_id_invalido_falla_claro(entorno: Config):
    from pdpipe.phase1_data import PDBIDInvalido

    with pytest.raises(PDBIDInvalido):
        fase1.fetch(entorno, pdb_ids=["NOPE"])


def test_fetch_por_uniprot_expande_a_sus_estructuras(entorno: Config):
    """Una accesión se expande a todos los códigos PDB que la referencian.

    El fixture de P00698 trae 5 referencias a PDB (recortadas de las cientos
    reales). Ninguna tiene fixture propia, así que todas fallan sin red: lo
    que se verifica acá es la expansión, no la descarga.
    """
    resultado = fase1.fetch(entorno, uniprot_ids=["P00698"])

    intentadas = set(resultado.descargadas) | {p for p, _ in resultado.fallidas}
    assert intentadas == {"132L", "193L", "194L", "1A2Y", "1AKI"}


def test_fetch_por_uniprot_sin_estructuras_lo_reporta(entorno: Config, monkeypatch):
    from pdpipe.phase1_data.models import RegistroUniProt

    monkeypatch.setattr(
        "pdpipe.phase1_data.uniprot.ClienteUniProt.obtener",
        lambda self, acc: RegistroUniProt(accesion=acc, pdb_ids=[]),
    )
    resultado = fase1.fetch(entorno, uniprot_ids=["P99999"])
    assert not resultado.descargadas
    assert "no tiene estructuras" in resultado.fallidas[0][1]


def test_una_falla_no_corta_el_lote(entorno: Config):
    """Un id sin fixture falla, pero el resto del lote se procesa igual."""
    resultado = fase1.fetch(entorno, pdb_ids=["1UBQ", "9ZZZ", "1LYZ"])

    assert set(resultado.descargadas) == {"1UBQ", "1LYZ"}
    assert [p for p, _ in resultado.fallidas] == ["9ZZZ"]


# ------------------------------------------------------------- bioseguridad


def test_fetch_rechaza_por_bioseguridad(entorno: Config, monkeypatch):
    """Una estructura rechazada se registra pero no se descarga."""
    from pdpipe.phase1_data.models import RegistroUniProt

    def uniprot_toxico(self, accesion):
        return RegistroUniProt(
            accesion=accesion,
            nombre="Botulinum neurotoxin type A",
            organismo="Clostridium botulinum",
            keywords=["Toxin", "Neurotoxin"],
            longitud=1296,
        )

    monkeypatch.setattr(
        "pdpipe.phase1_data.uniprot.ClienteUniProt.obtener", uniprot_toxico
    )

    resultado = fase1.fetch(entorno, pdb_ids=["1UBQ"])

    assert not resultado.descargadas
    assert len(resultado.rechazadas) == 1
    pdb_id, motivo = resultado.rechazadas[0]
    assert pdb_id == "1UBQ"
    assert "bioseguridad" in motivo

    # Queda registrada con su motivo, pero sin archivo en disco.
    with BaseDatos(entorno.resolved_paths()["database"]) as db:
        proteina = db.obtener_proteina("1UBQ")
    assert proteina.curada is False
    assert proteina.archivo_path is None
    assert not (entorno.resolved_paths()["data_raw"] / "1UBQ.pdb").exists()


def test_se_puede_desactivar_la_verificacion(entorno: Config, monkeypatch):
    from pdpipe.phase1_data.models import RegistroUniProt

    def uniprot_toxico(self, accesion):
        return RegistroUniProt(accesion=accesion, nombre="Ricin A chain")

    monkeypatch.setattr(
        "pdpipe.phase1_data.uniprot.ClienteUniProt.obtener", uniprot_toxico
    )
    sin_check = entorno.model_copy(
        update={"data": entorno.data.model_copy(update={"biosafety_check": False})}
    )
    resultado = fase1.fetch(sin_check, pdb_ids=["1UBQ"])
    assert resultado.descargadas == ["1UBQ"]


# ------------------------------------------------------------------- curate


def test_curate_acepta_lo_que_cumple(entorno: Config):
    fase1.fetch(entorno, pdb_ids=["1UBQ", "1LYZ"])
    resultado = fase1.curate(entorno)

    assert set(resultado.aceptadas) == {"1UBQ", "1LYZ"}
    assert not resultado.rechazadas


def test_curate_rechaza_por_resolucion(entorno: Config):
    fase1.fetch(entorno, pdb_ids=["1UBQ", "1LYZ"])
    # 1UBQ está a 1.8 Å y 1LYZ a 2.0 Å.
    resultado = fase1.curate(entorno, resolucion_max=1.9)

    assert resultado.aceptadas == ["1UBQ"]
    assert [p for p, _ in resultado.rechazadas] == ["1LYZ"]


def test_curate_rechaza_por_organismo(entorno: Config):
    fase1.fetch(entorno, pdb_ids=["1UBQ", "1LYZ"])
    resultado = fase1.curate(entorno, organismos=["Homo sapiens"])

    assert resultado.aceptadas == ["1UBQ"]
    assert [p for p, _ in resultado.rechazadas] == ["1LYZ"]


def test_curate_persiste_el_veredicto(entorno: Config):
    fase1.fetch(entorno, pdb_ids=["1UBQ", "1LYZ"])
    fase1.curate(entorno, resolucion_max=1.9)

    with BaseDatos(entorno.resolved_paths()["database"]) as db:
        assert db.obtener_proteina("1UBQ").curada is True
        rechazada = db.obtener_proteina("1LYZ")
    assert rechazada.curada is False
    assert "resolución" in rechazada.motivo_rechazo


def test_curate_es_repetible_con_otros_criterios(entorno: Config):
    """Ajustar umbrales no debe requerir re-descargar."""
    fase1.fetch(entorno, pdb_ids=["1UBQ", "1LYZ"])

    estricto = fase1.curate(entorno, resolucion_max=1.5)
    assert not estricto.aceptadas

    laxo = fase1.curate(entorno, resolucion_max=3.0)
    assert len(laxo.aceptadas) == 2


def test_curate_sin_datos_no_rompe(entorno: Config):
    resultado = fase1.curate(entorno)
    assert resultado.total == 0


def test_curate_no_revisa_los_rechazos_por_bioseguridad(entorno: Config):
    """Un rechazo por bioseguridad no se revierte aflojando la resolución."""
    from pdpipe.phase1_data.models import Proteina

    with BaseDatos(entorno.resolved_paths()["database"]) as db:
        db.guardar_proteina(
            Proteina(
                pdb_id="9TOX",
                nombre="Ricin A chain",
                metodo="X-RAY DIFFRACTION",
                resolucion=1.0,
                longitud=267,
                organismo="Ricinus communis",
                curada=False,
                motivo_rechazo="[bioseguridad/organismo] organismo excluido",
            )
        )

    resultado = fase1.curate(entorno, resolucion_max=5.0)
    assert "9TOX" not in resultado.aceptadas
    assert any(p == "9TOX" for p, _ in resultado.rechazadas)

    with BaseDatos(entorno.resolved_paths()["database"]) as db:
        assert db.obtener_proteina("9TOX").curada is False


def test_curate_registra_los_criterios_en_el_manifiesto(entorno: Config):
    from pdpipe.utils.manifest import RunManifest

    fase1.fetch(entorno, pdb_ids=["1UBQ"])
    manifest = RunManifest.start(command="curate", seed=1)
    fase1.curate(entorno, resolucion_max=1.9, manifest=manifest)

    assert manifest.params["criterios"]["resolution_max"] == 1.9
