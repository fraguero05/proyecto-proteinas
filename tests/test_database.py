"""Tests del esquema SQLite y su capa de acceso (Fase 1)."""

from __future__ import annotations

from pathlib import Path

import sqlite3

import pytest

from pdpipe.phase1_data.database import BaseDatos, abrir_base
from pdpipe.phase1_data.models import Funcion, Interaccion, PTM, Proteina


@pytest.fixture
def db(tmp_path: Path) -> BaseDatos:
    with BaseDatos(tmp_path / "test.sqlite") as base:
        yield base


@pytest.fixture
def proteina() -> Proteina:
    return Proteina(
        pdb_id="1LYZ",
        uniprot_id="P00698",
        nombre="Lysozyme C",
        organismo="Gallus gallus",
        tax_id=9031,
        metodo="X-RAY DIFFRACTION",
        resolucion=2.0,
        longitud=147,
        secuencia="KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRW",
        archivo_path="data/raw/1LYZ.pdb",
        sha256="a" * 64,
        funciones=[
            Funcion(
                tipo="go_molecular_function",
                termino_go="GO:0003796",
                descripcion="lysozyme activity",
                evidencia="IDA:UniProt",
            ),
            Funcion(tipo="keyword", descripcion="Hydrolase"),
        ],
        interacciones=[
            Interaccion(
                partner_uniprot="P0C0V0",
                partner_gen="degP",
                tipo="binaria",
                fuente="IntAct",
                n_experimentos=11,
            )
        ],
        ptms=[
            PTM(posicion=24, posicion_fin=145, tipo="Disulfide bond"),
            PTM(posicion=53, tipo="Modified residue", descripcion="Phosphoserine"),
        ],
    )


# -------------------------------------------------------------------- esquema


def test_crea_las_cuatro_tablas(db: BaseDatos):
    tablas = {
        f["name"]
        for f in db.conexion.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"proteinas", "funciones", "interacciones", "ptms"} <= tablas


def test_crear_esquema_es_idempotente(db: BaseDatos, proteina: Proteina):
    db.guardar_proteina(proteina)
    db.crear_esquema()  # no debe borrar nada ni fallar
    assert db.contar()["proteinas"] == 1


def test_crea_el_directorio_de_la_base(tmp_path: Path):
    ruta = tmp_path / "a" / "b" / "pdpipe.sqlite"
    with BaseDatos(ruta):
        pass
    assert ruta.is_file()


def test_las_claves_foraneas_estan_activas(db: BaseDatos):
    activas = db.conexion.execute("PRAGMA foreign_keys").fetchone()[0]
    assert activas == 1


def test_pdb_id_es_unico(db: BaseDatos, proteina: Proteina):
    db.guardar_proteina(proteina)
    filas = db.conexion.execute(
        "SELECT COUNT(*) FROM proteinas WHERE pdb_id = '1LYZ'"
    ).fetchone()[0]
    assert filas == 1


# ----------------------------------------------------------------- escritura


def test_guarda_y_recupera_una_proteina(db: BaseDatos, proteina: Proteina):
    db.guardar_proteina(proteina)
    recuperada = db.obtener_proteina("1LYZ")

    assert recuperada is not None
    assert recuperada.pdb_id == "1LYZ"
    assert recuperada.uniprot_id == "P00698"
    assert recuperada.organismo == "Gallus gallus"
    assert recuperada.resolucion == pytest.approx(2.0)
    assert recuperada.longitud == 147


def test_guarda_las_anotaciones(db: BaseDatos, proteina: Proteina):
    db.guardar_proteina(proteina)
    recuperada = db.obtener_proteina("1LYZ")

    assert len(recuperada.funciones) == 2
    assert len(recuperada.interacciones) == 1
    assert len(recuperada.ptms) == 2


def test_conserva_los_detalles_de_las_anotaciones(db: BaseDatos, proteina: Proteina):
    db.guardar_proteina(proteina)
    recuperada = db.obtener_proteina("1LYZ")

    go = next(f for f in recuperada.funciones if f.tipo == "go_molecular_function")
    assert go.termino_go == "GO:0003796"
    assert go.evidencia == "IDA:UniProt"

    interaccion = recuperada.interacciones[0]
    assert interaccion.partner_uniprot == "P0C0V0"
    assert interaccion.n_experimentos == 11

    puente = next(p for p in recuperada.ptms if p.tipo == "Disulfide bond")
    assert puente.posicion == 24
    assert puente.posicion_fin == 145


def test_busqueda_es_insensible_a_mayusculas(db: BaseDatos, proteina: Proteina):
    db.guardar_proteina(proteina)
    assert db.obtener_proteina("1lyz") is not None


def test_proteina_inexistente_devuelve_none(db: BaseDatos):
    assert db.obtener_proteina("9ZZZ") is None


def test_devuelve_el_id_de_la_fila(db: BaseDatos, proteina: Proteina):
    assert db.guardar_proteina(proteina) > 0


# ------------------------------------------------------------------- upsert


def test_reguardar_actualiza_en_vez_de_duplicar(db: BaseDatos, proteina: Proteina):
    id1 = db.guardar_proteina(proteina)
    id2 = db.guardar_proteina(proteina.model_copy(update={"resolucion": 1.5}))

    assert id1 == id2
    assert db.contar()["proteinas"] == 1
    assert db.obtener_proteina("1LYZ").resolucion == pytest.approx(1.5)


def test_reguardar_no_acumula_anotaciones(db: BaseDatos, proteina: Proteina):
    """Una re-descarga debe dejar la base como una descarga limpia."""
    db.guardar_proteina(proteina)
    db.guardar_proteina(proteina)

    conteos = db.contar()
    assert conteos["funciones"] == 2
    assert conteos["interacciones"] == 1
    assert conteos["ptms"] == 2


def test_reguardar_con_menos_anotaciones_borra_las_viejas(
    db: BaseDatos, proteina: Proteina
):
    db.guardar_proteina(proteina)
    db.guardar_proteina(proteina.model_copy(update={"funciones": [], "ptms": []}))

    conteos = db.contar()
    assert conteos["funciones"] == 0
    assert conteos["ptms"] == 0
    assert conteos["interacciones"] == 1


# ------------------------------------------------------------------- cascada


def test_borrar_una_proteina_borra_sus_anotaciones(db: BaseDatos, proteina: Proteina):
    proteina_id = db.guardar_proteina(proteina)
    with db.conexion:
        db.conexion.execute("DELETE FROM proteinas WHERE id = ?", (proteina_id,))

    conteos = db.contar()
    assert conteos["proteinas"] == 0
    assert conteos["funciones"] == 0
    assert conteos["interacciones"] == 0
    assert conteos["ptms"] == 0


def test_no_se_puede_anotar_una_proteina_inexistente(db: BaseDatos):
    with pytest.raises(sqlite3.IntegrityError):
        with db.conexion:
            db.conexion.execute(
                "INSERT INTO funciones (proteina_id, tipo, descripcion) "
                "VALUES (9999, 'keyword', 'x')"
            )


# ------------------------------------------------------------------ curación


def test_marcar_curada(db: BaseDatos, proteina: Proteina):
    db.guardar_proteina(proteina)
    db.marcar_curada("1LYZ", True)
    assert db.obtener_proteina("1LYZ").curada is True


def test_marcar_rechazada_guarda_el_motivo(db: BaseDatos, proteina: Proteina):
    db.guardar_proteina(proteina)
    db.marcar_curada("1LYZ", False, "resolución 2.00 Å peor que el máximo 1.50 Å")

    recuperada = db.obtener_proteina("1LYZ")
    assert recuperada.curada is False
    assert "resolución" in recuperada.motivo_rechazo


def test_las_rechazadas_se_conservan(db: BaseDatos, proteina: Proteina):
    """Saber qué quedó afuera y por qué es parte del resultado."""
    db.guardar_proteina(
        proteina.model_copy(update={"curada": False, "motivo_rechazo": "test"})
    )
    assert db.contar()["proteinas"] == 1
    assert db.contar()["rechazadas"] == 1


# ------------------------------------------------------------------- listados


def test_listar_vacia(db: BaseDatos):
    assert db.listar_proteinas() == []


def test_listar_ordena_por_pdb_id(db: BaseDatos, proteina: Proteina):
    for pdb_id in ("9ZZZ", "1UBQ", "4HHB"):
        db.guardar_proteina(proteina.model_copy(update={"pdb_id": pdb_id}))
    assert [p.pdb_id for p in db.listar_proteinas()] == ["1UBQ", "4HHB", "9ZZZ"]


def test_listar_solo_curadas(db: BaseDatos, proteina: Proteina):
    db.guardar_proteina(proteina.model_copy(update={"pdb_id": "1AAA", "curada": True}))
    db.guardar_proteina(proteina.model_copy(update={"pdb_id": "1BBB", "curada": False}))

    curadas = db.listar_proteinas(solo_curadas=True)
    assert [p.pdb_id for p in curadas] == ["1AAA"]


def test_contar(db: BaseDatos, proteina: Proteina):
    db.guardar_proteina(proteina.model_copy(update={"pdb_id": "1AAA", "curada": True}))
    db.guardar_proteina(
        proteina.model_copy(
            update={"pdb_id": "1BBB", "curada": False, "motivo_rechazo": "resolución"}
        )
    )

    conteos = db.contar()
    assert conteos["proteinas"] == 2
    assert conteos["curadas"] == 1
    assert conteos["rechazadas"] == 1
    assert conteos["sin_curar"] == 0
    assert conteos["funciones"] == 4  # 2 por proteína


def test_sin_curar_no_cuenta_como_rechazada(db: BaseDatos, proteina: Proteina):
    """Recién descargada no es lo mismo que rechazada.

    Antes, `contar` derivaba rechazadas = proteinas - curadas, así que un
    `fetch` sin `curate` reportaba rechazos que nunca habían ocurrido.
    """
    db.guardar_proteina(proteina)  # curada=False, motivo_rechazo=None

    conteos = db.contar()
    assert conteos["sin_curar"] == 1
    assert conteos["rechazadas"] == 0
    assert conteos["curadas"] == 0


def test_los_tres_estados_suman_el_total(db: BaseDatos, proteina: Proteina):
    db.guardar_proteina(proteina.model_copy(update={"pdb_id": "1AAA", "curada": True}))
    db.guardar_proteina(
        proteina.model_copy(update={"pdb_id": "1BBB", "motivo_rechazo": "x"})
    )
    db.guardar_proteina(proteina.model_copy(update={"pdb_id": "1CCC"}))

    conteos = db.contar()
    assert (
        conteos["curadas"] + conteos["rechazadas"] + conteos["sin_curar"]
        == conteos["proteinas"]
    )


# ------------------------------------------------------------------- persiste


def test_los_datos_sobreviven_a_cerrar_la_base(tmp_path: Path, proteina: Proteina):
    ruta = tmp_path / "persistente.sqlite"
    with BaseDatos(ruta) as db:
        db.guardar_proteina(proteina)
    with BaseDatos(ruta) as db:
        assert db.obtener_proteina("1LYZ") is not None


def test_abrir_base_cierra_al_salir(tmp_path: Path, proteina: Proteina):
    with abrir_base(tmp_path / "x.sqlite") as db:
        db.guardar_proteina(proteina)
    with pytest.raises(sqlite3.ProgrammingError):
        db.conexion.execute("SELECT 1")


def test_campos_opcionales_pueden_ser_nulos(db: BaseDatos):
    minima = Proteina(pdb_id="9XXX")
    db.guardar_proteina(minima)

    recuperada = db.obtener_proteina("9XXX")
    assert recuperada.uniprot_id is None
    assert recuperada.resolucion is None
    assert recuperada.secuencia is None
