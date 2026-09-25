"""Base SQLite local de la Fase 1.

Cuatro tablas: ``proteinas`` como raíz, y ``funciones``, ``interacciones`` y
``ptms`` colgando de ella con ``ON DELETE CASCADE``.

Se guardan también las proteínas **rechazadas**, con su ``motivo_rechazo``:
saber qué quedó afuera y por qué es parte del resultado de la curación, no un
descarte. Es lo que permite justificar en la tesis por qué el conjunto final
tiene el tamaño que tiene.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import contextlib
import sqlite3

from pdpipe.phase1_data.models import Funcion, Interaccion, PTM, Proteina
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

ESQUEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS proteinas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pdb_id          TEXT    NOT NULL UNIQUE,
    uniprot_id      TEXT,
    nombre          TEXT,
    organismo       TEXT,
    tax_id          INTEGER,
    metodo          TEXT,
    resolucion      REAL,
    longitud        INTEGER,
    secuencia       TEXT,
    archivo_path    TEXT,
    sha256          TEXT,
    fecha_descarga  TEXT    NOT NULL,
    curada          INTEGER NOT NULL DEFAULT 0,
    motivo_rechazo  TEXT
);

CREATE TABLE IF NOT EXISTS funciones (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    proteina_id  INTEGER NOT NULL REFERENCES proteinas(id) ON DELETE CASCADE,
    tipo         TEXT    NOT NULL,
    termino_go   TEXT,
    descripcion  TEXT    NOT NULL,
    evidencia    TEXT
);

CREATE TABLE IF NOT EXISTS interacciones (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    proteina_id      INTEGER NOT NULL REFERENCES proteinas(id) ON DELETE CASCADE,
    partner_uniprot  TEXT,
    partner_gen      TEXT,
    tipo             TEXT    NOT NULL,
    fuente           TEXT    NOT NULL,
    n_experimentos   INTEGER,
    evidencia        TEXT
);

CREATE TABLE IF NOT EXISTS ptms (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    proteina_id   INTEGER NOT NULL REFERENCES proteinas(id) ON DELETE CASCADE,
    posicion      INTEGER NOT NULL,
    posicion_fin  INTEGER,
    tipo          TEXT    NOT NULL,
    descripcion   TEXT,
    evidencia     TEXT
);

CREATE INDEX IF NOT EXISTS idx_proteinas_uniprot   ON proteinas(uniprot_id);
CREATE INDEX IF NOT EXISTS idx_proteinas_curada    ON proteinas(curada);
CREATE INDEX IF NOT EXISTS idx_funciones_proteina  ON funciones(proteina_id);
CREATE INDEX IF NOT EXISTS idx_funciones_go        ON funciones(termino_go);
CREATE INDEX IF NOT EXISTS idx_interacciones_prot  ON interacciones(proteina_id);
CREATE INDEX IF NOT EXISTS idx_ptms_proteina       ON ptms(proteina_id);
"""


class BaseDatos:
    """Acceso a la base SQLite del pipeline.

    Se usa como context manager::

        with BaseDatos(ruta) as db:
            db.guardar_proteina(proteina)
    """

    def __init__(self, ruta: str | Path) -> None:
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.conexion = sqlite3.connect(self.ruta)
        self.conexion.row_factory = sqlite3.Row
        # SQLite trae las claves foráneas desactivadas por defecto: sin esto,
        # el ON DELETE CASCADE del esquema no hace nada.
        self.conexion.execute("PRAGMA foreign_keys = ON")
        self.crear_esquema()

    def crear_esquema(self) -> None:
        """Crea tablas e índices si no existen. Idempotente."""
        self.conexion.executescript(ESQUEMA)
        self.conexion.commit()

    # ------------------------------------------------------------ escritura

    def guardar_proteina(self, proteina: Proteina) -> int:
        """Inserta o actualiza una proteína con todas sus anotaciones.

        Si el ``pdb_id`` ya existe, se reemplaza junto con sus anotaciones:
        una re-descarga tiene que dejar la base en el mismo estado que una
        descarga limpia, no acumular duplicados.

        Returns:
            El ``id`` de la fila en ``proteinas``.
        """
        with self.conexion:  # una sola transacción para las cuatro tablas
            cursor = self.conexion.execute(
                """
                INSERT INTO proteinas (
                    pdb_id, uniprot_id, nombre, organismo, tax_id, metodo,
                    resolucion, longitud, secuencia, archivo_path, sha256,
                    fecha_descarga, curada, motivo_rechazo
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(pdb_id) DO UPDATE SET
                    uniprot_id     = excluded.uniprot_id,
                    nombre         = excluded.nombre,
                    organismo      = excluded.organismo,
                    tax_id         = excluded.tax_id,
                    metodo         = excluded.metodo,
                    resolucion     = excluded.resolucion,
                    longitud       = excluded.longitud,
                    secuencia      = excluded.secuencia,
                    archivo_path   = excluded.archivo_path,
                    sha256         = excluded.sha256,
                    fecha_descarga = excluded.fecha_descarga,
                    curada         = excluded.curada,
                    motivo_rechazo = excluded.motivo_rechazo
                """,
                (
                    proteina.pdb_id,
                    proteina.uniprot_id,
                    proteina.nombre,
                    proteina.organismo,
                    proteina.tax_id,
                    proteina.metodo,
                    proteina.resolucion,
                    proteina.longitud,
                    proteina.secuencia,
                    proteina.archivo_path,
                    proteina.sha256,
                    proteina.fecha_descarga.isoformat(),
                    int(proteina.curada),
                    proteina.motivo_rechazo,
                ),
            )

            # ON CONFLICT DO UPDATE no devuelve lastrowid útil, así que se
            # resuelve el id por el pdb_id, que es único.
            fila = self.conexion.execute(
                "SELECT id FROM proteinas WHERE pdb_id = ?", (proteina.pdb_id,)
            ).fetchone()
            proteina_id = int(fila["id"])
            _ = cursor

            # Se borran las anotaciones anteriores antes de reinsertarlas.
            for tabla in ("funciones", "interacciones", "ptms"):
                self.conexion.execute(
                    f"DELETE FROM {tabla} WHERE proteina_id = ?", (proteina_id,)
                )

            self.conexion.executemany(
                """INSERT INTO funciones
                   (proteina_id, tipo, termino_go, descripcion, evidencia)
                   VALUES (?,?,?,?,?)""",
                [
                    (proteina_id, f.tipo, f.termino_go, f.descripcion, f.evidencia)
                    for f in proteina.funciones
                ],
            )
            self.conexion.executemany(
                """INSERT INTO interacciones
                   (proteina_id, partner_uniprot, partner_gen, tipo, fuente,
                    n_experimentos, evidencia)
                   VALUES (?,?,?,?,?,?,?)""",
                [
                    (
                        proteina_id,
                        i.partner_uniprot,
                        i.partner_gen,
                        i.tipo,
                        i.fuente,
                        i.n_experimentos,
                        i.evidencia,
                    )
                    for i in proteina.interacciones
                ],
            )
            self.conexion.executemany(
                """INSERT INTO ptms
                   (proteina_id, posicion, posicion_fin, tipo, descripcion, evidencia)
                   VALUES (?,?,?,?,?,?)""",
                [
                    (
                        proteina_id,
                        p.posicion,
                        p.posicion_fin,
                        p.tipo,
                        p.descripcion,
                        p.evidencia,
                    )
                    for p in proteina.ptms
                ],
            )

        return proteina_id

    def marcar_curada(
        self, pdb_id: str, curada: bool, motivo: str | None = None
    ) -> None:
        """Actualiza el veredicto de curación de una proteína ya guardada."""
        with self.conexion:
            self.conexion.execute(
                "UPDATE proteinas SET curada = ?, motivo_rechazo = ? WHERE pdb_id = ?",
                (int(curada), motivo, pdb_id.upper()),
            )

    # ------------------------------------------------------------- lectura

    def obtener_proteina(self, pdb_id: str) -> Proteina | None:
        """Una proteína con todas sus anotaciones, o ``None`` si no está."""
        fila = self.conexion.execute(
            "SELECT * FROM proteinas WHERE pdb_id = ?", (pdb_id.upper(),)
        ).fetchone()
        if fila is None:
            return None
        return self._fila_a_proteina(fila, con_anotaciones=True)

    def listar_proteinas(self, solo_curadas: bool = False) -> list[Proteina]:
        """Todas las proteínas, sin anotaciones (para listados rápidos)."""
        consulta = "SELECT * FROM proteinas"
        if solo_curadas:
            consulta += " WHERE curada = 1"
        consulta += " ORDER BY pdb_id"
        return [
            self._fila_a_proteina(f, con_anotaciones=False)
            for f in self.conexion.execute(consulta)
        ]

    def contar(self) -> dict[str, int]:
        """Cantidad de filas por tabla, más el estado de curación.

        Distingue tres estados, no dos: una proteína recién descargada está
        **sin curar**, que no es lo mismo que **rechazada**. Contar toda no
        curada como rechazada hacía que un ``fetch`` sin ``curate`` reportara
        rechazos que nunca ocurrieron.
        """
        conteos = {}
        for tabla in ("proteinas", "funciones", "interacciones", "ptms"):
            conteos[tabla] = int(
                self.conexion.execute(f"SELECT COUNT(*) AS n FROM {tabla}").fetchone()["n"]
            )
        conteos["curadas"] = int(
            self.conexion.execute(
                "SELECT COUNT(*) AS n FROM proteinas WHERE curada = 1"
            ).fetchone()["n"]
        )
        conteos["rechazadas"] = int(
            self.conexion.execute(
                "SELECT COUNT(*) AS n FROM proteinas "
                "WHERE curada = 0 AND motivo_rechazo IS NOT NULL"
            ).fetchone()["n"]
        )
        conteos["sin_curar"] = (
            conteos["proteinas"] - conteos["curadas"] - conteos["rechazadas"]
        )
        return conteos

    def _fila_a_proteina(self, fila: sqlite3.Row, con_anotaciones: bool) -> Proteina:
        from datetime import datetime

        proteina_id = int(fila["id"])
        datos: dict[str, Any] = {
            "pdb_id": fila["pdb_id"],
            "uniprot_id": fila["uniprot_id"],
            "nombre": fila["nombre"],
            "organismo": fila["organismo"],
            "tax_id": fila["tax_id"],
            "metodo": fila["metodo"],
            "resolucion": fila["resolucion"],
            "longitud": fila["longitud"],
            "secuencia": fila["secuencia"],
            "archivo_path": fila["archivo_path"],
            "sha256": fila["sha256"],
            "fecha_descarga": datetime.fromisoformat(fila["fecha_descarga"]),
            "curada": bool(fila["curada"]),
            "motivo_rechazo": fila["motivo_rechazo"],
        }

        if con_anotaciones:
            datos["funciones"] = [
                Funcion(
                    tipo=f["tipo"],
                    termino_go=f["termino_go"],
                    descripcion=f["descripcion"],
                    evidencia=f["evidencia"],
                )
                for f in self.conexion.execute(
                    "SELECT * FROM funciones WHERE proteina_id = ? ORDER BY id",
                    (proteina_id,),
                )
            ]
            datos["interacciones"] = [
                Interaccion(
                    partner_uniprot=i["partner_uniprot"],
                    partner_gen=i["partner_gen"],
                    tipo=i["tipo"],
                    fuente=i["fuente"],
                    n_experimentos=i["n_experimentos"],
                    evidencia=i["evidencia"],
                )
                for i in self.conexion.execute(
                    "SELECT * FROM interacciones WHERE proteina_id = ? ORDER BY id",
                    (proteina_id,),
                )
            ]
            datos["ptms"] = [
                PTM(
                    posicion=p["posicion"],
                    posicion_fin=p["posicion_fin"],
                    tipo=p["tipo"],
                    descripcion=p["descripcion"],
                    evidencia=p["evidencia"],
                )
                for p in self.conexion.execute(
                    "SELECT * FROM ptms WHERE proteina_id = ? ORDER BY posicion",
                    (proteina_id,),
                )
            ]

        return Proteina(**datos)

    # ----------------------------------------------------------------- ciclo

    def close(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self.conexion.close()

    def __enter__(self) -> "BaseDatos":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@contextlib.contextmanager
def abrir_base(ruta: str | Path) -> Iterator[BaseDatos]:
    """Abre la base y la cierra al salir del bloque."""
    db = BaseDatos(ruta)
    try:
        yield db
    finally:
        db.close()
