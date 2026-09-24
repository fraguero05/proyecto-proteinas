"""Tests de los hashes SHA-256 usados para trazar entradas y salidas."""

from __future__ import annotations

from pathlib import Path

import hashlib

import pytest

from pdpipe.utils.checksums import file_sha256, hash_directory, short_hash


def test_hash_coincide_con_hashlib(tmp_path: Path):
    archivo = tmp_path / "datos.txt"
    contenido = b"ATOM      1  N   ALA A   1\n"
    archivo.write_bytes(contenido)
    assert file_sha256(archivo) == hashlib.sha256(contenido).hexdigest()


def test_hash_de_archivo_vacio(tmp_path: Path):
    archivo = tmp_path / "vacio.txt"
    archivo.touch()
    assert file_sha256(archivo) == hashlib.sha256(b"").hexdigest()


def test_hash_es_estable_entre_llamadas(pdb_fixture: Path):
    assert file_sha256(pdb_fixture) == file_sha256(pdb_fixture)


def test_hash_cambia_con_el_contenido(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("uno", encoding="utf-8")
    b.write_text("dos", encoding="utf-8")
    assert file_sha256(a) != file_sha256(b)


def test_archivos_grandes_se_leen_por_bloques(tmp_path: Path):
    """Más de un chunk de 1 MiB: verifica que el streaming no corrompa el hash."""
    archivo = tmp_path / "grande.bin"
    contenido = b"x" * (1024 * 1024 + 517)
    archivo.write_bytes(contenido)
    assert file_sha256(archivo) == hashlib.sha256(contenido).hexdigest()


def test_archivo_inexistente_falla(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "fantasma.pdb")


def test_directorio_no_es_archivo(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path)


def test_hash_directory_recursivo(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "uno.pdb").write_text("A", encoding="utf-8")
    (tmp_path / "sub" / "dos.pdb").write_text("B", encoding="utf-8")

    resultado = hash_directory(tmp_path)
    assert set(resultado) == {"uno.pdb", "sub/dos.pdb"}  # rutas POSIX siempre


def test_hash_directory_con_patron(tmp_path: Path):
    (tmp_path / "estructura.pdb").write_text("A", encoding="utf-8")
    (tmp_path / "notas.txt").write_text("B", encoding="utf-8")

    resultado = hash_directory(tmp_path, pattern="*.pdb")
    assert set(resultado) == {"estructura.pdb"}


def test_hash_directory_no_recursivo(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "uno.pdb").write_text("A", encoding="utf-8")
    (tmp_path / "sub" / "dos.pdb").write_text("B", encoding="utf-8")

    resultado = hash_directory(tmp_path, recursive=False)
    assert set(resultado) == {"uno.pdb"}


def test_hash_directory_ordenado(tmp_path: Path):
    for nombre in ("c.pdb", "a.pdb", "b.pdb"):
        (tmp_path / nombre).write_text(nombre, encoding="utf-8")
    assert list(hash_directory(tmp_path)) == ["a.pdb", "b.pdb", "c.pdb"]


def test_hash_directory_en_ruta_invalida(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        hash_directory(tmp_path / "no_existe")


def test_short_hash():
    completo = "a" * 64
    assert short_hash(completo) == "aaaaaaaa"
    assert len(short_hash(completo, length=12)) == 12
