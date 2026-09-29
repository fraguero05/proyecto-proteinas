"""Guarda contra caracteres que la consola de Windows no puede imprimir.

El proyecto se desarrolla y se corre en Windows, donde la consola usa la
codepage cp1252. Un carácter fuera de ese conjunto no sale mal: **revienta**
con ``UnicodeEncodeError`` a mitad de la salida, y se lleva el comando entero.

Pasó de verdad con una flecha ``U+2192`` en una tabla de la CLI. La Å y el ²
sí están en cp1252, así que las unidades no son el problema; las flechas, los
signos matemáticos y los emoji sí.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RAIZ = Path(__file__).parent.parent / "src"
FUENTES = sorted(RAIZ.rglob("*.py"))


def _imprimibles_en_consola(texto: str) -> list[tuple[int, int]]:
    """Devuelve (línea, codepoint) de cada carácter que cp1252 no soporta."""
    problemas: list[tuple[int, int]] = []
    for i, ch in enumerate(texto):
        if ord(ch) < 128:
            continue
        try:
            ch.encode("cp1252")
        except UnicodeEncodeError:
            problemas.append((texto[:i].count("\n") + 1, ord(ch)))
    return problemas


@pytest.mark.parametrize("fuente", FUENTES, ids=lambda p: p.name)
def test_sin_caracteres_que_rompan_la_consola_de_windows(fuente: Path):
    problemas = _imprimibles_en_consola(fuente.read_text(encoding="utf-8"))

    detalle = ", ".join(f"línea {ln}: U+{cp:04X}" for ln, cp in problemas)
    assert not problemas, (
        f"{fuente.name} tiene caracteres que cp1252 no puede codificar "
        f"({detalle}). En la consola de Windows esto lanza UnicodeEncodeError."
    )


def test_el_detector_encuentra_una_flecha():
    """El guard tiene que servir para algo: se verifica con el caso real."""
    problemas = _imprimibles_en_consola("Atomos 660 \u2192 602")

    assert problemas == [(1, 0x2192)]


def test_la_a_con_anillo_y_el_cuadrado_si_estan_permitidos():
    """Las unidades que usa el análisis de MD no son el problema."""
    assert _imprimibles_en_consola("RMSD en \u00c5, SASA en \u00c5\u00b2") == []
