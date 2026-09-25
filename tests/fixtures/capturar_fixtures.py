"""Captura respuestas reales de RCSB y UniProt como fixtures de test.

Se ejecuta a mano, no desde pytest: los tests no tocan la red. Cuando una API
cambie de forma, se vuelve a correr esto y se revisa el diff.

    python tests/fixtures/capturar_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import json

import requests

AQUI = Path(__file__).parent
TIMEOUT = 30

# Proteínas de uso académico estándar, las mismas del Hito 1.
ENTRADAS_PDB = ["1UBQ", "1LYZ"]
ACCESIONES_UNIPROT = {
    "P0CG48": "poliubiquitina C humana",
    "P00698": "lisozima C de clara de huevo",
}

# Las respuestas de UniProt traen cientos de referencias cruzadas; para los
# tests alcanza con una muestra representativa de cada base.
MAX_XREFS_POR_BASE = 5

# Los GO son la fuente de la tabla `funciones` y se reparten en tres
# ontologías: con solo 5 se pierden ontologías enteras del fixture.
MAX_XREFS_GO = 40

# Entidades poliméricas a capturar: de ahí salen la longitud y la secuencia
# de la cadena efectivamente cristalizada.
ENTIDADES = [("1UBQ", "1"), ("1LYZ", "1")]

# Modelos de AlphaFold DB (Fase 2). Se guardan los metadatos y el .pdb, del
# que se extrae el pLDDT del campo B-factor.
ACCESIONES_ALPHAFOLD = ["P00698"]


def _get(url: str) -> dict:
    respuesta = requests.get(url, timeout=TIMEOUT)
    respuesta.raise_for_status()
    return respuesta.json()


def _recortar_uniprot(data: dict) -> dict:
    """Deja una muestra de xrefs por base de datos para que el fixture no pese MB."""
    xrefs = data.get("uniProtKBCrossReferences", [])
    por_base: dict[str, list] = {}
    for ref in xrefs:
        por_base.setdefault(ref.get("database", "?"), []).append(ref)

    recortadas = []
    for base, refs in por_base.items():
        tope = MAX_XREFS_GO if base == "GO" else MAX_XREFS_POR_BASE
        recortadas.extend(refs[:tope])
    data["uniProtKBCrossReferences"] = recortadas
    data["_fixture_nota"] = (
        f"xrefs recortadas a {MAX_XREFS_POR_BASE} por base de datos "
        f"(originales: {len(xrefs)})"
    )
    return data


def main() -> None:
    for pdb_id in ENTRADAS_PDB:
        data = _get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
        destino = AQUI / f"rcsb_entry_{pdb_id}.json"
        destino.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{destino.name}: {destino.stat().st_size // 1024} KiB")

        estructura = requests.get(
            f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=TIMEOUT
        )
        estructura.raise_for_status()
        pdb_destino = AQUI / f"{pdb_id}.pdb"
        pdb_destino.write_bytes(estructura.content)
        print(f"{pdb_destino.name}: {pdb_destino.stat().st_size // 1024} KiB")

    for pdb_id, entity_id in ENTIDADES:
        data = _get(
            f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
        )
        destino = AQUI / f"rcsb_entity_{pdb_id}_{entity_id}.json"
        destino.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{destino.name}: {destino.stat().st_size // 1024} KiB")

    for accesion, descripcion in ACCESIONES_UNIPROT.items():
        data = _recortar_uniprot(_get(f"https://rest.uniprot.org/uniprotkb/{accesion}.json"))
        destino = AQUI / f"uniprot_{accesion}.json"
        destino.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{destino.name} ({descripcion}): {destino.stat().st_size // 1024} KiB")

    for accesion in ACCESIONES_ALPHAFOLD:
        data = _get(f"https://alphafold.ebi.ac.uk/api/prediction/{accesion}")
        destino = AQUI / f"alphafold_{accesion}.json"
        destino.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{destino.name}: {destino.stat().st_size // 1024} KiB")

        prediccion = data[0] if isinstance(data, list) else data
        modelo = requests.get(prediccion["pdbUrl"], timeout=TIMEOUT)
        modelo.raise_for_status()
        nombre = prediccion["pdbUrl"].rsplit("/", 1)[-1]
        (AQUI / nombre).write_bytes(modelo.content)
        print(f"{nombre}: {(AQUI / nombre).stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()
