"""Preparación del sistema para dinámica molecular (Fase 3, parte A).

Dos mitades bien distintas:

* **Limpieza de la estructura** (:func:`limpiar_estructura`) — corre con
  Biopython, sin GROMACS, así que se puede probar en cualquier máquina. Saca
  aguas cristalográficas, heteroátomos, hidrógenos preexistentes y
  conformaciones alternativas, que son las cuatro cosas que hacen fallar a
  ``pdb2gmx``.
* **Armado del sistema** (:func:`preparar_sistema`) — cinco llamadas a
  GROMACS: topología, caja, solvatación y iones. Necesita ``gmx``.

Lo que sale de acá es un sistema neutro, solvatado y con la topología
correspondiente, listo para las cuatro etapas de la parte B.
"""

from __future__ import annotations

from pathlib import Path

from pdpipe.config import Config
from pdpipe.phase3_md.gromacs import ClienteGromacs
from pdpipe.phase3_md.mdp import escribir_todos
from pdpipe.phase3_md.models import ResultadoLimpieza, SistemaPreparado
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

# Nombres con los que el PDB identifica el agua. DOD es agua deuterada, que
# aparece en estructuras de difracción de neutrones.
AGUAS = {"HOH", "WAT", "SOL", "TIP3", "DOD", "H2O"}

# Iones monoatómicos que sí conviene conservar cuando se piden: son parte del
# sitio activo en muchas enzimas, no contaminación del cristal.
IONES_ESTRUCTURALES = {"ZN", "MG", "CA", "FE", "MN", "CU", "NA", "K", "CL"}

# Cómo nombran los campos de fuerza a los iones en la sección [ molecules ].
# AMBER usa NA/CL; CHARMM, SOD/CLA; algunos escriben la carga en el nombre.
NOMBRES_DE_ION = {
    "NA", "CL", "K", "MG", "CA", "ZN", "LI", "BR", "I", "F",
    "NA+", "CL-", "K+", "SOD", "CLA", "POT",
}


def limpiar_estructura(
    entrada: str | Path,
    salida: str | Path,
    quitar_aguas: bool = True,
    quitar_heteroatomos: bool = True,
    quitar_hidrogenos: bool = True,
) -> ResultadoLimpieza:
    """Deja la estructura en condiciones de pasar por ``pdb2gmx``.

    Args:
        entrada: PDB o mmCIF de partida.
        salida: PDB limpio a escribir.
        quitar_aguas: saca las aguas cristalográficas. El solvente se agrega
            después con ``solvate``, así que las del cristal no aportan.
        quitar_heteroatomos: saca ligandos, cofactores e iones. ``pdb2gmx``
            no los conoce y falla; parametrizarlos es otro trabajo.
        quitar_hidrogenos: saca los hidrógenos que ya estuvieran. ``pdb2gmx``
            los agrega según el campo de fuerza, y los preexistentes pueden
            tener otra convención de nombres.

    Returns:
        Un :class:`ResultadoLimpieza` con el detalle de lo que se sacó.

    Raises:
        ValueError: si el archivo no se puede leer o queda sin átomos.
    """
    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select

    entrada = Path(entrada)
    salida = Path(salida)
    if not entrada.is_file():
        raise ValueError(f"No existe la estructura de entrada: {entrada}")

    parser = (
        MMCIFParser(QUIET=True)
        if entrada.suffix.lower() == ".cif"
        else PDBParser(QUIET=True)
    )
    try:
        estructura = parser.get_structure("entrada", str(entrada))
    except Exception as exc:  # noqa: BLE001 - Biopython lanza de todo
        raise ValueError(f"No se pudo leer {entrada.name}: {exc}") from exc

    modelos = list(estructura)
    if not modelos:
        raise ValueError(f"{entrada.name} no tiene modelos")

    conteo = {
        "aguas": 0,
        "hetero": 0,
        "hidrogenos": 0,
        "altloc": 0,
    }
    heteroatomos: dict[str, int] = {}
    iniciales = sum(1 for _ in estructura.get_atoms())

    class _Filtro(Select):
        """Decide átomo por átomo qué sobrevive."""

        def accept_model(self, model):
            # Una estructura de RMN trae varios modelos; se simula uno.
            return model.id == modelos[0].id

        def accept_residue(self, residue):
            nombre = residue.get_resname().strip().upper()
            hetflag = residue.get_id()[0]

            if nombre in AGUAS or hetflag == "W":
                if quitar_aguas:
                    conteo["aguas"] += 1
                    return False
                return True

            if hetflag.startswith("H_"):
                if quitar_heteroatomos:
                    conteo["hetero"] += 1
                    heteroatomos[nombre] = heteroatomos.get(nombre, 0) + 1
                    return False
                return True

            return True

        def accept_atom(self, atom):
            if quitar_hidrogenos and atom.element == "H":
                conteo["hidrogenos"] += 1
                return False
            # Conformaciones alternativas: se conserva la primera. GROMACS no
            # sabe qué hacer con dos posiciones para el mismo átomo.
            altloc = atom.get_altloc()
            if altloc not in (" ", "", "A"):
                conteo["altloc"] += 1
                return False
            return True

    salida.parent.mkdir(parents=True, exist_ok=True)
    escritor = PDBIO()
    escritor.set_structure(estructura)
    escritor.save(str(salida), select=_Filtro())

    finales = _contar_atomos_pdb(salida)
    if finales == 0:
        raise ValueError(
            f"La limpieza dejó {salida.name} sin átomos. "
            "¿La estructura es solo agua y heteroátomos?"
        )

    advertencias: list[str] = []
    estructurales = sorted(set(heteroatomos) & IONES_ESTRUCTURALES)
    if estructurales:
        advertencias.append(
            "Se quitaron iones que suelen ser estructurales: "
            f"{', '.join(estructurales)}. Si alguno es parte del sitio activo, "
            "la simulación no lo va a tener en cuenta."
        )
    otros = sorted(set(heteroatomos) - IONES_ESTRUCTURALES)
    if otros:
        advertencias.append(
            f"Se quitaron estos heteroátomos: {', '.join(otros)}. "
            "Parametrizar un ligando para AMBER es un trabajo aparte."
        )
    if len(modelos) > 1:
        advertencias.append(
            f"La estructura tenía {len(modelos)} modelos y se conservó el primero."
        )

    resultado = ResultadoLimpieza(
        entrada=entrada,
        salida=salida,
        atomos_iniciales=iniciales,
        atomos_finales=finales,
        aguas_quitadas=conteo["aguas"],
        heteroatomos_quitados=conteo["hetero"],
        hidrogenos_quitados=conteo["hidrogenos"],
        altloc_descartadas=conteo["altloc"],
        modelos_descartados=max(0, len(modelos) - 1),
        heteroatomos=heteroatomos,
        advertencias=advertencias,
    )

    logger.info(
        "Limpieza: %d -> %d átomos (%d aguas, %d heteroátomos, %d hidrógenos)",
        iniciales,
        finales,
        conteo["aguas"],
        conteo["hetero"],
        conteo["hidrogenos"],
    )
    for aviso in advertencias:
        logger.warning("%s", aviso)
    return resultado


def _contar_atomos_pdb(ruta: Path) -> int:
    """Cuenta registros ATOM/HETATM de un PDB ya escrito."""
    with ruta.open(encoding="utf-8") as handle:
        return sum(1 for ln in handle if ln.startswith(("ATOM", "HETATM")))


def preparar_sistema(
    config: Config,
    estructura: str | Path,
    directorio: str | Path,
    cliente: ClienteGromacs | None = None,
) -> SistemaPreparado:
    """Arma el sistema solvatado y neutro (parte A completa).

    Los cinco pasos, en orden:

    1. ``pdb2gmx`` — topología y agregado de hidrógenos según el campo de fuerza.
    2. ``editconf`` — mete la proteína en una caja con el margen configurado.
    3. ``solvate`` — llena la caja de agua.
    4. ``grompp`` — arma el ``.tpr`` que necesita ``genion``.
    5. ``genion`` — reemplaza aguas por iones hasta neutralizar.

    Raises:
        GromacsNoDisponible: si falta ``gmx``.
        ErrorDeGromacs: si alguno de los pasos falla.
    """
    md = config.md
    gmx = cliente or ClienteGromacs(md.gromacs_bin)
    gmx.verificar_disponible()

    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    limpio = directorio / "limpio.pdb"
    limpieza = limpiar_estructura(estructura, limpio)

    # 1. Topología. -ignh porque los hidrógenos los pone el campo de fuerza.
    gmx.correr(
        "pdb2gmx",
        "-f", limpio.name,
        "-o", "procesado.gro",
        "-p", "topol.top",
        "-i", "posre.itp",
        "-ff", md.force_field,
        "-water", md.water_model,
        "-ignh",
        directorio=directorio,
        etiqueta="pdb2gmx (topología)",
    )

    # 2. Caja. -c centra la proteína, -d deja el margen mínimo hasta el borde:
    # si queda más cerca, la proteína interactúa con su propia imagen
    # periódica y el resultado no significa nada.
    gmx.correr(
        "editconf",
        "-f", "procesado.gro",
        "-o", "caja.gro",
        "-c",
        "-d", str(md.box_padding_nm),
        "-bt", md.box_shape.value,
        directorio=directorio,
        etiqueta="editconf (caja)",
    )

    # 3. Solvatación con la caja de agua preequilibrada que trae GROMACS.
    gmx.correr(
        "solvate",
        "-cp", "caja.gro",
        "-cs", "spc216.gro",
        "-o", "solvatado.gro",
        "-p", "topol.top",
        directorio=directorio,
        etiqueta="solvate (agua)",
    )

    mdps = escribir_todos(md, directorio, seed=config.seed)

    # 4. El .tpr intermedio que genion necesita para conocer las cargas.
    # maxwarn 1 porque el sistema todavía no es neutro y grompp lo avisa:
    # es justamente lo que genion va a arreglar en el paso siguiente.
    gmx.correr(
        "grompp",
        "-f", mdps["minim"].name,
        "-c", "solvatado.gro",
        "-p", "topol.top",
        "-o", "iones.tpr",
        "-maxwarn", "1",
        directorio=directorio,
        etiqueta="grompp (previo a iones)",
    )

    # 5. Iones. Se responde "SOL" por stdin: es el grupo del que genion saca
    # moléculas de agua para reemplazarlas por iones.
    gmx.correr(
        "genion",
        "-s", "iones.tpr",
        "-o", "sistema.gro",
        "-p", "topol.top",
        "-pname", "NA",
        "-nname", "CL",
        "-neutral",
        "-conc", str(md.ion_concentration_m),
        entrada="SOL\n",
        directorio=directorio,
        etiqueta="genion (iones)",
    )

    sistema = directorio / "sistema.gro"
    n_atomos, n_aguas = _resumen_gro(sistema)

    resultado = SistemaPreparado(
        directorio=directorio,
        estructura=sistema,
        topologia=directorio / "topol.top",
        restricciones=directorio / "posre.itp",
        mdp=mdps,
        campo_de_fuerza=md.force_field,
        modelo_de_agua=md.water_model,
        forma_de_caja=md.box_shape.value,
        n_atomos=n_atomos,
        n_aguas=n_aguas,
        iones=_iones_de_topologia(directorio / "topol.top"),
        limpieza=limpieza,
        version_gromacs=gmx.version(),
    )

    logger.info(
        "Sistema listo: %d átomos, %d aguas, iones %s",
        n_atomos,
        n_aguas,
        resultado.iones or "(ninguno)",
    )
    return resultado


def _resumen_gro(ruta: Path) -> tuple[int, int]:
    """Cuenta átomos y moléculas de agua de un ``.gro``.

    El formato ``.gro`` pone el total de átomos en la segunda línea y un átomo
    por línea después, con el nombre de residuo en las columnas 6 a 10.
    """
    if not ruta.is_file():
        return 0, 0

    with ruta.open(encoding="utf-8") as handle:
        handle.readline()  # título
        try:
            total = int(handle.readline().strip())
        except ValueError:
            return 0, 0
        aguas = 0
        for _ in range(total):
            linea = handle.readline()
            if not linea:
                break
            if linea[5:10].strip().upper() in AGUAS:
                aguas += 1
    # Cada agua TIP3P son tres átomos.
    return total, aguas // 3


def _iones_de_topologia(topologia: Path) -> dict[str, int]:
    """Lee los iones de la sección ``[ molecules ]`` del ``topol.top``.

    Antes esto se sacaba raspando la salida de consola de ``genion`` con un
    regex, y contaba de más: ``genion`` menciona la misma cantidad en varias
    líneas, así que un sistema con 25 NA y 25 CL se reportaba como 25 y 50.
    El error no rompía nada —la simulación estaba bien— pero ensuciaba el
    ``run_manifest.json``, que es material de tesis.

    El ``topol.top`` es el registro autoritativo: es lo que ``grompp`` va a
    leer después, así que si dice 25 CL, hay 25 CL.
    """
    if not topologia.is_file():
        return {}

    iones: dict[str, int] = {}
    en_moleculas = False
    for linea in topologia.read_text(encoding="utf-8").splitlines():
        limpia = linea.split(";", 1)[0].strip()
        if not limpia:
            continue
        if limpia.startswith("["):
            en_moleculas = limpia.replace(" ", "").lower() == "[molecules]"
            continue
        if not en_moleculas:
            continue

        partes = limpia.split()
        if len(partes) < 2:
            continue
        nombre = partes[0].upper()
        if nombre in NOMBRES_DE_ION:
            try:
                iones[nombre] = iones.get(nombre, 0) + int(partes[1])
            except ValueError:
                continue
    return iones


__all__ = [
    "AGUAS",
    "IONES_ESTRUCTURALES",
    "NOMBRES_DE_ION",
    "limpiar_estructura",
    "preparar_sistema",
]
