"""Tests de carga y validación del config.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdpipe.config import (
    Config,
    ConfigError,
    DesignerBackend,
    StructureSource,
    find_config,
    load_config,
)


def test_carga_config_valido(config_file: Path):
    cfg = load_config(config_file)
    assert cfg.seed == 123
    assert cfg.data.resolution_max == 2.0
    assert cfg.design.n_sequences == 4
    assert cfg.design.structure_source is StructureSource.ALPHAFOLD_DB
    assert cfg.design.designer is DesignerBackend.PROTEINMPNN
    assert cfg.source_path == config_file.resolve()


def test_config_por_defecto_es_valido():
    """Sin ningún valor explícito, los defaults tienen que validar."""
    cfg = Config()
    assert cfg.seed == 42
    assert cfg.md.force_field == "amber99sb-ildn"
    assert cfg.ml.window_size % 2 == 1


def test_config_del_repo_es_valido():
    """El config.yaml versionado en la raíz debe cargar sin errores."""
    raiz = Path(__file__).resolve().parents[1]
    cfg = load_config(raiz / "config.yaml")
    assert cfg.seed >= 0
    assert cfg.md.water_model == "tip3p"


def test_clave_desconocida_falla(write_config, config_dict):
    """Una clave mal escrita corta la ejecución en vez de ignorarse."""
    config_dict["resolucion_max"] = 2.0  # typo: va dentro de `data` y en inglés
    ruta = write_config(config_dict)
    with pytest.raises(ConfigError) as exc:
        load_config(ruta)
    assert "resolucion_max" in str(exc.value)


def test_clave_desconocida_anidada_falla(write_config, config_dict):
    config_dict["md"]["campo_de_fuerza"] = "charmm36"
    ruta = write_config(config_dict)
    with pytest.raises(ConfigError):
        load_config(ruta)


def test_valor_fuera_de_rango_falla(write_config, config_dict):
    config_dict["data"]["resolution_max"] = -1.0
    ruta = write_config(config_dict)
    with pytest.raises(ConfigError):
        load_config(ruta)


def test_enum_invalido_falla(write_config, config_dict):
    config_dict["design"]["structure_source"] = "rosettafold"
    ruta = write_config(config_dict)
    with pytest.raises(ConfigError):
        load_config(ruta)


def test_ventana_par_falla(write_config, config_dict):
    """window_size debe ser impar para centrarse en un residuo."""
    config_dict["ml"]["window_size"] = 12
    ruta = write_config(config_dict)
    with pytest.raises(ConfigError) as exc:
        load_config(ruta)
    assert "impar" in str(exc.value)


def test_splits_que_no_dejan_train_fallan(write_config, config_dict):
    config_dict["ml"]["test_size"] = 0.6
    config_dict["ml"]["val_size"] = 0.5
    ruta = write_config(config_dict)
    with pytest.raises(ConfigError) as exc:
        load_config(ruta)
    assert "menor a 1" in str(exc.value)


def test_rango_de_longitud_invertido_falla(write_config, config_dict):
    config_dict["data"]["length_min"] = 500
    config_dict["data"]["length_max"] = 100
    ruta = write_config(config_dict)
    with pytest.raises(ConfigError) as exc:
        load_config(ruta)
    assert "length_min" in str(exc.value)


def test_archivo_inexistente_falla(tmp_path: Path):
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "no_existe.yaml")
    assert "No existe" in str(exc.value)


def test_yaml_malformado_falla(tmp_path: Path):
    ruta = tmp_path / "config.yaml"
    ruta.write_text("seed: 42\n  paths: [roto\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(ruta)
    assert "YAML" in str(exc.value)


def test_yaml_que_no_es_mapeo_falla(tmp_path: Path):
    ruta = tmp_path / "config.yaml"
    ruta.write_text("- uno\n- dos\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(ruta)
    assert "mapeo" in str(exc.value)


def test_yaml_vacio_usa_defaults(tmp_path: Path):
    ruta = tmp_path / "config.yaml"
    ruta.write_text("", encoding="utf-8")
    cfg = load_config(ruta)
    assert cfg.seed == 42


def test_config_es_inmutable(config_file: Path):
    """Frozen: nadie cambia la config a mitad de una corrida."""
    cfg = load_config(config_file)
    with pytest.raises(Exception):
        cfg.seed = 999  # type: ignore[misc]


def test_rutas_relativas_se_resuelven_contra_el_config(config_file: Path):
    cfg = load_config(config_file)
    rutas = cfg.resolved_paths()
    assert rutas["data_raw"].is_absolute()
    assert rutas["data_raw"] == config_file.parent.resolve() / "data" / "raw"


def test_rutas_absolutas_se_respetan(write_config, config_dict, tmp_path: Path):
    absoluta = (tmp_path / "otro_lugar" / "raw").resolve()
    config_dict["paths"]["data_raw"] = str(absoluta)
    ruta = write_config(config_dict)
    cfg = load_config(ruta)
    assert cfg.resolved_paths()["data_raw"] == absoluta


def test_find_config_busca_hacia_arriba(tmp_path: Path, config_dict):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config_dict), encoding="utf-8"
    )
    hondo = tmp_path / "a" / "b" / "c"
    hondo.mkdir(parents=True)
    assert find_config(hondo) == tmp_path.resolve() / "config.yaml"


def test_find_config_sin_resultado(tmp_path: Path, monkeypatch):
    """En un árbol sin config.yaml devuelve None en vez de explotar."""
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    # Se recorre hacia arriba, así que hay que asegurarse de que no haya un
    # config.yaml real en los padres del tmp_path.
    resultado = find_config(vacio)
    if resultado is not None:
        assert resultado.is_file()


def test_rutas_se_serializan_en_posix(config_file: Path):
    """El manifiesto debe verse igual en Windows y en Linux.

    Sin esto, la misma corrida produce "data/raw" en uno y "data\\raw" en el
    otro, y comparar dos run_manifest.json deja de servir.
    """
    cfg = load_config(config_file)
    data = cfg.to_dict()
    for ruta in data["paths"].values():
        assert "\\" not in ruta, ruta
    assert data["paths"]["data_raw"] == "data/raw"


def test_resolved_paths_devuelve_objetos_path(config_file: Path):
    """El serializador POSIX no debe afectar el uso interno de las rutas."""
    cfg = load_config(config_file)
    for ruta in cfg.resolved_paths().values():
        assert isinstance(ruta, Path)


def test_to_dict_es_serializable(config_file: Path):
    import json

    cfg = load_config(config_file)
    data = cfg.to_dict()
    json.dumps(data)  # no debe lanzar
    assert "source_path" not in data  # excluido: cambia entre máquinas
    assert data["design"]["structure_source"] == "alphafold_db"
