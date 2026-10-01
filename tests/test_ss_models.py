"""Tests de los modelos y las métricas de estructura secundaria (Fase 4).

Sin DSSP y sin el dataset real: las proteínas son sintéticas y la respuesta
correcta se conoce por construcción. Los modelos se entrenan unas pocas
épocas sobre problemas triviales, lo justo para ver que aprenden.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from pdpipe.phase4_ml import classifiers as clf
from pdpipe.phase4_ml.metrics import evaluar, f1_macro, matriz_de_confusion
from pdpipe.phase4_ml.models import Proteina, Split

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

# Por construcción: estos residuos son hélice, estos hebra, el resto coil.
HELICE = "AELMQK"
HEBRA = "VIYFWT"
COIL = "GPNDSC"


def sintetica(identificador: str, largo: int, rng: random.Random) -> Proteina:
    """Proteína cuya estructura se deduce del propio residuo.

    Un modelo que no aprende esto no aprende nada; uno que lo aprende llega a
    Q3 = 1. Sirve para probar el circuito completo sin depender de DSSP.
    """
    secuencia = "".join(rng.choice(HELICE + HEBRA + COIL) for _ in range(largo))
    estructura = "".join(
        "H" if r in HELICE else "E" if r in HEBRA else "C" for r in secuencia
    )
    return Proteina(identificador=identificador, secuencia=secuencia, estructura=estructura)


@pytest.fixture
def conjunto() -> list[Proteina]:
    rng = random.Random(0)
    return [sintetica(f"P{i:02d}", rng.randint(20, 60), rng) for i in range(24)]


# ---------------------------------------------------------------- métricas


def test_una_prediccion_perfecta():
    y = np.array([0, 1, 2, 2, 1, 0])
    m = evaluar(y, y, "perfecto")
    assert m.q3 == 1.0
    assert m.f1 == {"H": 1.0, "E": 1.0, "C": 1.0}
    assert m.confusion == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_el_q3_es_la_fraccion_de_aciertos():
    real = np.array([0, 0, 0, 1, 2])
    pred = np.array([0, 0, 1, 1, 0])
    assert evaluar(real, pred, "m").q3 == 0.6


def test_precision_y_recall_por_clase():
    # H: 3 reales, 2 acertadas, 1 predicha de más (de C) -> P=2/3, R=2/3
    real = np.array([0, 0, 0, 2, 2])
    pred = np.array([0, 0, 2, 0, 2])
    m = evaluar(real, pred, "m")
    assert m.precision["H"] == pytest.approx(2 / 3, abs=1e-4)
    assert m.recall["H"] == pytest.approx(2 / 3, abs=1e-4)
    assert m.f1["H"] == pytest.approx(2 / 3, abs=1e-4)


def test_una_clase_nunca_predicha_da_cero_y_no_nan():
    real = np.array([0, 1, 2])
    pred = np.array([0, 0, 2])
    m = evaluar(real, pred, "m")
    assert m.precision["E"] == 0.0
    assert m.f1["E"] == 0.0


def test_las_filas_de_la_matriz_son_las_clases_reales():
    matriz = matriz_de_confusion(np.array([1, 1]), np.array([0, 0]))
    assert matriz[1, 0] == 2
    assert matriz.sum() == 2


def test_largos_distintos_fallan():
    with pytest.raises(ValueError):
        matriz_de_confusion(np.array([0, 1]), np.array([0]))


def test_el_piso_usa_la_clase_que_se_le_pasa():
    real = np.array([0, 2, 2, 2])
    pred = real.copy()
    # Mayoritaria del propio conjunto (C): 3/4.
    assert evaluar(real, pred, "m").q3_base == 0.75
    # Mayoritaria del entrenamiento (H): solo 1/4 en este conjunto.
    assert evaluar(real, pred, "m", clase_base=0).q3_base == 0.25


def test_f1_macro():
    m = evaluar(np.array([0, 1, 2]), np.array([0, 1, 1]), "m")
    assert f1_macro(m) == pytest.approx((1 + 2 / 3 + 0) / 3, abs=1e-3)


# ---------------------------------------------------------------- etiquetas


def test_las_etiquetas_se_concatenan_en_orden():
    a = Proteina(identificador="a", secuencia="AA", estructura="HE")
    b = Proteina(identificador="b", secuencia="A", estructura="C")
    assert clf.etiquetas([a, b]).tolist() == [0, 1, 2]
    assert clf.etiquetas([]).shape == (0,)


# ---------------------------------------------------------------- baseline


@pytest.mark.parametrize("tipo", ["logistic", "random_forest"])
def test_el_baseline_aprende_un_problema_trivial(tipo, conjunto):
    modelo = clf.Baseline(tipo, ventana=3, seed=0)
    modelo.entrenar(conjunto[:18], conjunto[18:])
    pred = modelo.predecir(conjunto[18:])
    assert len(pred) == sum(len(p) for p in conjunto[18:])
    assert np.mean(pred == clf.etiquetas(conjunto[18:])) > 0.95


def test_un_baseline_desconocido_falla():
    with pytest.raises(clf.ErrorDeModelo):
        clf.Baseline("svm")


def test_predecir_sin_entrenar_falla(conjunto):
    with pytest.raises(clf.ErrorDeModelo):
        clf.Baseline("logistic").predecir(conjunto)


def test_el_baseline_se_guarda(tmp_path: Path, conjunto):
    modelo = clf.Baseline("logistic", ventana=3)
    modelo.entrenar(conjunto[:18], [])
    archivo = modelo.guardar(tmp_path)
    assert archivo.name == "modelo_logistic.pkl"
    assert archivo.stat().st_size > 0


# ---------------------------------------------------------------- BiLSTM


def test_invertir_dentro_del_largo_deja_el_relleno_al_final():
    x = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0, 5.0]]).unsqueeze(-1)
    largos = torch.tensor([3, 5])
    invertido = clf.invertir_dentro_del_largo(x, largos).squeeze(-1)
    assert invertido.tolist() == [[3.0, 2.0, 1.0, 0.0, 0.0], [5.0, 4.0, 3.0, 2.0, 1.0]]
    # Dos veces es la identidad.
    assert torch.equal(clf.invertir_dentro_del_largo(clf.invertir_dentro_del_largo(x, largos), largos), x)


def test_el_relleno_no_cambia_la_prediccion(conjunto):
    """Una proteína tiene que dar lo mismo sola que en un lote con otras más largas.

    Es la prueba de que la dirección inversa no lee el relleno: si lo leyera,
    la salida de la proteína corta dependería de con quién comparte lote.
    """
    modelo = clf.ClasificadorBiLSTM(oculto=8, capas=2, dropout=0.0, lote=8, seed=0)
    modelo.red = clf._red(22, 8, 2, 0.0, 3)
    modelo.red.eval()

    corta = min(conjunto, key=len)
    largas = sorted(conjunto, key=len)[-3:]

    with torch.no_grad():
        _, x, l, _ = next(modelo._lotes([corta], mezclar=False))
        sola = modelo.red(x, l)[0, : len(corta)]
        _, x, l, _ = next(modelo._lotes([corta, *largas], mezclar=False))
        en_lote = modelo.red(x, l)[0, : len(corta)]

    assert x.shape[1] > len(corta)  # de verdad hubo relleno
    assert torch.allclose(sola, en_lote, atol=1e-5)


def test_el_bilstm_aprende_un_problema_trivial(conjunto):
    modelo = clf.ClasificadorBiLSTM(
        oculto=16, capas=1, dropout=0.0, lote=4, epocas=25, tasa=1e-2, seed=0
    )
    historial = modelo.entrenar(conjunto[:18], conjunto[18:])
    pred = modelo.predecir(conjunto[18:])
    assert len(pred) == sum(len(p) for p in conjunto[18:])
    assert np.mean(pred == clf.etiquetas(conjunto[18:])) > 0.9
    assert historial.mejor_epoca >= 1


def test_el_bilstm_usa_el_contexto_hacia_atras():
    """La etiqueta de cada residuo es la del residuo *siguiente*.

    La LSTM hacia adelante, en la posición i, todavía no vio i+1: solo la
    dirección inversa tiene esa información. Si la inversión dentro del largo
    estuviera mal, el modelo no pasaría de adivinar.
    """
    rng = random.Random(1)
    proteinas = []
    for i in range(24):
        base = sintetica(f"S{i:02d}", rng.randint(20, 60), rng)
        corrida = base.estructura[1:] + "C"
        proteinas.append(
            Proteina(identificador=base.identificador, secuencia=base.secuencia, estructura=corrida)
        )
    modelo = clf.ClasificadorBiLSTM(
        oculto=16, capas=1, dropout=0.0, lote=4, epocas=40, tasa=1e-2, seed=0
    )
    modelo.entrenar(proteinas[:18], proteinas[18:])
    q3 = np.mean(modelo.predecir(proteinas[18:]) == clf.etiquetas(proteinas[18:]))
    assert q3 > 0.9


def test_el_bilstm_se_queda_con_la_mejor_epoca(conjunto):
    modelo = clf.ClasificadorBiLSTM(oculto=8, capas=1, lote=4, epocas=6, seed=0, paciencia=2)
    historial = modelo.entrenar(conjunto[:18], conjunto[18:])
    mejor = max(historial.epocas, key=lambda e: e.q3_val)
    assert historial.mejor_epoca == mejor.numero
    # El modelo final es el de la mejor época, no el de la última.
    q3 = float(np.mean(modelo.predecir(conjunto[18:]) == clf.etiquetas(conjunto[18:])))
    assert q3 == pytest.approx(mejor.q3_val, abs=1e-4)


def test_el_bilstm_es_reproducible(conjunto):
    def correr():
        modelo = clf.ClasificadorBiLSTM(oculto=8, capas=1, lote=4, epocas=2, seed=7)
        modelo.entrenar(conjunto[:18], conjunto[18:])
        return modelo.predecir(conjunto[18:])

    assert np.array_equal(correr(), correr())


def test_el_bilstm_se_guarda(tmp_path: Path, conjunto):
    modelo = clf.ClasificadorBiLSTM(oculto=8, capas=1, lote=4, epocas=1)
    modelo.entrenar(conjunto[:18], [])
    archivo = modelo.guardar(tmp_path)
    guardado = torch.load(archivo, weights_only=False)
    assert guardado["clases"] == ("H", "E", "C")
    assert guardado["hiperparametros"]["oculto"] == 8


def test_cuda_sin_gpu_cae_a_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert clf.resolver_dispositivo("cuda") == "cpu"
    assert clf.resolver_dispositivo("auto") == "cpu"
    assert clf.resolver_dispositivo("cpu") == "cpu"


# ---------------------------------------------------------------- entrenamiento


@pytest.fixture
def dataset_en_disco(tmp_path: Path, conjunto, config_dict):
    """Escribe un dataset sintético donde lo buscaría `pdpipe train`."""
    from pdpipe.config import Config

    procesados = tmp_path / "processed"
    procesados.mkdir()
    (procesados / "dataset_ss.json").write_text(
        json.dumps(
            [{"id": p.identificador, "secuencia": p.secuencia, "estructura": p.estructura}
             for p in conjunto]
        ),
        encoding="utf-8",
    )
    ids = [p.identificador for p in conjunto]
    split = Split(train=ids[:16], val=ids[16:20], test=ids[20:], n_grupos=len(ids))
    (procesados / "dataset_ss_split.json").write_text(split.model_dump_json(), encoding="utf-8")

    config_dict["paths"]["data_processed"] = str(procesados)
    config_dict["ml"].update(
        {"window_size": 3, "hidden_size": 8, "num_layers": 1, "batch_size": 4, "epochs": 2}
    )
    return Config.model_validate(config_dict)


def test_entrenar_varios_modelos_sobre_el_mismo_split(tmp_path: Path, dataset_en_disco):
    from pdpipe.phase4_ml.training import entrenar_modelos

    destino = tmp_path / "corrida"
    resultados = entrenar_modelos(
        dataset_en_disco, ["logistic", "bilstm"], destino, con_figuras=True
    )

    assert [r.nombre for r in resultados] == ["logistic", "bilstm"]
    # Mismo test para los dos: mismos residuos evaluados y mismo piso.
    assert resultados[0].test.n_residuos == resultados[1].test.n_residuos
    assert resultados[0].test.q3_base == resultados[1].test.q3_base

    metricas = json.loads((destino / "metricas_ss.json").read_text(encoding="utf-8"))
    assert [m["modelo"] for m in metricas["modelos"]] == ["logistic", "bilstm"]
    assert (destino / "modelo_logistic.pkl").is_file()
    assert (destino / "modelo_bilstm.pt").is_file()
    assert (destino / "confusion_bilstm.png").is_file()
    assert (destino / "aprendizaje_bilstm.png").is_file()
    # El baseline no tiene curva: se entrena de una.
    assert not (destino / "aprendizaje_logistic.png").exists()


def test_las_epocas_se_pueden_pisar(tmp_path: Path, dataset_en_disco):
    from pdpipe.phase4_ml.training import entrenar_modelos

    [resultado] = entrenar_modelos(
        dataset_en_disco, ["bilstm"], tmp_path / "c", epocas=1, con_figuras=False
    )
    assert len(resultado.historial["epocas"]) == 1


def test_un_split_inconsistente_falla(tmp_path: Path, dataset_en_disco):
    from pdpipe.phase4_ml.pipeline import ErrorDeDataset
    from pdpipe.phase4_ml.training import entrenar_modelos

    archivo = Path(dataset_en_disco.paths.data_processed) / "dataset_ss_split.json"
    archivo.write_text(Split(train=["NO_EXISTE"], test=["P00"]).model_dump_json(), encoding="utf-8")
    with pytest.raises(ErrorDeDataset, match="no están en el dataset"):
        entrenar_modelos(dataset_en_disco, ["logistic"], tmp_path / "c")


def test_sin_dataset_falla_con_instrucciones(tmp_path: Path, config_dict):
    from pdpipe.config import Config
    from pdpipe.phase4_ml.pipeline import ErrorDeDataset
    from pdpipe.phase4_ml.training import entrenar_modelos

    config_dict["paths"]["data_processed"] = str(tmp_path / "vacio")
    with pytest.raises(ErrorDeDataset, match="pdpipe dataset"):
        entrenar_modelos(Config.model_validate(config_dict), ["logistic"], tmp_path / "c")


def test_un_modelo_desconocido_falla_antes_de_cargar(tmp_path: Path, dataset_en_disco):
    from pdpipe.phase4_ml.training import entrenar_modelos

    with pytest.raises(clf.ErrorDeModelo):
        entrenar_modelos(dataset_en_disco, ["svm"], tmp_path / "c")
