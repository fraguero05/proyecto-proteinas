"""Clasificadores de estructura secundaria (Fase 4).

Dos familias, con la misma interfaz para que el resto del pipeline no sepa
cuál está usando:

* **Baseline clásico** (:class:`Baseline`) — regresión logística o random
  forest de scikit-learn sobre ventanas deslizantes one-hot. Cada residuo se
  clasifica mirando solo sus ``ventana`` vecinos. Es el método de los
  predictores de primera generación y el piso que el modelo profundo tiene
  que superar para justificar su costo.
* **BiLSTM** (:class:`ClasificadorBiLSTM`) — red recurrente bidireccional en
  PyTorch sobre la secuencia completa. Lee la cadena en los dos sentidos, así
  que cada residuo puede usar contexto de toda la proteína y no solo de una
  ventana fija.

Los dos devuelven sus predicciones **concatenadas por residuo, en el orden de
las proteínas recibidas**, que es el mismo orden en que :func:`etiquetas`
arma las reales. Eso es lo que permite evaluarlos con el mismo código.
"""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from pdpipe.phase4_ml import dataset as ds
from pdpipe.phase4_ml.dssp import CLASES_Q3
from pdpipe.phase4_ml.models import Proteina
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

#: Épocas sin mejora del Q3 de validación antes de cortar el entrenamiento.
#: El modelo que se guarda es siempre el de la mejor época, no el último.
PACIENCIA = 5

#: Etiqueta de las posiciones de relleno en un lote del BiLSTM. Es el valor
#: que ``CrossEntropyLoss`` ignora por defecto.
IGNORAR = -100


class ErrorDeModelo(Exception):
    """No se pudo entrenar o usar un modelo."""


def etiquetas(proteinas: list[Proteina]) -> np.ndarray:
    """Clases reales concatenadas por residuo, en el orden de las proteínas."""
    if not proteinas:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(
        [
            np.array([ds.INDICE_Q3[c] for c in p.estructura], dtype=np.int64)
            for p in proteinas
        ]
    )


@dataclass
class Epoca:
    """Una fila de la curva de aprendizaje."""

    numero: int
    #: ``None`` para los modelos que no se entrenan por épocas.
    perdida_train: float | None
    q3_val: float


@dataclass
class Historial:
    """Qué pasó durante el entrenamiento."""

    epocas: list[Epoca] = field(default_factory=list)
    mejor_epoca: int = 0
    corte_temprano: bool = False

    def a_dict(self) -> dict:
        return {
            "mejor_epoca": self.mejor_epoca,
            "corte_temprano": self.corte_temprano,
            "epocas": [e.__dict__ for e in self.epocas],
        }


class Clasificador(Protocol):
    nombre: str

    def entrenar(self, train: list[Proteina], val: list[Proteina]) -> Historial: ...

    def predecir(self, proteinas: list[Proteina]) -> np.ndarray: ...

    def guardar(self, destino: Path) -> Path: ...


# ---------------------------------------------------------------------------
# Baseline clásico
# ---------------------------------------------------------------------------


class Baseline:
    """Regresión logística o random forest sobre ventanas deslizantes."""

    TIPOS = ("logistic", "random_forest")

    def __init__(self, tipo: str, ventana: int = 17, seed: int = 42) -> None:
        if tipo not in self.TIPOS:
            raise ErrorDeModelo(
                f"Baseline desconocido: '{tipo}'. Opciones: {', '.join(self.TIPOS)}"
            )
        self.nombre = tipo
        self.ventana = ventana
        self.seed = seed
        self.modelo = None

    def _crear(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - depende del extra `ml`
            raise ErrorDeModelo(
                'scikit-learn no está instalado. Instalá el extra: uv pip install -e ".[ml]"'
            ) from exc

        if self.nombre == "logistic":
            # Sin regularización explícita más allá de la de lbfgs por
            # defecto (C=1): con 374 entradas binarias y decenas de miles de
            # filas no hay riesgo real de sobreajuste.
            return LogisticRegression(max_iter=1000, random_state=self.seed)

        # min_samples_leaf evita hojas de un solo residuo, que en este
        # problema son memorización pura de una ventana concreta.
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=self.seed,
        )

    def entrenar(self, train: list[Proteina], val: list[Proteina]) -> Historial:
        """Ajusta el modelo. Validación no se usa para elegir nada, solo se mide."""
        X, y, _ = ds.a_ventanas(train, self.ventana)
        if len(y) == 0:
            raise ErrorDeModelo("El conjunto de entrenamiento está vacío")

        logger.info("%s: entrenando sobre %d residuos", self.nombre, len(y))
        self.modelo = self._crear()
        self.modelo.fit(X, y)

        historial = Historial(mejor_epoca=1)
        if val:
            q3 = float(np.mean(self.predecir(val) == etiquetas(val)))
            historial.epocas.append(Epoca(1, None, round(q3, 4)))
        return historial

    def predecir(self, proteinas: list[Proteina]) -> np.ndarray:
        if self.modelo is None:
            raise ErrorDeModelo(f"{self.nombre}: hay que entrenar antes de predecir")
        X, _, _ = ds.a_ventanas(proteinas, self.ventana)
        if len(X) == 0:
            return np.zeros((0,), dtype=np.int64)
        return self.modelo.predict(X).astype(np.int64)

    def guardar(self, destino: Path) -> Path:
        destino = Path(destino) / f"modelo_{self.nombre}.pkl"
        with destino.open("wb") as f:
            pickle.dump(
                {
                    "tipo": self.nombre,
                    "ventana": self.ventana,
                    "alfabeto": ds.ALFABETO,
                    "clases": CLASES_Q3,
                    "modelo": self.modelo,
                },
                f,
            )
        return destino


# ---------------------------------------------------------------------------
# BiLSTM
# ---------------------------------------------------------------------------


def resolver_dispositivo(pedido: str) -> str:
    """Traduce ``cpu | cuda | auto`` al dispositivo que realmente se usa.

    Si se pide CUDA y no hay, se avisa y se sigue en CPU en vez de fallar: el
    resultado es el mismo modelo, solo más lento, y el dispositivo real queda
    registrado en el manifiesto.
    """
    import torch

    pedido = str(getattr(pedido, "value", pedido)).lower()
    hay_cuda = torch.cuda.is_available()
    if pedido == "auto":
        return "cuda" if hay_cuda else "cpu"
    if pedido == "cuda" and not hay_cuda:
        logger.warning("Se pidió CUDA pero no hay GPU disponible; se usa CPU")
        return "cpu"
    return pedido


def invertir_dentro_del_largo(x, largos):
    """Invierte cada secuencia del lote dentro de su propio largo.

    ``[a, b, c, -, -]`` con largo 3 queda ``[c, b, a, -, -]``: el relleno se
    queda al final. Aplicarla dos veces devuelve el lote original.
    """
    import torch

    maximo = x.shape[1]
    posiciones = torch.arange(maximo, device=x.device).unsqueeze(0)
    largos = largos.to(x.device).unsqueeze(1)
    indices = torch.where(posiciones < largos, largos - 1 - posiciones, posiciones)
    return x.gather(1, indices.unsqueeze(-1).expand_as(x))


def _red(n_entrada: int, oculto: int, capas: int, dropout: float, n_clases: int):
    """Construye la red. Se arma en una función para no importar torch al cargar el módulo.

    La bidireccionalidad se arma a mano, con dos LSTM unidireccionales por
    capa, en vez de usar ``nn.LSTM(bidirectional=True)``. El motivo es el
    relleno: con lotes de proteínas de distinto largo, la dirección inversa
    de una cadena corta arrancaría leyendo el relleno y llegaría a sus
    residuos con un estado contaminado por posiciones que no existen.

    La solución estándar es ``pack_padded_sequence``, pero en CPU es once
    veces más lenta (71 s contra 6 s por época, medido sobre el dataset del
    proyecto). Acá la dirección inversa recibe cada secuencia invertida
    *dentro de su largo* (:func:`invertir_dentro_del_largo`), así que el
    relleno siempre queda al final, donde solo afecta salidas que la pérdida
    ignora. El resultado es equivalente al empaquetado.
    """
    import torch
    from torch import nn

    class RedBiLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adelante = nn.ModuleList()
            self.atras = nn.ModuleList()
            for capa in range(capas):
                entrada = n_entrada if capa == 0 else 2 * oculto
                self.adelante.append(nn.LSTM(entrada, oculto, batch_first=True))
                self.atras.append(nn.LSTM(entrada, oculto, batch_first=True))
            self.dropout = nn.Dropout(dropout)
            self.salida = nn.Linear(2 * oculto, n_clases)

        def forward(self, x: torch.Tensor, largos: torch.Tensor) -> torch.Tensor:
            h = x
            for capa, (fwd, bwd) in enumerate(zip(self.adelante, self.atras)):
                ida, _ = fwd(h)
                vuelta, _ = bwd(invertir_dentro_del_largo(h, largos))
                h = torch.cat([ida, invertir_dentro_del_largo(vuelta, largos)], dim=-1)
                # Dropout entre capas y antes de la salida, como en nn.LSTM.
                h = self.dropout(h)
            return self.salida(h)

    return RedBiLSTM()


class ClasificadorBiLSTM:
    """BiLSTM sobre la secuencia completa, con corte temprano por Q3 de validación."""

    def __init__(
        self,
        oculto: int = 128,
        capas: int = 2,
        dropout: float = 0.3,
        lote: int = 8,
        epocas: int = 30,
        tasa: float = 1e-3,
        dispositivo: str = "cpu",
        seed: int = 42,
        paciencia: int = PACIENCIA,
    ) -> None:
        self.nombre = "bilstm"
        self.hiper = {
            "oculto": oculto,
            "capas": capas,
            "dropout": dropout,
            "lote": lote,
            "epocas": epocas,
            "tasa": tasa,
            "paciencia": paciencia,
        }
        self.dispositivo = resolver_dispositivo(dispositivo)
        self.seed = seed
        self.red = None

    # ------------------------------------------------------------ lotes

    def _lotes(self, proteinas: list[Proteina], mezclar: bool, generador=None):
        """Agrupa proteínas en lotes rellenos con su largo real."""
        import torch

        entradas, salidas, _ = ds.a_secuencias(proteinas)
        orden = list(range(len(entradas)))
        if mezclar:
            orden = torch.randperm(len(entradas), generator=generador).tolist()

        tam = self.hiper["lote"]
        for inicio in range(0, len(orden), tam):
            indices = orden[inicio : inicio + tam]
            largos = torch.tensor([len(entradas[i]) for i in indices])
            maximo = int(largos.max())
            x = torch.zeros((len(indices), maximo, len(ds.ALFABETO)))
            y = torch.full((len(indices), maximo), IGNORAR, dtype=torch.long)
            for fila, i in enumerate(indices):
                n = len(entradas[i])
                x[fila, :n] = torch.from_numpy(entradas[i])
                y[fila, :n] = torch.from_numpy(salidas[i])
            yield indices, x.to(self.dispositivo), largos, y.to(self.dispositivo)

    # ------------------------------------------------------------ interfaz

    def entrenar(self, train: list[Proteina], val: list[Proteina]) -> Historial:
        """Entrena con Adam y se queda con los pesos de la mejor época en validación.

        Elegir la época por validación —y no mirar el test— es lo que permite
        que el número del test sea una estimación honesta.
        """
        import torch
        from torch import nn

        if not train:
            raise ErrorDeModelo("El conjunto de entrenamiento está vacío")

        torch.manual_seed(self.seed)
        generador = torch.Generator().manual_seed(self.seed)

        h = self.hiper
        self.red = _red(len(ds.ALFABETO), h["oculto"], h["capas"], h["dropout"], len(CLASES_Q3))
        self.red.to(self.dispositivo)
        optimizador = torch.optim.Adam(self.red.parameters(), lr=h["tasa"])
        perdida_fn = nn.CrossEntropyLoss(ignore_index=IGNORAR)

        historial = Historial()
        mejor_q3 = -1.0
        mejor_estado = None
        sin_mejora = 0
        y_val = etiquetas(val)

        for epoca in range(1, h["epocas"] + 1):
            self.red.train()
            suma, residuos = 0.0, 0
            for _, x, largos, y in self._lotes(train, mezclar=True, generador=generador):
                optimizador.zero_grad()
                logits = self.red(x, largos)
                perdida = perdida_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                perdida.backward()
                # Las LSTM son propensas a gradientes explosivos en cadenas
                # largas; recortarlos estabiliza las primeras épocas.
                nn.utils.clip_grad_norm_(self.red.parameters(), max_norm=1.0)
                optimizador.step()
                n = int((y != IGNORAR).sum())
                suma += float(perdida.detach()) * n
                residuos += n

            q3_val = float(np.mean(self.predecir(val) == y_val)) if val else 0.0
            historial.epocas.append(
                Epoca(epoca, round(suma / max(residuos, 1), 4), round(q3_val, 4))
            )
            logger.info(
                "bilstm época %d: pérdida %.4f, Q3 val %.4f",
                epoca, suma / max(residuos, 1), q3_val,
            )

            if q3_val > mejor_q3:
                mejor_q3 = q3_val
                mejor_estado = copy.deepcopy(self.red.state_dict())
                historial.mejor_epoca = epoca
                sin_mejora = 0
            else:
                sin_mejora += 1
                if sin_mejora >= h["paciencia"]:
                    historial.corte_temprano = True
                    logger.info("Corte temprano en la época %d", epoca)
                    break

        if mejor_estado is not None:
            self.red.load_state_dict(mejor_estado)
        return historial

    def predecir(self, proteinas: list[Proteina]) -> np.ndarray:
        import torch

        if self.red is None:
            raise ErrorDeModelo("bilstm: hay que entrenar antes de predecir")
        if not proteinas:
            return np.zeros((0,), dtype=np.int64)

        self.red.eval()
        por_proteina: list[np.ndarray] = [None] * len(proteinas)  # type: ignore[list-item]
        with torch.no_grad():
            for indices, x, largos, _ in self._lotes(proteinas, mezclar=False):
                clases = self.red(x, largos).argmax(dim=-1).cpu().numpy()
                for fila, i in enumerate(indices):
                    por_proteina[i] = clases[fila, : int(largos[fila])]
        return np.concatenate(por_proteina).astype(np.int64)

    def guardar(self, destino: Path) -> Path:
        import torch

        destino = Path(destino) / "modelo_bilstm.pt"
        torch.save(
            {
                "hiperparametros": self.hiper,
                "alfabeto": ds.ALFABETO,
                "clases": CLASES_Q3,
                "estado": self.red.state_dict(),
            },
            destino,
        )
        return destino


MODELOS = ("bilstm", *Baseline.TIPOS)


def crear(nombre: str, config) -> Clasificador:
    """Instancia un clasificador con los hiperparámetros del config."""
    ml = config.ml
    if nombre == "bilstm":
        return ClasificadorBiLSTM(
            oculto=ml.hidden_size,
            capas=ml.num_layers,
            dropout=ml.dropout,
            lote=ml.batch_size,
            epocas=ml.epochs,
            tasa=ml.learning_rate,
            dispositivo=ml.device,
            seed=config.seed,
        )
    if nombre in Baseline.TIPOS:
        return Baseline(nombre, ventana=ml.window_size, seed=config.seed)
    raise ErrorDeModelo(f"Modelo desconocido: '{nombre}'. Opciones: {', '.join(MODELOS)}")


__all__ = [
    "MODELOS",
    "PACIENCIA",
    "Baseline",
    "Clasificador",
    "ClasificadorBiLSTM",
    "Epoca",
    "ErrorDeModelo",
    "Historial",
    "crear",
    "etiquetas",
    "resolver_dispositivo",
]
