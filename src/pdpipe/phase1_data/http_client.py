"""Cliente HTTP con reintentos, timeouts y caché en disco (Fase 1).

RCSB y UniProt son servicios públicos y gratuitos: se caen, devuelven 503 bajo
carga y cortan conexiones. Un script que no reintenta falla a la mitad de una
descarga de cincuenta estructuras y hay que empezar de nuevo.

La caché en disco cumple dos funciones: evita re-descargar lo mismo, y deja el
crudo exactamente como llegó, que es lo que el ``run_manifest.json`` hashea.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pdpipe import __version__
from pdpipe.utils.logging import get_logger

logger = get_logger(__name__)

# Las APIs públicas piden identificarse. Si algo de este pipeline genera
# tráfico anómalo, que sepan a quién escribirle antes de bloquear el rango.
USER_AGENT = (
    f"pdpipe/{__version__} (pipeline académico de diseño de proteínas; "
    f"https://github.com/fraguero05/proyecto-proteinas)"
)

# Códigos que vale la pena reintentar: throttling y errores transitorios del
# servidor. Un 404 no se reintenta, no va a aparecer solo.
_CODIGOS_REINTENTABLES = (429, 500, 502, 503, 504)


class ErrorDeRed(Exception):
    """Falló una petición después de agotar los reintentos."""


class RecursoNoEncontrado(ErrorDeRed):
    """El servidor respondió 404: el identificador no existe."""


class ClienteHTTP:
    """Sesión HTTP reutilizable con reintentos exponenciales y caché opcional.

    Args:
        timeout_s: timeout por petición.
        max_retries: reintentos ante errores transitorios.
        cache_dir: si se pasa, las respuestas se guardan y se releen de ahí.
        usar_cache: permite desactivar la caché sin cambiar la ruta.
    """

    def __init__(
        self,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        cache_dir: str | Path | None = None,
        usar_cache: bool = True,
    ) -> None:
        self.timeout_s = timeout_s
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.usar_cache = usar_cache and self.cache_dir is not None

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

        # backoff_factor=1 -> espera 1s, 2s, 4s entre reintentos. Suficiente
        # para que un 503 momentáneo se resuelva sin castigar al servidor.
        reintentos = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=_CODIGOS_REINTENTABLES,
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )
        adaptador = HTTPAdapter(max_retries=reintentos)
        self.session.mount("https://", adaptador)
        self.session.mount("http://", adaptador)

    # ------------------------------------------------------------- caché

    def _ruta_cache(self, nombre: str) -> Path | None:
        if not self.usar_cache or self.cache_dir is None:
            return None
        return self.cache_dir / nombre

    def _leer_cache(self, nombre: str) -> bytes | None:
        ruta = self._ruta_cache(nombre)
        if ruta is not None and ruta.is_file():
            logger.debug("Caché: %s", ruta.name)
            return ruta.read_bytes()
        return None

    def _escribir_cache(self, nombre: str, contenido: bytes) -> Path | None:
        ruta = self._ruta_cache(nombre)
        if ruta is None:
            return None
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(contenido)
        return ruta

    # ---------------------------------------------------------- peticiones

    def get_bytes(self, url: str, nombre_cache: str | None = None) -> bytes:
        """GET que devuelve el cuerpo crudo, pasando por la caché si hay."""
        if nombre_cache:
            cacheado = self._leer_cache(nombre_cache)
            if cacheado is not None:
                return cacheado

        logger.debug("GET %s", url)
        try:
            respuesta = self.session.get(url, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise ErrorDeRed(f"No se pudo conectar a {url}: {exc}") from exc

        if respuesta.status_code == 404:
            raise RecursoNoEncontrado(f"404 en {url}")
        if not respuesta.ok:
            raise ErrorDeRed(
                f"{respuesta.status_code} {respuesta.reason} en {url}"
            )

        if nombre_cache:
            self._escribir_cache(nombre_cache, respuesta.content)
        return respuesta.content

    def get_json(self, url: str, nombre_cache: str | None = None) -> dict[str, Any]:
        """GET que parsea la respuesta como JSON."""
        crudo = self.get_bytes(url, nombre_cache=nombre_cache)
        try:
            return json.loads(crudo)
        except json.JSONDecodeError as exc:
            raise ErrorDeRed(f"{url} no devolvió JSON válido: {exc}") from exc

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST con cuerpo JSON. No se cachea: las búsquedas cambian seguido."""
        logger.debug("POST %s", url)
        try:
            respuesta = self.session.post(url, json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise ErrorDeRed(f"No se pudo conectar a {url}: {exc}") from exc

        # La Search API del RCSB devuelve 204 cuando no hay resultados.
        if respuesta.status_code == 204:
            return {}
        if not respuesta.ok:
            raise ErrorDeRed(
                f"{respuesta.status_code} {respuesta.reason} en {url}: "
                f"{respuesta.text[:200]}"
            )

        try:
            return respuesta.json()
        except json.JSONDecodeError as exc:
            raise ErrorDeRed(f"{url} no devolvió JSON válido: {exc}") from exc

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "ClienteHTTP":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
