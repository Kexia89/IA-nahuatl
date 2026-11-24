from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


BASE_DIR = Path(__file__).resolve().parent
MODES = {"aprendizaje": "Aprender náhuatl", "conversaciones": "Conversar en náhuatl"}
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_MESSAGE_LENGTH = 2000
MAX_BODY_BYTES = 16384
LOGGER = logging.getLogger("interfaz")


class ModelUnavailable(Exception):
    pass


class ModelStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self._predictors: dict[str, tuple[tuple[int, int], Any]] = {}
        self._locks = {mode: threading.Lock() for mode in MODES}

    def status(self) -> dict[str, Any]:
        models = []
        for mode, label in MODES.items():
            model = self.directory / mode / "modelo.h5"
            models.append({
                "mode": mode,
                "label": label,
                "available": model.is_file(),
                "loaded": mode in self._predictors,
            })
        return {"ok": True, "models": models, "max_message_length": MAX_MESSAGE_LENGTH}

    def predict(self, mode: str, message: str) -> dict[str, Any]:
        with self._locks[mode]:
            model_dir = self.directory / mode
            model_file = model_dir / "modelo.h5"
            if not model_file.is_file():
                raise ModelUnavailable(
                    "Todavía no hay un modelo entrenado para este modo. "
                    f"Ejecuta python entrenar.py --modo {mode} y actualiza el estado."
                )
            stat = model_file.stat()
            fingerprint = (stat.st_mtime_ns, stat.st_size)
            cached = self._predictors.get(mode)
            if cached is None or cached[0] != fingerprint:
                from motor import Predictor

                predictor = Predictor(model_dir)
                self._predictors[mode] = (fingerprint, predictor)
            else:
                predictor = cached[1]
            return {**predictor.predict(message), "mode": mode}


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, models_dir: Path) -> None:
        self.models = ModelStore(models_dir)
        super().__init__(("127.0.0.1", port), RequestHandler)


class RequestHandler(BaseHTTPRequestHandler):
    server: LocalServer
    server_version = "NahuatlLocal/1.0"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def _trusted_request(self) -> bool:
        port = self.server.server_port
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed_hosts.update({"127.0.0.1", "localhost"})
        if self.headers.get("Host", "").lower() not in allowed_hosts:
            self._json(HTTPStatus.FORBIDDEN, {"error": "Usa la dirección local del servidor."})
            return False
        origin = self.headers.get("Origin")
        if origin and origin.lower() not in {f"http://{host}" for host in allowed_hosts}:
            self._json(HTTPStatus.FORBIDDEN, {"error": "La petición debe venir de esta interfaz local."})
            return False
        return True

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, data, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if not self._trusted_request():
            return
        route = urlsplit(self.path).path
        if route in {"/health", "/api/health"}:
            self._json(HTTPStatus.OK, {"ok": True})
        elif route == "/api/status":
            self._json(HTTPStatus.OK, self.server.models.status())
        elif route in STATIC_FILES:
            filename, content_type = STATIC_FILES[route]
            try:
                data = (BASE_DIR / "web" / filename).read_bytes()
            except OSError:
                LOGGER.exception("No se pudo leer el archivo de interfaz %s", filename)
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Falta un archivo de la interfaz. Revisa la carpeta web."})
                return
            self._send(HTTPStatus.OK, data, content_type)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "No se encontró esta página."})

    def do_POST(self) -> None:
        if not self._trusted_request():
            return
        if urlsplit(self.path).path != "/api/chat":
            self._json(HTTPStatus.NOT_FOUND, {"error": "No se encontró esta función."})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Envía el mensaje en formato JSON."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE if length > MAX_BODY_BYTES else HTTPStatus.BAD_REQUEST, {"error": "El tamaño de la petición no es válido."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "El mensaje no tiene un formato JSON válido."})
            return
        if not isinstance(payload, dict):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "El mensaje debe ser un objeto JSON."})
            return
        mode = payload.get("mode")
        message = payload.get("message")
        if not isinstance(mode, str) or mode not in MODES:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Elige aprendizaje o conversaciones."})
            return
        if not isinstance(message, str) or not message.strip():
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Escribe un mensaje antes de enviar."})
            return
        if len(message) > MAX_MESSAGE_LENGTH:
            self._json(HTTPStatus.BAD_REQUEST, {"error": f"El mensaje debe tener como máximo {MAX_MESSAGE_LENGTH} caracteres."})
            return
        try:
            result = self.server.models.predict(mode, message.strip())
            self._json(HTTPStatus.OK, result)
        except ModelUnavailable as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc), "code": "model_missing", "mode": mode})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except (ImportError, ModuleNotFoundError):
            LOGGER.exception("Faltan dependencias para cargar el modelo")
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Faltan dependencias para usar el modelo. Instálalas con python -m pip install -r requirements.txt y reinicia la interfaz."})
        except Exception:
            LOGGER.exception("No se pudo predecir en el modo %s", mode)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "No se pudo usar el modelo. Revisa la consola del servidor y confirma que su carpeta contiene todos los archivos del entrenamiento."})

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interfaz local para practicar náhuatl de la Huasteca.")
    parser.add_argument("--port", type=int, default=8000, help="Puerto local; por defecto 8000.")
    parser.add_argument("--models-dir", type=Path, default=BASE_DIR / "models", help="Directorio que contiene aprendizaje/ y conversaciones/.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("El puerto debe estar entre 1 y 65535.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        server = LocalServer(args.port, args.models_dir)
    except OSError as exc:
        parser.exit(1, f"No se pudo iniciar el servidor: {exc}\nPrueba otro puerto con --port 8001.\n")
    print(f"\nPráctica de náhuatl: http://127.0.0.1:{server.server_port}\nModelos: {args.models_dir.resolve()}\nPara detener: Ctrl+C\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
