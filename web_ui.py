"""Tiny local Web UI for the Stable Diffusion CLI engine (generate.py).

Standard library only (http.server, json, urllib, pathlib, threading).
The pipeline is loaded ONCE on the first generation request and reused;
generation jobs are serialized so only one GPU job runs at a time.

Run:
    .venv\\Scripts\\python.exe web_ui.py

Then open http://127.0.0.1:8000 in a browser.
"""
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import generate

HOST = "127.0.0.1"
PORT = 8000

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
OUTPUT_DIR = ROOT / "outputs"

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}

# Safety limits (also hinted client-side; the server is authoritative).
MAX_STEPS = 50
MAX_WIDTH = 768
MAX_HEIGHT = 768
MAX_COUNT = 8
MAX_PROMPT_LEN = 500
# Shared pipeline, loaded lazily and kept for the whole process lifetime.
_pipe = None
_pipe_device = None
_pipe_lock = threading.Lock()   # guards pipeline initialization
_gen_lock = threading.Lock()    # allows only one generation job at a time


def get_pipeline():
    """Return the shared (pipe, device); the model is loaded only once.

    Raises RuntimeError if the model cannot be loaded or does not fit in VRAM.
    """
    global _pipe, _pipe_device
    with _pipe_lock:
        if _pipe is None:
            _pipe_device = generate.pick_device()
            _pipe = generate.load_pipeline(_pipe_device)
            print(f"[webui] pipeline loaded once on {_pipe_device}", flush=True)
        return _pipe, _pipe_device


def _as_int(value, name, minimum, maximum):
    """Return int(value) if it is an int in [minimum, maximum]; else raise."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def _as_dimension(value, name, maximum):
    """Return a valid image dimension, or raise ValueError with a clear reason.

    Dimension rules live in generate.dimension_error() so the CLI and the API
    can never drift apart: integers, >= 64, divisible by 8, <= maximum.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole number of pixels")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number of pixels") from None
    error = generate.dimension_error(number)
    if error:
        raise ValueError(f"{name} {error}")
    if number > maximum:
        raise ValueError(
            f"{name} must be at most {maximum} pixels on this GPU (got {number})"
        )
    return number


def validate_payload(payload):
    """Validate a /api/generate payload; return a normalized dict or raise."""
    if not isinstance(payload, dict):
        raise ValueError("body must be a JSON object")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if len(prompt) > MAX_PROMPT_LEN:
        raise ValueError(f"prompt too long (maximum {MAX_PROMPT_LEN} characters)")
    prompt = prompt.strip()

    negative = payload.get("negative") or ""
    if not isinstance(negative, str):
        raise ValueError("negative must be a string")
    negative = negative.strip()

    steps = _as_int(payload.get("steps", 25), "steps", 1, MAX_STEPS)

    guidance = payload.get("guidance", 7.5)
    if isinstance(guidance, bool):
        raise ValueError("guidance must be a number")
    try:
        guidance = float(guidance)
    except (TypeError, ValueError):
        raise ValueError("guidance must be a number") from None
    if guidance <= 0:
        raise ValueError("guidance must be greater than 0")

    width = _as_dimension(payload.get("width", 512), "width", MAX_WIDTH)
    height = _as_dimension(payload.get("height", 512), "height", MAX_HEIGHT)
    count = _as_int(payload.get("count", 1), "count", 1, MAX_COUNT)

    seed = payload.get("seed")
    if seed is None or seed == "":
        seed = None
    else:
        if isinstance(seed, bool):
            raise ValueError("seed must be an integer or blank")
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError("seed must be an integer or blank") from None

    return {
        "prompt": prompt,
        "negative": negative,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "seed": seed,
        "count": count,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalTxt2Img/1.0"

    # ----- response helpers -----
    def _send(self, status, body, mime):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status, payload):
        self._send(status, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _send_text(self, status, text):
        self._send(status, text.encode("utf-8"), "text/plain; charset=utf-8")

    def _serve_file(self, path, root, allowed_suffixes=None):
        """Serve `path` only if it resolves inside `root` (and, when given,
        has an allowed suffix). Never follows paths from the browser."""
        try:
            resolved = path.resolve()
            root_resolved = root.resolve()
        except OSError:
            self._send_text(404, "Not found")
            return
        if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
            self._send_text(404, "Not found")
            return
        if allowed_suffixes and resolved.suffix.lower() not in allowed_suffixes:
            self._send_text(404, "Not found")
            return
        mime = MIME_TYPES.get(resolved.suffix.lower(), "application/octet-stream")
        self._send(200, resolved.read_bytes(), mime)

    # ----- routes -----
    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        if path == "/":
            self._serve_file(STATIC_DIR / "index.html", STATIC_DIR)
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            if not rel or "/" in rel or "\\" in rel:
                self._send_text(404, "Not found")
                return
            self._serve_file(STATIC_DIR / rel, STATIC_DIR)
        elif path.startswith("/outputs/"):
            rel = path[len("/outputs/"):]
            if not rel or "/" in rel or "\\" in rel:
                self._send_text(404, "Not found")
                return
            self._serve_file(OUTPUT_DIR / rel, OUTPUT_DIR, allowed_suffixes={".png"})
        else:
            self._send_text(404, "Not found")

    def do_POST(self):
        if urlparse(self.path).path != "/api/generate":
            self._send_json(404, {"success": False, "error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(min(length, 1 << 20)) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"success": False, "error": "Invalid JSON body"})
            return

        try:
            params = validate_payload(payload)
        except ValueError as exc:
            self._send_json(400, {"success": False, "error": str(exc)})
            return

        if not _gen_lock.acquire(blocking=False):
            self._send_json(409, {"success": False,
                                  "error": "Generation already in progress"})
            return

        try:
            pipe, device = get_pipeline()
            t0 = time.perf_counter()
            images = generate.generate_images(
                params["prompt"],
                negative=params["negative"],
                steps=params["steps"],
                guidance=params["guidance"],
                width=params["width"],
                height=params["height"],
                seed=params["seed"],
                count=params["count"],
                device=device,
                pipe=pipe,
            )
            total_time = round(time.perf_counter() - t0, 2)
            self._send_json(200, {
                "success": True,
                "device": device,
                "total_time": total_time,
                "images": [
                    {"filename": im["filename"],
                     "url": f"/outputs/{im['filename']}",
                     "seed": im["seed"]}
                    for im in images
                ],
            })
        except RuntimeError as exc:
            # Our own actionable failures. Out-of-memory text contains only
            # dimensions and advice, so it is safe and useful to show. Anything
            # else (e.g. model missing) may embed a local path and stays in the
            # console.
            message = str(exc)
            if "out of memory" in message.lower():
                print(f"[webui] {message}", file=sys.stderr, flush=True)
                self._send_json(507, {"success": False, "error": message})
            else:
                print(f"[webui] generation failed: {exc!r}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                self._send_json(500, {
                    "success": False,
                    "error": ("Generation failed on the server. "
                              "Check the server console for details."),
                })
        except Exception as exc:  # one bad job must not kill the server
            # Full detail (paths, traceback) goes to THIS server's console only;
            # the browser gets a short, path-free message.
            print(f"[webui] generation failed: {exc!r}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            self._send_json(500, {
                "success": False,
                "error": ("Generation failed on the server. "
                          "Check the server console for details."),
            })
        finally:
            _gen_lock.release()


class NoReuseHTTPServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer that does NOT set SO_REUSEADDR.

    On Windows that flag lets a second process bind the same port while the
    first is still listening, silently creating ambiguous duplicate servers
    (the exact failure this project hit). With it off, the second bind fails
    with WSAEADDRINUSE and the clear 'could not bind' message is printed.
    """

    allow_reuse_address = False


def main():
    try:
        server = NoReuseHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        print(f"[webui] could not bind {HOST}:{PORT} - {exc}")
        raise SystemExit(1) from exc
    print("Local Text-to-Image UI", flush=True)
    print(f"http://{HOST}:{PORT}", flush=True)
    print(f"[webui] device: {generate.pick_device()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
