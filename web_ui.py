"""Tiny local Web UI for the Stable Diffusion CLI engine (generate.py).

Standard library only (http.server, json, urllib, pathlib, threading).
The pipeline is loaded ONCE on the first generation request and reused;
generation jobs are serialized so only one GPU job runs at a time.

Run:
    .venv\\Scripts\\python.exe web_ui.py

Then open http://127.0.0.1:8000 in a browser.
"""
import json
import os
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
HISTORY_FILE = OUTPUT_DIR / "history.json"

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
MAX_HISTORY = 20
_history: list[dict] = []   # newest first

# --- Request-level timeout for generation ---
# A safe request/response guard: the HTTP handler waits for the generation
# worker to complete within this window. If it does not, a 504 is returned
# to the client while the worker continues holding _gen_lock and using the GPU.
# The actual CUDA operation is NEVER forcibly terminated.
GENERATION_TIMEOUT = 600  # seconds

# --- Graceful shutdown ---
_shutting_down = threading.Event()

# --- Generation progress (thread-safe) ---
_progress_lock = threading.Lock()
_progress_active = False
_progress_current_step = 0
_progress_total_steps = 0


def _reset_progress(total_steps=0):
    with _progress_lock:
        global _progress_active, _progress_current_step, _progress_total_steps
        _progress_active = total_steps > 0
        _progress_current_step = 0
        _progress_total_steps = total_steps


def _set_progress_step(step):
    global _progress_current_step
    with _progress_lock:
        _progress_current_step = step


def _get_progress():
    with _progress_lock:
        return {
            "active": _progress_active,
            "current_step": _progress_current_step,
            "total_steps": _progress_total_steps,
        }



def _load_history():
    """Load history from disk. Returns empty list if missing/corrupt."""
    if not HISTORY_FILE.is_file():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text())
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _save_history():
    """Persist history to disk. Best-effort — never raises."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(_history))
    except OSError:
        pass


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

    negative = payload.get("negative_prompt") or payload.get("negative") or ""
    if not isinstance(negative, str):
        raise ValueError("negative_prompt must be a string")
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

    scheduler = payload.get("scheduler", "pndm")
    if not isinstance(scheduler, str):
        raise ValueError("scheduler must be a string")
    scheduler = scheduler.lower()
    if scheduler not in ("pndm", "lcm"):
        raise ValueError("scheduler must be 'pndm' or 'lcm'")

    preset = payload.get("preset")
    if preset is not None:
        if not isinstance(preset, str):
            raise ValueError("preset must be a string")
        preset = preset.lower()
        if preset not in generate.PRESETS:
            raise ValueError("preset must be 'quality', 'balanced', or 'fast'")

    seed = payload.get("seed")
    if seed is None or seed == "":
        seed = -1
    else:
        if isinstance(seed, bool):
            raise ValueError("seed must be an integer >= -1")
        if isinstance(seed, float):
            raise ValueError("seed must be an integer >= -1")
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError("seed must be an integer >= -1") from None
    if seed < -1:
        raise ValueError("seed must be >= -1")

    return {
        "prompt": prompt,
        "negative": negative,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "seed": seed,
        "count": count,
        "scheduler": scheduler,
        "preset": preset,
        "enhanced_prompt": payload.get("enhanced_prompt") or None,
    }


def _progress_callback(pipeline, step, timestep, callback_kwargs):
    """diffusers callback_on_step_end: called after each denoising step.

    Returns callback_kwargs unchanged. Updates shared progress state.
    Runs on the generation worker thread.
    """
    _set_progress_step(step)
    return callback_kwargs


def _enhance_prompt(original_prompt):
    """Call Groq API to enhance the prompt. Returns enhanced text or raises."""
    try:
        from groq import Groq, APITimeoutError, APIConnectionError
    except ImportError:
        raise RuntimeError("Groq SDK not installed")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Groq API key not configured")

    client = Groq()
    try:
        completion = client.with_options(timeout=30.0).chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a prompt enhancement expert for Stable Diffusion 1.5 image generation. Improve the given image-generation prompt to be more descriptive, specific, and effective. Preserve the user's intended subject and style. Do not invent an unrelated concept. Return ONLY the enhanced prompt text — no explanation, no markdown, no surrounding quotes."},
                {"role": "user", "content": original_prompt},
            ],
            model="openai/gpt-oss-120b",
            max_completion_tokens=512,
        )
    except APITimeoutError:
        raise RuntimeError("Prompt enhancement timed out")
    except APIConnectionError:
        raise RuntimeError("Prompt enhancement service unavailable")
    except Exception:
        raise RuntimeError("Prompt enhancement failed")

    enhanced = completion.choices[0].message.content
    if not enhanced or not isinstance(enhanced, str) or not enhanced.strip():
        raise RuntimeError("Invalid enhancement response")

    return enhanced.strip()


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
        elif path == "/api/history":
            self._send_json(200, list(_history))
        elif path == "/api/outputs":
            self._send_json(200, self._list_outputs())
        elif path == "/api/progress":
            self._send_json(200, _get_progress())
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

    def do_DELETE(self):
        path = unquote(urlparse(self.path).path)

        if not path.startswith("/api/outputs/"):
            self._send_json(404, {"success": False, "error": "Not found"})
            return

        filename = path[len("/api/outputs/"):]
        if not filename or "/" in filename or "\\" in filename:
            self._send_json(400, {"success": False, "error": "Invalid filename"})
            return
        if not filename.endswith(".png"):
            self._send_json(400, {"success": False, "error": "Only PNG files can be deleted"})
            return

        target = OUTPUT_DIR / filename
        try:
            resolved = target.resolve()
            root_resolved = OUTPUT_DIR.resolve()
        except OSError:
            self._send_json(400, {"success": False, "error": "Invalid path"})
            return
        if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
            self._send_json(404, {"success": False, "error": "File not found"})
            return

        try:
            resolved.unlink()
        except OSError:
            self._send_json(500, {"success": False, "error": "Failed to delete file"})
            return

        self._send_json(200, {"success": True, "filename": filename})

    def _list_outputs(self):
        """List generated PNG outputs with basic metadata."""
        if not OUTPUT_DIR.is_dir():
            return []
        files = []
        for f in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.suffix.lower() == ".png" and f.name != "history.json":
                stat = f.stat()
                files.append({
                    "filename": f.name,
                    "url": f"/outputs/{f.name}",
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
        return files

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/enhance-prompt":
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

            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                self._send_json(400, {"success": False, "error": "prompt is required"})
                return
            if len(prompt) > MAX_PROMPT_LEN:
                self._send_json(400, {"success": False, "error": f"prompt too long (maximum {MAX_PROMPT_LEN})"})
                return

            try:
                enhanced = _enhance_prompt(prompt.strip())
            except RuntimeError as exc:
                message = str(exc)
                if "not configured" in message:
                    self._send_json(401, {"success": False, "error": message})
                elif "timed out" in message:
                    self._send_json(504, {"success": False, "error": message})
                elif "unavailable" in message:
                    self._send_json(502, {"success": False, "error": "Prompt enhancement service unavailable"})
                elif "Invalid enhancement response" in message:
                    self._send_json(500, {"success": False, "error": "Prompt enhancement returned an invalid response"})
                else:
                    self._send_json(500, {"success": False, "error": "Prompt enhancement failed"})
                return

            self._send_json(200, {
                "original_prompt": prompt.strip(),
                "enhanced_prompt": enhanced,
            })
            return

        if path != "/api/generate":
            self._send_json(404, {"success": False, "error": "Not found"})
            return

        # --- Shutdown guard: reject new generation requests ---
        if _shutting_down.is_set():
            self._send_json(503, {
                "success": False,
                "error": "Server is shutting down. Try again shortly.",
            })
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

        # --- Shutdown guard after validation ---
        if _shutting_down.is_set():
            self._send_json(503, {
                "success": False,
                "error": "Server is shutting down. Try again shortly.",
            })
            return

        # --- Concurrency guard ---
        if not _gen_lock.acquire(blocking=False):
            self._send_json(409, {"success": False,
                                  "error": "Generation already in progress"})
            return

        # --- Prepare progress state for this generation ---
        steps = params.get("steps", 25)
        _reset_progress(steps)

        # --- Generation worker state ---
        state = {
            "complete": threading.Event(),
            "result": None,
            "error": None,
            "error_type": None,
        }

        def generation_worker():
            """Run generation in a dedicated thread.

            The calling thread (handler) does NOT touch _gen_lock — it is
            released here in the finally block, guaranteeing the lock stays
            held until the GPU operation actually stops.
            """
            try:
                # Resolve preset: if provided, override scheduler and steps.
                scheduler = params["scheduler"]
                steps = params["steps"]
                preset = params.get("preset")
                if preset is not None:
                    scheduler, steps = generate.PRESETS[preset]
                t0 = time.perf_counter()
                images = generate.generate_images(
                    params["prompt"],
                    negative_prompt=params["negative"],
                    steps=steps,
                    guidance=params["guidance"],
                    width=params["width"],
                    height=params["height"],
                    seed=params["seed"],
                    count=params["count"],
                    device=device,
                    pipe=pipe,
                    scheduler=scheduler,
                    preset=preset,
                    callback_on_step_end=_progress_callback,
                )
                total_time = round(time.perf_counter() - t0, 2)
                if images:
                    first = images[0]
                    _history.insert(0, {
                        "url": f"/outputs/{first['filename']}",
                        "prompt": params["prompt"],
                        "negative_prompt": params["negative"],
                        "seed": params["seed"],
                        "preset": params.get("preset"),
                        "scheduler": scheduler,
                        "steps": steps,
                        "guidance": params["guidance"],
                        "width": params["width"],
                        "height": params["height"],
                        "device": device,
                        "generation_time": total_time,
                        "timestamp": time.time(),
                        "enhanced_prompt": params.get("enhanced_prompt"),
                    })
                    if len(_history) > MAX_HISTORY:
                        _history.pop()
                    _save_history()
                state["result"] = {
                    "success": True,
                    "device": device,
                    "total_time": total_time,
                    "images": [
                        {"filename": im["filename"],
                         "url": f"/outputs/{im['filename']}",
                         "seed": im["seed"]}
                        for im in images
                    ],
                }
            except RuntimeError as exc:
                message = str(exc)
                if "out of memory" in message.lower():
                    state["error_type"] = "oom"
                    state["error"] = message
                else:
                    state["error_type"] = "server"
                    state["error"] = ("Generation failed on the server. "
                                      "Check the server console for details.")
            except Exception:
                state["error_type"] = "server"
                state["error"] = ("Generation failed on the server. "
                                  "Check the server console for details.")
            finally:
                _reset_progress()
                state["complete"].set()
                _gen_lock.release()

        # --- Load pipeline (before worker; fast, no GPU inference) ---
        try:
            pipe, device = get_pipeline()
        except RuntimeError as exc:
            message = str(exc)
            _gen_lock.release()
            if "Model not found" in message:
                self._send_json(503, {
                    "success": False,
                    "error": "Model not found. Run download_model.py to download the model.",
                })
            else:
                self._send_json(500, {
                    "success": False,
                    "error": ("Generation failed on the server. "
                              "Check the server console for details."),
                })
            return

        # --- Start generation worker ---
        try:
            t = threading.Thread(target=generation_worker)
            t.start()
        except Exception:
            _gen_lock.release()
            self._send_json(500, {
                "success": False,
                "error": ("Generation failed on the server. "
                          "Check the server console for details."),
            })
            return

        # --- Wait for completion with safe timeout ---
        # SAFETY: The handler waits for the worker to finish within
        # GENERATION_TIMEOUT seconds. If the timeout fires, a 504 is
        # returned to the client while the worker continues running,
        # still holding _gen_lock and still using the GPU. The CUDA
        # operation is NEVER forcibly terminated. The lock is released
        # only when the worker's finally block runs (generation done).
        try:
            if state["complete"].wait(timeout=GENERATION_TIMEOUT):
                if state["error_type"] == "oom":
                    self._send_json(507, {
                        "success": False,
                        "error": state["error"],
                    })
                elif state["error_type"] == "server":
                    self._send_json(500, {
                        "success": False,
                        "error": state["error"],
                    })
                else:
                    self._send_json(200, state["result"])
            else:
                self._send_json(504, {
                    "success": False,
                    "error": "Generation timed out. The operation is still running on the server.",
                })
        finally:
            if _shutting_down.is_set():
                state["complete"].wait()


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
    global _history
    _history = _load_history()
    print("Local Text-to-Image UI", flush=True)
    print(f"http://{HOST}:{PORT}", flush=True)
    print(f"[webui] device: {generate.pick_device()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _shutting_down.set()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
