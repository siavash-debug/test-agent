# test-agent v0.2

A minimal, fully local **Text-to-Image** engine. A text prompt goes in, a PNG
file comes out.

- **No paid APIs.** No network access at generation time.
- **Local inference only.** Runs entirely on your own GPU.
- **Zero cost.** Open weights, open tooling.

Inference is Stable Diffusion 1.5 (fp16) via `diffusers`, tuned to fit a
6 GB laptop GPU.

---

## Hardware requirements

| | Minimum | Notes |
|---|---|---|
| GPU | NVIDIA GTX 1060 6 GB | Verified target: Surface Book 2 15", Pascal compute capability 6.1 |
| VRAM | 4 GB free | ~2.9 GB peak at 512×512 |
| RAM | 8 GB | Pipeline loaded on GPU when CUDA is available; CPU fallback otherwise |
| Disk | ~6 GB | ~2 GB model + PyTorch |
| CUDA | Driver 450+ | Verified with driver 546.33 |

CPU-only inference works as a fallback, but is very slow (minutes per image).

## Installation

Requires **Python 3.11** (64-bit) on Windows or Linux.

### 1. Virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

On Linux/macOS use `source .venv/bin/activate`.

### 2. Dependencies

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` pins the **CUDA 11.8 (`+cu118`)** builds of `torch` and
`torchvision` and points pip at the PyTorch index, so a plain install cannot
silently downgrade you to CPU-only wheels.

### 3. Verify the GPU

```powershell
python gpu_test.py
```

Expect `CUDA available: True`, your GPU name, and `GPU tensor test: PASSED`.

## Model location

The weights are **not** in this repository. Download them once (~2 GB):

```powershell
python download_model.py
```

This fetches the fp16 variant of Stable Diffusion 1.5 into:

```
models/sd-1-5/
```

Re-running is safe: already-downloaded files are skipped. The NSFW safety
checker is intentionally not downloaded (unused, saves ~600 MB).

## CLI usage

```powershell
python generate.py "a small robot on Mars"
```

The prompt can be given positionally or with `--prompt` (both are equivalent):

```powershell
python generate.py --prompt "a small robot on Mars"
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--negative TEXT` | none | Negative prompt |
| `--seed N` | -1 | Reproducible seed (`-1` = random) |
| `--steps N` | 25 | Denoising steps |
| `--guidance F` | 7.5 | Classifier-free guidance scale |
| `--scheduler {pndm,lcm,dpmpp_2m_karras}` | pndm | Scheduler (see below) |
| `--preset {quality,balanced,fast}` | none | Preset (overrides scheduler and steps) |
| `--width N` | 512 | Multiple of 8, 64–768 |
| `--height N` | 512 | Multiple of 8, 64–768 |
| `--count N` | 1 | Images from one prompt; seeds advance by 1 |

### Presets

Presets combine a scheduler and step count for common workflows:

| Preset | Scheduler | Steps | Purpose |
|---|---|---:|---|
| `quality` | pndm | 25 | Best current quality |
| `balanced` | pndm | 8 | Faster PNDM |
| `fast` | lcm | 8 | Fastest recommended mode |

If a preset is explicitly provided, it determines both scheduler and steps.
If no preset is provided, the existing behavior is preserved
(scheduler defaults to pndm; steps default to 25).

### Scheduler

| Value | Description |
|---|---|
| `pndm` | Default. Standard PNDM scheduler. |
| `lcm` | Optional fast mode using LCMScheduler with standard SD 1.5 weights. |
| `dpmpp_2m_karras` | DPM++ 2M Karras (`DPMSolverMultistepScheduler`, `dpmsolver++`, Karras sigmas), built from the model's own scheduler config. |

**LCM notes:**

- Uses the standard SD 1.5 fp16 weights via `LCMScheduler` — this is **not** an LCM-distilled model.
- No additional model download is required.
- Benchmark on GTX 1060 6 GB: ~7.75 s/image at 8 steps vs ~23.6 s/image for PNDM 25 steps (~3× faster).
- **8 steps is recommended.** Fewer than 8 steps may reduce image quality or introduce artifacts.
- Peak VRAM remains ~2938 MiB.

Example:

```powershell
python generate.py "a futuristic city at sunset" --negative "blurry, low quality" --seed 42 --steps 8 --scheduler lcm --count 2
```

Images are written to `outputs/` as timestamped PNGs.

## Web UI usage

```powershell
python web_ui.py
```

Then open <http://127.0.0.1:8000>.

The server is standard-library only and binds to `127.0.0.1` (localhost).
The model is loaded once on the first request and reused; generation requests
are serialized, so only one job uses the GPU at a time.

## Server behavior

### Generation timeout
The server enforces a per-request generation timeout of 600 seconds. If a
generation does not complete within this window, the server returns HTTP 504
to the client. The underlying generation worker is **not** forcibly killed —
CUDA inference continues running safely in the background until it finishes,
and the generation lock remains held until the worker completes.

### Safe shutdown
Pressing Ctrl+C initiates a graceful server shutdown. During shutdown, new
generation requests are rejected immediately with HTTP 503. Any generation
already in progress is allowed to finish safely; no CUDA or thread operations
are forcibly terminated.

### Generation concurrency
Only one generation job may use the GPU at a time. A concurrent generation
request receives HTTP 409 (`Generation already in progress`). There is no
generation queue — the request fails immediately rather than waiting.

### Progress
The server reports generation progress via `GET /api/progress`, which returns
the current denoising step and total step count. The frontend polls this
endpoint while a generation is active. Progress state is reset when
generation finishes, errors, or the server shuts down.

### Model loading errors
If the model is not found, the server returns HTTP 503 with an actionable
message instructing the user to run `download_model.py`. No traceback or
local file path is exposed to the client.

## Seed

| Value | Behavior |
|---|---|
| `-1` | Random seed (default) |
| `>= 0` | Deterministic generation |

Same prompt + same settings + same non-negative seed will always produce the same image.

## Negative Prompt

A negative prompt guides the model away from undesired qualities (e.g. `"blurry, low quality"`).

## Reproducing an image

Every PNG embeds its own recipe as PNG text metadata:

`prompt`, `negative_prompt`, `seed`, `steps`, `guidance_scale`, `width`,
`height`, `model`, `scheduler`, `preset`.

No sidecar files are needed. To read the metadata:

```powershell
python -c "from PIL import Image; print(Image.open('outputs/<file>.png').text)"
```

## History

- **Persistent.** History is saved to `outputs/history.json` and restored when the server starts.
- **Maximum 20 entries.** Older entries are discarded when the limit is reached.
- **Reuse Settings.** Click **Reuse Settings** on any history item to restore its prompt, negative prompt, seed, preset, scheduler, and steps into the form. Generation does not start automatically.

## Prompt Enhancement

Prompt enhancement is an optional feature that uses an AI model to improve your image-generation prompt before it is sent to the local Stable Diffusion pipeline.

- **Optional.** You can ignore the enhancement and generate directly with your original prompt.
- **Image generation remains local.** Only the text enhancement is processed externally; the actual image is generated on your GPU.
- **Groq** is used only for text enhancement. No image data is sent externally.
- **Model:** `openai/gpt-oss-120b`

When you click **Enhance Prompt**, your current prompt is sent to Groq. The enhanced text appears in a preview where you can review it and choose to use it or dismiss it. Your original prompt is always preserved.

## API Key

Prompt enhancement requires a Groq API key provided via the environment variable `GROQ_API_KEY`.

**PowerShell:**

```powershell
$env:GROQ_API_KEY="your-key"
```

**CMD:**

```cmd
set GROQ_API_KEY=your-key
```

The key is used server-side only and is never exposed to the browser. If the key is not configured, enhancement fails with a clear message and image generation continues to work normally.

## Known GTX 1060 limitations

- **512×512 is the sweet spot** (~1.1 s/step, ~2.9 GB peak). 768×768 works but
  is ~3× slower and pushes reserved VRAM to ~5.3 GB of 6 GB — risky while the
  display is also using the card.
- **SDXL and newer models do not fit.** 6 GB is not enough without CPU offload,
  which is far too slow to be useful.
- **fp16 is required.** fp32 weights fit but peak at ~4.9 GB and invite
  out-of-memory errors.
- **Display overhead matters.** Desktop composition already holds ~2 GB. Close
  browsers and games before generating at higher resolutions.
- **`empty_cache()` is avoided during batches** on purpose — the caching
  allocator reuses blocks more efficiently than returning them to the driver.
- **Batch size is fixed at 1.** Multiple images are generated sequentially;
  this is what keeps peak VRAM low.
- **xformers is deliberately not used.** PyTorch SDPA is already the attention
  backend, and xformers adds a dependency without benefit on Pascal.

## Project layout

```
generate.py         CLI + reusable generation core (load, generate, save)
web_ui.py           Local HTTP server + JSON API (stdlib only)
download_model.py   One-time SD 1.5 fp16 weight downloader
gpu_test.py         CUDA smoke test
requirements.txt    Pinned, reproducible dependency set
constraints.txt     Guard pin keeping torch on the +cu118 build
static/             Web UI (index.html, app.js, style.css)
models/             Downloaded weights (git-ignored)
outputs/            Generated PNGs (git-ignored)
```
