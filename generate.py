"""Minimal local Text-to-Image engine (CLI) for a GTX 1060 6GB.

TEXT PROMPT -> Stable Diffusion 1.5 (fp16) -> PNG. 100% local inference:
no APIs, no paid services, no network traffic at generation time.

Model: Stable Diffusion v1.5 (fp16 variant), loaded once from models/sd-1-5.

Usage:
    .venv\\Scripts\\python.exe generate.py "a futuristic city at sunset"
    .venv\\Scripts\\python.exe generate.py --prompt "a futuristic city at sunset"
    .venv\\Scripts\\python.exe generate.py "a futuristic city at sunset" --count 4

Exit codes:
    0  success
    2  invalid arguments (bad dimensions, missing prompt)
    1  runtime failure (model missing, CUDA out of memory)

Optional flags:
    --negative TEXT   negative prompt
    --seed N          integer seed (default: random starting seed, printed so
                      the images can be reproduced later)
    --steps N         denoising steps (default 25)
    --guidance F      classifier-free guidance scale (default 7.5)
    --scheduler {pndm,lcm}  scheduler (default: pndm; lcm is faster but
                      may reduce quality below 8 steps)
    --preset {quality,balanced,fast}  preset (overrides scheduler and steps)
    --width N         image width in pixels (default 512). Must be 64-768 and
                      divisible by 8 (512, 640, 768, ...).
    --height N        image height in pixels (default 512). Same rules as --width.
    --count N         number of images from the same prompt (default 1);
                      seeds advance sequentially: seed, seed+1, seed+2, ...

Example:
    .venv\\Scripts\\python.exe generate.py "a futuristic city at sunset" ^
        --negative "blurry, low quality" --seed 42 --steps 25 --count 4
"""
import argparse
import secrets
import time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, LCMScheduler
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "sd-1-5"
OUTPUT_DIR = ROOT / "outputs"

# Model identity, recorded in every PNG so an image is traceable to its weights.
MODEL_NAME = "stable-diffusion-v1-5 (fp16)"

# Supported schedulers. PNDM is the default; LCM provides faster inference
# at the cost of potential quality tradeoffs at very low step counts.
SCHEDULERS = ("pndm", "lcm")

# Generation presets: each maps to a (scheduler, steps) pair.
PRESETS = {
    "quality": ("pndm", 25),
    "balanced": ("pndm", 8),
    "fast": ("lcm", 8),
}
PRESET_NAMES = tuple(PRESETS.keys())

# Image dimension rules. The VAE downsamples by 8, so dimensions that are not
# multiples of 8 make diffusers reject the request outright. 768 is the tested
# ceiling for a 6 GB card; anything larger is rejected before it can OOM.
MIN_DIM = 64
MAX_DIM = 768
DIM_STEP = 8


def positive_int(text: str) -> int:
    """argparse type: only positive integers (steps, count)."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer (got '{text}')")
    return value


def image_dim(text: str) -> int:
    """argparse type for --width / --height.

    Integers only, within [MIN_DIM, MAX_DIM], divisible by DIM_STEP (8).
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be a whole number of pixels (got '{text}')"
        ) from None
    error = dimension_error(value)
    if error:
        raise argparse.ArgumentTypeError(error)
    return value


def dimension_error(value: object) -> str | None:
    """Return a human-readable problem with `value`, or None when it is valid.

    Shared by the CLI (argparse) and the Web API so both enforce one rule set.
    """
    if isinstance(value, bool):
        return "must be a whole number of pixels"
    if not isinstance(value, int):
        return "must be a whole number of pixels"
    # Bounds first: for 63 or 999 "too small/too large" is the more useful
    # complaint than the divisibility rule they also break.
    if value < MIN_DIM:
        return f"must be at least {MIN_DIM} pixels (got {value})"
    if value > MAX_DIM:
        return f"must be at most {MAX_DIM} pixels (got {value})"
    if value % DIM_STEP != 0:
        nearest = round(value / DIM_STEP) * DIM_STEP
        return (f"{value} is not divisible by {DIM_STEP}; "
                f"use a multiple of {DIM_STEP} such as {nearest}")
    return None


def positive_float(text: str) -> float:
    """argparse type: only positive numbers (guidance scale)."""
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number (got '{text}')")
    return value


def scheduler_choice(text: str) -> str:
    """argparse type: only supported scheduler names."""
    value = text.lower()
    if value not in SCHEDULERS:
        raise argparse.ArgumentTypeError(
            f"must be one of {', '.join(SCHEDULERS)} (got '{text}')"
        )
    return value


def preset_choice(text: str) -> str:
    """argparse type: only supported preset names."""
    value = text.lower()
    if value not in PRESET_NAMES:
        raise argparse.ArgumentTypeError(
            f"must be one of {', '.join(PRESET_NAMES)} (got '{text}')"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local text-to-image engine: TEXT PROMPT -> SD 1.5 -> PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt_positional", nargs="?", default=None,
                        metavar="PROMPT", help="text prompt for the image")
    parser.add_argument("--prompt", dest="prompt_option", default=None,
                        help="text prompt (alternative to the positional form)")
    parser.add_argument("--negative", default=None, help="negative prompt")
    parser.add_argument("--seed", type=int, default=-1,
                        help="integer seed (-1 = random, default)")
    parser.add_argument("--steps", type=positive_int, default=25,
                        help="denoising steps")
    parser.add_argument("--guidance", type=positive_float, default=7.5,
                        help="guidance scale (CFG)")
    parser.add_argument("--width", type=image_dim, default=512,
                        help=f"image width in pixels (multiple of {DIM_STEP})")
    parser.add_argument("--height", type=image_dim, default=512,
                        help=f"image height in pixels (multiple of {DIM_STEP})")
    parser.add_argument("--count", type=positive_int, default=1,
                        help="number of images from the same prompt")
    parser.add_argument("--scheduler", type=scheduler_choice, default="pndm",
                        help="scheduler: pndm (default) or lcm")
    parser.add_argument("--preset", type=preset_choice, default=None,
                        help="generation preset (overrides scheduler and steps)")

    args = parser.parse_args()

    if args.seed < -1:
        parser.error("seed must be >= -1")

    # Accept the prompt either positionally or as --prompt, but not both, then
    # normalize to a single `prompt` attribute for the rest of the program.
    if args.prompt_positional and args.prompt_option:
        parser.error("give the prompt either positionally or with --prompt, not both")
    args.prompt = args.prompt_positional or args.prompt_option
    if not args.prompt or not args.prompt.strip():
        parser.error("a prompt is required (positional or --prompt)")
    return args


def pick_device() -> str:
    """CUDA when available; otherwise plain CPU fallback."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(device: str) -> StableDiffusionPipeline:
    """Load the local model once. fp16 on CUDA, fp32 on CPU.

    Raises RuntimeError (never SystemExit) so callers such as the Web UI can
    treat a failure to load as an ordinary recoverable error.
    """
    if not MODEL_DIR.is_dir():
        raise RuntimeError(
            f"Model not found at {MODEL_DIR}.\n"
            "Run: .venv\\Scripts\\python.exe download_model.py  (one-time, ~2 GB)"
        )

    if device == "cuda":
        # GTX 1060 6GB: fp16 halves weights (~2 GB total) and Pascal sm_61
        # handles fp16 compute. Weights load directly in fp16 (no fp32 spike).
        pipe = StableDiffusionPipeline.from_pretrained(
            str(MODEL_DIR),
            variant="fp16",
            dtype=torch.float16,
            safety_checker=None,          # skipped at download time too
            requires_safety_checker=False,
        ).to("cuda")
        # Attention slicing cuts peak activation VRAM and is the standard
        # 6 GB setting. That alone keeps peak usage ~2.9 GB at 512x512, so
        # nothing more is needed: CPU offload would slow things down for no
        # reason, and PyTorch SDPA is already the attention backend.
        pipe.enable_attention_slicing()
    else:
        # Honest CPU fallback: same fp16-variant weight files on disk, but
        # loaded as fp32 (diffusers upcasts on load). No extra tricks.
        pipe = StableDiffusionPipeline.from_pretrained(
            str(MODEL_DIR),
            variant="fp16",               # only fp16-variant files exist locally
            safety_checker=None,
            requires_safety_checker=False,
        )
    return pipe


def out_of_memory_message(width: int, height: int, steps: int) -> str:
    """Actionable text for a CUDA OOM, tailored to the request that failed."""
    step_word = "step" if steps == 1 else "steps"
    return (
        f"CUDA ran out of memory generating {width}x{height} in {steps} {step_word}.\n"
        f"Try: --width {max(MIN_DIM, width // 2)} --height {max(MIN_DIM, height // 2)} "
        "to reduce resolution, lower --steps, or close other GPU apps "
        "(browsers and games hold VRAM)."
    )


def unique_output_path() -> Path:
    """Timestamped filename; appends a counter if the name already exists
    (two runs within the same second)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = OUTPUT_DIR / f"{stamp}.png"
    counter = 2
    while candidate.exists():
        candidate = OUTPUT_DIR / f"{stamp}-{counter}.png"
        counter += 1
    return candidate


def build_png_metadata(prompt: str, negative: str | None, seed: int, steps: int,
                       guidance: float, width: int, height: int,
                       scheduler: str = "pndm",
                       preset: str | None = None) -> PngInfo:
    """Metadata embedded directly in the PNG (no sidecar file).

    Storing the full recipe makes any image reproducible from the file alone.
    """
    meta = PngInfo()
    meta.add_text("prompt", prompt)
    meta.add_text("negative_prompt", negative if negative else "")
    meta.add_text("seed", str(seed))
    meta.add_text("steps", str(steps))
    meta.add_text("guidance_scale", str(guidance))
    meta.add_text("width", str(width))
    meta.add_text("height", str(height))
    meta.add_text("model", MODEL_NAME)
    meta.add_text("scheduler", scheduler)
    if preset is not None:
        meta.add_text("preset", preset)
    return meta


def generate_images(prompt: str, *, negative_prompt: str = "",
                      steps: int = 25, guidance: float = 7.5,
                      width: int = 512, height: int = 512,
                      seed: int = -1, count: int = 1,
                      device: str | None = None,
                      pipe: StableDiffusionPipeline | None = None,
                      scheduler: str = "pndm",
                      preset: str | None = None,
                      callback_on_step_end=None) -> list[dict]:
    """Generate `count` images sequentially from `prompt`, saving each to
    OUTPUT_DIR with a unique timestamped filename and embedded PNG metadata.

    Seeds advance by one per image, starting from `seed` (or a random base
    seed when seed is -1). `pipe` may be an already loaded pipeline (the Web UI
    reuses it); when None the model is loaded here once. `device` selects the
    torch generator / CUDA bookkeeping and defaults to auto-detection.

    Returns a list of dicts: {"filename", "seed", "time"}.

    Raises RuntimeError on CUDA out-of-memory, with resolution/step advice.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if pipe is None:
        device = pick_device() if device is None else device
        pipe = load_pipeline(device)
    elif device is None:
        device = pick_device()

    base_seed = secrets.randbelow(2**32) if seed == -1 else seed
    images: list[dict] = []

    if scheduler not in SCHEDULERS:
        raise ValueError(f"unknown scheduler '{scheduler}'")

    original_scheduler = pipe.scheduler
    if scheduler == "lcm":
        pipe.scheduler = LCMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            steps_offset=0,
        )
        print(f"[generate] scheduler: lcm", flush=True)

    try:
        for i in range(count):
            current_seed = base_seed + i
            generator = torch.Generator(device=device).manual_seed(current_seed)
            metadata = build_png_metadata(prompt, negative_prompt, current_seed, steps,
                                             guidance, width, height, scheduler,
                                             preset=preset)

            t1 = time.perf_counter()
            try:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=height,
                    width=width,
                    generator=generator,
                    callback_on_step_end=callback_on_step_end,
                )
            except torch.cuda.OutOfMemoryError as exc:
                if device == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(out_of_memory_message(width, height, steps)) from exc
            if device == "cuda":
                torch.cuda.synchronize()
            t_gen = time.perf_counter() - t1

            out_path = unique_output_path()
            image = result.images[0]
            image.save(out_path, pnginfo=metadata)
            images.append({"filename": out_path.name, "seed": current_seed, "time": t_gen})

            del image, result

            # Sequential: keep one image in memory at a time so peak VRAM stays
            # low. The pipeline itself stays loaded for the next image. The caching
            # allocator reuses these blocks, so no empty_cache() is needed here -
            # releasing to the driver every step would only add overhead.
    finally:
        pipe.scheduler = original_scheduler

    return images


def main() -> None:
    args = parse_args()

    # Resolve preset: if explicitly provided, it determines scheduler and steps.
    scheduler = args.scheduler
    steps = args.steps
    preset = args.preset
    if preset is not None:
        scheduler, steps = PRESETS[preset]
        print(f"[preset] {preset} -> scheduler: {scheduler}, steps: {steps}", flush=True)

    device = pick_device()
    if device == "cuda":
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)} (fp16)")
    else:
        print("[device] CUDA unavailable - falling back to CPU (fp32, slow)")

    t0 = time.perf_counter()
    pipe = load_pipeline(device)
    t_load = time.perf_counter() - t0
    print(f"[load] {t_load:.1f}s")

    # Reset peak stats AFTER the weights are resident, so the peak reported
    # below is the true cost of a generation step (not the load spike).
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    images = generate_images(
        args.prompt,
        negative_prompt=args.negative or "",
        steps=steps,
        guidance=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed,
        count=args.count,
        device=device,
        pipe=pipe,
        scheduler=scheduler,
        preset=preset,
    )

    gen_times = [im["time"] for im in images]
    total_gen = sum(gen_times)
    seeds = [im["seed"] for im in images]

    print(f"[prompt] {args.prompt}")
    print(f"[negative] {args.negative if args.negative else '(none)'}")
    if preset is not None:
        print(f"[preset] {preset}")
    print(f"[scheduler] {scheduler}")
    if args.count == 1:
        print(f"[seed] {seeds[0]}")
    else:
        print(f"[seeds] {', '.join(map(str, seeds))}")
    print(f"[steps] {steps}")
    print(f"[guidance] {args.guidance}")
    print(f"[resolution] {args.width}x{args.height}")
    print(f"[device] {device}")
    if args.count == 1:
        print(f"[time] load {t_load:.1f}s | generation {gen_times[0]:.1f}s "
              f"({steps} steps, {gen_times[0] / steps:.2f}s/step)")
    else:
        print(f"[time] load {t_load:.1f}s | total batch {total_gen:.1f}s "
              f"({args.count} images, {total_gen / args.count:.2f}s avg)")
    if device == "cuda":
        print(f"[vram] held {torch.cuda.memory_allocated() / 1024**2:.0f} MiB | "
              f"peak {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")
    for im in images:
        print(f"[saved] {OUTPUT_DIR / im['filename']}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        # Model-missing and CUDA out-of-memory report as a clean message, not
        # a stack trace: both are user-actionable conditions.
        raise SystemExit(f"[error] {exc}")
