"""Steps benchmark runner: runs 5 step configs x 6 prompts = 30 generations.

Reads prompts from benchmark/prompts.json, uses seeds 42-47 (same as baseline).
Records generation time and peak VRAM for each generation.
"""
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "sd-1-5"
OUTPUT_DIR = ROOT / "outputs"
PROMPTS_FILE = ROOT / "benchmark" / "prompts.json"
REPORTS_DIR = ROOT / "benchmark" / "reports"

SEEDS = [42, 43, 44, 45, 46, 47]
STEP_CONFIGS = [8, 15, 25, 35, 50]


def load_prompts():
    data = json.loads(PROMPTS_FILE.read_text())
    return data["prompts"]


def build_metadata(prompt, negative, seed, steps, guidance):
    meta = PngInfo()
    meta.add_text("prompt", prompt)
    meta.add_text("negative_prompt", negative)
    meta.add_text("seed", str(seed))
    meta.add_text("steps", str(steps))
    meta.add_text("guidance_scale", str(guidance))
    meta.add_text("width", "512")
    meta.add_text("height", "512")
    meta.add_text("model", "stable-diffusion-v1-5 (fp16)")
    meta.add_text("scheduler", "pndm")
    return meta


def main():
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        str(MODEL_DIR),
        variant="fp16",
        dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.enable_attention_slicing()

    prompts = load_prompts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for steps in STEP_CONFIGS:
        print(f"\n=== Steps: {steps} ===", flush=True)
        torch.cuda.reset_peak_memory_stats()

        for prompt_info in prompts:
            prompt_id = prompt_info["id"]
            prompt = prompt_info["prompt"]
            negative = prompt_info.get("negative_prompt", "")
            seed = SEEDS[prompts.index(prompt_info)]

            generator = torch.Generator(device=device).manual_seed(seed)

            t1 = time.perf_counter()
            try:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=generator,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [FAIL] {prompt_id}: CUDA OOM at {steps} steps", flush=True)
                all_results.append({
                    "steps": steps, "prompt_id": prompt_id, "seed": seed,
                    "time_seconds": None, "peak_vram_mb": None,
                    "status": "OOM_FAIL",
                })
                continue

            torch.cuda.synchronize()
            t_gen = time.perf_counter() - t1
            peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 2

            image = result.images[0]
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"steps{steps}-{prompt_id}-{timestamp}.png"
            out_path = OUTPUT_DIR / filename
            meta = build_metadata(prompt, negative, seed, steps, 7.5)
            image.save(out_path, pnginfo=meta)

            print(f"  [{prompt_id}] {t_gen:.2f}s, VRAM={peak_vram:.0f}MB, seed={seed}, file={filename}", flush=True)

            all_results.append({
                "steps": steps,
                "prompt_id": prompt_id,
                "category": prompt_info.get("category", ""),
                "seed": seed,
                "time_seconds": round(t_gen, 2),
                "peak_vram_mb": round(peak_vram, 0),
                "filename": filename,
                "status": "OK",
            })

            del image, result

    report = {
        "benchmark_id": f"v0.9.0-steps-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "engine_version": "v0.9.0",
        "git_commit": "9612cf4",
        "timestamp": datetime.now().isoformat(),
        "model": "stable-diffusion-1.5-fp16",
        "scheduler": "pndm",
        "device": "GTX 1060 6GB (CUDA)",
        "steps_configs": STEP_CONFIGS,
        "guidance": 7.5,
        "width": 512,
        "height": 512,
        "seeds": SEEDS,
        "prompt_suite_version": "1.0",
        "results": all_results,
    }

    report_path = REPORTS_DIR / "v0.9.0-steps-raw-results.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nRaw results saved to {report_path}", flush=True)

    # Summary by steps
    print("\n=== SUMMARY ===", flush=True)
    for steps in STEP_CONFIGS:
        items = [r for r in all_results if r["steps"] == steps and r["status"] == "OK"]
        if items:
            avg_time = sum(r["time_seconds"] for r in items) / len(items)
            avg_vram = sum(r["peak_vram_mb"] for r in items) / len(items)
            print(f"  {steps} steps: {len(items)}/6 OK, avg time={avg_time:.2f}s, avg VRAM={avg_vram:.0f}MB")
        else:
            print(f"  {steps} steps: ALL FAILED")


if __name__ == "__main__":
    main()
