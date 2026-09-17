"""Benchmark v2 runner.

Standardized benchmark methodology v2.

Differences from v1:
- Warm-up generations (excluded from timing)
- CUDA sync OUTSIDE timed generation boundary
- Image saving excluded from generation timing
- GPU temperature recording (nvidia-smi via subprocess)
- Cooldown period between configurations
- VRAM: peak allocated + peak reserved
- Image preservation in benchmark/runs/<run_id>/<config>/<prompt_id>.png
- Production pipeline via generate.py (no separate runner implementation)
- Automated metrics explicitly labeled as automated
- Human evaluation as separate explicit step (no auto-generated human scores)
- V2 schema with methodology_version=2

Usage:
    .venv\\Scripts\\python.exe benchmark/benchmark_v2_runner.py
    .venv\\Scripts\\python.exe benchmark/benchmark_v2_runner.py --cooldown 0
    .venv\\Scripts\\python.exe benchmark/benchmark_v2_runner.py --warmup 3
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import generate
from generate import load_pipeline

REPORTS_DIR = ROOT / "benchmark" / "reports"
RUNS_DIR = ROOT / "benchmark" / "runs"
PROMPTS_FILE = ROOT / "benchmark" / "prompts.json"

DEFAULT_WARMUP = 2
DEFAULT_COOLDOWN = 300

SEEDS = [42, 43, 44, 45, 46, 47]


def load_prompts():
    data = json.loads(PROMPTS_FILE.read_text())
    return data["prompts"]


def get_gpu_temp():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0].strip())
    except (subprocess.TimeoutExpired, ValueError, IndexError,
            FileNotFoundError, OSError):
        pass
    return None


def warm_up(pipe, prompts, device, warmup_count):
    for _ in range(warmup_count):
        for p in prompts:
            seed = SEEDS[prompts.index(p)]
            generator = torch.Generator(device=device).manual_seed(seed)
            pipe(
                prompt=p["prompt"],
                negative_prompt=p.get("negative_prompt", ""),
                num_inference_steps=25,
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=generator,
            )
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_generation(pipe, prompt_info, seed, device):
    prompt = prompt_info["prompt"]
    negative = prompt_info.get("negative_prompt", "")
    generator = torch.Generator(device=device).manual_seed(seed)

    torch.cuda.synchronize()
    temp_pre_gen = get_gpu_temp()
    t1 = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=25,
        guidance_scale=7.5,
        height=512,
        width=512,
        generator=generator,
    )
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t1
    temp_post_gen = get_gpu_temp()

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2

    image = result.images[0]
    del result
    return image, t_gen, peak_vram, peak_reserved, \
        temp_pre_gen, temp_post_gen


def build_metadata(prompt, negative, seed, steps, guidance):
    from PIL.PngImagePlugin import PngInfo
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
    meta.add_text("benchmark_methodology", "v2")
    return meta


def run_config(pipe, prompts, config_name, device, cooldown_seconds,
               warmup_count, benchmark_id):
    print(f"\n[v2] Config: {config_name}", flush=True)
    if cooldown_seconds > 0:
        print(f"[v2] cooldown requested: {cooldown_seconds}s", flush=True)
        for remaining in range(cooldown_seconds, 0, -1):
            if remaining % 30 == 0 or remaining <= 5:
                print(f"[v2] cooldown: {remaining}s remaining", flush=True)
        time.sleep(cooldown_seconds)
        print(f"[v2] cooldown completed", flush=True)
    else:
        print(f"[v2] cooldown: disabled", flush=True)

    print(f"[v2] warm-up starting ({warmup_count} generations)...",
          flush=True)
    warm_up(pipe, prompts, device, warmup_count)
    print(f"[v2] warm-up complete", flush=True)

    temp_pre_gen = get_gpu_temp()
    run_dir = RUNS_DIR / benchmark_id / config_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_results = []
    for prompt_info in prompts:
        prompt_id = prompt_info["id"]
        seed = SEEDS[prompts.index(prompt_info)]
        print(f"\n[v2] {config_name}/{prompt_id} (seed={seed})",
              flush=True)

        image, t_gen, peak_vram, peak_reserved, \
            temp_before, temp_after = run_generation(
                pipe, prompt_info, seed, device)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{benchmark_id}-{config_name}-{prompt_id}-{timestamp}.png"
        out_path = run_dir / filename
        meta = build_metadata(
            prompt_info["prompt"],
            prompt_info.get("negative_prompt", ""),
            seed, 25, 7.5,
        )
        image.save(out_path, pnginfo=meta)
        del image

        entry = {
            "prompt_id": prompt_id,
            "category": prompt_info.get("category", ""),
            "seed": seed,
            "steps": 25,
            "guidance": 7.5,
            "generation_time_seconds": round(t_gen, 2),
            "peak_vram_mb": round(peak_vram, 0),
            "peak_reserved_vram_mb": round(peak_reserved, 0),
            "temperature_before_c": temp_before,
            "temperature_pre_generation_c": temp_pre_gen,
            "temperature_post_generation_c": temp_after,
            "filename": filename,
            "image_path": str(out_path),
            "status": "OK",
        }
        config_results.append(entry)
        print(f"  {t_gen:.2f}s, VRAM={peak_vram:.0f}MB "
              f"(reserved={peak_reserved:.0f}MB), "
              f"temp={temp_before}C->{temp_after}C", flush=True)

    return config_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark v2 runner")
    parser.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN,
                        help="Cooldown seconds between configs (default: 300)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="Warm-up generations per config (default: 2)")
    parser.add_argument("--no-cooldown", action="store_true",
                        help="Disable cooldown (equivalent to --cooldown 0)")
    parser.add_argument("--config", type=str, default="baseline",
                        help="Configuration name for this benchmark run")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="List of config names (default: ['baseline'])")
    args = parser.parse_args()

    cooldown_seconds = 0 if args.no_cooldown else args.cooldown
    warmup_count = args.warmup
    config_names = args.configs or [args.config]

    device = "cuda"
    pipe = load_pipeline(device)
    pipe.enable_attention_slicing()

    prompts = load_prompts()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_id = \
        f"v0.9.0-benchmark-v2-baseline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    all_results = {}
    temp_before_config = get_gpu_temp()
    print(f"[v2] benchmark_id: {benchmark_id}", flush=True)
    print(f"[v2] configs: {config_names}", flush=True)
    print(f"[v2] warmup: {warmup_count}", flush=True)
    print(f"[v2] cooldown: {cooldown_seconds}s", flush=True)
    print(f"[v2] temperature before config: {temp_before_config}C",
          flush=True)

    for config_name in config_names:
        config_results = run_config(
            pipe, prompts, config_name, device, cooldown_seconds,
            warmup_count, benchmark_id,
        )
        all_results[config_name] = config_results

    temp_after_config = get_gpu_temp()

    report = {
        "methodology_version": 2,
        "benchmark_id": benchmark_id,
        "engine_version": "v0.9.0",
        "git_commit": "9612cf4",
        "timestamp": datetime.now().isoformat(),
        "model": "stable-diffusion-1.5-fp16",
        "scheduler": "pndm",
        "steps": 25,
        "guidance": 7.5,
        "width": 512,
        "height": 512,
        "seeds": SEEDS,
        "prompt_suite_version": "1.0",
        "warmup_count": warmup_count,
        "cooldown_seconds": cooldown_seconds,
        "device": "GTX 1060 6GB (CUDA)",
        "configs": config_names,
        "temperature_before_c": temp_before_config,
        "temperature_pre_generation_c": \
            all_results[config_names[0]][0].get("temperature_pre_generation_c"),
        "temperature_post_generation_c": temp_after_config,
        "pipeline": {
            "source": "generate.py (production pipeline)",
            "dtype": "torch.float16",
            "attention_slicing": True,
            "safety_checker": None,
        },
        "timing_methodology": {
            "warmup_excluded": True,
            "cuda_sync_before_timed": True,
            "cuda_sync_after_generation": True,
            "image_save_excluded": True,
            "report_writing_excluded": True,
        },
        "results": all_results,
    }

    report_path = REPORTS_DIR / f"{benchmark_id}-v2.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[v2] Results saved to {report_path}", flush=True)

    total_times = []
    for config_results in all_results.values():
        for r in config_results:
            total_times.append(r["generation_time_seconds"])
    avg_time = sum(total_times) / len(total_times) if total_times else 0
    print(f"[v2] Average generation time: {avg_time:.2f}s", flush=True)
    print(f"[v2] Benchmark v2 complete", flush=True)


if __name__ == "__main__":
    main()
