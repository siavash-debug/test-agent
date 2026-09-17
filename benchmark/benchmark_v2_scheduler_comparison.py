"""Benchmark v2 scheduler + step-count comparison runner.

Compares PNDM vs DPM++ 2M Karras at 25/35/50 steps using
Benchmark V2 methodology.

Configurations (fixed order):
    PNDM 25, PNDM 35, PNDM 50
    DPM++ 2M Karras 25, DPM++ 2M Karras 35, DPM++ 2M Karras 50

6 configurations x 6 prompts = 36 generations.

Usage:
    .venv\\Scripts\\python.exe benchmark/benchmark_v2_scheduler_comparison.py
    .venv\\Scripts\\python.exe benchmark/benchmark_v2_scheduler_comparison.py --cooldown 0
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler

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

DPMPP_2M_KARRAS_KWARGS = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "steps_offset": 0,
    "algorithm_type": "dpmsolver++",
    "use_karras_sigmas": True,
}

CONFIGS = [
    {"name": "pndm_25_steps", "scheduler": "pndm", "steps": 25},
    {"name": "pndm_35_steps", "scheduler": "pndm", "steps": 35},
    {"name": "pndm_50_steps", "scheduler": "pndm", "steps": 50},
    {"name": "dpmpp_25_steps", "scheduler": "dpmpp_2m_karras", "steps": 25},
    {"name": "dpmpp_35_steps", "scheduler": "dpmpp_2m_karras", "steps": 35},
    {"name": "dpmpp_50_steps", "scheduler": "dpmpp_2m_karras", "steps": 50},
]


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


def make_scheduler(scheduler_name, production_scheduler):
    """Return the scheduler instance for `scheduler_name`.

    PNDM reuses the production scheduler object loaded by
    generate.load_pipeline() from models/sd-1-5/scheduler/scheduler_config.json,
    so the benchmark PNDM cannot drift from the engine. DPM++ 2M Karras is
    benchmark-only and is built from DPMPP_2M_KARRAS_KWARGS.
    """
    if scheduler_name == "pndm":
        return production_scheduler
    elif scheduler_name == "dpmpp_2m_karras":
        return DPMSolverMultistepScheduler(**DPMPP_2M_KARRAS_KWARGS)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def warm_up(pipe, prompts, device, warmup_count, steps):
    for _ in range(warmup_count):
        for p in prompts:
            seed = SEEDS[prompts.index(p)]
            generator = torch.Generator(device=device).manual_seed(seed)
            pipe(
                prompt=p["prompt"],
                negative_prompt=p.get("negative_prompt", ""),
                num_inference_steps=steps,
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=generator,
            )
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_generation(pipe, prompt_info, seed, device, steps):
    prompt = prompt_info["prompt"]
    negative = prompt_info.get("negative_prompt", "")
    generator = torch.Generator(device=device).manual_seed(seed)

    torch.cuda.synchronize()
    temp_pre_gen = get_gpu_temp()
    t1 = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
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


def build_metadata(prompt, negative, seed, steps, guidance, scheduler):
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
    meta.add_text("scheduler", scheduler)
    meta.add_text("benchmark_methodology", "v2")
    if scheduler == "dpmpp_2m_karras":
        meta.add_text("scheduler_details",
                      "DPMSolverMultistepScheduler dpmsolver++ karras")
    return meta


def run_config(pipe, prompts, config, device, cooldown_seconds,
               warmup_count, benchmark_id):
    config_name = config["name"]
    scheduler_name = config["scheduler"]
    steps = config["steps"]
    print(f"\n[v2] Config: {config_name} (scheduler={scheduler_name}, "
          f"steps={steps})", flush=True)

    if cooldown_seconds > 0:
        print(f"[v2] cooldown requested: {cooldown_seconds}s", flush=True)
        for remaining in range(cooldown_seconds, 0, -1):
            if remaining % 30 == 0 or remaining <= 5:
                print(f"[v2] cooldown: {remaining}s remaining", flush=True)
        time.sleep(cooldown_seconds)
        print(f"[v2] cooldown completed", flush=True)
    else:
        print(f"[v2] cooldown: disabled", flush=True)

    original_scheduler = pipe.scheduler
    pipe.scheduler = make_scheduler(scheduler_name, original_scheduler)
    print(f"[v2] scheduler set: {scheduler_name}", flush=True)

    print(f"[v2] warm-up starting ({warmup_count} generations "
          f"@ {steps} steps)...", flush=True)
    warm_up(pipe, prompts, device, warmup_count, steps)
    print(f"[v2] warm-up complete", flush=True)

    temp_pre_gen = get_gpu_temp()
    run_dir = RUNS_DIR / benchmark_id / config_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_results = []
    for prompt_idx, prompt_info in enumerate(prompts):
        prompt_id = prompt_info["id"]
        seed = SEEDS[prompts.index(prompt_info)]
        print(f"\n[v2] {config_name}/{prompt_id} (seed={seed}, "
              f"steps={steps})", flush=True)

        image, t_gen, peak_vram, peak_reserved, \
            temp_before, temp_after = run_generation(
                pipe, prompt_info, seed, device, steps)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{benchmark_id}-{config_name}-{prompt_id}-{timestamp}.png"
        out_path = run_dir / filename
        meta = build_metadata(
            prompt_info["prompt"],
            prompt_info.get("negative_prompt", ""),
            seed, steps, 7.5, scheduler_name,
        )
        image.save(out_path, pnginfo=meta)
        del image

        entry = {
            "config": config_name,
            "prompt_idx": prompt_idx,
            "prompt_id": prompt_id,
            "category": prompt_info.get("category", ""),
            "seed": seed,
            "steps": steps,
            "guidance": 7.5,
            "scheduler": scheduler_name,
            "scheduler_details": get_scheduler_details(scheduler_name),
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

    pipe.scheduler = original_scheduler
    print(f"[v2] scheduler restored: {type(original_scheduler).__name__}",
          flush=True)

    return config_results


def get_scheduler_details(scheduler_name):
    if scheduler_name == "pndm":
        return {"class": "PNDMScheduler", "algorithm_type": "pndm"}
    elif scheduler_name == "dpmpp_2m_karras":
        return {
            "class": "DPMSolverMultistepScheduler",
            "algorithm_type": "dpmsolver++",
            "use_karras_sigmas": True,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "num_train_timesteps": 1000,
            "steps_offset": 0,
        }
    return {}


def scheduler_parameters(scheduler):
    """JSON-safe copy of a scheduler's effective config (private keys dropped)."""
    return {key: value for key, value in scheduler.config.items()
            if not key.startswith("_")}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark v2 scheduler comparison")
    parser.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN,
                        help="Cooldown seconds between configs")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="Warm-up generations per config")
    parser.add_argument("--no-cooldown", action="store_true",
                        help="Disable cooldown")
    args = parser.parse_args()

    cooldown_seconds = 0 if args.no_cooldown else args.cooldown
    warmup_count = args.warmup

    device = "cuda"
    pipe = load_pipeline(device)
    pipe.enable_attention_slicing()

    prompts = load_prompts()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_id = \
        f"v0.9.0-benchmark-v2-scheduler-comparison-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    all_results = {}
    config_schedules = []
    temp_before_config = get_gpu_temp()
    print(f"[v2] benchmark_id: {benchmark_id}", flush=True)
    print(f"[v2] configs:", flush=True)
    for c in CONFIGS:
        line = f"  {c['name']}: scheduler={c['scheduler']}, steps={c['steps']}"
        print(line, flush=True)
        config_schedules.append(line.strip())
    print(f"[v2] warmup: {warmup_count}", flush=True)
    print(f"[v2] cooldown: {cooldown_seconds}s", flush=True)
    print(f"[v2] temperature before config: {temp_before_config}C",
          flush=True)

    for config in CONFIGS:
        config_results = run_config(
            pipe, prompts, config, device, cooldown_seconds,
            warmup_count, benchmark_id,
        )
        all_results[config["name"]] = config_results

    temp_after_config = get_gpu_temp()

    report = {
        "methodology_version": 2,
        "benchmark_id": benchmark_id,
        "engine_version": "v0.9.0",
        "git_commit": "9612cf4",
        "timestamp": datetime.now().isoformat(),
        "model": "stable-diffusion-1.5-fp16",
        "guidance": 7.5,
        "width": 512,
        "height": 512,
        "seeds": SEEDS,
        "prompt_suite_version": "1.0",
        "warmup_count": warmup_count,
        "cooldown_seconds": cooldown_seconds,
        "device": "GTX 1060 6GB (CUDA)",
        "temperature_before_c": temp_before_config,
        "temperature_pre_generation_c": \
            all_results[CONFIGS[0]["name"]][0].get("temperature_pre_generation_c"),
        "temperature_post_generation_c": temp_after_config,
        "configurations": [
            {"name": c["name"], "scheduler": c["scheduler"],
             "steps": c["steps"]} for c in CONFIGS
        ],
        "config_schedule": config_schedules,
        "pipeline": {
            "source": "generate.py (production pipeline)",
            "dtype": "torch.float16",
            "attention_slicing": True,
            "safety_checker": None,
        },
        "scheduler_configs": {
            "pndm": {
                "class": type(pipe.scheduler).__name__,
                "source": "production model scheduler_config.json "
                          "(reused via generate.load_pipeline)",
                "parameters": scheduler_parameters(pipe.scheduler),
            },
            "dpmpp_2m_karras": {
                "class": "DPMSolverMultistepScheduler",
                "parameters": DPMPP_2M_KARRAS_KWARGS,
            },
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

    print("\n" + "=" * 70, flush=True)
    print("[v2] SCHEDULER + STEP COMPARISON SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for config_name, results in all_results.items():
        times = [r["generation_time_seconds"] for r in results]
        avg_time = sum(times) / len(times) if times else 0
        vrams = [r["peak_vram_mb"] for r in results]
        avg_vram = sum(vrams) / len(vrams) if vrams else 0
        sched = results[0]["scheduler"]
        print(f"\n{config_name} ({sched}):", flush=True)
        print(f"  avg generation time: {avg_time:.2f}s", flush=True)
        print(f"  avg peak VRAM: {avg_vram:.0f} MB", flush=True)
        for r in results:
            print(f"  {r['prompt_id']}: {r['generation_time_seconds']:.2f}s, "
                  f"VRAM={r['peak_vram_mb']:.0f}MB, "
                  f"temp={r['temperature_before_c']}C->{r['temperature_post_generation_c']}C",
                  flush=True)

    pndm_50_times = [r["generation_time_seconds"]
                     for r in all_results.get("pndm_50_steps", [])]
    pndm_50_avg = sum(pndm_50_times) / len(pndm_50_times) if pndm_50_times else 0
    if pndm_50_avg > 0:
        print("\n" + "=" * 70, flush=True)
        print("[v2] SPEEDUP RELATIVE TO PNDM 50", flush=True)
        print("=" * 70, flush=True)
        for config_name, results in all_results.items():
            times = [r["generation_time_seconds"] for r in results]
            avg_time = sum(times) / len(times) if times else 0
            speedup = pndm_50_avg / avg_time if avg_time > 0 else 0
            ratio = avg_time / pndm_50_avg if pndm_50_avg > 0 else 0
            print(f"  {config_name}: {avg_time:.2f}s "
                  f"(speedup={speedup:.2f}x, ratio={ratio:.3f})",
                  flush=True)

    print(f"\n[v2] Benchmark v2 scheduler comparison complete", flush=True)


if __name__ == "__main__":
    main()
