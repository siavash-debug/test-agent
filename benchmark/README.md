# Benchmark History

## Benchmark Methodology v2

### Overview

Benchmark v2 is the current methodology for measuring image-generation quality and performance. It was designed to address methodological limitations in v1 that produced misleading quality conclusions.

Key v2 improvements over v1:
- **Warm-up generations** excluded from reported timing (v1 had none)
- **CUDA synchronization** surrounding timed generation (v1 had sync inside timing)
- **Image saving** excluded from generation timing (v1 was correct here)
- **GPU temperature** recorded via nvidia-smi (v1 had none)
- **Cooldown period** between configurations (v1 had none)
- **VRAM**: peak allocated + peak reserved (v1 had only allocated)
- **Production pipeline** used via `generate.py` (v1 used custom runners for steps/CFG)
- **Automated metrics** explicitly labeled as automated (v1 labeled them as quality scores)
- **Human evaluation** is a separate explicit step (v1 auto-inferred scores)
- **Image preservation** in `benchmark/runs/<run_id>/<config>/<prompt_id>.png` (v1 used flat outputs/)
- **V2 schema** with `methodology_version: 2` and all new fields (v1 had minimal schema)

### Timing Boundaries (V2)

1. Pipeline loaded once
2. Warm-up: 2 non-timed generations per configuration (configurable via `--warmup`)
3. `torch.cuda.synchronize()` before timed generation
4. Start high-resolution timer (`time.perf_counter()`)
5. Run generation via `pipe()` (production pipeline from `generate.py`)
6. `torch.cuda.synchronize()` immediately after generation
7. Stop timer
8. Save image (NOT included in generation time)
9. Write metadata/report (NOT included in generation time)

### Warm-Up

Default: 2 non-timed generations per configuration.

Warm-up generations serve to:
- Stabilize GPU clocks and memory allocator state
- Pre-populate CUDA kernels
- Reduce first-generation variance

Warm-up is excluded from all reported timing and VRAM measurements.

Make warm-up count configurable via `--warmup N`.

### Cooldown

Default: 300 seconds (5 minutes) between benchmark configurations.

Cooldown serves to:
- Allow GPU temperature to stabilize
- Let the thermal management system recover
- Ensure each configuration starts from similar thermal conditions

The runner explicitly reports:
- `cooldown requested: 300s`
- Countdown every 30 seconds (and at final 5 seconds)
- `cooldown completed`

For development/testing, disable cooldown with `--cooldown 0` or `--no-cooldown`.

### GPU Temperature

Recorded via `nvidia-smi` (subprocess, no additional Python dependency).

For each configuration, the following temperatures are recorded:
- `temperature_before_c` — temperature before the configuration starts
- `temperature_pre_generation_c` — temperature immediately before timed generation
- `temperature_post_generation_c` — temperature immediately after timed generation

If temperature cannot be read (nvidia-smi unavailable, timeout, error):
- Value is `null` in the report
- Benchmark does NOT fail
- No fake values are invented

### VRAM Measurement

Two measurements per generation:
- `peak_vram_mb` — `torch.cuda.max_memory_allocated()` (actual allocations)
- `peak_reserved_vram_mb` — `torch.cuda.max_memory_reserved()` (reserved by CUDA allocator)

Both measured after `torch.cuda.synchronize()`. `torch.cuda.reset_peak_memory_stats()` called per configuration before generation begins.

### Image Preservation

All benchmark images are preserved in a structured directory tree:

```
benchmark/runs/<benchmark_run_id>/<configuration_name>/<prompt_id>.png
```

Example:
```
benchmark/runs/v0.9.0-benchmark-v2-baseline-20260915-233333/baseline/benchmark-portrait-001.png
```

Images include embedded PNG metadata with: prompt, negative prompt, seed, steps, guidance, model, scheduler, benchmark_methodology=v2.

Images are NEVER deleted or overwritten. Each benchmark run gets a unique run ID with timestamp.

### Automated Metrics (Explicitly Labeled)

Automated metrics are computed from image pixel statistics:
- Laplacian sharpness
- Gradient density
- Contrast (grayscale std)
- Color variance

These are ALWAYS labeled as "automated" in reports and evaluations.

**Automated metrics are NOT human visual quality scores.**
They are NEVER presented as:
- human score
- visual quality score
- realism score

They may be labeled as:
- automated_sharpness
- automated_contrast
- automated_gradient
- automated_metric_score

### Human Evaluation

Human evaluation is a SEPARATE EXPLICIT STEP. It is never automated or inferred.

Workflow:
1. Generate images using the benchmark runner
2. Run `benchmark_evaluate_v2.py --run-id <id> --human-template`
3. A human evaluator inspects each image and scores on each dimension
4. Save scores to `<run_id>-human-scores.json`
5. Run `benchmark_evaluate_v2.py --run-id <id>` to load and aggregate human scores

Scoring dimensions (1-10 rubric):
- Prompt adherence
- Composition
- Detail
- Realism (1 = clearly unrealistic, 10 = highly realistic)
- Anatomy (null when irrelevant)
- Artifacts (10 = artifact-free, 1 = severe artifacts)
- Overall (human overall assessment)

Rules:
- Scores must be entered by a human evaluator
- Scores are NEVER auto-generated
- If no human has evaluated, human_evaluation = null
- Report `human_evaluators = 1` (or actual count if multiple)
- Do NOT fabricate confidence intervals from single evaluator

### Production Pipeline

Benchmark v2 uses `generate.py` as the production pipeline. The benchmark runner imports `generate.load_pipeline()` and calls `pipe()` directly with V2 timing boundaries. This avoids having a separate benchmark implementation with subtle behavioral differences from production.

### VRAM and Performance Results from V2 Baseline

V2 baseline (PNDM, 25 steps, CFG 7.5, 512x512, 6 prompts, seeds 42-47):
- Average generation time: 26.58s (includes post-warm-up steady state)
- Peak VRAM: 2938MB (allocated), 4028MB (reserved)
- GPU temperature: 77C (GTX 1060 at thermal limit before benchmark started)
- All 6 prompts completed successfully

Note: The V2 baseline was run after previous benchmark sessions, so GPU temperature was elevated. The 26.58s includes warm-up overhead. Fresh-run timings may differ.

### Difference Between V1 and V2

| Aspect | V1 | V2 |
|---|---|---|
| Timing warm-up | None (cold start) | 2 generations (excluded from timing) |
| CUDA sync | Inside timing | Before timer, after generation |
| Image saving | After timing (correct) | After timing (correct) |
| GPU temperature | Not recorded | Recorded (nvidia-smi) |
| Cooldown | None | 300s between configs (configurable) |
| VRAM | peak_allocated only | peak_allocated + peak_reserved |
| Pipeline source | Mixed (generate.py + custom) | generate.py (production) |
| Quality scores | Inferred/automated | Human explicit (automated labeled separately) |
| Image storage | outputs/ (flat) | benchmark/runs/<id>/<config>/ (structured) |
| Schema | Minimal | Extended with methodology_version=2 |
| Seed mapping | 42-47 | 42-47 (unchanged) |

### How to Run a Benchmark

```bash
# V2 baseline (default: warmup=2, cooldown=300s)
.venv\Scripts\python.exe benchmark/benchmark_v2_runner.py

# Disable cooldown for development
.venv\Scripts\python.exe benchmark/benchmark_v2_runner.py --cooldown 0

# Custom warm-up
.venv\Scripts\python.exe benchmark/benchmark_v2_runner.py --warmup 3

# Custom config name
.venv\Scripts\python.exe benchmark/benchmark_v2_runner.py --config my-config

# Multiple configs (cooldown applies between each)
.venv\Scripts\python.exe benchmark/benchmark_v2_runner.py --configs baseline fast quality
```

### How to Run Human Evaluation

```bash
# Generate evaluation template (inspect images, fill in scores)
.venv\Scripts\python.exe benchmark/benchmark_evaluate_v2.py --run-id <run_id> --human-template

# After manual scoring, save to <run_id>-human-scores.json
# Then evaluate:
.venv\Scripts\python.exe benchmark/benchmark_evaluate_v2.py --run-id <run_id>
```

### Limitations

- GPU temperature requires nvidia-smi; if unavailable, temperature fields are null
- V2 baseline human scores remain UNSET pending human evaluation
- Automated metrics do not assess perceptual quality (realism, aesthetic composition)
- Single-evaluator scores have no confidence interval
- Cooldown timing adds significant wall-clock time for multi-config benchmarks
- Thermal conditions affect absolute timing; V2 timing is valid for comparison only when thermal conditions are similar

### Old Results (Methodology V1)

Historical v1 benchmark results are preserved in `benchmark/reports/` and `benchmark/history.json`. These are labeled as methodology_version 1 (or implicitly v1 based on their format). Do not compare v1 and v2 scores directly without understanding the methodological differences documented above.

### Reports

The `reports/` directory stores generated benchmark report files. Reports are derived from raw data and should be append-only.

The `runs/` directory stores preserved benchmark images organized by run ID and configuration.
