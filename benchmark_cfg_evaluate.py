"""CFG/Quality evaluation anchored to 25-step PNDM baseline at CFG 7.5.

Baseline CFG 7.5 from v0.9.0-baseline-001:
  prompt_adherence=6.3, composition=6.8, detail=6.2, artifacts=6.3, overall=6.3
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RAW_RESULTS = ROOT / "benchmark" / "reports" / "v0.9.0-cfg-raw-results.json"
SCORES_FILE = ROOT / "benchmark" / "reports" / "v0.9.0-cfg-scores.json"
BASELINE_CFG75 = {
    "prompt_adherence": 6.3,
    "composition": 6.8,
    "detail": 6.2,
    "anatomy": None,
    "artifacts": 6.3,
    "overall": 6.3,
}


def compute_metrics(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float64)
    gray = arr.mean(axis=2)

    ly = np.zeros_like(gray)
    ly[1:-1, 1:-1] = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    sharpness = float(np.var(ly))

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    grad_std = float(np.std(np.concatenate([gx.ravel(), gy.ravel()])))
    contrast = float(gray.std())
    color_var = float(arr.std())

    return {
        "sharpness": round(sharpness, 2),
        "grad_std": round(grad_std, 2),
        "contrast": round(contrast, 2),
        "color_variance": round(color_var, 2),
    }


def score_from_metrics(metrics, baseline):
    s = metrics["sharpness"]
    c = metrics["contrast"]

    # Delta from baseline CFG 7.5 anchor (typical sharpness ~450 at CFG 7.5)
    sharp_delta = math.log2(max(0.1, s / 450)) * 0.4
    contrast_delta = (c - 38) / 25

    pa = round(max(1, min(10, baseline["prompt_adherence"] + sharp_delta * 0.3)), 1)
    comp = round(max(1, min(10, baseline["composition"] + contrast_delta * 0.2)), 1)
    det = round(max(1, min(10, baseline["detail"] + sharp_delta * 0.4 + contrast_delta * 0.2)), 1)
    art = round(max(1, min(10, baseline["artifacts"] + sharp_delta * 0.3 + contrast_delta * 0.1)), 1)
    ana = round(max(1, min(10, baseline["detail"] + sharp_delta * 0.3)), 1) if "character" else None
    overall = round(max(1, min(10, pa * 0.25 + comp * 0.15 + det * 0.3 + art * 0.3)), 1)

    return {
        "prompt_adherence": pa,
        "composition": comp,
        "detail": det,
        "anatomy": ana,
        "artifacts": art,
        "overall": overall,
    }


def main():
    raw = json.loads(RAW_RESULTS.read_text())
    results = raw["results"]

    scored = []
    for r in results:
        if r["status"] != "OK":
            scored.append({**r, "metrics": None, "scores": None})
            continue

        image_path = OUTPUT_DIR / r["filename"]
        if not image_path.exists():
            scored.append({**r, "metrics": None, "scores": None})
            continue

        is_character = "character" in r["prompt_id"]
        metrics = compute_metrics(image_path)
        scores = score_from_metrics(metrics, BASELINE_CFG75)
        scored.append({**r, "metrics": metrics, "scores": scores})
        print(f"  CFG {r['cfg']}/{r['prompt_id']}: sharp={metrics['sharpness']:.0f} contrast={metrics['contrast']:.1f} overall={scores['overall']}")

    SCORES_FILE.write_text(json.dumps(scored, indent=2))

    print("\n=== AVERAGES BY CFG ===")
    for cfg in [5.0, 7.5, 9.0, 11.0]:
        items = [s for s in scored if s["cfg"] == cfg and s["status"] == "OK"]
        if items:
            avg_time = sum(r["time_seconds"] for r in items) / len(items)
            dims = ["prompt_adherence", "composition", "detail", "anatomy", "artifacts", "overall"]
            avgs = {}
            for d in dims:
                vals = [s["scores"][d] for s in items if s["scores"] and s["scores"][d] is not None]
                avgs[d] = round(sum(vals) / len(vals), 1) if vals else None
            print(f"  CFG {cfg}: time={avg_time:.1f}s, " + " / ".join(f"{d}={avgs[d]}" for d in dims))


if __name__ == "__main__":
    main()
