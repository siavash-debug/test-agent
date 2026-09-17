"""Benchmark v2 evaluation framework.

Separates automated metrics from human evaluation.
Automated metrics are explicitly labeled as such.
Human scores are NEVER auto-generated.

Usage (automated metrics only):
    .venv\\Scripts\\python.exe benchmark/benchmark_evaluate_v2.py --run-id <id>

Usage (human evaluation template):
    .venv\\Scripts\\python.exe benchmark/benchmark_evaluate_v2.py --run-id <id> --human-template
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "benchmark" / "reports"
RUNS_DIR = ROOT / "benchmark" / "runs"


def compute_automated_metrics(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float64)
    gray = arr.mean(axis=2)

    ly = np.zeros_like(gray)
    ly[1:-1, 1:-1] = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    sharpness = float(np.var(ly))

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    grad_std = float(np.std(np.concatenate([gx.ravel(), gy.ravel()])))
    color_var = float(arr.std())
    contrast = float(gray.std())

    return {
        "automated_sharpness": round(sharpness, 2),
        "automated_gradient_std": round(grad_std, 2),
        "automated_color_variance": round(color_var, 2),
        "automated_contrast": round(contrast, 2),
    }


def compute_automated_metric_score(metrics):
    """Compute a single automated metric score for comparison purposes.

    This is NOT a human visual quality score. It is an automated metric
    derived from image pixel statistics only.
    """
    sharpness = metrics["automated_sharpness"]
    contrast = metrics["automated_contrast"]

    sharp_delta = math.log2(max(0.1, sharpness / 450)) * 0.4
    contrast_delta = (contrast - 38) / 25

    score = 5.0 + sharp_delta + contrast_delta
    return round(max(1, min(10, score)), 1)


def evaluate_run(run_id, run_dir=None):
    if run_dir is None:
        run_dir = RUNS_DIR / run_id

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", flush=True)
        return None

    findings = []
    for image_path in sorted(run_dir.rglob("*.png")):
        metrics = compute_automated_metrics(image_path)
        auto_score = compute_automated_metric_score(metrics)
        findings.append({
            "image": str(image_path),
            "filename": image_path.name,
            "automated_metrics": metrics,
            "automated_metric_score": auto_score,
            "human_evaluation": None,
        })
        print(f"  {image_path.name}: "
              f"sharp={metrics['automated_sharpness']:.0f} "
              f"contrast={metrics['automated_contrast']:.1f} "
              f"auto_score={auto_score}", flush=True)

    return findings


def generate_human_template(run_id, findings, output_path=None):
    if output_path is None:
        output_path = REPORTS_DIR / f"{run_id}-human-evaluation-template.md"

    lines = [
        f"# Human Evaluation Template: {run_id}",
        "",
        "## Instructions",
        "",
        "Score each image on a 1-10 scale using the benchmark rubric:",
        "- 1-2 = Unusable",
        "- 3-4 = Poor",
        "- 5-6 = Acceptable",
        "- 7-8 = Good",
        "- 9 = Very good",
        "- 10 = Exceptional",
        "",
        "For realism: 1 = clearly unrealistic, 10 = highly realistic for the prompt.",
        "For artifacts: 10 = essentially artifact-free, 1 = severe artifacts.",
        "",
        "**IMPORTANT: These scores must come from actual visual inspection of the images.",
        "Do not infer scores from automated metrics or expected behavior.",
        "",
        "---\n",
    ]

    for i, f in enumerate(findings):
        lines.append(f"## Image {i+1}: {f['filename']}")
        lines.append(f"- Path: `{f['image']}`")
        lines.append(f"- Automated sharpness: {f['automated_metrics']['automated_sharpness']}")
        lines.append(f"- Automated contrast: {f['automated_metrics']['automated_contrast']}")
        lines.append(f"- Automated metric score: {f['automated_metric_score']} "
                     f"(NOT a human score)")
        lines.append("")
        lines.append("| Dimension | Score (1-10) |")
        lines.append("|---|---|")
        lines.append("| Prompt adherence | |")
        lines.append("| Composition | |")
        lines.append("| Detail | |")
        lines.append("| Realism | |")
        lines.append("| Anatomy (if applicable) | |")
        lines.append("| Artifacts | |")
        lines.append("| Overall | |")
        lines.append("")
        lines.append("Evaluator: _______________  Date: _______________\n")

    template_path = Path(output_path)
    template_path.write_text("\n".join(lines))
    print(f"Human evaluation template: {template_path}", flush=True)


def load_human_scores(run_id, evaluator_name="unknown"):
    """Load human scores from a completed evaluation template.

    This function reads a JSON file containing human scores.
    It never generates scores automatically.
    """
    score_file = REPORTS_DIR / f"{run_id}-human-scores.json"
    if not score_file.exists():
        print(f"No human score file found: {score_file}", flush=True)
        return None
    data = json.loads(score_file.read_text())
    data.setdefault("human_evaluators", 1)
    data.setdefault("evaluator_name", evaluator_name)
    return data


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark v2 evaluation")
    parser.add_argument("--run-id", type=str, required=True,
                        help="Benchmark run ID")
    parser.add_argument("--human-template", action="store_true",
                        help="Generate human evaluation template")
    parser.add_argument("--evaluator", type=str, default="unknown",
                        help="Evaluator name for human scores")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id

    if args.human_template:
        findings = evaluate_run(args.run_id, run_dir)
        if findings:
            generate_human_template(args.run_id, findings)
        return

    findings = evaluate_run(args.run_id, run_dir)
    if findings:
        print(f"\nAutomated evaluation of {len(findings)} images complete.", flush=True)
        print("Note: All scores above are AUTOMATED METRICS, not human evaluations.", flush=True)
        print("Use --human-template to generate a template for human evaluation.", flush=True)


if __name__ == "__main__":
    main()
