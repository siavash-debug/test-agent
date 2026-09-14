"""One-time downloader: fetches Stable Diffusion v1.5 (fp16 variant) into models/sd-1-5.

Downloads once; re-running skips files that are already complete on disk.
The NSFW safety checker weights are deliberately skipped (not needed and saves ~608 MB).

Run:
    .venv\\Scripts\\python.exe download_model.py
"""
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LOCAL_DIR = Path(__file__).resolve().parent / "models" / "sd-1-5"

# Inference-critical files only (fp16 variant + configs/scheduler/tokenizer).
ALLOW = [
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/*",
    "scheduler/scheduler_config.json",
    "feature_extractor/preprocessor_config.json",
]


def main() -> None:
    print(f"Downloading {REPO_ID} -> {LOCAL_DIR}")
    path = snapshot_download(
        repo_id=REPO_ID,
        # No `variant=` filter needed: ALLOW already pins the exact fp16
        # weight filenames, and the kwarg is unreliable across hub versions.
        allow_patterns=ALLOW,
        local_dir=str(LOCAL_DIR),
        # .incomplete files let an interrupted download resume on re-run.
        max_workers=2,
    )
    print("Model ready at:", path)

    # Report what we actually keep on disk.
    total = sum(f.stat().st_size for f in LOCAL_DIR.rglob("*") if f.is_file())
    print(f"Local model directory size: {total / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
