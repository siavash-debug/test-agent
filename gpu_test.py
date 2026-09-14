"""GPU smoke test for the GTX 1060 (Pascal, compute capability 6.1) inside .venv.

Run with:
    .venv\Scripts\python.exe gpu_test.py
"""
import sys

import torch

print("Python/PyTorch test")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("Compute capability:", f"{cap[0]}.{cap[1]}")

    device = torch.device("cuda")
    x = torch.randn((2048, 2048), device=device)
    y = torch.randn((2048, 2048), device=device)
    z = x @ y

    torch.cuda.synchronize()

    print("GPU tensor test: PASSED")
    print("Result device:", z.device)
    print("Result shape:", tuple(z.shape))
    # .item() forces a device -> host readback, proving the computation
    # really executed on the GPU and produced data.
    print("Result mean (host readback):", z.mean().item())
    print("Allocated VRAM:", torch.cuda.memory_allocated(0), "bytes")
else:
    print("GPU test: FAILED - CUDA is not available")
