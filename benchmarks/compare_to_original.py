from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from statistics import mean

import numpy as np

from speexdsp import EchoCanceller

ORIGINAL_SPEC = "speexdsp==0.1.1"
DEFAULT_FRAME_SIZE = 256
DEFAULT_FILTER_LENGTH = 2048
DEFAULT_SAMPLE_RATE = 16000
WARMUP_ITERS = 50


def _make_inputs(frame_size: int) -> tuple[np.ndarray, np.ndarray]:
    near = np.zeros(frame_size, dtype=np.int16)
    far = np.zeros(frame_size, dtype=np.int16)
    return near, far


def measure_current(frame_size: int, iterations: int) -> dict[str, float]:
    t0 = time.perf_counter()
    ec = EchoCanceller.create(frame_size, DEFAULT_FILTER_LENGTH, DEFAULT_SAMPLE_RATE)
    create_s = time.perf_counter() - t0

    near, far = _make_inputs(frame_size)
    for _ in range(WARMUP_ITERS):
        ec.process(near, far)

    tracemalloc.start()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        ec.process(near, far)
        timings.append(time.perf_counter() - start)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t1 = time.perf_counter()
    ec.destroy()
    destroy_s = time.perf_counter() - t1

    reset_s = 0.0
    t2 = time.perf_counter()
    ec = EchoCanceller.create(frame_size, DEFAULT_FILTER_LENGTH, DEFAULT_SAMPLE_RATE)
    ec.reset()
    reset_s = time.perf_counter() - t2
    ec.destroy()

    return {
        "create_us": create_s * 1e6,
        "process_avg_us": mean(timings) * 1e6,
        "process_p95_us": sorted(timings)[max(0, int(len(timings) * 0.95) - 1)] * 1e6,
        "destroy_us": destroy_s * 1e6,
        "reset_us": reset_s * 1e6,
        "peak_alloc_kb": peak / 1024.0,
        "current_alloc_kb": current / 1024.0,
    }


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def measure_original(frame_size: int, iterations: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="speexdsp-original-") as tmp:
        tmpdir = Path(tmp)
        venv_dir = tmpdir / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        py = _venv_python(venv_dir)

        subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(py), "-m", "pip", "install", ORIGINAL_SPEC], check=True)

        code = f"""
import json
import time
import tracemalloc
from statistics import mean

from speexdsp import EchoCanceller

frame_size = {frame_size}
iterations = {iterations}
chunk = b'\\0\\0' * frame_size

start = time.perf_counter()
ec = EchoCanceller.create(frame_size, {DEFAULT_FILTER_LENGTH}, {DEFAULT_SAMPLE_RATE})
create_s = time.perf_counter() - start

for _ in range({WARMUP_ITERS}):
    ec.process(chunk, chunk)

tracemalloc.start()
timings = []
for _ in range(iterations):
    t0 = time.perf_counter()
    ec.process(chunk, chunk)
    timings.append(time.perf_counter() - t0)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

payload = {{
    'create_us': create_s * 1e6,
    'process_avg_us': mean(timings) * 1e6,
    'process_p95_us': sorted(timings)[max(0, int(len(timings) * 0.95) - 1)] * 1e6,
    'peak_alloc_kb': peak / 1024.0,
    'current_alloc_kb': current / 1024.0,
}}
print(json.dumps(payload))
""".strip()
        raw = subprocess.check_output([str(py), "-c", code], text=True).strip()
        return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the current SpeexDSP binding against the original PyPI release")
    parser.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    cur = measure_current(args.frame_size, args.iterations)
    old = measure_original(args.frame_size, args.iterations)

    speedup = old["process_avg_us"] / cur["process_avg_us"] if cur["process_avg_us"] else 0.0

    print(f"Original spec: {ORIGINAL_SPEC}")
    print(f"Frame size:    {args.frame_size}")
    print(f"Iterations:    {args.iterations}")
    print(f"Current create: {cur['create_us']:.2f} us")
    print(f"Original create:{old['create_us']:.2f} us")
    print(f"Current avg:    {cur['process_avg_us']:.2f} us/frame")
    print(f"Original avg:   {old['process_avg_us']:.2f} us/frame")
    print(f"Current p95:    {cur['process_p95_us']:.2f} us/frame")
    print(f"Original p95:   {old['process_p95_us']:.2f} us/frame")
    print(f"Current peak KB: {cur['peak_alloc_kb']:.2f}")
    print(f"Original peak KB:{old['peak_alloc_kb']:.2f}")
    print(f"Current reset:   {cur['reset_us']:.2f} us")
    print(f"Current destroy: {cur['destroy_us']:.2f} us")
    print(f"Speedup:        {speedup:.3f}x")


if __name__ == "__main__":
    main()
