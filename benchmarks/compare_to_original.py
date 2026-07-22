from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean

import numpy as np

from speexdsp import EchoCanceller

ORIGINAL_SPEC = "speexdsp==0.1.1"


def current_avg(frame_size: int, iterations: int) -> float:
    ec = EchoCanceller.create(frame_size, 2048, 16000)
    near = np.zeros(frame_size, dtype=np.int16)
    far = np.zeros(frame_size, dtype=np.int16)
    for _ in range(50):
        ec.process(near, far)
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        ec.process(near, far)
        samples.append(time.perf_counter() - t0)
    return mean(samples)


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def original_avg(frame_size: int, iterations: int) -> float:
    with tempfile.TemporaryDirectory(prefix="speexdsp-original-") as tmp:
        tmpdir = Path(tmp)
        venv_dir = tmpdir / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        py = _venv_python(venv_dir)

        subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(py), "-m", "pip", "install", ORIGINAL_SPEC], check=True)

        code = f"""
import time
from statistics import mean

from speexdsp import EchoCanceller

frame_size = {frame_size}
iterations = {iterations}
chunk = b'\\0\\0' * frame_size

ec = EchoCanceller.create(frame_size, 2048, 16000)
for _ in range(50):
    ec.process(chunk, chunk)

samples = []
for _ in range(iterations):
    t0 = time.perf_counter()
    ec.process(chunk, chunk)
    samples.append(time.perf_counter() - t0)

print(mean(samples))
""".strip()
        out = subprocess.check_output([str(py), "-c", code], text=True).strip()
        return float(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the current SpeexDSP binding against the original PyPI release")
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    cur = current_avg(args.frame_size, args.iterations)
    old = original_avg(args.frame_size, args.iterations)

    print(f"Original spec: {ORIGINAL_SPEC}")
    print(f"Current avg:   {cur * 1e6:.2f} us/frame")
    print(f"Original avg:  {old * 1e6:.2f} us/frame")
    print(f"Speedup:       {old / cur:.3f}x")


if __name__ == "__main__":
    main()
