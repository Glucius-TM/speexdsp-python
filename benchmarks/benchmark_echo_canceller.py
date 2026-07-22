from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean

import numpy as np

from speexdsp import EchoCanceller


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_current_benchmark(frame_size: int, iterations: int) -> float:
    ec = EchoCanceller.create(frame_size, 2048, 16000)

    near = np.zeros(frame_size, dtype=np.int16)
    far = np.zeros(frame_size, dtype=np.int16)

    for _ in range(50):
        ec.process(near, far)

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        ec.process(near, far)
        timings.append(time.perf_counter() - start)

    return mean(timings)


def _python_executable(venv_dir: Path) -> Path:
    if os.name == 'nt':
        return venv_dir / 'Scripts' / 'python.exe'
    return venv_dir / 'bin' / 'python'


def _git(cmd: list[str], cwd: Path) -> None:
    subprocess.run(['git', '-C', str(cwd), *cmd], check=True)


def run_original_benchmark(frame_size: int, iterations: int, baseline_ref: str) -> float:
    if shutil.which('git') is None:
        raise RuntimeError('git is required to benchmark the original baseline')

    with tempfile.TemporaryDirectory(prefix='speexdsp-baseline-') as tmp:
        tmpdir = Path(tmp)
        baseline_dir = tmpdir / 'baseline'
        venv_dir = tmpdir / 'venv'

        _git(['worktree', 'add', '--detach', str(baseline_dir), baseline_ref], REPO_ROOT)
        try:
            subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
            py = _python_executable(venv_dir)

            subprocess.run([str(py), '-m', 'pip', 'install', '--upgrade', 'pip'], cwd=baseline_dir, check=True)
            subprocess.run([str(py), '-m', 'pip', 'install', 'numpy'], cwd=baseline_dir, check=True)
            subprocess.run([str(py), '-m', 'pip', 'install', '.'], cwd=baseline_dir, check=True)

            code = f"""
import time
from statistics import mean

from speexdsp import EchoCanceller

frame_size = {frame_size}
iterations = {iterations}

ec = EchoCanceller.create(frame_size, 2048, 16000)
chunk = '\\0\\0' * frame_size

for _ in range(50):
    ec.process(chunk, chunk)

timings = []
for _ in range(iterations):
    start = time.perf_counter()
    ec.process(chunk, chunk)
    timings.append(time.perf_counter() - start)

print(mean(timings))
""".strip()
            result = subprocess.run([str(py), '-c', code], cwd=baseline_dir, check=True, capture_output=True, text=True)
            return float(result.stdout.strip())
        finally:
            subprocess.run(['git', '-C', str(REPO_ROOT), 'worktree', 'remove', '--force', str(baseline_dir)], check=False)


def _baseline_ref() -> str:
    result = subprocess.run(
        ['git', '-C', str(REPO_ROOT), 'rev-list', '--max-parents=0', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    )
    first_line = result.stdout.splitlines()[0].strip()
    if not first_line:
        raise RuntimeError('unable to determine the original baseline commit')
    return first_line


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark SpeexDSP echo cancellation')
    parser.add_argument('--frame-size', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--baseline-ref', type=str, default='')
    args = parser.parse_args()

    baseline_ref = args.baseline_ref or _baseline_ref()

    current_avg = run_current_benchmark(args.frame_size, args.iterations)
    original_avg = run_original_benchmark(args.frame_size, args.iterations, baseline_ref)

    speedup = (original_avg / current_avg) if current_avg else 0.0

    print(f'Frame size:   {args.frame_size}')
    print(f'Iterations:   {args.iterations}')
    print(f'Baseline ref:  {baseline_ref}')
    print(f'Current avg:   {current_avg * 1e6:.2f} us/frame')
    print(f'Original avg:  {original_avg * 1e6:.2f} us/frame')
    print(f'Speedup:       {speedup:.3f}x')


if __name__ == '__main__':
    main()
