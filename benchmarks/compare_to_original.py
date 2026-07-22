from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean

import numpy as np

from speexdsp import EchoCanceller

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def first_commit() -> str:
    out = subprocess.check_output(
        ['git', '-C', str(REPO_ROOT), 'rev-list', '--max-parents=0', 'HEAD'],
        text=True,
    ).strip()
    if not out:
        raise RuntimeError('unable to determine the original baseline commit')
    return out.splitlines()[0]


def original_avg(frame_size: int, iterations: int, baseline_ref: str) -> float:
    with tempfile.TemporaryDirectory(prefix='speexdsp-baseline-') as tmp:
        tmpdir = Path(tmp)
        baseline_dir = tmpdir / 'baseline'
        site_dir = tmpdir / 'site'
        subprocess.run(['git', '-C', str(REPO_ROOT), 'worktree', 'add', '--detach', str(baseline_dir), baseline_ref], check=True)
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(site_dir), 'numpy'], cwd=baseline_dir, check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(site_dir), '.'], cwd=baseline_dir, check=True)
            env = os.environ.copy()
            env['PYTHONPATH'] = str(site_dir)
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
samples = []
for _ in range(iterations):
    t0 = time.perf_counter()
    ec.process(chunk, chunk)
    samples.append(time.perf_counter() - t0)
print(mean(samples))
""".strip()
            out = subprocess.check_output([sys.executable, '-c', code], cwd=baseline_dir, env=env, text=True).strip()
            return float(out)
        finally:
            subprocess.run(['git', '-C', str(REPO_ROOT), 'worktree', 'remove', '--force', str(baseline_dir)], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare the current SpeexDSP binding against the original baseline')
    parser.add_argument('--frame-size', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--baseline-ref', type=str, default='')
    args = parser.parse_args()

    baseline_ref = args.baseline_ref or first_commit()
    cur = current_avg(args.frame_size, args.iterations)
    old = original_avg(args.frame_size, args.iterations, baseline_ref)

    print(f'Baseline ref:  {baseline_ref}')
    print(f'Current avg:   {cur * 1e6:.2f} us/frame')
    print(f'Original avg:  {old * 1e6:.2f} us/frame')
    print(f'Speedup:       {old / cur:.3f}x')


if __name__ == '__main__':
    main()
