from __future__ import annotations

import argparse
import time
from statistics import mean

import numpy as np

from speexdsp import EchoCanceller


def run_benchmark(frame_size: int, iterations: int) -> float:
    ec = EchoCanceller.create(frame_size, 2048, 16000)

    near = np.zeros(frame_size, dtype=np.int16)
    far = np.zeros(frame_size, dtype=np.int16)

    # Warm-up
    for _ in range(50):
        ec.process(near, far)

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        ec.process(near, far)
        timings.append(time.perf_counter() - start)

    return mean(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SpeexDSP echo cancellation")
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    avg = run_benchmark(args.frame_size, args.iterations)

    print(f"Frame size: {args.frame_size}")
    print(f"Iterations:  {args.iterations}")
    print(f"avg:         {avg * 1e6:.2f} us/frame")
    print(f"fps:         {(1.0 / avg) if avg else 0.0:.2f}")


if __name__ == "__main__":
    main()
