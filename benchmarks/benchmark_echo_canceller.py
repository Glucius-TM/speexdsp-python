from __future__ import annotations

import argparse
import time
from statistics import mean

import numpy as np

from speexdsp import EchoCanceller


def run_benchmark(frame_size: int, iterations: int, mode: str) -> float:
    ec = EchoCanceller.create(frame_size, 2048, 16000)

    near = np.zeros(frame_size, dtype=np.int16)
    far = np.zeros(frame_size, dtype=np.int16)
    near_bytes = near.tobytes()
    far_bytes = far.tobytes()

    # Warm-up
    for _ in range(50):
        if mode == "ndarray":
            ec.process(near, far)
        else:
            ec.process(near_bytes, far_bytes)

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        if mode == "ndarray":
            ec.process(near, far)
        else:
            ec.process(near_bytes, far_bytes)
        timings.append(time.perf_counter() - start)

    return mean(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SpeexDSP echo cancellation")
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    ndarray_avg = run_benchmark(args.frame_size, args.iterations, "ndarray")
    bytes_avg = run_benchmark(args.frame_size, args.iterations, "bytes")

    print(f"Frame size: {args.frame_size}")
    print(f"Iterations:  {args.iterations}")
    print(f"ndarray avg: {ndarray_avg * 1e6:.2f} us/frame")
    print(f"bytes avg:   {bytes_avg * 1e6:.2f} us/frame")
    print(f"speedup:     {((bytes_avg / ndarray_avg) if ndarray_avg else 0.0):.3f}x")


if __name__ == "__main__":
    main()
