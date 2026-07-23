from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from statistics import mean, median

import numpy as np

from speexdsp import EchoCanceller

ORIGINAL_SPEC = "speexdsp==0.1.1"
DEFAULT_FRAME_SIZE = 256
DEFAULT_FILTER_LENGTH = 2048
DEFAULT_SAMPLE_RATE = 16000
WARMUP_ITERS = 50
DEFAULT_REPEATS = 3


def _make_inputs(frame_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    near = np.zeros(frame_size, dtype=np.int16)
    far = np.zeros(frame_size, dtype=np.int16)
    out = np.empty(frame_size, dtype=np.int16)
    return near, far, out


def _timed_loop(fn, iterations: int) -> tuple[list[float], float, float]:
    tracemalloc.start()
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return timings, current / 1024.0, peak / 1024.0


def _summarize_timings(timings: list[float], current_kb: float, peak_kb: float) -> dict[str, float]:
    ordered = sorted(timings)
    p95_index = max(0, int(len(ordered) * 0.95) - 1)
    return {
        "avg_us": mean(timings) * 1e6,
        "p95_us": ordered[p95_index] * 1e6,
        "current_kb": current_kb,
        "peak_kb": peak_kb,
    }


def _measure_current_process(frame_size: int, iterations: int) -> dict[str, dict[str, float]]:
    near, far, _ = _make_inputs(frame_size)

    t0 = time.perf_counter()
    ec = EchoCanceller.create(frame_size, DEFAULT_FILTER_LENGTH, DEFAULT_SAMPLE_RATE)
    create_s = time.perf_counter() - t0

    for _ in range(WARMUP_ITERS):
        ec.process(near, far)

    process_timings, process_current_kb, process_peak_kb = _timed_loop(
        lambda: ec.process(near, far),
        iterations,
    )

    return {
        "create": {"us": create_s * 1e6},
        "process": {
            **_summarize_timings(process_timings, process_current_kb, process_peak_kb),
        },
    }


def _measure_current_process_into(frame_size: int, iterations: int) -> dict[str, dict[str, float]]:
    near, far, out = _make_inputs(frame_size)

    ec = EchoCanceller.create(frame_size, DEFAULT_FILTER_LENGTH, DEFAULT_SAMPLE_RATE)
    for _ in range(WARMUP_ITERS):
        ec.process_into(near, far, out)

    process_into_timings, process_into_current_kb, process_into_peak_kb = _timed_loop(
        lambda: ec.process_into(near, far, out),
        iterations,
    )

    return {
        "process_into": {
            **_summarize_timings(process_into_timings, process_into_current_kb, process_into_peak_kb),
        }
    }


def _measure_current_lifecycle(frame_size: int) -> dict[str, dict[str, float]]:
    t1 = time.perf_counter()
    ec = EchoCanceller.create(frame_size, DEFAULT_FILTER_LENGTH, DEFAULT_SAMPLE_RATE)
    create_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    ec.reset()
    reset_s = time.perf_counter() - t2

    t3 = time.perf_counter()
    ec.destroy()
    destroy_s = time.perf_counter() - t3

    return {
        "create": {"us": create_s * 1e6},
        "reset": {"us": reset_s * 1e6},
        "destroy": {"us": destroy_s * 1e6},
    }


def measure_current(frame_size: int, iterations: int) -> dict[str, dict[str, float]]:
    current_process = _measure_current_process(frame_size, iterations)
    current_into = _measure_current_process_into(frame_size, iterations)
    lifecycle = _measure_current_lifecycle(frame_size)

    return {
        "create": current_process["create"],
        "process": current_process["process"],
        "process_into": current_into["process_into"],
        "reset": lifecycle["reset"],
        "destroy": lifecycle["destroy"],
    }


def _aggregate_reports(reports: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    def _median_key(path: tuple[str, str]) -> float:
        outer, inner = path
        values = [report[outer][inner] for report in reports]
        return float(median(values))

    keys = {
        "create": ("create", "us"),
        "process_avg": ("process", "avg_us"),
        "process_p95": ("process", "p95_us"),
        "process_current_kb": ("process", "current_kb"),
        "process_peak_kb": ("process", "peak_kb"),
        "process_into_avg": ("process_into", "avg_us"),
        "process_into_p95": ("process_into", "p95_us"),
        "process_into_current_kb": ("process_into", "current_kb"),
        "process_into_peak_kb": ("process_into", "peak_kb"),
        "reset": ("reset", "us"),
        "destroy": ("destroy", "us"),
    }

    aggregated = {
        "create": {"us": _median_key(keys["create"])},
        "process": {
            "avg_us": _median_key(keys["process_avg"]),
            "p95_us": _median_key(keys["process_p95"]),
            "current_kb": _median_key(keys["process_current_kb"]),
            "peak_kb": _median_key(keys["process_peak_kb"]),
        },
        "process_into": {
            "avg_us": _median_key(keys["process_into_avg"]),
            "p95_us": _median_key(keys["process_into_p95"]),
            "current_kb": _median_key(keys["process_into_current_kb"]),
            "peak_kb": _median_key(keys["process_into_peak_kb"]),
        },
        "reset": {"us": _median_key(keys["reset"])},
        "destroy": {"us": _median_key(keys["destroy"])},
    }
    return aggregated


def _measure_current_repeated(frame_size: int, iterations: int, repeats: int) -> dict[str, object]:
    runs = [measure_current(frame_size, iterations) for _ in range(repeats)]
    return {"aggregate": _aggregate_reports(runs), "runs": runs}


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
    'avg_us': mean(timings) * 1e6,
    'p95_us': sorted(timings)[max(0, int(len(timings) * 0.95) - 1)] * 1e6,
    'current_kb': current / 1024.0,
    'peak_kb': peak / 1024.0,
}}
print(json.dumps(payload))
""".strip()
        raw = subprocess.check_output([str(py), "-c", code], text=True).strip()
        return json.loads(raw)


def _profile_current(frame_size: int, iterations: int, profile_output: Path | None) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    _measure_current_process_into(frame_size, iterations)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(20)
    profile_text = stream.getvalue()

    if profile_output is not None:
        profile_output.write_text(profile_text, encoding="utf-8")

    return profile_text


def _render_report(frame_size: int, iterations: int, current: dict[str, dict[str, float]], original: dict[str, float]) -> str:
    process_speedup = original["avg_us"] / current["process"]["avg_us"] if current["process"]["avg_us"] else 0.0
    process_into_speedup = original["avg_us"] / current["process_into"]["avg_us"] if current["process_into"]["avg_us"] else 0.0

    lines = [
        f"Original spec: {ORIGINAL_SPEC}",
        f"Frame size:    {frame_size}",
        f"Iterations:    {iterations}",
        "",
        "Current binding",
        f"  create:       {current['create']['us']:.2f} us",
        f"  process avg:  {current['process']['avg_us']:.2f} us/frame",
        f"  process p95:  {current['process']['p95_us']:.2f} us/frame",
        f"  process peak: {current['process']['peak_kb']:.2f} KB",
        f"  process_into avg: {current['process_into']['avg_us']:.2f} us/frame",
        f"  process_into p95: {current['process_into']['p95_us']:.2f} us/frame",
        f"  process_into peak: {current['process_into']['peak_kb']:.2f} KB",
        f"  reset:        {current['reset']['us']:.2f} us",
        f"  destroy:      {current['destroy']['us']:.2f} us",
        "",
        "Original release",
        f"  create:       {original['create_us']:.2f} us",
        f"  avg:          {original['avg_us']:.2f} us/frame",
        f"  p95:          {original['p95_us']:.2f} us/frame",
        f"  peak:         {original['peak_kb']:.2f} KB",
        "",
        f"Speedup vs original (process):      {process_speedup:.3f}x",
        f"Speedup vs original (process_into): {process_into_speedup:.3f}x",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the current SpeexDSP binding against the original PyPI release")
    parser.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Repeat the benchmark and use the median run")
    parser.add_argument("--json-output", type=str, default="", help="Optional path for machine-readable JSON output")
    parser.add_argument("--profile", action="store_true", help="Write cProfile output for the benchmarked current fast path")
    parser.add_argument("--profile-output", type=str, default="", help="Path for the optional profile text output")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    current_report = _measure_current_repeated(args.frame_size, args.iterations, args.repeats)
    current = current_report["aggregate"]
    original = measure_original(args.frame_size, args.iterations)

    report_text = _render_report(args.frame_size, args.iterations, current, original)
    print(report_text)

    if args.json_output:
        payload = {
            "frame_size": args.frame_size,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "original_spec": ORIGINAL_SPEC,
            "current": current,
            "current_runs": current_report["runs"],
            "original": original,
        }
        Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.profile:
        profile_path = Path(args.profile_output) if args.profile_output else None
        profile_text = _profile_current(args.frame_size, args.iterations, profile_path)
        print()
        print("cProfile (current benchmark run)")
        print(profile_text.rstrip())


if __name__ == "__main__":
    main()
