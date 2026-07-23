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

ORIGINAL_SPEC = "speexdsp==0.1.1"
DEFAULT_FRAME_SIZE = 256
DEFAULT_FILTER_LENGTH = 2048
DEFAULT_SAMPLE_RATE = 16000
WARMUP_ITERS = 50
DEFAULT_REPEATS = 3
REPO_ROOT = Path(__file__).resolve().parents[1]


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _shared_code(frame_size: int, iterations: int, repeats: int) -> str:
    return f"""
import gc
import json
import time
import tracemalloc
from statistics import mean, median

import numpy as np
from speexdsp import EchoCanceller

frame_size = {frame_size}
iterations = {iterations}
repeats = {repeats}

create_times = []
process_avgs = []
process_p95s = []
process_current_kbs = []
process_peak_kbs = []

for _ in range(repeats):
    near = np.zeros(frame_size, dtype=np.int16)
    far = np.zeros(frame_size, dtype=np.int16)

    t0 = time.perf_counter()
    ec = EchoCanceller.create(frame_size, {DEFAULT_FILTER_LENGTH}, {DEFAULT_SAMPLE_RATE})
    create_times.append((time.perf_counter() - t0) * 1e6)

    for _ in range({WARMUP_ITERS}):
        ec.process(near, far)

    tracemalloc.start()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        ec.process(near, far)
        timings.append(time.perf_counter() - start)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    process_avgs.append(mean(timings) * 1e6)
    ordered = sorted(timings)
    process_p95s.append(ordered[max(0, int(len(ordered) * 0.95) - 1)] * 1e6)
    process_current_kbs.append(current / 1024.0)
    process_peak_kbs.append(peak / 1024.0)

    del ec
    gc.collect()

payload = {{
    'create_us': median(create_times),
    'process_avg_us': median(process_avgs),
    'process_p95_us': median(process_p95s),
    'process_current_kb': median(process_current_kbs),
    'process_peak_kb': median(process_peak_kbs),
}}
print(json.dumps(payload))
""".strip()


def _current_extras_code(frame_size: int, iterations: int, repeats: int) -> str:
    return f"""
import gc
import json
import time
import tracemalloc
from statistics import mean, median

import numpy as np
from speexdsp import EchoCanceller

frame_size = {frame_size}
iterations = {iterations}
repeats = {repeats}

process_into_avgs = []
process_into_p95s = []
process_into_current_kbs = []
process_into_peak_kbs = []
reset_times = []
destroy_times = []

for _ in range(repeats):
    near = np.zeros(frame_size, dtype=np.int16)
    far = np.zeros(frame_size, dtype=np.int16)
    out = np.empty(frame_size, dtype=np.int16)

    ec = EchoCanceller.create(frame_size, {DEFAULT_FILTER_LENGTH}, {DEFAULT_SAMPLE_RATE})
    for _ in range({WARMUP_ITERS}):
        ec.process_into(near, far, out)

    tracemalloc.start()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        ec.process_into(near, far, out)
        timings.append(time.perf_counter() - start)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    process_into_avgs.append(mean(timings) * 1e6)
    ordered = sorted(timings)
    process_into_p95s.append(ordered[max(0, int(len(ordered) * 0.95) - 1)] * 1e6)
    process_into_current_kbs.append(current / 1024.0)
    process_into_peak_kbs.append(peak / 1024.0)

    t0 = time.perf_counter()
    ec.reset()
    reset_times.append((time.perf_counter() - t0) * 1e6)

    t1 = time.perf_counter()
    ec.destroy()
    destroy_times.append((time.perf_counter() - t1) * 1e6)

    del ec
    gc.collect()

payload = {{
    'process_into_avg_us': median(process_into_avgs),
    'process_into_p95_us': median(process_into_p95s),
    'process_into_current_kb': median(process_into_current_kbs),
    'process_into_peak_kb': median(process_into_peak_kbs),
    'reset_us': median(reset_times),
    'destroy_us': median(destroy_times),
}}
print(json.dumps(payload))
""".strip()


def _install_target(py: Path, target: str) -> None:
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(py), "-m", "pip", "install", "numpy", "pybind11"])
    _run([str(py), "-m", "pip", "install", target], cwd=REPO_ROOT)


def _run_json_in_venv(target: str, code: str) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="speexdsp-bench-") as tmp:
        venv_dir = Path(tmp) / "venv"
        _run([sys.executable, "-m", "venv", str(venv_dir)])
        py = _venv_python(venv_dir)
        _install_target(py, target)
        raw = subprocess.check_output([str(py), "-c", code], cwd=REPO_ROOT, text=True).strip()
        return json.loads(raw)


def measure_shared(target: str, frame_size: int, iterations: int, repeats: int) -> dict[str, float]:
    code = _shared_code(frame_size, iterations, repeats)
    return _run_json_in_venv(target, code)


def measure_current_extras(frame_size: int, iterations: int, repeats: int) -> dict[str, float]:
    code = _current_extras_code(frame_size, iterations, repeats)
    return _run_json_in_venv(".", code)


def _render_report(frame_size: int, iterations: int, repeats: int, current_shared: dict[str, float], original_shared: dict[str, float], current_extras: dict[str, float]) -> str:
    process_speedup = original_shared["process_avg_us"] / current_shared["process_avg_us"] if current_shared["process_avg_us"] else 0.0

    lines = [
        f"Original spec: {ORIGINAL_SPEC}",
        f"Frame size:    {frame_size}",
        f"Iterations:    {iterations}",
        f"Repeats:       {repeats}",
        "",
        "Comparable benchmark (same harness for both packages)",
        f"  Current create:  {current_shared['create_us']:.2f} us",
        f"  Original create: {original_shared['create_us']:.2f} us",
        f"  Current process avg:  {current_shared['process_avg_us']:.2f} us/frame",
        f"  Original process avg: {original_shared['process_avg_us']:.2f} us/frame",
        f"  Current process p95:  {current_shared['process_p95_us']:.2f} us/frame",
        f"  Original process p95: {original_shared['process_p95_us']:.2f} us/frame",
        f"  Current peak KB:  {current_shared['process_peak_kb']:.2f}",
        f"  Original peak KB: {original_shared['process_peak_kb']:.2f}",
        f"  Relative speedup: {process_speedup:.3f}x",
        "",
        "Current-only fast paths",
        f"  process_into avg: {current_extras['process_into_avg_us']:.2f} us/frame",
        f"  process_into p95: {current_extras['process_into_p95_us']:.2f} us/frame",
        f"  process_into peak: {current_extras['process_into_peak_kb']:.2f} KB",
        f"  reset:          {current_extras['reset_us']:.2f} us",
        f"  destroy:        {current_extras['destroy_us']:.2f} us",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the current SpeexDSP binding against the original PyPI release with the same harness")
    parser.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Repeat each benchmark and use the median run")
    parser.add_argument("--json-output", type=str, default="", help="Optional path for machine-readable JSON output")
    parser.add_argument("--profile", action="store_true", help="Write cProfile output for the current process_into fast path")
    parser.add_argument("--profile-output", type=str, default="", help="Path for the optional profile text output")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    current_shared = measure_shared(".", args.frame_size, args.iterations, args.repeats)
    original_shared = measure_shared(ORIGINAL_SPEC, args.frame_size, args.iterations, args.repeats)
    current_extras = measure_current_extras(args.frame_size, args.iterations, args.repeats)

    report_text = _render_report(args.frame_size, args.iterations, args.repeats, current_shared, original_shared, current_extras)
    print(report_text)

    if args.json_output:
        payload = {
            "frame_size": args.frame_size,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "original_spec": ORIGINAL_SPEC,
            "current_shared": current_shared,
            "original_shared": original_shared,
            "current_extras": current_extras,
        }
        Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.profile:
        profile_path = Path(args.profile_output) if args.profile_output else None
        profile_text = _profile_current_extras(args.frame_size, args.iterations, profile_path)
        print()
        print("cProfile (current process_into fast path)")
        print(profile_text.rstrip())


def _profile_current_extras(frame_size: int, iterations: int, profile_output: Path | None) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    measure_current_extras(frame_size, iterations, 1)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(20)
    profile_text = stream.getvalue()

    if profile_output is not None:
        profile_output.write_text(profile_text, encoding="utf-8")

    return profile_text


if __name__ == "__main__":
    main()
