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

ORIGINAL_SPEC = "speexdsp==0.1.1"
DEFAULT_FRAME_SIZE = 256
DEFAULT_FILTER_LENGTH = 2048
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_WARMUP_ITERS = 250
DEFAULT_ITERATIONS = 25000
DEFAULT_REPEATS = 7
DEFAULT_MIN_SAMPLE_SECONDS = 1.5
REPO_ROOT = Path(__file__).resolve().parents[1]


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


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


def _shared_code(
    frame_size: int,
    iterations: int,
    repeats: int,
    warmup_iters: int,
    min_sample_seconds: float,
    *,
    api_kind: str,
) -> str:
    if api_kind == "current":
        input_setup = """
    near = np.zeros(frame_size, dtype=np.int16)
    far = np.zeros(frame_size, dtype=np.int16)
    call = lambda: ec.process(near, far)
"""
    elif api_kind == "original":
        input_setup = """
    chunk = b'\\0\\0' * frame_size
    call = lambda: ec.process(chunk, chunk)
"""
    else:
        raise ValueError(f"unknown api_kind: {api_kind}")

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
warmup_iters = {warmup_iters}
min_sample_seconds = {min_sample_seconds}

create_times = []
process_avgs = []
process_p95s = []
process_current_kbs = []
process_peak_kbs = []
process_counts = []

for _ in range(repeats):
{input_setup.rstrip()}
    t0 = time.perf_counter()
    ec = EchoCanceller.create(frame_size, {DEFAULT_FILTER_LENGTH}, {DEFAULT_SAMPLE_RATE})
    create_times.append((time.perf_counter() - t0) * 1e6)

    for _ in range(warmup_iters):
        call()

    tracemalloc.start()
    timings = []
    sample_count = 0
    pass_started = time.perf_counter()
    while sample_count < iterations or (time.perf_counter() - pass_started) < min_sample_seconds:
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
        sample_count += 1
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    process_counts.append(sample_count)
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
    'process_count': int(median(process_counts)),
}}
print(json.dumps(payload))
""".strip()


def _current_extras_code(
    frame_size: int,
    iterations: int,
    repeats: int,
    warmup_iters: int,
    min_sample_seconds: float,
) -> str:
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
warmup_iters = {warmup_iters}
min_sample_seconds = {min_sample_seconds}

process_into_avgs = []
process_into_p95s = []
process_into_current_kbs = []
process_into_peak_kbs = []
process_into_counts = []
reset_times = []
destroy_times = []

for _ in range(repeats):
    near = np.zeros(frame_size, dtype=np.int16)
    far = np.zeros(frame_size, dtype=np.int16)
    out = np.empty(frame_size, dtype=np.int16)

    ec = EchoCanceller.create(frame_size, {DEFAULT_FILTER_LENGTH}, {DEFAULT_SAMPLE_RATE})
    for _ in range(warmup_iters):
        ec.process_into(near, far, out)

    tracemalloc.start()
    timings = []
    sample_count = 0
    pass_started = time.perf_counter()
    while sample_count < iterations or (time.perf_counter() - pass_started) < min_sample_seconds:
        start = time.perf_counter()
        ec.process_into(near, far, out)
        timings.append(time.perf_counter() - start)
        sample_count += 1
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    process_into_counts.append(sample_count)
    process_into_avgs.append(mean(timings) * 1e6)
    ordered = sorted(timings)
    process_into_p95s.append(ordered[max(0, int(len(ordered) * 0.95) - 1)] * 1e6)
    process_into_current_kbs.append(current / 1024.0)
    process_into_peak_kbs.append(peak / 1024.0)

    reset_samples = []
    destroy_samples = []
    for _ in range(5):
        sample_ec = EchoCanceller.create(frame_size, {DEFAULT_FILTER_LENGTH}, {DEFAULT_SAMPLE_RATE})
        t0 = time.perf_counter()
        sample_ec.reset()
        reset_samples.append((time.perf_counter() - t0) * 1e6)
        t1 = time.perf_counter()
        sample_ec.destroy()
        destroy_samples.append((time.perf_counter() - t1) * 1e6)
        del sample_ec

    reset_times.append(median(reset_samples))
    destroy_times.append(median(destroy_samples))

    del ec
    gc.collect()

payload = {{
    'process_into_avg_us': median(process_into_avgs),
    'process_into_p95_us': median(process_into_p95s),
    'process_into_current_kb': median(process_into_current_kbs),
    'process_into_peak_kb': median(process_into_peak_kbs),
    'process_into_count': int(median(process_into_counts)),
    'reset_us': median(reset_times),
    'destroy_us': median(destroy_times),
}}
print(json.dumps(payload))
""".strip()


def measure_shared(
    target: str,
    frame_size: int,
    iterations: int,
    repeats: int,
    warmup_iters: int,
    min_sample_seconds: float,
) -> dict[str, float]:
    api_kind = "current" if target == "." else "original"
    code = _shared_code(frame_size, iterations, repeats, warmup_iters, min_sample_seconds, api_kind=api_kind)
    return _run_json_in_venv(target, code)


def measure_current_extras(
    frame_size: int,
    iterations: int,
    repeats: int,
    warmup_iters: int,
    min_sample_seconds: float,
) -> dict[str, float]:
    code = _current_extras_code(frame_size, iterations, repeats, warmup_iters, min_sample_seconds)
    return _run_json_in_venv(".", code)


def _profile_current_extras(
    frame_size: int,
    iterations: int,
    warmup_iters: int,
    min_sample_seconds: float,
    profile_output: Path | None,
) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    measure_current_extras(frame_size, iterations, 1, warmup_iters, min_sample_seconds)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(20)
    profile_text = stream.getvalue()

    if profile_output is not None:
        profile_output.write_text(profile_text, encoding="utf-8")

    return profile_text


def _render_report(
    frame_size: int,
    iterations: int,
    repeats: int,
    warmup_iters: int,
    min_sample_seconds: float,
    current_shared: dict[str, float],
    original_shared: dict[str, float],
    current_extras: dict[str, float],
) -> str:
    process_speedup = original_shared["process_avg_us"] / current_shared["process_avg_us"] if current_shared["process_avg_us"] else 0.0

    lines = [
        f"Original spec: {ORIGINAL_SPEC}",
        f"Frame size:    {frame_size}",
        f"Timed calls:   at least {iterations} per pass and at least {min_sample_seconds:.1f}s",
        f"Repeats:       {repeats}",
        f"Warmup iters:   {warmup_iters}",
        "",
        "Comparable benchmark (same harness, API-specific adapter)",
        f"  Current create:  {current_shared['create_us']:.2f} us",
        f"  Original create: {original_shared['create_us']:.2f} us",
        f"  Current process avg:  {current_shared['process_avg_us']:.2f} us/frame",
        f"  Original process avg: {original_shared['process_avg_us']:.2f} us/frame",
        f"  Current process p95:  {current_shared['process_p95_us']:.2f} us/frame",
        f"  Original process p95: {original_shared['process_p95_us']:.2f} us/frame",
        f"  Current samples: {int(current_shared['process_count'])}",
        f"  Original samples: {int(original_shared['process_count'])}",
        f"  Current peak KB:  {current_shared['process_peak_kb']:.2f}",
        f"  Original peak KB: {original_shared['process_peak_kb']:.2f}",
        f"  Relative speedup: {process_speedup:.3f}x",
        "",
        "Current-only fast paths",
        f"  process_into avg: {current_extras['process_into_avg_us']:.2f} us/frame",
        f"  process_into p95: {current_extras['process_into_p95_us']:.2f} us/frame",
        f"  process_into samples: {int(current_extras['process_into_count'])}",
        f"  process_into peak: {current_extras['process_into_peak_kb']:.2f} KB",
        f"  reset:          {current_extras['reset_us']:.2f} us",
        f"  destroy:        {current_extras['destroy_us']:.2f} us",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the current SpeexDSP binding against the original PyPI release with a longer, more stable harness")
    parser.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Minimum timed calls per pass")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Repeat each benchmark and use the median run")
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS, help="Warmup calls before timing each pass")
    parser.add_argument("--min-sample-seconds", type=float, default=DEFAULT_MIN_SAMPLE_SECONDS, help="Minimum wall time for each timed pass")
    parser.add_argument("--json-output", type=str, default="", help="Optional path for machine-readable JSON output")
    parser.add_argument("--profile", action="store_true", help="Write cProfile output for the current process_into fast path")
    parser.add_argument("--profile-output", type=str, default="", help="Path for the optional profile text output")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters must be at least 0")
    if args.min_sample_seconds <= 0:
        raise SystemExit("--min-sample-seconds must be greater than 0")

    current_shared = measure_shared(
        ".",
        args.frame_size,
        args.iterations,
        args.repeats,
        args.warmup_iters,
        args.min_sample_seconds,
    )
    original_shared = measure_shared(
        ORIGINAL_SPEC,
        args.frame_size,
        args.iterations,
        args.repeats,
        args.warmup_iters,
        args.min_sample_seconds,
    )
    current_extras = measure_current_extras(
        args.frame_size,
        args.iterations,
        args.repeats,
        args.warmup_iters,
        args.min_sample_seconds,
    )

    report_text = _render_report(
        args.frame_size,
        args.iterations,
        args.repeats,
        args.warmup_iters,
        args.min_sample_seconds,
        current_shared,
        original_shared,
        current_extras,
    )
    print(report_text)

    if args.json_output:
        payload = {
            "frame_size": args.frame_size,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "warmup_iters": args.warmup_iters,
            "min_sample_seconds": args.min_sample_seconds,
            "original_spec": ORIGINAL_SPEC,
            "current_shared": current_shared,
            "original_shared": original_shared,
            "current_extras": current_extras,
        }
        Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.profile:
        profile_path = Path(args.profile_output) if args.profile_output else None
        profile_text = _profile_current_extras(
            args.frame_size,
            args.iterations,
            args.warmup_iters,
            args.min_sample_seconds,
            profile_path,
        )
        print()
        print("cProfile (current process_into fast path)")
        print(profile_text.rstrip())


if __name__ == "__main__":
    main()
