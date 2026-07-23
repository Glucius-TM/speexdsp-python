speexdsp for Python
===================

Modern Python bindings for SpeexDSP AEC.

## Requirements

- Python 3.8+
- C++ toolchain
- `libspeexdsp`
- `numpy`

## Install

### From the repository

```bash
pip install .
```

### System package example (Debian/Ubuntu)

```bash
sudo apt install libspeexdsp-dev
```

## Quick start

```python
import numpy as np
from speexdsp import EchoCanceller

frame_size = 256
sample_rate = 16000
filter_length = 2048

ec = EchoCanceller.create(frame_size, filter_length, sample_rate)

# Contiguous int16 PCM blocks, one frame each
near = np.zeros(frame_size, dtype=np.int16)
far = np.zeros(frame_size, dtype=np.int16)
out = np.empty(frame_size, dtype=np.int16)

clean = ec.process(near, far)
ec.process_into(near, far, out)
print(type(clean), clean.dtype, clean.shape)
print(out.dtype, out.shape)
```

## Notes

- Use one canceller instance per channel.
- Keep near-end and far-end aligned in time.
- Inputs must be contiguous `numpy.ndarray[int16]` buffers.
- The returned array is a zero-copy view over an internal reusable buffer; copy it if you need to keep the data after the next `process()` call.
- For the lowest allocation path, reuse an output array and call `ec.process_into(near, far, out)`.
- Recommended frame size: 10–20 ms.
- Recommended filter tail: 100–500 ms.

## Benchmark

Latest verified comparative benchmark run on Ubuntu with Python 3.11, frame size 256, and 2000 iterations.

| Path | Avg time | Relative |
| --- | ---: | ---: |
| Current build (`process`) | 45.94 us/frame | 0.801x |
| Current build (`process_into`) | 38.26 us/frame | 0.962x |
| Original release (`speexdsp==0.1.1`) | 36.80 us/frame | 1.000x |

In the latest verified run, `process_into()` was the closest path to the original, but it was still about 3.8% slower.

The benchmark now runs multiple repetitions, uses the median report for stability, can write machine-readable JSON, and still reports creation time, p95 latency, peak Python allocations, `process_into()` timing, and current lifecycle overhead (`reset()` / `destroy()`). It can also emit `cProfile` output for the benchmarked current path when profiling is enabled.

## API

- `EchoCanceller.create(frame_size=256, filter_length=2048, sample_rate=16000, mics=1, speakers=1)`
- `ec.process(near, far)`
- `ec.process_into(near, far, out)`
- `ec.reset()`
- `ec.destroy()`
- Properties: `ok`, `frame_size`, `filter_length`, `sample_rate`, `mics`, `speakers`
