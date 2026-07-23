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

The benchmark harness compares the current binding against the original release using the same isolated harness for both packages. Both sides are measured with the same inputs, warmup, repetition count, and median aggregation, so the shared comparison is fair. The script also reports current-only fast-path metrics (`process_into()`, `reset()`, and `destroy()`), machine-readable JSON, and optional `cProfile` output.

## API

- `EchoCanceller.create(frame_size=256, filter_length=2048, sample_rate=16000, mics=1, speakers=1)`
- `ec.process(near, far)`
- `ec.process_into(near, far, out)`
- `ec.reset()`
- `ec.destroy()`
- Properties: `ok`, `frame_size`, `filter_length`, `sample_rate`, `mics`, `speakers`
