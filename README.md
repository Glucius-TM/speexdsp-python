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

clean = ec.process(near, far)
print(type(clean), clean.dtype, clean.shape)
```

## Notes

- Use one canceller instance per channel.
- Keep near-end and far-end aligned in time.
- Recommended frame size: 10–20 ms.
- Recommended filter tail: 100–500 ms.

## API

- `EchoCanceller.create(frame_size=256, filter_length=2048, sample_rate=16000, mics=1, speakers=1)`
- `ec.process(near, far)`
- `ec.reset()`
- `ec.destroy()`
- Properties: `ok`, `frame_size`, `filter_length`, `sample_rate`, `mics`, `speakers`
