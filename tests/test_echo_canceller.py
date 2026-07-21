import numpy as np

from speexdsp import EchoCanceller


def test_echo_canceller():
    frames = 64
    echo_canceller = EchoCanceller.create(frames, 256, 16000)

    chunk = np.zeros(frames, dtype=np.int16)
    for _ in range(16):
        out = echo_canceller.process(chunk, chunk)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int16
        assert out.shape == (frames,)
