import numpy as np

from speexdsp import EchoCanceller


def test_echo_canceller_smoke():
    frame_size = 256
    sample_rate = 16000
    filter_length = 2048

    ec = EchoCanceller.create(frame_size, filter_length, sample_rate)

    assert ec.ok is True
    assert ec.frame_size == frame_size
    assert ec.filter_length == filter_length
    assert ec.sample_rate == sample_rate

    near = np.zeros(frame_size, dtype=np.int16)
    far = np.zeros(frame_size, dtype=np.int16)

    out = ec.process(near, far)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int16
    assert out.shape == (frame_size,)

    ec.reset()
    out2 = ec.process(near, far)
    assert out2.shape == (frame_size,)

    ec.destroy()
    assert ec.ok is False
