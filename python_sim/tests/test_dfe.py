import numpy as np
import pytest

from antenna_diversity.config import default_config
from antenna_diversity.dfe.decimator import design_decimator
from antenna_diversity.dfe.processing import process_channels


def test_decimator_has_unity_dc_gain_and_integer_group_delay():
    """Catch filter normalization or odd-length group-delay regressions."""
    h, info = design_decimator(16.368e6, 4, 1.8e6, 129)

    assert h.shape == (129,)
    assert np.sum(h) == pytest.approx(1.0, abs=1e-12)
    assert info["group_delay_input_samples"] == 64
    assert info["fs_out_hz"] == pytest.approx(4.092e6, abs=1e-9)


def test_dfe_moves_if_to_dc_and_preserves_interchannel_phase():
    """Catch a missing DDC, incorrect delay slice, or separate channel NCOs."""
    cfg = default_config()
    cfg.dfe.output_bits = 0
    n_samples = round(cfg.dfe.fs_in_hz * 2e-3)
    n = np.arange(n_samples)
    tone = 100 * np.exp(
        1j * 2 * np.pi * cfg.dfe.if_hz * n / cfg.dfe.fs_in_hz
    )
    x = np.vstack([tone, tone * np.exp(1j * 0.37)])

    y, info = process_channels(x, cfg.dfe)

    expected_length = (
        n_samples - info["group_delay_input_samples"] - 1
    ) // cfg.dfe.decimation + 1
    assert y.shape == (2, expected_length)
    assert info["fs_out_hz"] == pytest.approx(4.092e6, abs=1e-9)
    assert np.mean(y[1] / y[0]) == pytest.approx(
        np.exp(1j * 0.37), abs=1e-10
    )
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(y[0])))
    frequency = np.fft.fftshift(np.fft.fftfreq(y.shape[1], 1 / info["fs_out_hz"]))
    peak_frequency = frequency[np.argmax(spectrum)]
    assert abs(peak_frequency) <= info["fs_out_hz"] / y.shape[1]


def test_dfe_output_quantizer_uses_binary_shift_and_signed_limits():
    """Catch omission of the post-filter binary shift or component clipping."""
    cfg = default_config()
    cfg.dfe.fs_in_hz = 4.092e6
    cfg.dfe.if_hz = 0.0
    cfg.dfe.decimation = 1
    cfg.dfe.cutoff_hz = 1.8e6
    cfg.dfe.num_taps = 3
    cfg.dfe.output_bits = 3
    cfg.dfe.output_binary_shift = 1
    x = np.full((1, 16), 100.0 + 100.0j)

    y, info = process_channels(x, cfg.dfe)

    assert info["output_scale"] == 2
    assert np.all(y.real == 3)
    assert np.all(y.imag == 3)
