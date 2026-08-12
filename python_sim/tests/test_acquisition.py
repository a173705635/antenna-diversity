import numpy as np

from antenna_diversity.config import default_config
from antenna_diversity.gnss.ca_code import ca_code
from antenna_diversity.receiver.acquisition import acquire


def make_acquisition_signal(cfg, delay_samples=600, noise_sigma=0.05, seed=7):
    fs_hz = cfg.dfe.fs_out_hz
    samples_per_ms = round(fs_hz * 1e-3)
    duration_ms = (
        cfg.acquisition.coherent_ms * cfg.acquisition.noncoherent_ms
    )
    n = np.arange(duration_ms * samples_per_ms)
    delay_chips = delay_samples * cfg.signal.code_rate_hz / fs_hz
    code = ca_code(cfg.signal.prn)
    code_phase = np.mod(
        n * cfg.signal.code_rate_hz / fs_hz - delay_chips,
        1023.0,
    )
    sampled_code = code[np.floor(code_phase).astype(int)]
    signal = sampled_code * np.exp(
        1j * 2 * np.pi * cfg.signal.doppler_hz * n / fs_hz
    )
    rng = np.random.Generator(np.random.MT19937(seed))
    noise = noise_sigma / np.sqrt(2) * (
        rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size)
    )
    return signal + noise


def test_fft_acquisition_estimates_doppler_and_code_phase():
    """Catch carrier-wipeoff sign, FFT conjugation, or code-index errors."""
    cfg = default_config()
    fs_hz = cfg.dfe.fs_out_hz
    samples_per_ms = round(fs_hz * 1e-3)
    x = make_acquisition_signal(cfg)

    acq = acquire(x, fs_hz, cfg.signal.prn, cfg.acquisition)

    assert acq["success"]
    assert abs(acq["doppler_hz"] - cfg.signal.doppler_hz) <= 250
    phase_error = abs(acq["code_phase_samples"] - 600)
    circular_error = min(phase_error, samples_per_ms - phase_error)
    assert circular_error <= 1
    assert acq["second_peak_distance_samples"] > acq["exclusion_samples"]
    assert acq["search_power"].shape == (
        cfg.acquisition.doppler_bins_hz.size,
        cfg.acquisition.coherent_ms * samples_per_ms,
    )


def test_acquisition_requires_all_configured_integration_samples():
    """Catch silent zero-padding of an incomplete noncoherent integration."""
    cfg = default_config()
    required = (
        round(cfg.dfe.fs_out_hz * 1e-3)
        * cfg.acquisition.coherent_ms
        * cfg.acquisition.noncoherent_ms
    )

    try:
        acquire(
            np.zeros(required - 1, dtype=np.complex128),
            cfg.dfe.fs_out_hz,
            cfg.signal.prn,
            cfg.acquisition,
        )
    except ValueError as error:
        assert "needs" in str(error)
    else:
        raise AssertionError("incomplete acquisition input was accepted")
