import numpy as np

from antenna_diversity.config import default_config
from antenna_diversity.gnss.ca_code import ca_code
from antenna_diversity.receiver.acquisition import acquire
from antenna_diversity.receiver.tracking import track


def make_tracking_signal(
    cfg,
    duration_ms=40,
    delay_samples=600,
    noise_sigma=0.15,
    seed=9,
):
    fs_hz = cfg.dfe.fs_out_hz
    n = np.arange(duration_ms * round(fs_hz * 1e-3))
    delay_chips = delay_samples * cfg.signal.code_rate_hz / fs_hz
    code = ca_code(cfg.signal.prn)
    code_phase = np.mod(
        n * cfg.signal.code_rate_hz / fs_hz - delay_chips,
        1023.0,
    )
    sampled_code = code[np.floor(code_phase).astype(int)]
    carrier = np.exp(
        1j
        * (
            2 * np.pi * cfg.signal.doppler_hz * n / fs_hz
            + 0.4
        )
    )
    rng = np.random.Generator(np.random.MT19937(seed))
    noise = noise_sigma / np.sqrt(2) * (
        rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size)
    )
    return sampled_code * carrier + noise


def test_tracking_converges_after_successful_acquisition():
    """Catch code-epoch indexing, discriminator signs, or unstable loop updates."""
    cfg = default_config()
    x = make_tracking_signal(cfg)
    acq = acquire(x, cfg.dfe.fs_out_hz, cfg.signal.prn, cfg.acquisition)

    trk = track(
        x,
        cfg.dfe.fs_out_hz,
        cfg.signal.prn,
        acq,
        cfg.tracking,
        cfg.signal.carrier_hz,
        cfg.signal.code_rate_hz,
    )

    assert trk["success"]
    assert len(trk["prompt"]) >= 20
    assert np.all(np.isfinite(trk["carrier_frequency_hz"]))
    assert np.all(np.isfinite(trk["code_rate_hz"]))
    assert np.mean(np.abs(trk["prompt"][-10:])) > 100
    assert (
        abs(trk["carrier_frequency_hz"][-1] - cfg.signal.doppler_hz)
        <= cfg.tracking.max_final_doppler_error_hz
    )
    assert np.sqrt(np.mean(trk["dll_error"][-10:] ** 2)) < 0.5
    assert np.sqrt(np.mean(trk["pll_error_rad"][-10:] ** 2)) < 0.5


def test_tracking_returns_failed_status_without_successful_acquisition():
    """Catch accidental loop entry when acquisition has not passed."""
    cfg = default_config()

    trk = track(
        np.ones(20_000, dtype=np.complex128),
        cfg.dfe.fs_out_hz,
        cfg.signal.prn,
        {"success": False},
        cfg.tracking,
        cfg.signal.carrier_hz,
        cfg.signal.code_rate_hz,
    )

    assert not trk["success"]
    assert trk["status"] == "acquisition_failed"
    assert trk["prompt"].size == 0
