import numpy as np
import pytest

from antenna_diversity.config import default_config
from antenna_diversity.gnss.ca_code import ca_code
from antenna_diversity.receiver.acquisition import acquire
from antenna_diversity.receiver.tracking import estimate_cn0_db_hz, track


def test_prompt_window_cn0_matches_coherent_to_noise_power_ratio():
    cn0_db_hz = estimate_cn0_db_hz(
        np.array([1.0 + 0.0j, 3.0 + 0.0j]),
        coherent_interval_s=1e-3,
    )

    assert cn0_db_hz == pytest.approx(10.0 * np.log10(4000.0))


@pytest.mark.parametrize(
    "prompt_window",
    [np.array([], dtype=complex), np.array([1.0 + 0.0j])],
)
def test_prompt_window_cn0_is_unavailable_before_two_samples(prompt_window):
    assert np.isnan(estimate_cn0_db_hz(prompt_window, 1e-3))


@pytest.mark.parametrize("coherent_interval_s", [0.0, -1e-3, np.nan, np.inf])
def test_prompt_window_cn0_rejects_invalid_coherent_interval(coherent_interval_s):
    with pytest.raises(ValueError, match="coherent_interval_s"):
        estimate_cn0_db_hz(np.ones(2, dtype=complex), coherent_interval_s)


def test_prompt_window_cn0_rejects_nonvector_prompt_samples():
    with pytest.raises(ValueError, match="one-dimensional"):
        estimate_cn0_db_hz(np.ones((2, 1), dtype=complex), 1e-3)


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
    assert trk["carrier_nco_mode"] == "float"
    np.testing.assert_array_equal(
        trk["carrier_frequency_command_hz"],
        trk["carrier_frequency_hz"],
    )
    np.testing.assert_array_equal(
        trk["carrier_frequency_quantization_error_hz"],
        np.zeros_like(trk["carrier_frequency_hz"]),
    )
    assert trk["carrier_nco_frequency_lsb_hz"] == 0.0
    assert "carrier_fcw" not in trk
    assert "carrier_phase_word" not in trk


@pytest.mark.parametrize("word_bits", [30, 43])
def test_fixed_carrier_nco_tracks_and_exposes_hardware_state(word_bits):
    cfg = default_config()
    cfg.tracking.carrier_nco_mode = "fixed"
    cfg.tracking.carrier_nco_word_bits = word_bits
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
    assert trk["carrier_nco_mode"] == "fixed"
    assert trk["carrier_nco_frequency_lsb_hz"] == pytest.approx(
        6.6e6 / 2**word_bits
    )
    assert trk["carrier_fcw"].dtype == np.int64
    assert trk["carrier_phase_word"].dtype == np.uint64
    assert len(trk["carrier_fcw"]) == len(trk["prompt"])
    assert np.all(trk["carrier_phase_word"] < 2**word_bits)
    np.testing.assert_allclose(
        trk["carrier_frequency_hz"],
        trk["carrier_fcw"].astype(np.float64) * 6.6e6 / 2**word_bits,
        rtol=0.0,
        atol=np.finfo(float).eps * 6.6e6,
    )
    np.testing.assert_allclose(
        trk["carrier_frequency_quantization_error_hz"],
        trk["carrier_frequency_hz"]
        - trk["carrier_frequency_command_hz"],
        rtol=0.0,
        atol=np.finfo(float).eps * 6.6e6,
    )
    assert (
        abs(trk["carrier_frequency_hz"][-1] - cfg.signal.doppler_hz)
        <= cfg.tracking.max_final_doppler_error_hz
    )


def test_fixed_nco_quantization_error_stays_within_each_word_length_bound():
    for word_bits in (30, 43):
        cfg = default_config()
        cfg.tracking.carrier_nco_mode = "fixed"
        cfg.tracking.carrier_nco_word_bits = word_bits
        x = make_tracking_signal(cfg)
        acq = acquire(
            x, cfg.dfe.fs_out_hz, cfg.signal.prn, cfg.acquisition
        )

        trk = track(
            x,
            cfg.dfe.fs_out_hz,
            cfg.signal.prn,
            acq,
            cfg.tracking,
            cfg.signal.carrier_hz,
            cfg.signal.code_rate_hz,
        )

        assert np.max(
            np.abs(trk["carrier_frequency_quantization_error_hz"])
        ) <= (
            0.5 * trk["carrier_nco_frequency_lsb_hz"]
            + np.finfo(float).eps * 6.6e6
        )


def test_direct_tracking_call_rejects_unknown_carrier_nco_mode():
    cfg = default_config()
    cfg.tracking.carrier_nco_mode = "integer"
    x = make_tracking_signal(cfg)
    acq = acquire(x, cfg.dfe.fs_out_hz, cfg.signal.prn, cfg.acquisition)

    with pytest.raises(ValueError, match="carrier_nco_mode"):
        track(
            x,
            cfg.dfe.fs_out_hz,
            cfg.signal.prn,
            acq,
            cfg.tracking,
            cfg.signal.carrier_hz,
            cfg.signal.code_rate_hz,
        )


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
