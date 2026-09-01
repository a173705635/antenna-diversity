import numpy as np

from config import DFEConfig, RFConfig, RFStage
from dfe import design_decimator, process_dfe
from rf import process_rf


def test_rf_is_repeatable_and_adc_codes_are_signed_integers():
    x = np.zeros((2, 4096), dtype=complex)
    first, info = process_rf(x, RFConfig(), 50.0, 42)
    second, _ = process_rf(x, RFConfig(), 50.0, 42)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.real, np.round(first.real))
    np.testing.assert_array_equal(first.imag, np.round(first.imag))
    assert info["noise_figure_db"] > 0.0


def test_hard_limiter_clips_i_and_q_before_adc():
    cfg = RFConfig(
        temperature_k=0.0,
        stages=(RFStage("unity", 0.0, 0.0),),
        adc_bits=4,
        adc_vpp=1.0,
        enable_soft_limiting=False,
    )
    output, _ = process_rf(np.array([[2.0 + 2.0j]]), cfg, 50.0, 1)

    assert output[0, 0] == 7.0 + 7.0j


def test_dfe_moves_if_to_dc_and_preserves_interchannel_phase():
    cfg = DFEConfig(output_bits=0)
    fs_in_hz = 16.368e6
    signal_if_hz = 4.092e6
    n = np.arange(round(fs_in_hz * 0.004))
    tone = np.exp(1j * 2.0 * np.pi * signal_if_hz * n / fs_in_hz)
    channels = np.vstack([tone, tone * np.exp(1j * np.deg2rad(70.0))])

    output, info = process_dfe(channels, cfg, fs_in_hz, signal_if_hz)
    phase = np.angle(np.vdot(output[0], output[1]), deg=True)
    h, filter_info = design_decimator(
        fs_in_hz, cfg.decimation, cfg.cutoff_hz, cfg.num_taps
    )

    assert np.isclose(phase, 70.0, atol=0.1)
    assert info["fs_out_hz"] == fs_in_hz / cfg.decimation
    assert info["if_hz"] == signal_if_hz
    assert np.isclose(h.sum(), 1.0)
    assert filter_info["group_delay_input_samples"] == (cfg.num_taps - 1) // 2


def test_dfe_output_is_shifted_quantized_and_saturated():
    cfg = DFEConfig(
        decimation=1,
        cutoff_hz=1.0e6,
        num_taps=3,
        output_bits=8,
        output_binary_shift=2,
        nco_frequency_hz=0.0,
    )
    samples = np.full((1, 16), 40.0 + 600.0j)

    output, info = process_dfe(samples, cfg, 16.368e6, 4.092e6)

    assert output[0, -1] == 10.0 + 127.0j
    assert info["output_scale"] == 4


def test_dfe_nco_can_override_signal_if():
    cfg = DFEConfig(
        decimation=1,
        cutoff_hz=1.0e6,
        num_taps=3,
        output_bits=0,
        nco_frequency_hz=3.0e6,
    )

    _, info = process_dfe(
        np.ones((1, 16), dtype=complex),
        cfg,
        16.368e6,
        4.092e6,
    )

    assert info["if_hz"] == 3.0e6
