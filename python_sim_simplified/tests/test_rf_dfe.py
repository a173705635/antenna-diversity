import numpy as np
import pytest
from scipy.constants import Boltzmann
from scipy.signal import welch

from core.config import DFEConfig, RFConfig, RFStage
from core.dfe import design_decimator, process_dfe
from core.rf import design_rf_filter, filter_rf_input, generate_rf_noise, process_rf


FS = 16.368e6
IF = 4.092e6


def test_rf_noise_has_thermal_power_density_and_stationary_start():
    cfg = RFConfig(noise_filter_taps=129)
    h = design_rf_filter(FS, cfg.noise_bandwidth_hz, cfg.noise_filter_taps)
    psd = Boltzmann * cfg.temperature_k * 10 ** (1.2350293663855842 / 10)
    noise = generate_rf_noise(400_000, psd, 50.0, FS, IF, ('TL',), 23, h)[0]
    expected_power = psd * 50.0 * cfg.noise_bandwidth_hz
    np.testing.assert_allclose(np.mean(np.abs(noise) ** 2), expected_power, rtol=0.02)
    np.testing.assert_allclose(np.var(noise.real), expected_power / 2, rtol=0.03)
    np.testing.assert_allclose(np.var(noise.imag), expected_power / 2, rtol=0.03)
    np.testing.assert_allclose(np.mean(np.abs(noise[:1000]) ** 2), expected_power, rtol=0.15)
    np.testing.assert_allclose(FS * np.sum(h**2), cfg.noise_bandwidth_hz, rtol=1e-10)


def test_rf_noise_channel_identity_and_prefix_do_not_depend_on_layout():
    h = design_rf_filter(FS, 8.184e6, 129)
    args = (1e-20, 50.0, FS, IF)
    both = generate_rf_noise(4096, *args, ('TL', 'TR'), 42, h)
    reverse = generate_rf_noise(4096, *args, ('TR', 'TL'), 42, h)
    longer = generate_rf_noise(8192, *args, ('TL',), 42, h)
    np.testing.assert_array_equal(both[0], reverse[1])
    np.testing.assert_array_equal(both[1], reverse[0])
    np.testing.assert_array_equal(both[0], longer[0, :4096])


def test_rf_filter_rejects_stopband_without_tone_delay():
    h = design_rf_filter(FS, 8.184e6, 129)
    n = np.arange(4096)
    carrier = np.exp(2j * np.pi * IF * n / FS)
    outside = np.exp(2j * np.pi * (IF + 6e6) * n / FS)
    filtered = filter_rf_input(np.vstack((carrier, outside)), h, FS, IF)
    assert filtered.shape == (2, 4096)
    np.testing.assert_allclose(filtered[0, 256:-256], carrier[256:-256], atol=0.005)
    assert np.mean(np.abs(filtered[1, 256:-256]) ** 2) < 1e-4


def test_rf_noise_density_is_invariant_to_sample_rate():
    psd = 1e-20
    for fs in (16.368e6, 33e6):
        h = design_rf_filter(fs, 8.184e6, 129)
        noise = generate_rf_noise(300_000, psd, 50.0, fs, 4.092e6, ('TL',), 1, h)[0]
        np.testing.assert_allclose(np.mean(np.abs(noise) ** 2), psd * 50.0 * 8.184e6, rtol=0.025)
        frequencies, spectrum = welch(noise, fs=fs, nperseg=4096,
                                      return_onesided=False)
        offset_hz = (frequencies - 4.092e6 + fs / 2) % fs - fs / 2
        in_band = np.abs(offset_hz) < 1e6
        stop_band = np.abs(offset_hz) > 5.5e6
        np.testing.assert_allclose(np.mean(spectrum[in_band]), psd * 50.0, rtol=0.10)
        assert np.mean(spectrum[stop_band]) < psd * 50.0 * 0.01


def test_rf_fullband_filter_is_identity():
    np.testing.assert_array_equal(design_rf_filter(FS, FS, 129), [1.0])


@pytest.mark.parametrize('position', (0, 1024, 2047))
def test_rf_filter_preserves_envelope_time_origin_at_record_edges(position):
    h = design_rf_filter(FS, 8.184e6, 129)
    oscillator = np.exp(2j * np.pi * IF * np.arange(2048) / FS)
    envelope = np.zeros(2048, dtype=complex)
    envelope[position] = 0.4 + 0.7j
    filtered = filter_rf_input((envelope * oscillator)[None, :], h, FS, IF)[0]
    # Independent centered convolution checks delay, including finite-record ends.
    expected = np.convolve(envelope, h, mode='same') * oscillator
    np.testing.assert_allclose(filtered, expected, atol=1e-14, rtol=1e-12)


def test_rf_is_repeatable_and_adc_codes_are_signed_integers():
    x = np.zeros((2, 4096), dtype=complex)
    first, info = process_rf(x, RFConfig(), 50.0, 42, fs_hz=FS, if_hz=IF, channel_ids=('TL', 'TR'))
    second, _ = process_rf(x, RFConfig(), 50.0, 42, fs_hz=FS, if_hz=IF, channel_ids=('TL', 'TR'))

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.real, np.round(first.real))
    np.testing.assert_array_equal(first.imag, np.round(first.imag))
    assert info["noise_figure_db"] > 0.0


def test_hard_limiter_clips_i_and_q_before_adc():
    cfg = RFConfig(
        temperature_k=0.0,
        noise_bandwidth_hz=FS,
        stages=(RFStage("unity", 0.0, 0.0),),
        adc_bits=4,
        adc_vpp=1.0,
        enable_soft_limiting=False,
    )
    output, _ = process_rf(np.array([[2.0 + 2.0j]]), cfg, 50.0, 1,
                           fs_hz=FS, if_hz=IF, channel_ids=('TL',))

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
