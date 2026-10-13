"""Equivalent RF cascade, thermal noise, limiter, and complex ADC."""

import numpy as np
from scipy.optimize import brentq
from scipy.signal import firwin, lfilter

from .config import RFConfig
from .models import RFInfo


BOLTZMANN_J_PER_K = 1.380649e-23


def design_rf_filter(fs_hz: float, bandwidth_hz: float, num_taps: int) -> np.ndarray:
    """Unity-DC real lowpass, with odd num_taps and two-sided ENBW bandwidth_hz."""
    if bandwidth_hz == fs_hz:
        return np.array([1.0])

    def enbw_error(cutoff_hz: float) -> float:
        taps = firwin(num_taps, cutoff_hz, fs=fs_hz)
        return fs_hz * np.sum(taps**2) - bandwidth_hz

    cutoff_hz = brentq(enbw_error, 1.0, fs_hz / 2.0 * (1.0 - 1e-10))
    return firwin(num_taps, cutoff_hz, fs=fs_hz)


def filter_rf_input(
    samples: np.ndarray, coefficients: np.ndarray, fs_hz: float, if_hz: float
) -> np.ndarray:
    """Band-limit 2-D sampled IF and compensate delay; record ends are zero-padded."""
    delay = (len(coefficients) - 1) // 2
    sample_index = np.arange(samples.shape[-1])
    oscillator = np.exp(-2j * np.pi * if_hz * sample_index / fs_hz)
    output = np.empty_like(samples, dtype=complex)
    for row in range(samples.shape[0]):
        baseband = samples[row] * oscillator
        padded = np.pad(baseband, (delay, delay))
        filtered = lfilter(coefficients, [1.0], padded)
        aligned = filtered[2 * delay:2 * delay + samples.shape[-1]]
        output[row] = aligned * oscillator.conj()
    return output


def generate_rf_noise(
    n_samples: int,
    noise_psd_w_per_hz: float,
    reference_impedance_ohm: float,
    fs_hz: float,
    if_hz: float,
    channel_ids: tuple[str, ...],
    seed: int,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Stationary complex IF noise, with stable per-physical-channel I/Q streams."""
    delay = (len(coefficients) - 1) // 2
    white_std_v = np.sqrt(noise_psd_w_per_hz * reference_impedance_ohm * fs_hz / 2.0)
    sample_index = np.arange(n_samples)
    oscillator = np.exp(2j * np.pi * if_hz * sample_index / fs_hz)
    output = np.empty((len(channel_ids), n_samples), dtype=complex)
    for row, channel_id in enumerate(channel_ids):
        identity = channel_id.encode('utf-8')
        streams = []
        for component_id in (0, 1):
            sequence = np.random.SeedSequence([seed, len(identity), *identity, component_id])
            rng = np.random.Generator(np.random.MT19937(sequence))
            white = rng.standard_normal(n_samples + 2 * delay)
            filtered = lfilter(coefficients, [1.0], white)
            streams.append(filtered[2 * delay:2 * delay + n_samples])
        output[row] = white_std_v * (streams[0] + 1j * streams[1]) * oscillator
    return output


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def process_rf(
    x_antenna_v: np.ndarray,
    cfg: RFConfig,
    reference_impedance_ohm: float,
    seed: int,
    *,
    fs_hz: float,
    if_hz: float,
    channel_ids: tuple[str, ...],
) -> tuple[np.ndarray, RFInfo]:
    if x_antenna_v.ndim != 2 or x_antenna_v.shape[0] == 0:
        raise ValueError('RF input must have shape (channels, samples)')
    amplitude_db = np.broadcast_to(np.asarray(cfg.amplitude_error_db, dtype=float),
                                   (x_antenna_v.shape[0],))
    phase_deg = np.broadcast_to(np.asarray(cfg.phase_error_deg, dtype=float),
                               (x_antenna_v.shape[0],))
    if not np.all(np.isfinite(amplitude_db)) or not np.all(np.isfinite(phase_deg)):
        raise ValueError('RF amplitude/phase errors must be finite')
    with np.errstate(over='raise', invalid='raise'):
        error_response = 10.**(amplitude_db/20.) * np.exp(1j*np.deg2rad(phase_deg))
    gain_linear = 10.0 ** (
        np.array([stage.gain_db for stage in cfg.stages]) / 10.0
    )
    noise_factors = 10.0 ** (
        np.array([stage.noise_figure_db for stage in cfg.stages]) / 10.0
    )
    cascade_noise_factor = noise_factors[0]
    preceding_gain = gain_linear[0]
    for index in range(1, len(gain_linear)):
        cascade_noise_factor += (
            noise_factors[index] - 1.0
        ) / preceding_gain
        preceding_gain *= gain_linear[index]

    total_gain_linear = np.prod(gain_linear)
    coefficients = design_rf_filter(fs_hz, cfg.noise_bandwidth_hz, cfg.noise_filter_taps)
    measured_bandwidth_hz = fs_hz * np.sum(coefficients**2)
    noise_psd_w_per_hz = BOLTZMANN_J_PER_K * cfg.temperature_k * cascade_noise_factor
    input_noise_power_w = noise_psd_w_per_hz * measured_bandwidth_hz
    noise_rms_v = np.sqrt(input_noise_power_w * reference_impedance_ohm)
    filtered_signal = filter_rf_input(x_antenna_v, coefficients, fs_hz, if_hz)
    noise = generate_rf_noise(
        x_antenna_v.shape[1], noise_psd_w_per_hz, reference_impedance_ohm,
        fs_hz, if_hz, channel_ids, seed, coefficients,
    )
    before_limit = (filtered_signal + noise) * np.sqrt(total_gain_linear)
    del filtered_signal, noise
    if np.any(amplitude_db != 0.) or np.any(phase_deg != 0.):
        before_limit *= error_response[:, None]
    limit_v = cfg.adc_vpp / 2.0
    saturated = (np.abs(before_limit.real) > limit_v) | (
        np.abs(before_limit.imag) > limit_v
    )
    if cfg.enable_soft_limiting:
        adc_input = limit_v * np.tanh(before_limit.real / limit_v) + 1j * (
            limit_v * np.tanh(before_limit.imag / limit_v)
        )
    else:
        adc_input = np.clip(before_limit.real, -limit_v, limit_v) + 1j * np.clip(
            before_limit.imag, -limit_v, limit_v
        )

    lsb_v = cfg.adc_vpp / 2**cfg.adc_bits
    minimum = -(2 ** (cfg.adc_bits - 1))
    maximum = 2 ** (cfg.adc_bits - 1) - 1
    i_code = np.clip(
        _round_half_away_from_zero(adc_input.real / lsb_v), minimum, maximum
    )
    q_code = np.clip(
        _round_half_away_from_zero(adc_input.imag / lsb_v), minimum, maximum
    )
    adc_codes = (i_code + 1j * q_code).astype(complex)
    info: RFInfo = {
        "fs_hz": fs_hz,
        "if_hz": if_hz,
        "channel_ids": channel_ids,
        "filter_coefficients": coefficients,
        "noise_bandwidth_hz": cfg.noise_bandwidth_hz,
        "measured_noise_bandwidth_hz": measured_bandwidth_hz,
        "group_delay_compensation_samples": (len(coefficients) - 1) // 2,
        "noise_psd_w_per_hz": noise_psd_w_per_hz,
        "total_gain_db": 10.0 * np.log10(total_gain_linear),
        "channel_total_gain_db": 10.0 * np.log10(total_gain_linear) + amplitude_db,
        "amplitude_error_db": amplitude_db.copy(),
        "phase_error_deg": phase_deg.copy(),
        "noise_figure_db": 10.0 * np.log10(cascade_noise_factor),
        "input_noise_power_w": input_noise_power_w,
        "input_noise_rms_v": noise_rms_v,
        "adc_lsb_v": lsb_v,
        "saturation_fraction": np.mean(saturated, axis=1),
        "input_rms_v": np.sqrt(np.mean(np.abs(x_antenna_v) ** 2, axis=1)),
        "adc_input_rms_v": np.sqrt(np.mean(np.abs(before_limit) ** 2, axis=1)),
        "adc_output_rms_code": np.sqrt(np.mean(np.abs(adc_codes) ** 2, axis=1)),
        "adc_bits": cfg.adc_bits,
        "adc_vpp": cfg.adc_vpp,
    }
    return adc_codes, info
