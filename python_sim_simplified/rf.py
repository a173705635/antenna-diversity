"""Equivalent RF cascade, thermal noise, limiter, and complex ADC."""

import numpy as np

from config import RFConfig
from models import RFInfo


BOLTZMANN_J_PER_K = 1.380649e-23


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def process_rf(
    x_antenna_v: np.ndarray,
    cfg: RFConfig,
    reference_impedance_ohm: float,
    seed: int,
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
    input_noise_power_w = (
        BOLTZMANN_J_PER_K
        * cfg.temperature_k
        * cfg.noise_bandwidth_hz
        * cascade_noise_factor
    )
    noise_rms_v = np.sqrt(input_noise_power_w * reference_impedance_ohm)
    rng = np.random.Generator(np.random.MT19937(seed))
    noise = noise_rms_v / np.sqrt(2.0) * (
        rng.standard_normal(x_antenna_v.shape)
        + 1j * rng.standard_normal(x_antenna_v.shape)
    )
    before_limit = (x_antenna_v + noise) * np.sqrt(total_gain_linear)
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
