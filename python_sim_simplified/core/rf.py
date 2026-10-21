"""等效 RF 级联附加噪声、统一前端滤波、限幅与复数 ADC。"""

import numpy as np
from scipy.optimize import brentq
from scipy.signal import firwin, lfilter

from .config import RFConfig
from .models import RFInfo
from .noise import (
    BOLTZMANN_J_PER_K, NOISE_REFERENCE_TEMPERATURE_K, generate_complex_noise,
)


def design_rf_filter(fs_hz: float, bandwidth_hz: float, num_taps: int) -> np.ndarray:
    """设计单位直流增益低通，返回奇数抽头 FIR；bandwidth_hz 是双边等效噪声带宽。"""
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
    """滤波 [通道, RF采样] 复数 IF 电压并补偿群延时；边界补零。"""
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


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def process_rf(
    x_rf_input_v: np.ndarray,
    cfg: RFConfig,
    reference_impedance_ohm: float,
    seed: int,
    *,
    fs_hz: float,
    if_hz: float,
    channel_ids: tuple[str, ...],
    retain_analysis_flag: bool = False,
) -> tuple[np.ndarray, RFInfo]:
    """处理馈线后的 RF 输入 [通道, RF采样] 伏特，返回 ADC 码值与参数。

    本级只添加 T0*(级联噪声因子-1) 的未滤波噪声；全部输入统一滤波一次，
    再施加总增益、通道误差、限幅及量化。T0=290 K 是固定噪声参考温度。
    """
    if x_rf_input_v.ndim != 2 or x_rf_input_v.shape[0] == 0:
        raise ValueError('RF input must have shape (channels, samples)')
    amplitude_db = np.broadcast_to(np.asarray(cfg.amplitude_error_db, dtype=float),
                                   (x_rf_input_v.shape[0],))
    phase_deg = np.broadcast_to(np.asarray(cfg.phase_error_deg, dtype=float),
                               (x_rf_input_v.shape[0],))
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
    added_noise_temperature_k = float(
        NOISE_REFERENCE_TEMPERATURE_K * (cascade_noise_factor - 1.0)
    )
    added_noise_psd_w_per_hz = BOLTZMANN_J_PER_K * added_noise_temperature_k
    added_noise_power_w = added_noise_psd_w_per_hz * measured_bandwidth_hz
    added_noise_rms_v = np.sqrt(added_noise_power_w * reference_impedance_ohm)
    added_noise_v = generate_complex_noise(
        x_rf_input_v.shape[1], added_noise_temperature_k,
        reference_impedance_ohm, fs_hz, channel_ids, seed, "rf",
    )
    filtered_input = filter_rf_input(x_rf_input_v + added_noise_v, coefficients, fs_hz, if_hz)
    before_limit = filtered_input * np.sqrt(total_gain_linear)
    retained_added_noise_v = added_noise_v if retain_analysis_flag else None
    del filtered_input, added_noise_v
    if np.any(amplitude_db != 0.) or np.any(phase_deg != 0.):
        before_limit *= error_response[:, None]
    limit_v = cfg.adc_vpp / 2.0
    saturated_flag = (np.abs(before_limit.real) > limit_v) | (
        np.abs(before_limit.imag) > limit_v
    )
    if cfg.enable_soft_limiting_flag:
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
        "noise_reference_temperature_k": NOISE_REFERENCE_TEMPERATURE_K,
        "added_noise_temperature_k": added_noise_temperature_k,
        "added_noise_psd_w_per_hz": added_noise_psd_w_per_hz,
        "total_gain_db": 10.0 * np.log10(total_gain_linear),
        "channel_total_gain_db": 10.0 * np.log10(total_gain_linear) + amplitude_db,
        "amplitude_error_db": amplitude_db.copy(),
        "phase_error_deg": phase_deg.copy(),
        "noise_figure_db": 10.0 * np.log10(cascade_noise_factor),
        "added_noise_power_w": added_noise_power_w,
        "added_noise_rms_v": added_noise_rms_v,
        "adc_lsb_v": lsb_v,
        "saturation_fraction": np.mean(saturated_flag, axis=1),
        "input_rms_v": np.sqrt(np.mean(np.abs(x_rf_input_v) ** 2, axis=1)),
        "adc_input_rms_v": np.sqrt(np.mean(np.abs(before_limit) ** 2, axis=1)),
        "adc_output_rms_code": np.sqrt(np.mean(np.abs(adc_codes) ** 2, axis=1)),
        "adc_bits": cfg.adc_bits,
        "adc_vpp": cfg.adc_vpp,
    }
    if retain_analysis_flag:
        info["added_noise_v"] = retained_added_noise_v
    return adc_codes, info
