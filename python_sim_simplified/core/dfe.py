"""共用 NCO 下变频、FIR 抽取与 DFE 输出量化。"""

import numpy as np
from scipy.signal import lfilter

from .config import DFEConfig
from .models import DFEInfo


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def design_decimator(fs_in_hz, decimation, cutoff_hz, num_taps):
    """设计低通抽取 FIR，返回抽头和采样率/群延时信息（Hz、输入采样点）。"""
    delay = (num_taps - 1) // 2
    m = np.arange(-delay, delay + 1)
    h = (2.0 * cutoff_hz / fs_in_hz) * np.sinc(
        2.0 * cutoff_hz / fs_in_hz * m
    )
    h *= np.hamming(num_taps)
    h /= h.sum()
    return h, {
        "fs_in_hz": fs_in_hz,
        "fs_out_hz": fs_in_hz / decimation,
        "decimation": decimation,
        "cutoff_hz": cutoff_hz,
        "num_taps": num_taps,
        "group_delay_input_samples": delay,
    }


def linear_dfe(adc_codes: np.ndarray, info: DFEInfo) -> np.ndarray:
    """按本次 NCO/FIR/抽取处理 [通道, 输入采样]，返回线性 DFE 复数序列。

    可输入真实 ADC 码值或线性未量化码值；先按本次完整输入的采样序号混频，
    FIR 后从群延时处开始抽取，此函数不做 DFE 输出量化。
    """
    n = np.arange(adc_codes.shape[1])
    nco = np.exp(-1j * 2.0 * np.pi * info["if_hz"] * n / info["fs_in_hz"])
    mixed = adc_codes * nco[None, :]
    filtered = lfilter(info["filter_coefficients"], [1.0], mixed, axis=1)
    return filtered[:, info["group_delay_input_samples"] :: info["decimation"]]


def process_dfe(
    adc_codes: np.ndarray,
    cfg: DFEConfig,
    fs_in_hz: float,
    signal_if_hz: float,
) -> tuple[np.ndarray, DFEInfo]:
    """处理 [通道, ADC采样] 复数码值，返回抽取/量化后的序列与采样信息。"""
    nco_frequency_hz = (
        signal_if_hz
        if cfg.nco_frequency_hz is None
        else cfg.nco_frequency_hz
    )
    h, raw_info = design_decimator(
        fs_in_hz, cfg.decimation, cfg.cutoff_hz, cfg.num_taps
    )
    output = linear_dfe(adc_codes, {
        "fs_in_hz": fs_in_hz,
        "if_hz": nco_frequency_hz,
        "filter_coefficients": h,
        "group_delay_input_samples": raw_info["group_delay_input_samples"],
        "decimation": cfg.decimation,
    })

    output_scale = 1
    if cfg.output_bits > 0:
        output_scale = 2**cfg.output_binary_shift
        minimum = -(2 ** (cfg.output_bits - 1))
        maximum = 2 ** (cfg.output_bits - 1) - 1
        i_code = np.clip(
            _round_half_away_from_zero(output.real / output_scale),
            minimum,
            maximum,
        )
        q_code = np.clip(
            _round_half_away_from_zero(output.imag / output_scale),
            minimum,
            maximum,
        )
        output = (i_code + 1j * q_code).astype(complex)

    info: DFEInfo = {
        "fs_in_hz": raw_info["fs_in_hz"],
        "fs_out_hz": raw_info["fs_out_hz"],
        "decimation": raw_info["decimation"],
        "cutoff_hz": raw_info["cutoff_hz"],
        "num_taps": raw_info["num_taps"],
        "group_delay_input_samples": raw_info["group_delay_input_samples"],
        "if_hz": nco_frequency_hz,
        "output_bits": cfg.output_bits,
        "output_scale": output_scale,
        "output_length": output.shape[1],
        "filter_coefficients": h,
    }
    return output, info


def process_dfe_blocks(blocks, cfg, fs_in_hz, signal_if_hz):
    """逐块处理连续的 [通道, ADC采样]，生成与整段 DFE 等价的输出块。

    NCO 相位、FIR 状态、群延时和抽取相位跨块连续；录波中断需开启新流。
    """
    h, info = design_decimator(fs_in_hz, cfg.decimation, cfg.cutoff_hz, cfg.num_taps)
    frequency = signal_if_hz if cfg.nco_frequency_hz is None else cfg.nco_frequency_hz
    offset = 0
    state = None
    channels = None
    for block in blocks:
        block = np.asarray(block)
        if block.ndim != 2 or block.shape[1] == 0:
            raise ValueError('DFE blocks must be nonempty channel-by-sample arrays')
        if state is None:
            channels = block.shape[0]
            state = np.zeros((channels, len(h) - 1), complex)
        if block.shape[0] != channels:
            raise ValueError('DFE channel count changed')
        # 块间延续 NCO 相位和 FIR 状态，并按全局采样位置确定抽取相位。
        n = np.arange(offset, offset + block.shape[1])
        mixed = block * np.exp(-1j * 2 * np.pi * frequency * n / fs_in_hz)[None, :]
        filtered, state = lfilter(h, [1.], mixed, axis=1, zi=state)
        first = max(offset, info['group_delay_input_samples'])
        first += (info['group_delay_input_samples'] - first) % cfg.decimation
        output = filtered[:, first - offset::cfg.decimation]
        offset += block.shape[1]
        if cfg.output_bits > 0:
            scale = 2**cfg.output_binary_shift
            minimum = -(2**(cfg.output_bits - 1))
            maximum = -minimum - 1
            output = (np.clip(_round_half_away_from_zero(output.real / scale), minimum, maximum)
                      + 1j * np.clip(_round_half_away_from_zero(output.imag / scale), minimum, maximum))
        if output.shape[1]:
            yield output
