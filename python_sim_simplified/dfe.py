"""Common-NCO DDC, FIR decimation, and DFE output quantization."""

import numpy as np
from scipy.signal import lfilter


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def design_decimator(fs_in_hz, decimation, cutoff_hz, num_taps):
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


def process_dfe(adc_codes, cfg):
    h, info = design_decimator(
        cfg.fs_in_hz, cfg.decimation, cfg.cutoff_hz, cfg.num_taps
    )
    n = np.arange(adc_codes.shape[1])
    nco = np.exp(-1j * 2.0 * np.pi * cfg.if_hz * n / cfg.fs_in_hz)
    mixed = adc_codes * nco[None, :]
    filtered = lfilter(h, [1.0], mixed, axis=1)
    output = filtered[:, info["group_delay_input_samples"] :: cfg.decimation]

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

    info.update(
        {
            "if_hz": cfg.if_hz,
            "output_bits": cfg.output_bits,
            "output_scale": output_scale,
            "output_length": output.shape[1],
            "filter_coefficients": h,
        }
    )
    return output, info
