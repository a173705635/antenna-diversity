"""Common-NCO DDC, FIR filtering, decimation, and requantization."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter

from antenna_diversity.config import DFEConfig
from antenna_diversity.numeric import matlab_round

from .decimator import design_decimator


def process_channels(
    adc_codes: ArrayLike,
    cfg: DFEConfig,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Downconvert and decimate all antenna channels coherently."""

    x = np.asarray(adc_codes)
    if x.ndim != 2 or x.size == 0:
        raise ValueError("adc_codes must be a nonempty channels-by-samples matrix")

    h, info = design_decimator(
        cfg.fs_in_hz,
        cfg.decimation,
        cfg.cutoff_hz,
        cfg.num_taps,
    )
    n_samples = x.shape[1]
    n = np.arange(n_samples, dtype=np.float64)
    nco = np.exp(-1j * 2.0 * np.pi * cfg.if_hz * n / cfg.fs_in_hz)
    mixed = x.astype(np.complex128, copy=False) * nco[np.newaxis, :]
    filtered = lfilter(h, [1.0], mixed, axis=1)

    group_delay = info["group_delay_input_samples"]
    if group_delay >= n_samples:
        raise ValueError("input is shorter than the FIR group delay")
    aligned = filtered[:, group_delay:]
    x_out = aligned[:, :: cfg.decimation]

    if cfg.output_bits > 0:
        if (
            not isinstance(cfg.output_bits, (int, np.integer))
            or cfg.output_bits < 2
            or not isinstance(cfg.output_binary_shift, (int, np.integer))
            or cfg.output_binary_shift < 0
        ):
            raise ValueError("output bits and binary shift are invalid")
        output_scale = 2**cfg.output_binary_shift
        min_code = -(2 ** (cfg.output_bits - 1))
        max_code = 2 ** (cfg.output_bits - 1) - 1
        i_code = np.clip(
            matlab_round(x_out.real / output_scale), min_code, max_code
        )
        q_code = np.clip(
            matlab_round(x_out.imag / output_scale), min_code, max_code
        )
        x_out = (i_code + 1j * q_code).astype(np.complex128, copy=False)
    else:
        output_scale = 1
        x_out = x_out.astype(np.complex128, copy=False)

    info.update(
        {
            "if_hz": cfg.if_hz,
            "output_bits": cfg.output_bits,
            "output_scale": output_scale,
            "output_length": x_out.shape[1],
            "filter_coefficients": h,
        }
    )
    return x_out, info
