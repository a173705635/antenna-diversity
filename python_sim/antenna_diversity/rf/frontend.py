"""Equivalent RF gain, thermal noise, limiting, and complex ADC."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from antenna_diversity.config import RFConfig
from antenna_diversity.numeric import matlab_round

from .cascade import cascade_metrics


BOLTZMANN_J_PER_K = 1.380649e-23


def process_frontend(
    x_antenna_v: ArrayLike,
    cfg: RFConfig,
    random_seed: int,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Apply input-referred noise, voltage gain, rail limiting, and I/Q ADC."""

    x = np.asarray(x_antenna_v, dtype=np.complex128)
    if x.ndim != 2 or x.size == 0:
        raise ValueError("x_antenna_v must be a nonempty antennas-by-samples matrix")
    if (
        cfg.temperature_k < 0
        or cfg.noise_bandwidth_hz <= 0
        or cfg.resistance_ohm <= 0
        or not isinstance(cfg.adc_bits, (int, np.integer))
        or cfg.adc_bits < 2
        or cfg.adc_vpp <= 0
    ):
        raise ValueError("RF temperature, bandwidth, resistance, or ADC setup is invalid")

    cascade = cascade_metrics(cfg.stages)
    noise_power_input_w = (
        BOLTZMANN_J_PER_K
        * cfg.temperature_k
        * cfg.noise_bandwidth_hz
        * cascade["noise_factor"]
    )
    noise_rms_v = np.sqrt(noise_power_input_w * cfg.resistance_ohm)
    rng = np.random.Generator(np.random.MT19937(int(random_seed)))
    noise = noise_rms_v / np.sqrt(2.0) * (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    )

    voltage_gain = np.sqrt(cascade["total_gain_linear"])
    adc_input_before_limit = (x + noise) * voltage_gain
    component_limit_v = cfg.adc_vpp / 2.0
    saturated = (
        np.abs(adc_input_before_limit.real) > component_limit_v
    ) | (np.abs(adc_input_before_limit.imag) > component_limit_v)

    if cfg.enable_soft_limiting:
        adc_input_v = component_limit_v * np.tanh(
            adc_input_before_limit.real / component_limit_v
        ) + 1j * component_limit_v * np.tanh(
            adc_input_before_limit.imag / component_limit_v
        )
    else:
        adc_input_v = np.clip(
            adc_input_before_limit.real, -component_limit_v, component_limit_v
        ) + 1j * np.clip(
            adc_input_before_limit.imag, -component_limit_v, component_limit_v
        )

    lsb_v = cfg.adc_vpp / 2**cfg.adc_bits
    min_code = -(2 ** (cfg.adc_bits - 1))
    max_code = 2 ** (cfg.adc_bits - 1) - 1
    i_code = np.clip(matlab_round(adc_input_v.real / lsb_v), min_code, max_code)
    q_code = np.clip(matlab_round(adc_input_v.imag / lsb_v), min_code, max_code)
    adc_codes = (i_code + 1j * q_code).astype(np.complex128, copy=False)

    info: dict[str, Any] = {
        "total_gain_db": cascade["total_gain_db"],
        "noise_figure_db": cascade["noise_figure_db"],
        "input_noise_power_w": noise_power_input_w,
        "input_noise_rms_v": noise_rms_v,
        "adc_lsb_v": lsb_v,
        "saturation_fraction": np.mean(saturated, axis=1),
        "input_rms_v": np.sqrt(np.mean(np.abs(x) ** 2, axis=1)),
        "adc_input_rms_v": np.sqrt(
            np.mean(np.abs(adc_input_before_limit) ** 2, axis=1)
        ),
        "adc_output_rms_code": np.sqrt(
            np.mean(np.abs(adc_codes) ** 2, axis=1)
        ),
        "adc_bits": cfg.adc_bits,
        "adc_vpp": cfg.adc_vpp,
    }
    return adc_codes, info
