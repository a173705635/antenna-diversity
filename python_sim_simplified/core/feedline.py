"""馈线无源衰减和其物理温度对应的独立热噪声。"""

import numpy as np

from .config import FeedlineConfig
from .noise import BOLTZMANN_J_PER_K, generate_complex_noise


def process_feedline(
    antenna_output_v: np.ndarray,
    cfg: FeedlineConfig,
    reference_impedance_ohm: float,
    fs_hz: float,
    channel_ids: tuple[str, ...],
    seed: int,
    *,
    retain_analysis_flag: bool = False,
) -> tuple[np.ndarray, dict]:
    """整体衰减天线端电压，再添加馈线自身的未滤波端口噪声。"""
    power_transmission = 10.0 ** (-cfg.loss_db / 10.0)
    voltage_transmission = 10.0 ** (-cfg.loss_db / 20.0)
    added_noise_temperature_k = cfg.temperature_k * (1.0 - power_transmission)
    added_noise_v = generate_complex_noise(
        antenna_output_v.shape[1], added_noise_temperature_k,
        reference_impedance_ohm, fs_hz, channel_ids, seed, "feedline",
    )
    info = {
        "loss_db": cfg.loss_db,
        "temperature_k": cfg.temperature_k,
        "power_transmission": power_transmission,
        "voltage_transmission": voltage_transmission,
        "added_noise_temperature_k": added_noise_temperature_k,
        "added_noise_psd_w_per_hz": BOLTZMANN_J_PER_K * added_noise_temperature_k,
    }
    if retain_analysis_flag:
        info["added_noise_v"] = added_noise_v
    return antenna_output_v * voltage_transmission + added_noise_v, info
