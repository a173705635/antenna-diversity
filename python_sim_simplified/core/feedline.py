"""阵元与馈线无源损耗噪声，以及馈线幅相和传播时延。"""

import numpy as np

from .config import FeedlineConfig
from .channel_delay import apply_channel_delays
from .noise import BOLTZMANN_J_PER_K, generate_complex_noise


def feedline_parameters(
    cfg: FeedlineConfig, channel_ids: tuple[str, ...], *,
    use_assumed_state_flag: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """依物理通道顺序取功率损耗dB、载频总相位度和传播时延秒。"""
    losses = cfg.assumed_loss_by_antenna_db if use_assumed_state_flag else cfg.loss_by_antenna_db
    phases = cfg.assumed_phase_at_carrier_deg if use_assumed_state_flag else cfg.phase_at_carrier_deg
    delays = cfg.assumed_delay_ns if use_assumed_state_flag else cfg.delay_ns
    return (
        np.array([losses.get(name, cfg.loss_db) for name in channel_ids]),
        np.array([phases.get(name, 0.0) for name in channel_ids]),
        1e-9 * np.array([delays.get(name, 0.0) for name in channel_ids]),
    )


def feedline_response(
    cfg: FeedlineConfig, channel_ids: tuple[str, ...],
    carrier_hz: float, frequency_hz: float, *,
    use_assumed_state_flag: bool = False,
) -> np.ndarray:
    """B(f)=10^(-L/20) exp(jφcarrier) exp[-j2π(f-fc)τ]。"""
    losses_db, phases_deg, delays_s = feedline_parameters(
        cfg, channel_ids, use_assumed_state_flag=use_assumed_state_flag,
    )
    return (10.0 ** (-losses_db / 20.0)
            * np.exp(1j * np.deg2rad(phases_deg))
            * np.exp(-2j * np.pi * (frequency_hz - carrier_hz) * delays_s))


def apply_feedline_transfer(
    input_v: np.ndarray, cfg: FeedlineConfig, channel_ids: tuple[str, ...],
    fs_hz: float, if_hz: float,
) -> np.ndarray:
    """对宽带IF序列施加B；用carrier_hz=0抵消序列中心IF相位，避免双计载频相位。"""
    losses_db, phases_deg, delays_s = feedline_parameters(cfg, channel_ids)
    delayed_v = apply_channel_delays(input_v, delays_s, fs_hz, 0.0, if_hz)
    return delayed_v * (10.0 ** (-losses_db / 20.0)
                        * np.exp(1j * np.deg2rad(phases_deg)))[:, None]


def process_feedline(
    antenna_output_v: np.ndarray,
    cfg: FeedlineConfig,
    antenna_info: dict,
    reference_impedance_ohm: float,
    fs_hz: float,
    if_hz: float,
    channel_ids: tuple[str, ...],
    seed: int,
    *,
    retain_analysis_flag: bool = False,
) -> tuple[np.ndarray, dict]:
    """在阵元损耗后、互耦前生成损耗噪声，经 C、B 后加入馈线自身噪声。"""
    loss_db, phase_deg, _ = feedline_parameters(cfg, channel_ids)
    power_transmission = 10.0 ** (-loss_db / 10.0)
    voltage_transmission = np.sqrt(power_transmission)
    element_loss_temperature_k = antenna_info["element_loss_temperature_k"]
    # 各阵元自身损耗噪声位于 D 后、C 前；每路独立流可具有不同温度。
    element_loss_noise_v = np.vstack([
        generate_complex_noise(
            antenna_output_v.shape[1], temperature_k, reference_impedance_ohm,
            fs_hz, (name,), seed, "antenna_loss",
        )[0]
        for name, temperature_k in zip(channel_ids, element_loss_temperature_k)
    ])
    coupled_loss_noise_v = antenna_info["coupling_matrix"] @ element_loss_noise_v
    propagated_v = apply_feedline_transfer(
        antenna_output_v + coupled_loss_noise_v, cfg, channel_ids, fs_hz, if_hz,
    )
    added_noise_temperature_k = cfg.temperature_k * (1.0 - power_transmission)
    feedline_noise_v = np.vstack([
        generate_complex_noise(
            antenna_output_v.shape[1], temperature_k, reference_impedance_ohm,
            fs_hz, (name,), seed, "feedline",
        )[0]
        for name, temperature_k in zip(channel_ids, added_noise_temperature_k)
    ])
    info = {
        "loss_db": loss_db,
        "temperature_k": cfg.temperature_k,
        "power_transmission": power_transmission,
        "voltage_transmission": voltage_transmission,
        "center_response": voltage_transmission * np.exp(1j * np.deg2rad(phase_deg)),
        "element_loss_temperature_k": element_loss_temperature_k,
        "added_noise_temperature_k": added_noise_temperature_k,
        "added_noise_psd_w_per_hz": BOLTZMANN_J_PER_K * added_noise_temperature_k,
    }
    if retain_analysis_flag:
        info["propagated_element_loss_noise_v"] = apply_feedline_transfer(
            coupled_loss_noise_v, cfg, channel_ids, fs_hz, if_hz,
        )
        info["added_noise_v"] = feedline_noise_v
    return propagated_v + feedline_noise_v, info
