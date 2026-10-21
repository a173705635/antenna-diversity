"""独立接收噪声流；参考面和传播由对应物理阶段确定。"""

import numpy as np


BOLTZMANN_J_PER_K = 1.380649e-23
NOISE_REFERENCE_TEMPERATURE_K = 290.0


def generate_complex_noise(
    n_samples: int,
    temperature_k: float,
    reference_impedance_ohm: float,
    fs_hz: float,
    channel_ids: tuple[str, ...],
    seed: int,
    stage_name: str,
) -> np.ndarray:
    """按阶段、物理通道及 I/Q 派生独立流，返回未滤波 [通道, RF采样] 伏特。"""
    component_std_v = np.sqrt(
        BOLTZMANN_J_PER_K * temperature_k * reference_impedance_ohm * fs_hz / 2.0
    )
    stage_identity = stage_name.encode("utf-8")
    output = np.empty((len(channel_ids), n_samples), dtype=complex)
    for row, channel_id in enumerate(channel_ids):
        channel_identity = channel_id.encode("utf-8")
        components = []
        for component_id in (0, 1):
            sequence = np.random.SeedSequence([
                seed,
                len(stage_identity), *stage_identity,
                len(channel_identity), *channel_identity,
                component_id,
            ])
            rng = np.random.Generator(np.random.MT19937(sequence))
            components.append(rng.standard_normal(n_samples))
        output[row] = component_std_v * (components[0] + 1j * components[1])
    return output


def rf_input_noise_budget(
    center_frequency_hz: float, antenna_info: dict, feedline_info: dict,
    rf_added_temperature_k: float,
) -> dict:
    """由各参考面的独立噪声求RF输入载频中心点温度谱密度矩阵。

    此矩阵单位K，乘k得到W/Hz。馈线时延使宽带跨通道谱随频率变化，
    因而该中心频点矩阵不能当作整带零滞后采样协方差。
    """
    transfer = antenna_info["transfer_matrix"]
    coupling = antenna_info["coupling_matrix"]
    feedline = np.diag(feedline_info["center_response"])
    antenna_temperature = (
        antenna_info["input_temperature_k"] * transfer @ transfer.conj().T
        + coupling @ np.diag(antenna_info["element_loss_temperature_k"])
        @ coupling.conj().T
    )
    antenna_at_rf = feedline @ antenna_temperature @ feedline.conj().T
    total = (
        antenna_at_rf
        + np.diag(feedline_info["added_noise_temperature_k"])
        + np.eye(transfer.shape[0]) * rf_added_temperature_k
    )
    return {
        "reference_plane": "RF input; center carrier frequency",
        "center_frequency_hz": center_frequency_hz,
        "antenna_at_rf_input_temperature_k": np.real(np.diag(antenna_at_rf)),
        "antenna_at_rf_input_temperature_matrix_k": antenna_at_rf,
        "feedline_added_temperature_k": feedline_info["added_noise_temperature_k"],
        "rf_added_temperature_k": rf_added_temperature_k,
        "total_rf_input_temperature_k": np.real(np.diag(total)),
        "total_rf_input_temperature_matrix_k": total,
        "feedline_loss_db": feedline_info["loss_db"],
    }
