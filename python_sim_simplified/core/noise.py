"""独立接收噪声源；各阶段先生成未滤波复白噪声。"""

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


def add_antenna_noise(
    clean_scene_v: np.ndarray,
    temperature_k: float,
    reference_impedance_ohm: float,
    fs_hz: float,
    channel_ids: tuple[str, ...],
    seed: int,
    *,
    retain_analysis_flag: bool = False,
) -> tuple[np.ndarray, dict]:
    """在已汇总的天线端参考面加独立端口噪声；不经过方向响应或互耦。"""
    added_noise_v = generate_complex_noise(
        clean_scene_v.shape[1], temperature_k, reference_impedance_ohm,
        fs_hz, channel_ids, seed, "antenna",
    )
    info = {
        "temperature_k": temperature_k,
        "noise_psd_w_per_hz": BOLTZMANN_J_PER_K * temperature_k,
    }
    if retain_analysis_flag:
        info["added_noise_v"] = added_noise_v
    return clean_scene_v + added_noise_v, info
