"""公共单通道、等功率独立噪声下的 MRC 和 MVDR 合并。"""

import numpy as np

from .models import CombiningInfo


def combine_known(
    channels: np.ndarray,
    desired: np.ndarray,
    mode: str,
    diagonal_loading_factor: float,
) -> tuple[np.ndarray, CombiningInfo]:
    """将 [通道, 采样] 复数码值按名义导向合并，返回时间序列和求权信息。

    MRC 假设各路噪声独立且功率相等，w=a/(aᴴa) 使名义目标响应为 1。
    """
    covariance = channels @ channels.conj().T / channels.shape[1]
    loading = (
        diagonal_loading_factor
        * np.trace(covariance).real
        / channels.shape[0]
    )
    loaded = covariance + loading * np.eye(channels.shape[0])
    if mode == "single":
        weights = np.zeros(channels.shape[0], dtype=complex)
        weights[0] = 1.0 / np.conj(desired[0])
    elif mode == "mrc":
        weights = desired / np.vdot(desired, desired)
    elif mode == "mvdr":
        solved = np.linalg.solve(loaded, desired)
        weights = solved / np.vdot(desired, solved)
    else:
        raise ValueError(f"Unsupported known combining mode: {mode}")

    combined = weights.conj() @ channels
    info: CombiningInfo = {
        "mode": mode,
        "weights": weights,
        "constraint_steering": desired,
        "desired_response": np.vdot(weights, desired),
        "sample_covariance": covariance,
        "loaded_covariance": loaded,
        "diagonal_loading": loading,
        "weight_norm": np.linalg.norm(weights),
        "condition_number": np.linalg.cond(loaded),
        "weights_are_fixed_flag": True,
    }
    return combined, info
