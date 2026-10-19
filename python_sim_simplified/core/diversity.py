"""单通道实际接收，以及显式对照用的等幅和已知响应合并。"""

import numpy as np

from .models import CombiningInfo


def combine_single(dfe_channels: np.ndarray) -> tuple[np.ndarray, CombiningInfo]:
    """直接选取首个激活通道的 DFE 复采样，不读取目标幅相。"""
    weights = np.zeros(dfe_channels.shape[0], dtype=complex)
    weights[0] = 1.0
    combining_info: CombiningInfo = {
        "mode": "single",
        "weights": weights,
        "weights_are_fixed_flag": True,
    }
    return dfe_channels[0].copy(), combining_info


def combine_egc(
    dfe_channels: np.ndarray,
    phase_estimates_deg: np.ndarray,
) -> tuple[np.ndarray, CombiningInfo]:
    """仅用逐通道相位估计（度）等幅合并，w=exp(jφ)/N，输出为 wᴴX。"""
    weights = np.exp(1j * np.deg2rad(phase_estimates_deg)) / dfe_channels.shape[0]
    combining_info: CombiningInfo = {
        "mode": "egc",
        "weights": weights,
        "weights_are_fixed_flag": True,
        "weight_norm": np.linalg.norm(weights),
    }
    return weights.conj() @ dfe_channels, combining_info


def combine_mvdr(
    dfe_channels: np.ndarray,
    assumed_target_response: np.ndarray,
    diagonal_loading_factor: float,
) -> tuple[np.ndarray, CombiningInfo]:
    """仅供已知响应对照：按名义目标复响应和全段采样协方差求 MVDR 权值。"""
    covariance = dfe_channels @ dfe_channels.conj().T / dfe_channels.shape[1]
    loading = (
        diagonal_loading_factor
        * np.trace(covariance).real
        / dfe_channels.shape[0]
    )
    loaded = covariance + loading * np.eye(dfe_channels.shape[0])
    solved = np.linalg.solve(loaded, assumed_target_response)
    weights = solved / np.vdot(assumed_target_response, solved)

    combined_samples = weights.conj() @ dfe_channels
    combining_info: CombiningInfo = {
        "mode": "mvdr",
        "weights": weights,
        "constraint_steering": assumed_target_response,
        "desired_response": np.vdot(weights, assumed_target_response),
        "sample_covariance": covariance,
        "loaded_covariance": loaded,
        "diagonal_loading": loading,
        "weight_norm": np.linalg.norm(weights),
        "condition_number": np.linalg.cond(loaded),
        "covariance_snapshots": dfe_channels.shape[1],
        "weights_are_fixed_flag": True,
    }
    return combined_samples, combining_info
