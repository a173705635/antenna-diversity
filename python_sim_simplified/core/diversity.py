"""Shared single-channel, EGC and MVDR combining primitives."""

import numpy as np

from .models import CombiningInfo


def combine_known(
    channels: np.ndarray,
    desired: np.ndarray,
    mode: str,
    diagonal_loading_factor: float,
) -> tuple[np.ndarray, CombiningInfo]:
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
    elif mode == "egc":
        weights = desired / np.vdot(desired, desired)
    elif mode == "mvdr":
        solved = np.linalg.solve(loaded, desired)
        weights = solved / np.vdot(desired, solved)

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
        "weights_are_fixed": True,
    }
    return combined, info
