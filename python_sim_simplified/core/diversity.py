"""Single, EGC, MVDR, and sample-only Blind Null combining."""

import numpy as np

from .config import BlindNullConfig
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


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        abs(np.vdot(first, second))
        / np.sqrt(np.vdot(first, first).real)
        / np.sqrt(np.vdot(second, second).real)
    )


def combine_lcmv(
    channels: np.ndarray,
    los_steering: np.ndarray,
    multipath_steerings: list[np.ndarray],
    diagonal_loading_factor: float,
    max_steering_correlation: float,
) -> tuple[np.ndarray, CombiningInfo]:
    accepted: list[np.ndarray] = []
    rejected: list[int] = []
    for index, steering in enumerate(multipath_steerings):
        references = [los_steering, *accepted]
        independent = all(
            _normalized_correlation(reference, steering)
            <= max_steering_correlation
            for reference in references
        )
        if independent and len(accepted) < channels.shape[0] - 1:
            accepted.append(steering)
        else:
            rejected.append(index)

    if not accepted:
        combined, info = combine_known(
            channels,
            los_steering,
            "mvdr",
            diagonal_loading_factor,
        )
        info.update(
            {
                "mode": "multipath_lcmv",
                "multipath_constraint_steerings": [],
                "multipath_constraint_responses": [],
                "suppression_status": (
                    "spatially_unresolvable"
                    if multipath_steerings
                    else "no_multipath_detected"
                ),
                "rejected_constraint_indices": rejected,
            }
        )
        return combined, info

    covariance = channels @ channels.conj().T / channels.shape[1]
    loading = (
        diagonal_loading_factor
        * np.trace(covariance).real
        / channels.shape[0]
    )
    loaded = covariance + loading * np.eye(channels.shape[0])
    constraints = np.column_stack([los_steering, *accepted])
    responses = np.zeros(constraints.shape[1], dtype=complex)
    responses[0] = 1.0
    solved = np.linalg.solve(loaded, constraints)
    gram = constraints.conj().T @ solved
    weights = solved @ np.linalg.solve(gram, responses)
    combined = weights.conj() @ channels
    info: CombiningInfo = {
        "mode": "multipath_lcmv",
        "weights": weights,
        "constraint_steering": los_steering,
        "desired_response": np.vdot(weights, los_steering),
        "sample_covariance": covariance,
        "loaded_covariance": loaded,
        "diagonal_loading": loading,
        "weight_norm": np.linalg.norm(weights),
        "condition_number": np.linalg.cond(loaded),
        "weights_are_fixed": True,
        "constraint_matrix": constraints,
        "constraint_response": responses,
        "multipath_constraint_steerings": accepted,
        "multipath_constraint_responses": [
            np.vdot(weights, steering) for steering in accepted
        ],
        "suppression_status": "multipath_constraints_applied",
        "rejected_constraint_indices": rejected,
    }
    return combined, info


def combine_blind_null(
    channels: np.ndarray,
    cfg: BlindNullConfig,
) -> tuple[np.ndarray, CombiningInfo]:
    snapshots = channels.shape[1]
    covariance = channels @ channels.conj().T / snapshots
    covariance = (covariance + covariance.conj().T) / 2.0
    powers = np.real(np.diag(covariance))
    scales = np.sqrt(powers)
    correlation = covariance / np.outer(scales, scales)
    correlation = (correlation + correlation.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    score_db = 10.0 * np.log10(
        eigenvalues[-1] / np.mean(eigenvalues[:-1])
    )
    interference_vector = scales * eigenvectors[:, -1]
    interference_vector /= np.linalg.norm(interference_vector)
    reference = np.eye(channels.shape[0], dtype=complex)[:, cfg.reference_channel]

    info: CombiningInfo = {
        "mode": "blind_null",
        "sample_covariance": covariance,
        "condition_number": np.linalg.cond(covariance),
        "covariance_snapshots": snapshots,
        "channel_powers": powers,
        "normalized_correlation": correlation,
        "correlation_eigenvalues": eigenvalues,
        "detection_score_db": score_db,
        "detection_threshold_db": cfg.detection_threshold_db,
        "estimated_interference_vector": interference_vector,
        "reference_channel": cfg.reference_channel,
        "weights_are_fixed": True,
    }
    if score_db < cfg.detection_threshold_db:
        info.update(
            {
                "weights": reference,
                "weight_norm": 1.0,
                "interference_detected": False,
                "nulling_enabled": False,
                "reference_projection_power": 1.0,
            }
        )
        return channels[cfg.reference_channel].copy(), info

    projector = np.eye(channels.shape[0], dtype=complex) - np.outer(
        interference_vector, interference_vector.conj()
    )
    projected_reference = projector @ reference
    gamma = np.vdot(reference, projected_reference).real
    weights = projected_reference / gamma
    info.update(
        {
            "weights": weights,
            "weight_norm": np.linalg.norm(weights),
            "interference_detected": True,
            "nulling_enabled": True,
            "reference_projection_power": gamma,
            "estimated_interference_response": np.vdot(
                weights, interference_vector
            ),
        }
    )
    return weights.conj() @ channels, info
