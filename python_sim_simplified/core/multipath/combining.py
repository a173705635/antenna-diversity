"""Multipath constrained combining."""

import numpy as np

from ..diversity import combine_known
from ..models import CombiningInfo


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
