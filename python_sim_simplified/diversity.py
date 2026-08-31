"""Single, EGC, MVDR, and sample-only Blind Null combining."""

import numpy as np


def combine_known(channels, desired, mode, diagonal_loading_factor):
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
    return combined, {
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


def combine_blind_null(channels, cfg):
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

    info = {
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
