"""Blind-null combining for interferers."""

import numpy as np

from ..config import BlindNullConfig
from ..models import CombiningInfo


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
