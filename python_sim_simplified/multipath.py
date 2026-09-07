"""Code-delay and spatial multipath detection from DFE array samples."""

import numpy as np

from antenna import AntennaModel, antenna_response
from config import (
    AcquisitionConfig,
    Direction,
    MultipathDetectionConfig,
    WavePolarization,
)
from gnss import ca_code
from models import AcquisitionResult, MultipathCandidate, MultipathDetectionResult


def empty_detection(
    status: str,
    pilot_mode: str | None = None,
) -> MultipathDetectionResult:
    return {
        "status": status,
        "pilot_mode": pilot_mode,
        "delay_axis_chips": np.array([], dtype=float),
        "correlation_vectors": np.empty((0, 0), dtype=complex),
        "raw_power": np.array([], dtype=float),
        "residual_power": np.array([], dtype=float),
        "threshold_power": 0.0,
        "candidates": [],
    }


def project_out_los(vectors: np.ndarray, los: np.ndarray) -> np.ndarray:
    projector = np.eye(los.size) - np.outer(los, los.conj()) / np.vdot(los, los)
    return projector @ vectors


def _correlation_vectors(
    channels: np.ndarray,
    fs_hz: float,
    prn: int,
    code_rate_hz: float,
    acquisition: AcquisitionResult,
    acquisition_cfg: AcquisitionConfig,
    delay_axis_chips: np.ndarray,
) -> np.ndarray:
    n_samples = min(
        channels.shape[1],
        acquisition["coherent_samples"] * acquisition_cfg.noncoherent_ms,
    )
    n = np.arange(n_samples)
    code = ca_code(prn)
    phase = np.mod(n * code_rate_hz / fs_hz, 1023.0)
    local_code = code[np.floor(phase).astype(int)]
    wipeoff = np.exp(
        -1j * 2.0 * np.pi * acquisition["doppler_hz"] * n / fs_hz
    )
    wiped = channels[:, :n_samples] * wipeoff[None, :]
    vectors = np.empty((channels.shape[0], delay_axis_chips.size), dtype=complex)
    for index, delay_chips in enumerate(delay_axis_chips):
        offset = round(delay_chips * fs_hz / code_rate_hz)
        shift = acquisition["code_phase_samples"] + offset
        aligned_code = np.roll(local_code, shift)
        vectors[:, index] = np.mean(wiped * aligned_code[None, :], axis=1)
    return vectors


def _angle_scan(
    residual_vector: np.ndarray,
    los_steering: np.ndarray,
    antenna_model: AntennaModel,
    polarization: WavePolarization,
    cfg: MultipathDetectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Direction, np.ndarray]:
    azimuths = np.arange(0.0, 360.0, cfg.azimuth_step_deg)
    elevations = np.arange(
        0.0,
        90.0 + 0.5 * cfg.elevation_step_deg,
        cfg.elevation_step_deg,
    )
    scores = np.zeros((elevations.size, azimuths.size))
    steerings: list[list[np.ndarray]] = []
    for row, elevation in enumerate(elevations):
        steering_row = []
        for column, azimuth in enumerate(azimuths):
            direction = Direction(float(azimuth), float(elevation))
            steering, _ = antenna_response(
                antenna_model,
                direction,
                polarization,
                estimated=True,
            )
            projected = project_out_los(steering[:, None], los_steering)[:, 0]
            norm_squared = np.vdot(projected, projected).real
            if norm_squared > np.finfo(float).eps:
                scores[row, column] = (
                    abs(np.vdot(projected, residual_vector)) ** 2
                    / norm_squared
                )
            steering_row.append(steering)
        steerings.append(steering_row)
    best_row, best_column = np.unravel_index(np.argmax(scores), scores.shape)
    direction = Direction(
        float(azimuths[best_column]),
        float(elevations[best_row]),
    )
    return (
        azimuths,
        elevations,
        scores,
        direction,
        steerings[best_row][best_column],
    )


def _local_peaks(power: np.ndarray, threshold: float) -> list[int]:
    peaks = []
    for index, value in enumerate(power):
        left = power[index - 1] if index > 0 else -np.inf
        right = power[index + 1] if index + 1 < power.size else -np.inf
        if value >= threshold and value >= left and value >= right:
            peaks.append(index)
    return sorted(peaks, key=lambda index: power[index], reverse=True)


def detect_multipath(
    channels: np.ndarray,
    fs_hz: float,
    prn: int,
    code_rate_hz: float,
    acquisition: AcquisitionResult,
    acquisition_cfg: AcquisitionConfig,
    antenna_model: AntennaModel,
    los_direction: Direction,
    polarization: WavePolarization,
    cfg: MultipathDetectionConfig,
    max_steering_correlation: float,
) -> MultipathDetectionResult:
    config_values = np.array(
        [
            cfg.max_excess_delay_chips,
            cfg.delay_step_chips,
            cfg.detection_threshold_db,
            cfg.azimuth_step_deg,
            cfg.elevation_step_deg,
            max_steering_correlation,
        ]
    )
    if not np.all(np.isfinite(config_values)) or cfg.delay_step_chips <= 0.0:
        raise ValueError("multipath detection values must be finite and delay step positive")

    delay_axis = np.arange(
        cfg.delay_step_chips,
        cfg.max_excess_delay_chips + 0.5 * cfg.delay_step_chips,
        cfg.delay_step_chips,
    )
    vectors = _correlation_vectors(
        channels,
        fs_hz,
        prn,
        code_rate_hz,
        acquisition,
        acquisition_cfg,
        delay_axis,
    )
    los_steering, _ = antenna_response(
        antenna_model,
        los_direction,
        polarization,
        estimated=True,
    )
    residual_vectors = project_out_los(vectors, los_steering)
    raw_power = np.sum(np.abs(vectors) ** 2, axis=0)
    residual_power = np.sum(np.abs(residual_vectors) ** 2, axis=0)
    threshold = float(
        np.median(residual_power)
        * 10.0 ** (cfg.detection_threshold_db / 10.0)
    )
    numerical_floor = (
        np.finfo(float).eps * max(float(np.max(raw_power)), np.finfo(float).tiny)
    )
    if float(np.max(residual_power)) <= numerical_floor:
        peak_indices = []
    else:
        peak_indices = _local_peaks(residual_power, threshold)

    candidates: list[MultipathCandidate] = []
    for peak_index in peak_indices[: max(0, channels.shape[0] - 1)]:
        azimuths, elevations, angle_power, direction, steering = _angle_scan(
            residual_vectors[:, peak_index],
            los_steering,
            antenna_model,
            polarization,
            cfg,
        )
        correlation = float(
            abs(np.vdot(los_steering, steering))
            / np.sqrt(np.vdot(los_steering, los_steering).real)
            / np.sqrt(np.vdot(steering, steering).real)
        )
        median = max(float(np.median(residual_power)), np.finfo(float).tiny)
        candidates.append(
            {
                "excess_delay_chips": float(delay_axis[peak_index]),
                "residual_power": float(residual_power[peak_index]),
                "detection_score_db": float(
                    10.0 * np.log10(residual_power[peak_index] / median)
                ),
                "direction": direction,
                "steering_estimated": steering,
                "steering_correlation_with_los": correlation,
                "spatial_status": (
                    "spatially_unresolvable"
                    if correlation > max_steering_correlation
                    else "resolvable"
                ),
                "azimuths_deg": azimuths,
                "elevations_deg": elevations,
                "angle_power": angle_power,
            }
        )

    return {
        "status": "detected" if candidates else "no_multipath_detected",
        "pilot_mode": None,
        "delay_axis_chips": delay_axis,
        "correlation_vectors": vectors,
        "raw_power": raw_power,
        "residual_power": residual_power,
        "threshold_power": threshold,
        "candidates": candidates,
    }
