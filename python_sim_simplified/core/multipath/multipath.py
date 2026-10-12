"""Code-delay and spatial multipath detection from DFE array samples."""

import numpy as np
from scipy.optimize import minimize_scalar
from ..code_replica import bandlimited_code_replica, _sample_code

from ..antenna import AntennaModel, antenna_response, direction_vector
from ..config import (
    AcquisitionConfig,
    Direction,
    MultipathDetectionConfig,
    WavePolarization,
)
from ..gnss import ca_code
from ..models import (
    AcquisitionResult,
    MultipathCandidate,
    MultipathDetectionResult,
    MultipathEvaluation,
    MultipathTruth,
)


def refine_single_path_delays(
    channels, fs_hz, prn, code_rate_hz, carrier_hz, acquisition,
    los_steering, echo_steering, bandwidth_hz, max_excess_delay_chips,
):
    """Research-only two-direction demixing and continuous relative delay fit.

    Requires one estimated, spatially resolvable echo and an assumed LOS.
    Directions must be supplied by estimation, never scenario truth. This is
    not a multi-path-count estimator and is not enabled in the normal pipeline.
    """
    matrix = np.column_stack([los_steering, echo_steering])
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[-1] <= .05 * singular[0]:
        return {"status": "spatially_unresolvable", "uses_truth": False}
    if not 0 < bandwidth_hz < fs_hz/2 or max_excess_delay_chips <= 0:
        raise ValueError("valid bandwidth and positive delay search range required")
    # Skip the initial finite-record/filter transient; use up to three ms.
    first = round(.001*fs_hz)
    stop = min(channels.shape[1], round(.004*fs_hz))
    if stop-first < round(.001*fs_hz):
        return {"status": "insufficient_samples", "uses_truth": False}
    n = np.arange(first, stop)
    separated = np.linalg.pinv(matrix) @ channels[:,first:stop]
    separated *= np.exp(-2j*np.pi*acquisition["doppler_hz"]*n/fs_hz)
    received_rate = code_rate_hz*(1+acquisition["doppler_hz"]/carrier_hz)
    replica = bandlimited_code_replica(ca_code(prn), received_rate, bandwidth_hz)
    coarse_delay = acquisition["code_phase_samples"]*code_rate_hz/fs_hz
    phase = n*received_rate/fs_hz - coarse_delay

    def fit(samples, low, high):
        def score(offset):
            local = _sample_code(replica, phase-offset)
            return abs(np.sum(samples*local))**2 / np.vdot(local,local).real
        grid = np.linspace(low, high, int(np.ceil((high-low)/.025))+1)
        powers = np.array([score(offset) for offset in grid])
        index = int(np.argmax(powers))
        if powers[index] <= 0 or not np.isfinite(powers[index]):
            return None
        bounds = (grid[max(0,index-1)],grid[min(grid.size-1,index+1)])
        result = minimize_scalar(lambda offset: -score(offset)/powers[index],
                                 bounds=bounds,method="bounded",options={"xatol":1e-8})
        best = max([grid[index],float(result.x)],key=score)
        return best, bool(best <= low+1e-5 or best >= high-1e-5)

    direct = fit(separated[0],-.75,.75)
    reflected = fit(separated[1],-.75,max_excess_delay_chips+.75)
    if direct is None or reflected is None:
        return {"status": "zero_correlation", "uses_truth": False}
    excess = (reflected[0]-direct[0])*code_rate_hz/received_rate
    valid = not (direct[1] or reflected[1]) and 0 <= excess <= max_excess_delay_chips
    return {"status": "refined" if valid else "outside_search_range",
            "los_offset_from_pilot_chips": float(direct[0]),
            "echo_offset_from_pilot_chips": float(reflected[0]),
            "excess_delay_chips": float(excess),
            "spatial_condition_number": float(singular[0]/singular[-1]),
            "uses_truth": False}


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


def _direction_error_deg(first: Direction, second: Direction) -> float:
    cosine = float(
        np.clip(
            np.dot(direction_vector(first), direction_vector(second)),
            -1.0,
            1.0,
        )
    )
    return float(np.rad2deg(np.arccos(cosine)))


def evaluate_multipath(
    detection: MultipathDetectionResult,
    truth_paths: list[MultipathTruth],
    delay_tolerance_chips: float,
    angle_tolerance_deg: float,
) -> MultipathEvaluation:
    possible = []
    for detected_index, candidate in enumerate(detection["candidates"]):
        for truth_index, truth in enumerate(truth_paths):
            delay_error = abs(
                candidate["excess_delay_chips"]
                - truth["excess_delay_chips"]
            )
            direction_error = _direction_error_deg(
                candidate["direction"],
                truth["direction"],
            )
            possible.append(
                (
                    delay_error,
                    direction_error,
                    detected_index,
                    truth_index,
                )
            )
    matched_detections: set[int] = set()
    matched_truth: set[int] = set()
    matches = []
    for delay_error, direction_error, detected_index, truth_index in sorted(possible):
        if detected_index in matched_detections or truth_index in matched_truth:
            continue
        if delay_error > delay_tolerance_chips or direction_error > angle_tolerance_deg:
            continue
        matched_detections.add(detected_index)
        matched_truth.add(truth_index)
        matches.append(
            {
                "detected_index": detected_index,
                "truth_index": truth_index,
                "delay_error_chips": float(delay_error),
                "direction_error_deg": float(direction_error),
            }
        )
    return {
        "detected_count": len(matches),
        "missed_count": len(truth_paths) - len(matches),
        "false_alarm_count": len(detection["candidates"]) - len(matches),
        "matches": matches,
    }
