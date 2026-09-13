"""End-to-end RF → DFE → combining → acquisition → tracking chain."""

import numpy as np

from antenna import build_antenna_model
from acquisition import acquire
from config import SimulationConfig
from dfe import process_dfe
from diversity import combine_blind_null, combine_known, combine_lcmv
from gnss import generate_scenario
from metrics import compute_metrics
from models import (
    BranchResult,
    CombiningInfo,
    ScenarioTruth,
    SimulationResults,
    TrackingResult,
)
from multipath import detect_multipath, empty_detection, evaluate_multipath
from presets import default_config
from rf import process_rf
from tracking import track


def _run_branch(
    mode: str,
    dfe_channels: np.ndarray,
    truth: ScenarioTruth,
    cfg: SimulationConfig,
    fs_out_hz: float,
) -> BranchResult:
    if mode == "blind_null":
        combined, combining = combine_blind_null(
            dfe_channels,
            cfg.diversity.blind_null,
        )
    else:
        combined, combining = combine_known(
            dfe_channels,
            truth["desired_steering_assumed"],
            mode,
            cfg.diversity.diagonal_loading_factor,
        )

    return _finish_branch(
        combined,
        combining,
        truth,
        cfg,
        fs_out_hz,
        dfe_channels.shape[1],
    )


def _finish_branch(
    combined: np.ndarray,
    combining: CombiningInfo,
    truth: ScenarioTruth,
    cfg: SimulationConfig,
    fs_out_hz: float,
    covariance_snapshots: int,
) -> BranchResult:

    weights = combining["weights"]
    interferer_gains = [
        float(abs(np.vdot(weights, steering)))
        for steering in truth["interferer_steering_effective"]
    ]
    multipath_gains = [
        float(abs(np.vdot(weights, path["steering_true"])))
        for path in truth["multipath_paths"]
    ]
    combining.update(
        {
            "target_direction_gain": float(
                abs(np.vdot(weights, truth["desired_steering_true"]))
            ),
            "interferer_direction_gains": interferer_gains,
            "worst_interferer_gain": max(interferer_gains, default=0.0),
            "multipath_direction_gains": multipath_gains,
            "worst_multipath_gain": max(multipath_gains, default=0.0),
            "covariance_snapshots": covariance_snapshots,
            "weights_are_fixed": True,
        }
    )
    acquisition = acquire(
        combined,
        fs_out_hz,
        cfg.signal.prn,
        cfg.signal.code_rate_hz,
        cfg.acquisition,
    )
    if acquisition["success"]:
        tracking = track(
            combined,
            fs_out_hz,
            cfg.signal.prn,
            acquisition,
            cfg.tracking,
            cfg.signal.carrier_hz,
            cfg.signal.code_rate_hz,
        )
    else:
        tracking: TrackingResult = {
            "success": False,
            "status": "acquisition_failed",
            "prompt": np.array([], dtype=complex),
        }
    return {
        "combined": combined,
        "combining": combining,
        "acquisition": acquisition,
        "tracking": tracking,
    }


def run_end_to_end(cfg: SimulationConfig | None = None) -> SimulationResults:
    """Run the complete chain and return every stage needed for analysis."""

    cfg = default_config() if cfg is None else cfg
    scene_input_v, truth = generate_scenario(cfg)
    adc_codes, rf_info = process_rf(
        scene_input_v,
        cfg.rf,
        cfg.reference_impedance_ohm,
        cfg.random_seed,
    )
    dfe_channels, dfe_info = process_dfe(
        adc_codes,
        cfg.dfe,
        cfg.signal.fs_rf_hz,
        cfg.signal.if_hz,
    )
    branches = {
        mode: _run_branch(
            mode,
            dfe_channels,
            truth,
            cfg,
            dfe_info["fs_out_hz"],
        )
        for mode in cfg.diversity.modes
    }
    successful = [
        (mode, branch["acquisition"])
        for mode, branch in branches.items()
        if branch["acquisition"]["success"]
    ]
    pilot = (
        max(successful, key=lambda item: item[1]["metric"])
        if successful
        else None
    )
    if not cfg.multipath_detection.enable:
        detection = empty_detection("detection_disabled")
    elif pilot is None:
        detection = empty_detection("no_pilot_acquisition")
    else:
        pilot_mode, pilot_acquisition = pilot
        antenna_model = build_antenna_model(
            cfg.phone,
            cfg.signal.carrier_hz,
        )
        detection = detect_multipath(
            dfe_channels,
            dfe_info["fs_out_hz"],
            cfg.signal.prn,
            cfg.signal.code_rate_hz,
            pilot_acquisition,
            cfg.acquisition,
            antenna_model,
            cfg.signal.assumed_direction,
            cfg.signal.assumed_polarization,
            cfg.multipath_detection,
            cfg.multipath_suppression.max_steering_correlation,
        )
        detection["pilot_mode"] = pilot_mode

    if cfg.multipath_suppression.enable:
        estimated_multipath = [
            candidate["steering_estimated"]
            for candidate in detection["candidates"]
            if candidate["spatial_status"] == "resolvable"
        ]
        combined, combining = combine_lcmv(
            dfe_channels,
            truth["desired_steering_assumed"],
            estimated_multipath,
            cfg.multipath_suppression.diagonal_loading_factor,
            cfg.multipath_suppression.max_steering_correlation,
        )
        branches["multipath_lcmv"] = _finish_branch(
            combined,
            combining,
            truth,
            cfg,
            dfe_info["fs_out_hz"],
            dfe_channels.shape[1],
        )
    multipath_evaluation = evaluate_multipath(
        detection,
        truth["multipath_paths"],
        max(0.5, 2.0 * cfg.multipath_detection.delay_step_chips),
        2.0
        * max(
            cfg.multipath_detection.azimuth_step_deg,
            cfg.multipath_detection.elevation_step_deg,
        ),
    )
    results: SimulationResults = {
        "config": cfg,
        "truth": truth,
        "scene_input_v": scene_input_v,
        "adc_codes": adc_codes,
        "rf_info": rf_info,
        "dfe_channels": dfe_channels,
        "dfe_info": dfe_info,
        "branches": branches,
        "multipath_detection": detection,
        "multipath_evaluation": multipath_evaluation,
        "metrics": [],
    }
    results["metrics"] = compute_metrics(results)
    return results
