"""End-to-end RF → DFE → combining → acquisition → tracking chain."""

import numpy as np

from acquisition import acquire
from config import SimulationConfig
from dfe import process_dfe
from diversity import combine_blind_null, combine_known
from gnss import generate_scenario
from metrics import compute_metrics
from models import BranchResult, ScenarioTruth, SimulationResults, TrackingResult
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

    weights = combining["weights"]
    interferer_gains = [
        float(abs(np.vdot(weights, steering)))
        for steering in truth["interferer_steering_effective"]
    ]
    combining.update(
        {
            "target_direction_gain": float(
                abs(np.vdot(weights, truth["desired_steering_true"]))
            ),
            "interferer_direction_gains": interferer_gains,
            "worst_interferer_gain": max(interferer_gains, default=0.0),
            "covariance_snapshots": dfe_channels.shape[1],
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
    results: SimulationResults = {
        "config": cfg,
        "truth": truth,
        "scene_input_v": scene_input_v,
        "adc_codes": adc_codes,
        "rf_info": rf_info,
        "dfe_channels": dfe_channels,
        "dfe_info": dfe_info,
        "branches": branches,
        "metrics": [],
    }
    results["metrics"] = compute_metrics(results)
    return results
