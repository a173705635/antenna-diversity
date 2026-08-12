"""End-to-end GPS L1 C/A antenna-diversity simulation pipeline."""

from __future__ import annotations

import numpy as np

from antenna_diversity.analysis.metrics import compute_metrics
from antenna_diversity.config import SimulationConfig, default_config
from antenna_diversity.dfe.processing import process_channels
from antenna_diversity.diversity.combiner import combine
from antenna_diversity.gnss.scenario import generate_scenario
from antenna_diversity.models import BranchResult, SimulationResults
from antenna_diversity.receiver.acquisition import acquire
from antenna_diversity.receiver.tracking import track
from antenna_diversity.rf.frontend import process_frontend


def run_end_to_end(cfg: SimulationConfig | None = None) -> SimulationResults:
    """Simulate GPS L1 C/A from antenna input through closed-loop tracking."""

    if cfg is None:
        cfg = default_config()
    cfg.validate()

    scene_input_v, truth = generate_scenario(cfg)
    adc_codes, rf_info = process_frontend(
        scene_input_v, cfg.rf, cfg.random_seed
    )
    dfe_channels, dfe_info = process_channels(adc_codes, cfg.dfe)

    desired_steering = truth["desired_steering_nominal"]
    branches: dict[str, BranchResult] = {}
    for mode in cfg.diversity.modes:
        combined, combining_info = combine(
            dfe_channels,
            desired_steering,
            mode,
            cfg.diversity,
        )
        acquisition = acquire(
            combined,
            dfe_info["fs_out_hz"],
            cfg.signal.prn,
            cfg.acquisition,
        )
        if acquisition["success"]:
            tracking = track(
                combined,
                dfe_info["fs_out_hz"],
                cfg.signal.prn,
                acquisition,
                cfg.tracking,
                cfg.signal.carrier_hz,
                cfg.signal.code_rate_hz,
            )
        else:
            tracking = {
                "success": False,
                "status": "acquisition_failed",
                "prompt": np.array([], dtype=np.complex128),
            }
        branches[mode.lower()] = BranchResult(
            combined=combined,
            combining=combining_info,
            acquisition=acquisition,
            tracking=tracking,
        )

    results = SimulationResults(
        config=cfg,
        truth=truth,
        scene_input_v=scene_input_v,
        adc_codes=adc_codes,
        rf_info=rf_info,
        dfe_channels=dfe_channels,
        dfe_info=dfe_info,
        branches=branches,
        metrics=[],
    )
    results.metrics = compute_metrics(results, truth)
    return results
