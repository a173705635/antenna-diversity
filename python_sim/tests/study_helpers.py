from dataclasses import replace
from pathlib import Path

from antenna_diversity.studies.config import resolve_study_config
from antenna_diversity.studies.models import ResolvedStudyConfig


def resolved_for_test(*, preset: str = "quick", output_dir: Path | None = None, **study_changes: object) -> ResolvedStudyConfig:
    base = resolve_study_config(preset=preset)
    changes = dict(study_changes)
    if output_dir is not None:
        changes["output_dir"] = output_dir
    study = replace(base.study, **changes)
    return ResolvedStudyConfig.from_parts(study, base.simulation)


def completed_tables_for_reporting() -> dict[str, tuple[dict[str, object], ...]]:
    """Return the smallest complete, normalized study snapshot for reporting tests."""

    combination = ["TL", "BR"]
    direction = tuple(
        {
            "case_id": f"direction-{azimuth}-{elevation}",
            "active_antennas": combination,
            "display_label": "MVDR(TL+BR)",
            "mode": "mvdr",
            "azimuth_deg": float(azimuth),
            "elevation_deg": float(elevation),
            "azimuth_index": az_index,
            "elevation_index": el_index,
            "output_sinr_db": float(18 + az_index + el_index),
            "interferer_gain": float(0.1 + 0.01 * az_index),
        }
        for el_index, elevation in enumerate((10.0, 20.0))
        for az_index, azimuth in enumerate((90.0, 120.0))
    )
    return {
        "baseline_rankings": ({
            "active_antennas": combination,
            "display_label": "MVDR(TL+BR)",
            "mode": "mvdr",
            "p10_resolvable_db": 18.0,
            "median_resolvable_db": 19.5,
            "minimum_all_db": 2.0,
            "white_noise_gain": 1.0,
            "maximum_baseline_m": 0.178,
            "rank": 1,
        },),
        "direction_scan": direction,
        "power_scan": tuple({
            "active_antennas": combination,
            "display_label": f"{mode.upper()}(TL+BR)",
            "mode": mode,
            "power_dbm": power,
            "output_sinr_db": 20.0 - (power + 100.0) / 5.0 + (2.0 if mode == "mvdr" else 0.0),
        } for mode in ("egc", "mvdr") for power in (-100.0, -90.0)),
        "robustness_scans": tuple({
            "active_antennas": combination,
            "factor": "phase",
            "value": value,
            "normalized_severity": abs(value) / 5.0,
            "egc_output_sinr_db": 15.0,
            "mvdr_output_sinr_db": 18.0 - abs(value) / 5.0,
            "mvdr_minus_egc_sinr_db": 3.0 - abs(value) / 5.0,
        } for value in (-5.0, 0.0, 5.0)),
        "combined_errors": tuple({
            "active_antennas": combination,
            "level": level,
            "sample_index": 0,
            "output_sinr_db": value,
        } for level, value in (("light", 18.0), ("medium", 14.0), ("heavy", 8.0))),
        "multi_jammer": ({
            "active_antennas": combination,
            "template": "two_jammer_separated",
            "mode": "mvdr",
            "enabled_interferers": 2,
            "output_sinr_db": 12.0,
            "worst_interferer_gain": 0.04,
        },),
        "full_chain_validation": tuple({
            "case_id": "best-001",
            "selection_reasons": ["best"],
            "mode": mode,
            "run_index": 0,
            "seed": 7,
            "acquired": mode != "single",
            "tracking": mode != "single",
            "status": "tracking" if mode != "single" else "acquisition_failed",
            "acquisition_metric": 3.0 if mode != "single" else 1.0,
            "analytic_output_sinr_db": 18.0 if mode == "mvdr" else 10.0,
            "final_cn0_db_hz": 37.0 if mode != "single" else None,
            "final_doppler_error_hz": 5.0 if mode != "single" else None,
        } for mode in ("single", "egc", "mvdr")),
        "full_chain_cn0": tuple({
            "case_id": "best-001",
            "mode": mode,
            "run_index": 0,
            "seed": 7,
            "epoch_index": epoch,
            "time_ms": float(epoch),
            "cn0_db_hz": 36.0 + epoch,
        } for mode in ("egc", "mvdr") for epoch in (0, 1)),
    }
