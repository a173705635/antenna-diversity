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
            "target_azimuth_deg": 33.0,
            "target_elevation_deg": 22.0,
            "output_sinr_db": float(18 + az_index + el_index),
            "interferer_gain": float(0.1 + 0.01 * az_index),
        }
        for el_index, elevation in enumerate((10.0, 20.0))
        for az_index, azimuth in enumerate((90.0, 120.0))
    )
    diagnostic_fields = {
        "record_kind": None,
        "model_source": "parametric",
        "model_label": "synthetic_phone_corner_v1",
        "model_view": None,
        "element_name": None,
        "receiver": None,
        "source": None,
        "source_name": None,
        "azimuth_deg": None,
        "elevation_deg": None,
        "response_real": None,
        "response_imag": None,
        "response_db": None,
        "attenuation_db": None,
        "polarization_match_db": None,
        "measured_pattern_sha256": None,
    }
    corners = ("TL", "TR", "BL", "BR")
    pattern_rows = tuple({
        **diagnostic_fields,
        "record_kind": "pattern_grid",
        "model_view": view,
        "element_name": corner,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "response_real": 1.0 + corner_index * 0.1,
        "response_imag": 0.05 * corner_index,
        "response_db": gain + corner_index,
    } for view, gain in (("true", -2.0), ("assumed", -1.0))
      for corner_index, corner in enumerate(corners)
      for elevation in (5.0, 50.0)
      for azimuth in (0.0, 35.0, 210.0))
    blockage_rows = tuple({
        **diagnostic_fields,
        "record_kind": "blockage_grid",
        "model_view": view,
        "element_name": corner,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "response_db": -attenuation,
        "attenuation_db": attenuation,
    } for view, base_attenuation in (("true", 4.0), ("assumed", 1.0))
      for corner_index, corner in enumerate(corners)
      for elevation in (5.0, 50.0)
      for azimuth in (0.0, 35.0, 210.0)
      for attenuation in (base_attenuation + 0.1 * corner_index,))
    source_rows = tuple({
        **diagnostic_fields,
        "record_kind": "source_match",
        "model_view": view,
        "element_name": corner,
        "source_name": source_name,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "response_real": 0.8,
        "response_imag": 0.1,
        "response_db": -2.0,
        "polarization_match_db": match_db - corner_index,
    } for view in ("true", "assumed")
      for corner_index, corner in enumerate(corners)
      for source_name, azimuth, elevation, match_db in (
          ("gps", 33.0, 22.0, -0.5),
          ("jammer_0", 120.0, 10.0, -3.0),
      ))
    coupling_rows = tuple({
        **diagnostic_fields,
        "record_kind": "coupling_entry",
        "model_view": view,
        "receiver": receiver,
        "source": source,
        "response_real": 1.0 if receiver == source else 0.05,
        "response_imag": 0.0 if receiver == source else phase_sign * 0.02,
        "response_db": 0.0 if receiver == source else -25.0,
    } for view, phase_sign in (("true", 1.0), ("assumed", -1.0))
      for receiver in corners for source in corners)
    return {
        "baseline_rankings": ({
            "active_antennas": combination,
            "display_label": "MVDR(TL+BR)",
            "mode": "mvdr",
            "p10_resolvable_db": 18.0,
            "median_resolvable_db": 19.5,
            "p10_all_db": 17.0,
            "median_all_db": 18.5,
            "minimum_all_db": 2.0,
            "minimum_resolvable_db": 16.0,
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
        "antenna_diagnostics": (
            *pattern_rows, *source_rows, *coupling_rows, *blockage_rows,
        ),
    }
