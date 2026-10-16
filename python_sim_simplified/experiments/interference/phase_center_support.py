"""无干扰相位中心偏移的独立几何复算与跨seed统计。"""

import numpy as np
from scipy.stats import t as student_t

from core.config import SPEED_OF_LIGHT_MPS
from core.presets import mate70_phase_centers
from experiments.interference.coupling_support import coupling_row_fields, independent_manifold


def phase_center_geometry(cfg) -> dict:
    """按真实/名义坐标分别投影；输出相对TL的残余距离mm和相位deg。"""
    azimuth = np.deg2rad(cfg.signal.true_direction.azimuth_deg)
    elevation = np.deg2rad(cfg.signal.true_direction.elevation_deg)
    line_of_sight = np.array([np.cos(elevation) * np.cos(azimuth),
                              np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    origin = mate70_phase_centers()
    true_offsets = {}
    assumed_offsets = {}
    residual_los = {}
    for name in cfg.phone.active_antennas:
        true_offset = (np.asarray(cfg.phone.phase_centers_m[name]) - origin[name]) * 1e3
        assumed_offset = (np.asarray(cfg.phone.assumed_phase_centers_m[name]) - origin[name]) * 1e3
        true_offsets[name] = true_offset.tolist()
        assumed_offsets[name] = assumed_offset.tolist()
        residual_los[name] = float(np.dot(true_offset - assumed_offset, line_of_sight))
    relative_mm = residual_los["BR"] - residual_los["TL"]
    phase_deg = -360.0 * cfg.signal.carrier_hz / SPEED_OF_LIGHT_MPS * relative_mm * 1e-3
    return {
        "true_offset_xyz_mm_by_antenna": true_offsets,
        "assumed_offset_xyz_mm_by_antenna": assumed_offsets,
        "residual_los_mm_by_antenna": residual_los,
        "residual_relative_los_mm": float(relative_mm),
        "residual_relative_phase_deg": float(phase_deg),
    }


def strict_egc_loss_db(residual_phase_deg: float) -> float:
    """理想等幅独立白噪声下，双阵元严格EGC相对准确对齐的正损失dB。"""
    return float(-10.0 * np.log10(np.cos(np.deg2rad(residual_phase_deg) / 2) ** 2))


def phase_center_row_fields(case, frontend, channels, components, window, weights) -> dict:
    """复用独立源/噪声SNR核算，另核对位置投影与实际阵列响应。"""
    cfg = case.config
    # 无互耦、各路同RF且零通道时延，故源标量/热噪声协方差核算与互耦批次相同。
    fields = coupling_row_fields(case, frontend, channels, components, window, weights)
    geometry = phase_center_geometry(cfg)
    true = independent_manifold(cfg, estimated_flag=False)
    assumed = independent_manifold(cfg, estimated_flag=True)
    relative_response = np.angle((true[1] / true[0]) / (assumed[1] / assumed[0]), deg=True)
    difference = np.angle(np.exp(1j * np.deg2rad(
        relative_response - geometry["residual_relative_phase_deg"])), deg=True)
    fields.update(geometry)
    fields["residual_phase_formula_error_deg"] = float(abs(difference))
    fields["ideal_strict_egc_loss_db"] = strict_egc_loss_db(
        geometry["residual_relative_phase_deg"])
    fields["strict_egc_response_loss_error_db"] = (
        fields["true_response_gain_db"] + fields["ideal_strict_egc_loss_db"]
        if case.series_label == "egc_strict" else None)
    return fields


def _paired_statistics(values: list[float]) -> dict:
    """配对差值先按seed形成样本，CI仅描述该批seed的均值不确定度。"""
    mean = float(np.mean(values))
    interval = (float(student_t.ppf(0.975, len(values) - 1)
                      * np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else None)
    return {"mean_db": mean, "ci95_lower_db": mean - interval if interval is not None else None,
            "ci95_upper_db": mean + interval if interval is not None else None}


def summarize_phase_center(rows: list[dict]) -> dict:
    """逐PC/方法统计，配对PC001与同真实位置的PC009准确标定。"""
    indexed = {(row["case"], row["seed"]): row for row in rows}
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    cases = {}
    checks = {}
    for name, observations in grouped.items():
        first = observations[0]
        case_id = name[:6]
        mode = first["mode"]
        baseline_name = f"PC-001-{mode}" if case_id != "PC-001" else None
        accurate_name = (f"PC-009-{mode}" if case_id in
                         {"PC-006", "PC-010", "PC-011", "PC-012"} else None)
        snr_baseline = []
        snr_calibrated = []
        snr_single = []
        for row in observations:
            seed = row["seed"]
            numerical_error = max(row["manifold_max_abs_error"], row["weight_relative_error"],
                                  row["assumed_unit_constraint_error"],
                                  row["residual_phase_formula_error_deg"])
            snr_error = abs(row["thermal_snr_theory_error_db"])
            response_error = (abs(row["true_response_gain_db"] +
                                  row["ideal_strict_egc_loss_db"])
                              if mode == "egc_strict" else 0.0)
            passed_flag = bool(numerical_error <= 1e-8 and snr_error <= 1e-7
                               and response_error <= 1e-8)
            checks[f"phase_center_math:{name}:{seed}"] = {
                "status": "passed" if passed_flag else "failed", "passed_flag": passed_flag,
                "reason": (f"independent geometry/weight/SNR/strict-EGC response errors "
                           f"{numerical_error}, {snr_error}, {response_error} dB"),
            }
            if baseline_name is not None:
                snr_baseline.append(row["measured_thermal_snr_db"] -
                                    indexed[(baseline_name, seed)]["measured_thermal_snr_db"])
            if accurate_name is not None:
                reference = indexed[(accurate_name, seed)]
                snr_calibrated.append(row["measured_thermal_snr_db"] -
                                      reference["measured_thermal_snr_db"])
                same_flag = row["physical_dfe_sha256"] == reference["physical_dfe_sha256"]
                checks[f"phase_center_physical_pair:{name}:{seed}"] = {
                    "status": "passed" if same_flag else "failed", "passed_flag": same_flag,
                    "reason": "same-seed physical DFE equals accurately calibrated PC-009",
                }
            single = indexed[(f"{case_id}-single", seed)]
            snr_single.append(row["measured_thermal_snr_db"] - single["measured_thermal_snr_db"])
        finite_cn0 = [row["tracking_cn0_late_median_db_hz"] for row in observations
                      if row["tracking_cn0_late_median_db_hz"] is not None]
        correct_count = sum(row["acquired_flag"] and
                            abs(row["acquired_code_error_chips"]) <= row["acquisition_code_tolerance_chips"]
                            and abs(row["acquired_doppler_error_hz"]) <=
                            row["acquisition_doppler_half_bin_hz"] for row in observations)
        cases[name] = {
            "case": name, "mode": mode, "seeds": len(observations),
            "baseline_reference": baseline_name,
            "accurate_reference": accurate_name,
            "single_reference": f"{case_id}-single",
            "antennas": first["antennas"],
            "residual_relative_los_mm": first["residual_relative_los_mm"],
            "residual_relative_phase_deg": first["residual_relative_phase_deg"],
            "ideal_strict_egc_loss_db": first["ideal_strict_egc_loss_db"] if mode == "egc_strict" else None,
            "true_response_gain_db_mean": float(np.mean([row["true_response_gain_db"]
                                                          for row in observations])),
            "true_response_phase_deg_mean": float(np.angle(np.mean(np.exp(1j * np.deg2rad(
                [row["true_response_phase_deg"] for row in observations]))), deg=True)),
            "thermal_snr_mean_db": float(np.mean([row["measured_thermal_snr_db"]
                                                  for row in observations])),
            "acquisition_peak_ratio_mean": float(np.mean([row["acquisition_metric"]
                                                            for row in observations])),
            "acquisition_threshold_pass_count": sum(row["acquired_flag"] for row in observations),
            "acquisition_correct_count": correct_count,
            "tracking_success_count": sum(row["tracking_success_flag"] for row in observations),
            "tracking_cn0_mean_db_hz": float(np.mean(finite_cn0)) if finite_cn0 else None,
            "tracking_cn0_valid_seed_count": len(finite_cn0),
            "tracking_cn0_missing_count": len(observations) - len(finite_cn0),
            "max_manifold_abs_error": max(row["manifold_max_abs_error"] for row in observations),
            "max_weight_relative_error": max(row["weight_relative_error"] for row in observations),
            "max_snr_theory_error_db": max(abs(row["thermal_snr_theory_error_db"])
                                           for row in observations),
        }
        for label, values in (("snr_delta_vs_pc001", snr_baseline),
                              ("snr_delta_vs_accurate_calibration", snr_calibrated),
                              ("snr_gain_vs_same_condition_single", snr_single)):
            statistics = (_paired_statistics(values) if values else
                          {"mean_db": None, "ci95_lower_db": None, "ci95_upper_db": None})
            cases[name].update({f"{label}_{key}": value for key, value in statistics.items()})
        for antenna in first["antennas"].split(","):
            for key in ("true_offset_xyz_mm_by_antenna", "assumed_offset_xyz_mm_by_antenna"):
                for axis, value in zip("xyz", first[key][antenna]):
                    cases[name][f"{antenna}_{key.removesuffix('_by_antenna')}_{axis}_mm"] = value
            cases[name][f"{antenna}_residual_los_mm"] = first["residual_los_mm_by_antenna"][antenna]
        passed_flag = all(checks[f"phase_center_math:{name}:{row['seed']}"]["passed_flag"]
                          for row in observations)
        if accurate_name is not None:
            passed_flag = passed_flag and all(
                checks[f"phase_center_physical_pair:{name}:{row['seed']}"]["passed_flag"]
                for row in observations)
        cases[name]["status"] = "passed" if passed_flag else "failed"
        checks[f"phase_center_case:{name}"] = {
            "status": cases[name]["status"], "passed_flag": passed_flag,
            "reason": "independent geometry, SNR and same-physical-input checks",
        }
    return {"passed_flag": all(case["status"] == "passed" for case in cases.values()),
            "cases": cases, "checks": checks}
