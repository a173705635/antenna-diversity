"""无干扰相位中心偏移的独立几何复算与跨seed统计。"""

import numpy as np
from scipy.stats import t as student_t

from core.config import SPEED_OF_LIGHT_MPS
from core.presets import mate70_phase_centers
from experiments.interference.coupling_support import coupling_row_fields, independent_manifold


def phase_center_geometry(cfg) -> dict:
    """真实和名义完整坐标各按其方向投影，复算第二阵元相对第一阵元的残余距离。"""
    true_direction = cfg.signal.true_direction
    assumed_direction = cfg.signal.assumed_direction
    true_azimuth = np.deg2rad(true_direction.azimuth_deg)
    true_elevation = np.deg2rad(true_direction.elevation_deg)
    assumed_azimuth = np.deg2rad(assumed_direction.azimuth_deg)
    assumed_elevation = np.deg2rad(assumed_direction.elevation_deg)
    true_line_of_sight = np.array([
        np.cos(true_elevation) * np.cos(true_azimuth),
        np.cos(true_elevation) * np.sin(true_azimuth), np.sin(true_elevation),
    ])
    assumed_line_of_sight = np.array([
        np.cos(assumed_elevation) * np.cos(assumed_azimuth),
        np.cos(assumed_elevation) * np.sin(assumed_azimuth), np.sin(assumed_elevation),
    ])
    origin = mate70_phase_centers()
    true_offsets = {}
    assumed_offsets = {}
    residual_los = {}
    for name in cfg.phone.active_antennas:
        true_offset = (np.asarray(cfg.phone.phase_centers_m[name]) - origin[name]) * 1e3
        assumed_offset = (np.asarray(cfg.phone.assumed_phase_centers_m[name]) - origin[name]) * 1e3
        true_offsets[name] = true_offset.tolist()
        assumed_offsets[name] = assumed_offset.tolist()
        # 方位失配使预设位置也贡献残余相位，因此这里用完整坐标，而非只投影偏移。
        residual_los[name] = float(1e3 * (
            np.dot(cfg.phone.phase_centers_m[name], true_line_of_sight) -
            np.dot(cfg.phone.assumed_phase_centers_m[name], assumed_line_of_sight)
        ))
    first, second = cfg.phone.active_antennas[:2]
    relative_mm = residual_los[second] - residual_los[first]
    phase_deg = -360.0 * cfg.signal.carrier_hz / SPEED_OF_LIGHT_MPS * relative_mm * 1e-3
    return {
        "true_offset_xyz_mm_by_antenna": true_offsets,
        "assumed_offset_xyz_mm_by_antenna": assumed_offsets,
        "residual_los_mm_by_antenna": residual_los,
        "residual_relative_los_mm": float(relative_mm),
        "residual_relative_phase_deg": float(phase_deg),
    }


def phase_center_row_fields(case, frontend, channels, components, window, weights, mode: str) -> dict:
    """复用独立源/噪声SNR核算，另核对位置投影与实际阵列响应。"""
    cfg = case.config
    # 无互耦、各路同RF且零通道时延，故源标量/热噪声协方差核算与互耦批次相同。
    fields = coupling_row_fields(case, frontend, channels, components, window, weights, mode)
    geometry = phase_center_geometry(cfg)
    true = independent_manifold(cfg, estimated_flag=False)
    assumed = independent_manifold(cfg, estimated_flag=True)
    first, second = cfg.phone.active_antennas[:2]
    true_by_name = dict(zip(cfg.phone.active_antennas, true))
    assumed_by_name = dict(zip(cfg.phone.active_antennas, assumed))
    relative_response = np.angle((true_by_name[second] / true_by_name[first]) /
                                 (assumed_by_name[second] / assumed_by_name[first]), deg=True)
    difference = np.angle(np.exp(1j * np.deg2rad(
        relative_response - geometry["residual_relative_phase_deg"])), deg=True)
    fields.update(geometry)
    fields["residual_phase_formula_error_deg"] = float(abs(difference))
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
    """逐PC/方法统计；同物理位置的条件仅核验 DFE 一致性。"""
    indexed = {(row["case"], row["mode"], row["seed"]): row for row in rows}
    grouped = {}
    for row in rows:
        grouped.setdefault((row["case"], row["mode"]), []).append(row)
    cases = {}
    checks = {}
    for (case_id, mode), observations in grouped.items():
        name = f"{case_id}-{mode}"
        first = observations[0]
        physical_reference = ("PC-009" if case_id in
                         {"PC-006", "PC-010", "PC-011", "PC-012"} else None)
        snr_gain = []
        for row in observations:
            seed = row["seed"]
            numerical_error = max(row["manifold_max_abs_error"], row["weight_relative_error"],
                                  row["weight_constraint_error"],
                                  row["residual_phase_formula_error_deg"])
            snr_error = abs(row["thermal_snr_theory_error_db"])
            passed_flag = bool(numerical_error <= 1e-8 and snr_error <= 1e-7)
            checks[f"phase_center_math:{name}:{seed}"] = {
                "status": "passed" if passed_flag else "failed", "passed_flag": passed_flag,
                "reason": (f"independent geometry/weight/SNR errors "
                           f"{numerical_error}, {snr_error} dB"),
            }
            if physical_reference is not None:
                reference = indexed[(physical_reference, mode, seed)]
                same_flag = row["physical_dfe_sha256"] == reference["physical_dfe_sha256"]
                checks[f"phase_center_physical_pair:{name}:{seed}"] = {
                    "status": "passed" if same_flag else "failed", "passed_flag": same_flag,
                    "reason": "same-seed physical DFE equals accurately calibrated PC-009",
                }
            snr_gain.append(row["thermal_snr_gain_db"])
        finite_cn0 = [row["tracking_cn0_late_median_db_hz"] for row in observations
                      if row["tracking_cn0_late_median_db_hz"] is not None]
        correct_count = sum(row["acquired_flag"] and
                            abs(row["acquired_code_error_chips"]) <= row["acquisition_code_tolerance_chips"]
                            and abs(row["acquired_doppler_error_hz"]) <=
                            row["acquisition_doppler_half_bin_hz"] for row in observations)
        cases[name] = {
            "case": case_id, "mode": mode, "seeds": len(observations),
            "antennas": first["antennas"],
            "residual_relative_los_mm": first["residual_relative_los_mm"],
            "residual_relative_phase_deg": first["residual_relative_phase_deg"],
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
        statistics = _paired_statistics(snr_gain)
        cases[name].update({f"snr_gain_vs_same_condition_single_{key}": value
                            for key, value in statistics.items()})
        for antenna in first["antennas"].split(","):
            for key in ("true_offset_xyz_mm_by_antenna", "assumed_offset_xyz_mm_by_antenna"):
                for axis, value in zip("xyz", first[key][antenna]):
                    cases[name][f"{antenna}_{key.removesuffix('_by_antenna')}_{axis}_mm"] = value
            cases[name][f"{antenna}_residual_los_mm"] = first["residual_los_mm_by_antenna"][antenna]
        passed_flag = all(checks[f"phase_center_math:{name}:{row['seed']}"]["passed_flag"]
                          for row in observations)
        if physical_reference is not None:
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
