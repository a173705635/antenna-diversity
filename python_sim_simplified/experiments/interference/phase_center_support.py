"""相位中心位置投影与实际接收结果的跨 seed 统计。"""

import numpy as np

from core.config import SPEED_OF_LIGHT_MPS
from core.presets import mate70_phase_centers
from experiments.interference.coupling_support import sample_statistics


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


def phase_center_row_fields(case, true_steering, weights) -> dict:
    """记录真实/估计位置偏移、视线投影和真实目标复响应。"""
    true_response = np.vdot(weights, true_steering)
    return {
        "true_response_gain_db": float(20 * np.log10(abs(true_response))),
        "true_response_phase_deg": float(np.angle(true_response, deg=True)),
        **phase_center_geometry(case.config),
    }


def summarize_phase_center(rows: list[dict]) -> dict:
    """逐物理条件和方法统计真实目标响应及接收性能。"""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["case"], row["receiver_role"], row["mode"]), []).append(row)
    cases = {}
    for (case_id, receiver_role, mode), observations in grouped.items():
        name = f"{case_id}-{mode}"
        first = observations[0]
        finite_cn0 = [row["tracking_cn0_mean_db_hz"] for row in observations
                      if row["tracking_cn0_mean_db_hz"] is not None]
        cases[name] = {
            "case": case_id, "mode": mode, "receiver_role": receiver_role,
            "series": first["series"], "scan_value": first["scan_value"],
            "phase_estimate_source": first["phase_estimate_source"],
            "seeds": len(observations),
            "antennas": first["antennas"],
            "residual_relative_los_mm": first["residual_relative_los_mm"],
            "residual_relative_phase_deg": first["residual_relative_phase_deg"],
            "true_response_gain_db": sample_statistics(
                [row["true_response_gain_db"] for row in observations]),
            "true_response_phase_deg_mean": float(np.angle(np.mean(np.exp(1j * np.deg2rad(
                [row["true_response_phase_deg"] for row in observations]))), deg=True)),
            # 分量功率收益统计暂时停用。
            # "thermal_snr_db": sample_statistics([row["thermal_snr_db"] for row in observations]),
            # "thermal_snr_gain_db": sample_statistics(
            #     [row["thermal_snr_gain_db"] for row in observations]),
            "acquisition_peak_ratio": sample_statistics(
                [row["acquisition_metric"] for row in observations]),
            "acquisition_threshold_pass_count": sum(row["acquired_flag"] for row in observations),
            "acquisition_correct_count": sum(row["acquisition_correct_flag"] for row in observations),
            "tracking_success_count": sum(row["tracking_success_flag"] for row in observations),
            "tracking_cn0_db_hz": sample_statistics(finite_cn0),
            "tracking_cn0_missing_count": len(observations) - len(finite_cn0),
        }
        for antenna in first["antennas"].split(","):
            for key in ("true_offset_xyz_mm_by_antenna", "assumed_offset_xyz_mm_by_antenna"):
                for axis, value in zip("xyz", first[key][antenna]):
                    cases[name][f"{antenna}_{key.removesuffix('_by_antenna')}_{axis}_mm"] = value
            cases[name][f"{antenna}_residual_los_mm"] = first["residual_los_mm_by_antenna"][antenna]
    return {"cases": cases}
