"""互耦实验条件与实际接收结果的跨 seed 统计。"""

import numpy as np

def sample_statistics(values: list[float]) -> dict:
    """返回有效样本的均值、样本标准差及有效数；单样本不估计标准差。"""
    samples = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(samples)) if len(samples) else None,
        "std": float(np.std(samples, ddof=1)) if len(samples) > 1 else None,
        "valid_count": len(samples),
    }


def coupling_row_fields(case, true_steering: np.ndarray, weights: np.ndarray) -> dict:
    """记录互耦条件和实际权值对真实目标导向的复响应。"""
    cfg = case.config
    true_response = np.vdot(weights, true_steering)
    true_paths = cfg.phone.antenna_model.true_state.coupling_paths
    assumed_paths = cfg.phone.antenna_model.assumed_state.coupling_paths
    return {
        "true_coupling_db": true_paths[0].magnitude_db if true_paths else None,
        "true_coupling_phase_deg": true_paths[0].phase_deg if true_paths else None,
        "assumed_coupling_db": assumed_paths[0].magnitude_db if assumed_paths else None,
        "assumed_coupling_phase_deg": assumed_paths[0].phase_deg if assumed_paths else None,
        "true_response_gain_db": float(20 * np.log10(abs(true_response))),
        "true_response_phase_deg": float(np.angle(true_response, deg=True)),
    }


def summarize_coupling(rows: list[dict]) -> dict:
    """逐物理条件和方法汇总真实响应、捕获及跟踪。"""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["case"], row["receiver_role"], row["mode"]), []).append(row)
    cases = {}
    for (case_id, receiver_role, mode), observations in grouped.items():
        first = observations[0]
        cn0 = [row["tracking_cn0_mean_db_hz"] for row in observations
               if row["tracking_cn0_mean_db_hz"] is not None]
        phases = np.deg2rad([row["true_response_phase_deg"] for row in observations])
        cases[f"{case_id}-{mode}"] = {
            "case": case_id, "scene": first["scene"], "level": "link", "mode": mode,
            "series": first["series"], "scan_value": first["scan_value"],
            "receiver_role": receiver_role,
            "phase_estimate_source": first["phase_estimate_source"],
            "seeds": len(observations), "antennas": first["antennas"],
            "true_coupling_db": first["true_coupling_db"],
            "true_coupling_phase_deg": first["true_coupling_phase_deg"],
            "assumed_coupling_db": first["assumed_coupling_db"],
            "assumed_coupling_phase_deg": first["assumed_coupling_phase_deg"],
            "true_response_gain_db": sample_statistics(
                [row["true_response_gain_db"] for row in observations]),
            "true_response_phase_deg_mean": float(np.angle(np.mean(np.exp(1j * phases)), deg=True)),
            # 分量功率收益统计暂时停用。
            # "thermal_snr_db": sample_statistics(
            #     [row["thermal_snr_db"] for row in observations]),
            # "thermal_snr_gain_db": sample_statistics(
            #     [row["thermal_snr_gain_db"] for row in observations]),
            "acquisition_peak_ratio": sample_statistics(
                [row["acquisition_metric"] for row in observations]),
            "acquisition_threshold_pass_count": sum(row["acquired_flag"] for row in observations),
            "acquisition_correct_count": sum(row["acquisition_correct_flag"] for row in observations),
            "tracking_success_count": sum(row["tracking_success_flag"] for row in observations),
            "tracking_cn0_db_hz": sample_statistics(cn0),
            "tracking_cn0_missing_count": len(observations) - len(cn0),
        }
    return {"cases": cases}
