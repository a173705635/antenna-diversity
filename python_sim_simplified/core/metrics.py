"""Branch-comparison metrics for the simplified research simulation."""

import numpy as np

from .config import SPEED_OF_LIGHT_MPS
from .models import (
    AcquisitionResult,
    MetricRecord,
    ScenarioTruth,
    SimulationResults,
    TrackingResult,
)


# 日常结果的显示精度；只在导出时使用，统计和 JSON 仍保留原始数值。
METRIC_DECIMAL_PLACES = {
    "tracking_cn0_mean_db_hz": 2,
    "tracking_cn0_std_db_hz": 2,
    "acquisition_metric": 2,
    "acquired_doppler_error_hz": 2,
    "final_doppler_hz": 2,
    "acquired_code_error_chips": 3,
    "acquisition_code_bias_chips": 3,
    "tracking_code_bias_rms_m": 2,
}


def format_metric_value(name: str, value):
    """按指标单位生成显示文本；缺测值留空，条件、计数和布尔值原样保留。"""
    if name in METRIC_DECIMAL_PLACES and value is not None:
        return f"{value:.{METRIC_DECIMAL_PLACES[name]}f}"
    return value


def circular_chip_error(estimate, truth):
    return (
        np.asarray(estimate) - np.asarray(truth) + 511.5
    ) % 1023.0 - 511.5


def acquisition_code_bias_chips(
    acquisition: AcquisitionResult,
    truth: ScenarioTruth,
    fs_out_hz: float,
) -> float:
    estimated = (
        acquisition["code_phase_samples"]
        * truth["code_rate_hz"]
        / fs_out_hz
    )
    return float(circular_chip_error(estimated, truth["code_delay_chips"]))


def tracking_code_bias_chips(
    tracking: TrackingResult,
    truth: ScenarioTruth,
    fs_out_hz: float,
    coherent_ms: int,
) -> np.ndarray:
    if not tracking["success_flag"]:
        return np.array([], dtype=float)
    samples_per_epoch = round(fs_out_hz * coherent_ms * 1e-3)
    comparison_samples = tracking["epoch_start_samples"] + samples_per_epoch
    expected = np.mod(
        comparison_samples * truth["code_rate_hz"] / fs_out_hz
        - truth["code_delay_chips"],
        1023.0,
    )
    return circular_chip_error(tracking["code_phase_chips"], expected)


def tracking_cn0_mean(tracking: TrackingResult) -> tuple[float | None, int]:
    """对原跟踪序列后半段的有限 C/N₀ (dB-Hz) 求算术均值及有效历元数。"""
    if not tracking["success_flag"]:
        return None, 0
    cn0 = tracking["cn0_db_hz"]
    late_cn0 = cn0[len(cn0) // 2:]
    finite_cn0 = late_cn0[np.isfinite(late_cn0)]
    return (float(np.mean(finite_cn0)), int(finite_cn0.size)) if finite_cn0.size else (None, 0)


def compute_metrics(results: SimulationResults) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for mode, branch_result in results["branches"].items():
        acquisition = branch_result["acquisition"]
        tracking = branch_result["tracking"]
        # 方向增益仅在事后用真值评分，不回写实际接收分支的求权信息。
        weights = branch_result["combining_info"]["weights"]
        target_direction_gain = float(abs(np.vdot(
            weights, results["truth"]["desired_steering_true"],
        )))
        interferer_gains = [
            float(abs(np.vdot(weights, steering)))
            for steering in results["truth"]["interferer_steering_effective"]
        ]
        tracked_flag = bool(tracking["success_flag"])
        if tracked_flag:
            window = slice(max(0, tracking["prompt"].size - 10), None)
            final_doppler = float(tracking["carrier_frequency_hz"][-1])
            pll_rms = float(np.sqrt(np.mean(tracking["pll_error_rad"][window] ** 2)))
            dll_rms = float(np.sqrt(np.mean(tracking["dll_error"][window] ** 2)))
            mean_prompt = float(np.mean(np.abs(tracking["prompt"][window])))
            tracking_bias = tracking_code_bias_chips(
                tracking,
                results["truth"],
                results["dfe_info"]["fs_out_hz"],
                results["config"].tracking.coherent_ms,
            )
            tracking_bias_rms = float(
                np.sqrt(np.mean(tracking_bias**2))
            )
            tracking_bias_rms_m = (
                tracking_bias_rms
                * SPEED_OF_LIGHT_MPS
                / results["config"].signal.code_rate_hz
            )
        else:
            final_doppler = pll_rms = dll_rms = mean_prompt = None
            tracking_bias_rms = tracking_bias_rms_m = None
        records.append(
            {
                "mode": mode,
                "receiver_role": branch_result["receiver_role"],
                "phase_estimate_source": branch_result["phase_estimate_source"],
                "acquired_flag": bool(acquisition["success_flag"]),
                "acquisition_metric": float(acquisition["metric"]),
                "acquired_doppler_hz": float(acquisition["doppler_hz"]),
                "target_direction_gain": target_direction_gain,
                "worst_interferer_gain": max(interferer_gains, default=0.0),
                "tracking_flag": tracked_flag,
                "final_doppler_hz": final_doppler,
                "pll_rms_rad": pll_rms,
                "dll_rms": dll_rms,
                "mean_prompt_magnitude": mean_prompt,
                "tracking_cn0_mean_db_hz": tracking_cn0_mean(tracking)[0],
                "acquisition_code_bias_chips": acquisition_code_bias_chips(
                    acquisition,
                    results["truth"],
                    results["dfe_info"]["fs_out_hz"],
                ),
                "tracking_code_bias_rms_chips": tracking_bias_rms,
                "tracking_code_bias_rms_m": tracking_bias_rms_m,
            }
        )
    return records


def format_metrics(records: list[MetricRecord]) -> str:
    mode_labels = []
    for record in records:
        mode_label = record["mode"]
        if record["receiver_role"] == "known_response_comparison":
            if mode_label == "mvdr":
                mode_label = "MVDR (known resp)"
            elif mode_label == "egc":
                source = record["phase_estimate_source"]
                if source == "preset_assumed_target_response_phase":
                    mode_label = "EGC (preset phase)"
                elif source == "external_input":
                    mode_label = "EGC (external phase)"
                else:
                    mode_label = f"EGC ({source})"
            else:
                mode_label += f" ({record['phase_estimate_source']})"
        mode_labels.append(mode_label)
    mode_width = max([4, *(len(label) for label in mode_labels)])
    header = (
        f"{'Mode':>{mode_width}} {'ACQ Pass':>8} {'ACQ Ratio':>9} {'TRK':>5} "
        f"{'Dop (Hz)':>9} {'CN0 (dB-Hz)':>12} {'Code RMS (m)':>12}"
    )
    lines = [header, "-" * len(header)]
    for record, mode_label in zip(records, mode_labels):
        # CSV 和终端共用精度；终端沿用 nan 标记缺测，CSV 仍为空单元格。
        displayed_metrics = {
            name: format_metric_value(name, value) if value is not None else "nan"
            for name, value in record.items()
        }
        lines.append(
            f"{mode_label:>{mode_width}} {str(record['acquired_flag']):>8} "
            f"{displayed_metrics['acquisition_metric']:>9} "
            f"{str(record['tracking_flag']):>5} "
            f"{displayed_metrics['final_doppler_hz']:>9} "
            f"{displayed_metrics['tracking_cn0_mean_db_hz']:>12} "
            f"{displayed_metrics['tracking_code_bias_rms_m']:>12}"
        )
    return "\n".join(lines)
