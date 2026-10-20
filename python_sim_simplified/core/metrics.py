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


def _final_cn0(tracking: TrackingResult) -> float | None:
    if not tracking["success_flag"]:
        return None
    finite = tracking["cn0_db_hz"][np.isfinite(tracking["cn0_db_hz"])]
    return float(finite[-1]) if finite.size else None


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
                "final_cn0_db_hz": _final_cn0(tracking),
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
            source = (record["phase_estimate_source"] if mode_label == "egc"
                      else "known_response")
            mode_label += f" ({source})"
        mode_labels.append(mode_label)
    mode_width = max([10, *(len(label) for label in mode_labels)])
    header = (
        f"{'Mode':>{mode_width}} {'Acq':>6} {'Metric':>9} {'Tracking':>9} "
        f"{'Doppler':>10} {'C/N0':>9} {'CodeBias(m)':>12}"
    )
    lines = [header, "-" * len(header)]
    for record, mode_label in zip(records, mode_labels):
        doppler = record["final_doppler_hz"]
        cn0 = record["final_cn0_db_hz"]
        code_bias_m = record["tracking_code_bias_rms_m"]
        lines.append(
            f"{mode_label:>{mode_width}} {str(record['acquired_flag']):>6} "
            f"{record['acquisition_metric']:>9.3f} "
            f"{str(record['tracking_flag']):>9} "
            f"{doppler if doppler is not None else np.nan:>10.2f} "
            f"{cn0 if cn0 is not None else np.nan:>9.2f} "
            f"{code_bias_m if code_bias_m is not None else np.nan:>12.3f}"
        )
    return "\n".join(lines)
