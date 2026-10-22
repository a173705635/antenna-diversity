"""通用验证固定入口：读取条件 → 仿真与核验 → 保存配置和数据 → 绘图。

日常修改 validation_config.py；本文件只负责组织运行与输出，不实现接收算法。
在 python_sim_simplified 目录运行 python -m experiments.interference.validation。
"""

from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    # 支持直接运行本文件；模块方式启动时无需修改导入路径。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.metrics import format_metric_value
from core.provenance import source_hashes
from core.serialization import serializable
from experiments.interference.validation_config import build_cases, build_settings
from experiments.interference.validation_support import (
    ValidationSettings, run_validation,
)
from experiments.interference.coupling_support import sample_statistics
from experiments.interference.validation_plotting import plot_validation
from experiments.paths import reset_output


DETAIL_RESULT_COLUMNS = (
    "acquisition_metric", "acquired_flag", "acquisition_correct_flag",
    "acquired_code_error_chips", "acquired_doppler_error_hz",
    "tracking_success_flag", "tracking_status", "tracking_cn0_mean_db_hz",
    "tracking_cn0_valid_count",
)

CSV_LABELS = {
    "case": "Case", "mode": "Mode", "seed": "Seed", "scene": "Scene",
    "series": "Series", "seeds": "N",
    "acquisition_correct_count": "ACQ OK", "tracking_success_count": "TRK",
    "tracking_cn0_mean_db_hz": "CN0 (dB-Hz)",
    "tracking_cn0_std_db_hz": "CN0 SD (dB)",
    "tracking_cn0_valid_seed_count": "CN0 Seeds",
    "acquisition_metric": "ACQ Ratio", "acquired_flag": "ACQ Pass",
    "acquisition_correct_flag": "ACQ OK",
    "acquired_code_error_chips": "ACQ Code Err (chip)",
    "acquired_doppler_error_hz": "ACQ Dop Err (Hz)",
    "tracking_success_flag": "TRK", "tracking_status": "TRK Status",
    "tracking_cn0_valid_count": "CN0 Epochs",
    "los_offset_mm": "LOS Offset (mm)",
    "true_coupling_voltage_ratio": "Cpl True (V/V)",
    "true_coupling_phase_deg": "Cpl True (deg)",
    "assumed_coupling_db": "Cpl Nom (dB)",
    "assumed_coupling_phase_deg": "Cpl Nom (deg)",
    "DFE channel phase (deg)": "DFE Phase (deg)",
    "temperature_c": "Temperature (°C)",
    "gain_delta_db": "Gain Δ (dB)",
    "phase_delta_deg": "Phase Δ (deg)",
    "rf_group_delay_ns": "RF Delay (ns)",
}


def _write_json(output_path: Path, content: object) -> None:
    """将路径、NumPy 标量和嵌套结构转成 JSON 可保存的形式。"""
    def paths_as_text(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: paths_as_text(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [paths_as_text(item) for item in value]
        return value

    output_path.write_text(json.dumps(serializable(paths_as_text(content)), indent=2,
                               allow_nan=False), encoding="utf-8")


def _write_csv(output_path: Path, records: list[dict]) -> None:
    """按实际条件列的并集写表，异构扫描不丢列。"""
    columns = list(dict.fromkeys(key for record in records for key in record))
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        # 只格式化 CSV 的表头和数值；原记录供统计及 JSON 保存，不在此舍入。
        writer.writerow({key: _csv_label(key) for key in columns})
        writer.writerows({key: (f"{round(value, 3) + 0.0:g}" if value is not None and key in (
            "temperature_c", "gain_delta_db", "phase_delta_deg") else
            format_metric_value(key, value)) for key, value in record.items()}
                         for record in records)


def _csv_label(key: str) -> str:
    """保留内部字段名，只将 CSV 表头压成便于阅读的短标签。"""
    if key in CSV_LABELS:
        return CSV_LABELS[key]
    if key.endswith("_gain_db"):
        return f"{key.removesuffix('_gain_db')} Gain (dB)"
    return key.replace(" gain (", " Gain (")


def _condition_columns(row: dict, experiment_type: str) -> dict:
    """只提取日常表需要的物理条件；完整配置和诊断仍留在 JSON。"""
    if experiment_type == "antenna_temperature":
        temperature_info = row["temperature_offset"]
        return {"scene": row["scene"], "series": row["series"],
                "temperature_c": temperature_info["temperature_c"],
                "gain_delta_db": temperature_info["gain_delta_db"] if row["series"] == "gain" else None,
                "phase_delta_deg": temperature_info["phase_delta_deg"] if row["series"] == "phase" else None}
    if experiment_type == "antenna_gain":
        return {f"{antenna}_gain_db": gain for antenna, gain in zip(
            row["antennas"].split(","), row["input_gain_db_by_antenna"])}
    if experiment_type == "mutual_coupling":
        return {"series": row["series"],
                "true_coupling_voltage_ratio": (10 ** (row["true_coupling_db"] / 20)
                                                if row["true_coupling_db"] is not None else 0),
                "true_coupling_phase_deg": row["true_coupling_phase_deg"]}
    if experiment_type == "phase_center_offset":
        return {"series": row["series"], "los_offset_mm": row["scan_value"]}
    if experiment_type == "phase_offset":
        return {"scene": row["scene"], "DFE channel phase (deg)": row["scan_value"]}
    if experiment_type == "rf_group_delay":
        return {"scene": row["scene"], "rf_group_delay_ns": row["scan_value"]}
    conditions = {"scene": row["scene"], "series": row["series"]}
    if row["scan_value"] is not None:
        label = row["scan_label"]
        unit = row["scan_unit"]
        conditions[f"{label} ({unit})" if unit else label] = row["scan_value"]
    return conditions


def phase_center_summary_records(cases: list[dict]) -> list[dict]:
    """从当前 PCO 统计生成单因素主表，不修改完整统计。"""
    records = []
    for case in cases:
        records.append({
            "case": case["case"], "mode": case["mode"],
            "series": case["series"], "los_offset_mm": case["scan_value"],
            "seeds": case["seeds"],
            "acquisition_correct_count": case["acquisition_correct_count"],
            "tracking_success_count": case["tracking_success_count"],
            "tracking_cn0_mean_db_hz": case["tracking_cn0_db_hz"]["mean"],
            "tracking_cn0_std_db_hz": case["tracking_cn0_db_hz"]["std"],
            "tracking_cn0_valid_seed_count": case["tracking_cn0_db_hz"]["valid_count"],
        })
    return records


def general_summary_records(rows: list[dict], experiment_type: str = "phase_offset") -> list[dict]:
    """按用例和算法汇总普通扫描；缺测 C/N₀ 只计数，不填零。"""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["case"], row["mode"]), []).append(row)
    records = []
    for (case_name, mode), observations in grouped.items():
        first_row = observations[0]
        available_cn0 = [row["tracking_cn0_mean_db_hz"] for row in observations
                         if row["tracking_cn0_mean_db_hz"] is not None]
        cn0_statistics = sample_statistics(available_cn0)
        records.append({
            "case": case_name, "mode": mode,
            **_condition_columns(first_row, experiment_type),
            "seeds": len(observations),
            "acquisition_correct_count": sum(row["acquisition_correct_flag"] for row in observations),
            "tracking_success_count": sum(row["tracking_success_flag"] for row in observations),
            "tracking_cn0_mean_db_hz": cn0_statistics["mean"],
            "tracking_cn0_std_db_hz": cn0_statistics["std"],
            "tracking_cn0_valid_seed_count": cn0_statistics["valid_count"],
        })
    return records


def save_results(settings: ValidationSettings, summary: dict) -> None:
    """刷新本批次输出，保存执行配置、结果行和公共核验证据。"""
    # 仿真完成后才刷新本入口的专用输出，避免换扫描因素后残留旧图。
    if settings.workers == 1:
        reset_output(settings.output_dir)
    # 并行批次已在启动时刷新专用目录；进度文件在完整入口成功后清理。
    # 日常摘要只展示实验条件；完整底层配置只在manifest中保存一次。
    case_conditions = []
    for case in summary["cases"]:
        condition = {key: value for key, value in case.items() if key != "receiver_configs"}
        case_conditions.append(condition)
    manifest = {
        # 清单用于追溯参数与代码版本，不是下一次仿真的配置输入。
        "settings": asdict(settings),
        "cases": case_conditions,
        # run_validation 已按条件和 seed 保存实际执行配置，直接复用。
        "actual_receiver_configs": {
            case["name"]: case["receiver_configs"] for case in summary["cases"]
        },
        "methods": {row["mode"]: {
            "receiver_role": row["receiver_role"],
            "phase_estimate_source": row["phase_estimate_source"],
        } for row in summary["rows"]},
        "source_sha256": source_hashes(),
    }
    _write_json(settings.output_dir / "manifest.json", manifest)
    # summary 保存结果、统计和核验；配置与条件只在 manifest 保存。
    _write_json(settings.output_dir / "summary.json", {
        **{key: value for key, value in summary.items() if key != "cases"},
        "receiver_config_file": "manifest.json",
    })
    rows = summary["rows"]
    detail_records = [{
        "case": row["case"], "seed": row["seed"], "mode": row["mode"],
        **_condition_columns(row, settings.experiment_type),
        **{key: row[key] for key in DETAIL_RESULT_COLUMNS},
    } for row in rows]
    _write_csv(settings.output_dir / "summary.csv", detail_records)
    if "pattern_gain" in summary:
        # 一行一个 TC：跨seed汇总与逐seed原始行分开，便于直接筛选。
        records = []
        for result in summary["pattern_gain"]["cases"].values():
            tc = result["case"]
            observations = [row for row in rows if row["case"] == tc
                            and row["mode"] == result["mode"]]
            records.append({
                "case": tc, "mode": result["mode"],
                **_condition_columns(observations[0], settings.experiment_type),
                "seeds": result["seeds"],
                "acquisition_correct_count": result["acquisition_correct_count"],
                "tracking_success_count": result["tracking_success_count"],
                "tracking_cn0_mean_db_hz": result["tracking_cn0_observed_db_hz"]["mean"],
                "tracking_cn0_std_db_hz": result["tracking_cn0_observed_db_hz"]["std"],
                "tracking_cn0_valid_seed_count": result["tracking_cn0_observed_db_hz"]["valid_count"],
            })
        _write_csv(settings.output_dir / "case_summary.csv", records)
    if "mutual_coupling" in summary:
        records = []
        for case in summary["mutual_coupling"]["cases"].values():
            record = {
                "case": case["case"], "mode": case["mode"],
                "series": case["series"],
                "true_coupling_voltage_ratio": (10 ** (case["true_coupling_db"] / 20)
                                                if case["true_coupling_db"] is not None else 0),
                "true_coupling_phase_deg": case["true_coupling_phase_deg"],
                "seeds": case["seeds"],
                "acquisition_correct_count": case["acquisition_correct_count"],
                "tracking_success_count": case["tracking_success_count"],
                "tracking_cn0_mean_db_hz": case["tracking_cn0_db_hz"]["mean"],
                "tracking_cn0_std_db_hz": case["tracking_cn0_db_hz"]["std"],
                "tracking_cn0_valid_seed_count": case["tracking_cn0_db_hz"]["valid_count"],
            }
            records.append(record)
        _write_csv(settings.output_dir / "case_summary.csv", records)
    if "phase_center" in summary:
        # 日常只看精简主表；完整统计、坐标和核验信息仍保留在JSON。
        records = phase_center_summary_records(list(summary["phase_center"]["cases"].values()))
        _write_csv(settings.output_dir / "case_summary.csv", records)
    if not any(key in summary for key in ("pattern_gain", "mutual_coupling", "phase_center")):
        records = general_summary_records(rows, settings.experiment_type)
        _write_csv(settings.output_dir / "case_summary.csv", records)


def main() -> dict:
    """运行并保存数据，报告失败项；图片由配置显式开启。"""
    settings = build_settings()
    cases = build_cases()
    summary = run_validation(settings, cases)
    save_results(settings, summary)
    print(f"Saved {len(summary['rows'])} result rows to {settings.output_dir}")
    failed_checks = {
        name: check for name, check in summary["verification"]["checks"].items()
        if check["status"] == "failed"
    }
    print(f"Verification failed checks: {len(failed_checks)}")
    for name, check in failed_checks.items():
        print(f"  {name}: {check['reason']}")
    if settings.generate_plots_flag:
        plot_validation(summary["rows"], settings)
    if not summary["verification"]["passed_flag"]:
        # 这是实验验收失败，不是接收机防御分支；不能把失败批次静默当成功。
        raise AssertionError("validation verification failed; inspect summary.json")
    if settings.workers > 1:
        (settings.output_dir / "progress.jsonl").unlink()
    return summary


if __name__ == "__main__":
    main()
