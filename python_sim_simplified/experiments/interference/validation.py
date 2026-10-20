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

from core.provenance import source_hashes
from core.serialization import serializable
from experiments.interference.validation_config import build_cases, build_settings
from experiments.interference.validation_support import (
    ValidationSettings, run_validation,
)
from experiments.interference.validation_plotting import plot_validation
from experiments.paths import reset_output


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


def phase_center_summary_records(cases: list[dict]) -> list[dict]:
    """从当前 PCO 统计生成主表视图，不修改完整统计。"""
    columns = (
        "residual_relative_los_mm", "residual_relative_phase_deg",
        "true_response_phase_deg_mean",
        "acquisition_threshold_pass_count", "acquisition_correct_count",
        "tracking_success_count", "tracking_cn0_missing_count",
    )
    records = []
    for case in cases:
        descriptions = []
        for antenna in case["antennas"].split(","):
            vectors = []
            for state, label in (("true", "真实"), ("assumed", "估计")):
                # 条件从已保存的实际坐标读取，不按PC编号写死；仅说明文字舍入。
                values = [round(case[f"{antenna}_{state}_offset_xyz_mm_{axis}_mm"], 3)
                          for axis in "xyz"]
                xyz = ",".join(f"{value + 0.0:g}" for value in values)
                vectors.append(f"{label}({xyz})")
            descriptions.append(f"{antenna}: " + "/".join(vectors))
        records.append({
            "case": case["case"], "mode": case["mode"],
            "receiver_role": case["receiver_role"],
            "phase_estimate_source": case["phase_estimate_source"],
            "seeds": case["seeds"],
            "condition_xyz_mm": "; ".join(descriptions),
            **{key: case[key] for key in columns},
            # 分量功率及相对 SINGLE 收益列暂时停用。
            # "thermal_snr_mean_db": case["thermal_snr_db"]["mean"],
            # "thermal_snr_std_db": case["thermal_snr_db"]["std"],
            # "snr_gain_vs_same_condition_single_mean_db": case["thermal_snr_gain_db"]["mean"],
            # "snr_gain_vs_same_condition_single_std_db": case["thermal_snr_gain_db"]["std"],
            "acquisition_peak_ratio_mean": case["acquisition_peak_ratio"]["mean"],
            "acquisition_peak_ratio_std": case["acquisition_peak_ratio"]["std"],
            "tracking_cn0_mean_db_hz": case["tracking_cn0_db_hz"]["mean"],
            "tracking_cn0_std_db_hz": case["tracking_cn0_db_hz"]["std"],
            "tracking_cn0_valid_seed_count": case["tracking_cn0_db_hz"]["valid_count"],
        })
    return records


def save_results(settings: ValidationSettings, summary: dict) -> None:
    """刷新本批次输出，保存执行配置、结果行和公共核验证据。"""
    # 仿真完成后才刷新本入口的专用输出，避免换扫描因素后残留旧图。
    if settings.workers == 1:
        reset_output(settings.output_dir)
    # 并行批次已在启动时刷新专用目录；保留逐条件 progress.jsonl。
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
        "dfe_phase_definition": "Whole DFE row including target, every enabled interferer, thermal noise and frontend residual; estimated phase is an externally supplied inverse rotation.",
        "egc_phase_estimate_source": "preset_assumed_target_response_phase",
        "source_sha256": source_hashes(),
    }
    _write_json(settings.output_dir / "manifest.json", manifest)
    # summary 用于看数值/后续重绘；verification 用于逐项查看判据及失败原因。
    _write_json(settings.output_dir / "summary.json", {
        **summary, "cases": case_conditions, "receiver_config_file": "manifest.json",
    })
    _write_json(settings.output_dir / "verification.json", summary["verification"])
    rows = summary["rows"]
    with (settings.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        # 不同实验条件可有不同指标列；按所有行的字段并集写出。
        hidden_config_columns = {"acquisition_threshold"}
        columns = list(dict.fromkeys(key for row in rows for key in row
                                    if key not in hidden_config_columns))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items() if key in columns} for row in rows)
    if "pattern_gain" in summary:
        # 一行一个 TC：跨seed汇总与逐seed原始行分开，便于直接筛选。
        records = []
        for result in summary["pattern_gain"]["cases"].values():
            tc = result["case"]
            observations = [row for row in rows if row["case"] == tc
                            and row["mode"] == result["mode"]]
            # 按物理天线名输出本条注入增益。
            antenna_gains = dict(zip(
                observations[0]["antennas"].split(","),
                observations[0]["input_gain_db_by_antenna"]
            ))
            antenna_columns = {f"{antenna}_gain_db": gain
                               for antenna, gain in antenna_gains.items()}
            records.append({
                "tc": tc, "mode": result["mode"],
                "receiver_role": result["receiver_role"],
                "phase_estimate_source": result["phase_estimate_source"],
                "n": result["seeds"],
                **antenna_columns,
                "acquisition_peak_ratio_mean": result["acquisition_peak_ratio"]["mean"],
                "acquisition_peak_ratio_std": result["acquisition_peak_ratio"]["std"],
                "tracking_cn0_mean_db_hz": result["tracking_cn0_observed_db_hz"]["mean"],
                "tracking_cn0_std_db_hz": result["tracking_cn0_observed_db_hz"]["std"],
                "tracking_cn0_valid_seed_count": result["tracking_cn0_observed_db_hz"]["valid_count"],
                "tracking_cn0_missing_count": result["tracking_cn0_missing_count"],
                # 分量收益和热噪声 SNR 输出暂时停用。
                # "measured_gain_mean_db": result["thermal_snr_gain_db"]["mean"],
                # "measured_gain_std_db": result["thermal_snr_gain_db"]["std"],
                # "target_output_gain_mean_db": result["target_output_gain_db"]["mean"],
                # "target_output_gain_std_db": result["target_output_gain_db"]["std"],
                # "thermal_snr_mean_db": result["thermal_snr_db"]["mean"],
                # "thermal_snr_std_db": result["thermal_snr_db"]["std"],
                "acquisition_success_count": result["acquisition_success_count"],
                "acquisition_correct_count": result["acquisition_correct_count"],
                "tracking_success_count": result["tracking_success_count"],
            })
        with (settings.output_dir / "case_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in record.items()} for record in records)
    if "mutual_coupling" in summary:
        records = []
        for case in summary["mutual_coupling"]["cases"].values():
            record = {key: case[key] for key in (
                "case", "mode", "receiver_role", "phase_estimate_source",
                "antennas", "seeds", "true_coupling_db",
                "true_coupling_phase_deg", "assumed_coupling_db", "assumed_coupling_phase_deg",
                "acquisition_threshold_pass_count", "acquisition_correct_count",
                "tracking_success_count", "tracking_cn0_missing_count",
            )}
            record.update({
                # 分量收益和热噪声 SNR 输出暂时停用。
                # "thermal_snr_mean_db": case["thermal_snr_db"]["mean"],
                # "thermal_snr_std_db": case["thermal_snr_db"]["std"],
                # "thermal_snr_gain_mean_db": case["thermal_snr_gain_db"]["mean"],
                # "thermal_snr_gain_std_db": case["thermal_snr_gain_db"]["std"],
                "acquisition_peak_ratio_mean": case["acquisition_peak_ratio"]["mean"],
                "acquisition_peak_ratio_std": case["acquisition_peak_ratio"]["std"],
                "tracking_cn0_mean_db_hz": case["tracking_cn0_db_hz"]["mean"],
                "tracking_cn0_std_db_hz": case["tracking_cn0_db_hz"]["std"],
                "tracking_cn0_valid_seed_count": case["tracking_cn0_db_hz"]["valid_count"],
            })
            records.append(record)
        with (settings.output_dir / "case_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in record.items()} for record in records)
    if "phase_center" in summary:
        # 日常只看精简主表；完整统计、坐标和核验信息仍保留在JSON。
        records = phase_center_summary_records(list(summary["phase_center"]["cases"].values()))
        with (settings.output_dir / "case_summary.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in record.items()} for record in records)


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
        raise AssertionError("validation verification failed; inspect verification.json")
    return summary


if __name__ == "__main__":
    main()
