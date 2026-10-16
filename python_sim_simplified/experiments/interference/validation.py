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
    ValidationSettings, plot_validation, run_validation,
)
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


def save_results(settings: ValidationSettings, summary: dict) -> None:
    """刷新本批次输出，保存执行配置、结果行和验收证据供追溯。"""
    # 仿真完成后才刷新本入口的专用输出，避免换扫描因素后残留旧图。
    if settings.workers == 1:
        reset_output(settings.output_dir)
    # 并行批次已在启动时刷新专用目录；保留逐条件 progress.jsonl。
    manifest = {
        # 清单用于追溯参数与代码版本，不是下一次仿真的配置输入。
        "settings": asdict(settings),
        "cases": summary["cases"],
        # run_validation 已按条件和 seed 保存实际执行配置，直接复用。
        "actual_receiver_configs": {
            case["name"]: case["receiver_configs"] for case in summary["cases"]
        },
        "dfe_phase_definition": "Whole DFE row including target, every enabled interferer, thermal noise and frontend residual; estimated phase is an externally supplied inverse rotation.",
        "source_sha256": source_hashes(),
    }
    _write_json(settings.output_dir / "manifest.json", manifest)
    # summary 用于看数值/后续重绘；verification 用于逐项查看判据及失败原因。
    _write_json(settings.output_dir / "summary.json", summary)
    _write_json(settings.output_dir / "verification.json", summary["verification"])
    rows = summary["rows"]
    with (settings.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        # 配对条件比普通条件多恢复误差列；普通条件对应单元格留空。
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)
    if "pattern_gain" in summary:
        # 一行一个 TC：跨seed汇总与逐seed原始行分开，便于直接筛选。
        records = []
        for tc, result in summary["pattern_gain"]["cases"].items():
            observations = [row for row in rows if row["case"] == tc]
            # 按物理天线名输出本条注入增益；衰减为增益的相反数。
            antenna_gains = dict(zip(
                observations[0]["antennas"].split(","),
                observations[0]["input_gain_db_by_antenna"]
            ))
            antenna_columns = {
                column: value
                for antenna, gain in antenna_gains.items()
                for column, value in (
                    (f"{antenna}_gain_db", gain), (f"{antenna}_attenuation_db", -gain)
                )
            }
            cn0_statistics = result["tracking_cn0_observed_db_hz"]
            records.append({
                "tc": tc, "mode": result["mode"], "n": result["seeds"],
                **antenna_columns,
                # 捕获峰值比对所有seed取均值，不剔除低于门限的样本。
                "acquisition_peak_ratio_mean": float(np.mean([
                    row["acquisition_metric"] for row in observations
                ])),
                "tracking_cn0_mean_db_hz": (
                    cn0_statistics["mean"] if cn0_statistics is not None else None
                ),
                "tracking_cn0_valid_seed_count": (
                    result["seeds"] - result["tracking_cn0_missing_count"]
                ),
                "status": result["status"],
                "expected_gain_db": result["expected_gain_db"],
                "measured_gain_mean_db": result["measured_gain_db"]["mean"],
                "gain_error_mean_db": result["gain_error_db"]["mean"],
                "expected_absolute_change_db": result["expected_absolute_change_db"],
                "absolute_snr_change_mean_db": result["absolute_snr_change_mean_db"],
                "input_target_change_max_error_db": result["input_target_change_max_error_db"],
                "input_thermal_change_max_error_db": result["input_thermal_change_max_error_db"],
                "weight_max_relative_error": result["weight_max_relative_error"],
                "acquisition_success_count": result["acquisition_success_count"],
                "tracking_success_count": result["tracking_success_count"],
                "tracking_cn0_missing_count": result["tracking_cn0_missing_count"],
            })
        with (settings.output_dir / "case_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in record.items()} for record in records)
    if "mutual_coupling" in summary:
        records = list(summary["mutual_coupling"]["cases"].values())
        with (settings.output_dir / "case_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in record.items()} for record in records)
        with (settings.output_dir / "model_checks.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            model_records = list(summary["model_checks"].values())
            writer = csv.DictWriter(stream, fieldnames=list(model_records[0]))
            writer.writeheader()
            writer.writerows(model_records)
    if "phase_center" in summary:
        # 逐PC/方法一行；逐seed原始行仍在summary.csv与summary.json。
        records = list(summary["phase_center"]["cases"].values())
        with (settings.output_dir / "case_summary.csv").open(
            "w", newline="", encoding="utf-8"
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
