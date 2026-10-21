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
from experiments.interference.coupling_support import sample_statistics
from experiments.interference.validation_plotting import plot_validation
from experiments.paths import reset_output


DETAIL_RESULT_COLUMNS = (
    "acquisition_metric", "acquired_flag", "acquisition_correct_flag",
    "acquired_code_error_chips", "acquired_doppler_error_hz",
    "tracking_success_flag", "tracking_status", "tracking_cn0_mean_db_hz",
    "tracking_cn0_valid_count",
)


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
        writer.writeheader()
        writer.writerows(records)


def _phase_center_row_condition(row: dict) -> str:
    """按物理天线名展示真实/名义 XYZ 偏移，单位 mm。"""
    descriptions = []
    for antenna in row["antennas"].split(","):
        true_xyz = ",".join(
            f"{round(value, 3) + 0.0:g}"
            for value in row["true_offset_xyz_mm_by_antenna"][antenna]
        )
        assumed_xyz = ",".join(
            f"{round(value, 3) + 0.0:g}"
            for value in row["assumed_offset_xyz_mm_by_antenna"][antenna]
        )
        descriptions.append(f"{antenna}: 真实({true_xyz})/估计({assumed_xyz})")
    return "; ".join(descriptions)


def _condition_columns(row: dict, experiment_type: str) -> dict:
    """只提取日常表需要的物理条件；完整配置和诊断仍留在 JSON。"""
    if experiment_type == "antenna_gain":
        return {f"{antenna}_gain_db": gain for antenna, gain in zip(
            row["antennas"].split(","), row["input_gain_db_by_antenna"])}
    if experiment_type == "mutual_coupling":
        return {key: row[key] for key in (
            "true_coupling_db", "true_coupling_phase_deg",
            "assumed_coupling_db", "assumed_coupling_phase_deg")}
    if experiment_type == "phase_center_offset":
        return {"condition_xyz_mm": _phase_center_row_condition(row)}
    conditions = {"scene": row["scene"], "series": row["series"]}
    if row["scan_value"] is not None:
        label = row["scan_label"]
        unit = row["scan_unit"]
        conditions[f"{label} ({unit})" if unit else label] = row["scan_value"]
    return conditions


def phase_center_summary_records(cases: list[dict]) -> list[dict]:
    """从当前 PCO 统计生成主表视图，不修改完整统计。"""
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
            "condition_xyz_mm": "; ".join(descriptions),
            "seeds": case["seeds"],
            "acquisition_correct_count": case["acquisition_correct_count"],
            "tracking_success_count": case["tracking_success_count"],
            "tracking_cn0_mean_db_hz": case["tracking_cn0_db_hz"]["mean"],
            "tracking_cn0_std_db_hz": case["tracking_cn0_db_hz"]["std"],
            "tracking_cn0_valid_seed_count": case["tracking_cn0_db_hz"]["valid_count"],
        })
    return records


def general_summary_records(rows: list[dict]) -> list[dict]:
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
            **_condition_columns(first_row, "phase_direction_gain"),
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
                **{key: case[key] for key in (
                    "true_coupling_db", "true_coupling_phase_deg",
                    "assumed_coupling_db", "assumed_coupling_phase_deg")},
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
        records = general_summary_records(rows)
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
