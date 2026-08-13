"""Strict, evidence-only Markdown reporting for spatial-study artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


_REQUIRED_TABLES = (
    "baseline_rankings", "direction_scan", "power_scan", "robustness_scans",
    "combined_errors", "multi_jammer", "full_chain_validation", "full_chain_cn0",
)


def build_summary(
    tables: Mapping[str, Sequence[Mapping[str, object]]], manifest: Mapping[str, object], selection: Mapping[str, object],
) -> dict[str, object]:
    """Summarize saved tables without recomputing simulation or scientific values."""

    for name in _REQUIRED_TABLES:
        if name not in tables:
            raise ValueError(f"missing required table: {name}")
    rankings = sorted(tables["baseline_rankings"], key=lambda row: (int(row.get("rank", 10**9)), _combo(row)))
    by_level: dict[str, list[float]] = defaultdict(list)
    for row in tables["combined_errors"]:
        value = _finite(row.get("output_sinr_db", row.get("mvdr_output_sinr_db")))
        if value is not None:
            by_level[str(row.get("level", "unknown"))].append(value)
    full_chain: dict[str, dict[str, object]] = {}
    for mode in sorted({str(row.get("mode", "")) for row in tables["full_chain_validation"]}):
        rows = [row for row in tables["full_chain_validation"] if str(row.get("mode", "")) == mode]
        cn0 = [_finite(row.get("final_cn0_db_hz")) for row in rows]
        sinr = [_finite(row.get("analytic_output_sinr_db")) for row in rows]
        full_chain[mode] = {
            "run_count": len(rows), "acquisition_rate": _rate(rows, "acquired"), "tracking_rate": _rate(rows, "tracking"),
            "analytic_output_sinr_db": _statistics(sinr), "final_cn0_db_hz": _statistics(cn0),
            "unavailable_reasons": sorted({str(row.get("reason", row.get("status", "unavailable"))) for row in rows if not bool(row.get("tracking"))}),
        }
    cn0_by_mode: dict[str, object] = {}
    for mode in sorted({str(row.get("mode", "")) for row in tables["full_chain_cn0"]}):
        values = [_finite(row.get("cn0_db_hz")) for row in tables["full_chain_cn0"] if str(row.get("mode", "")) == mode]
        cn0_by_mode[mode] = _statistics(values)
    ranked = [_ranking_snapshot(row) for row in rankings]
    output: dict[str, object] = {
        "study": {"name": manifest.get("name"), "status": manifest.get("status"), "git_sha": manifest.get("git_sha"), "config_sha256": manifest.get("config_sha256", manifest.get("configuration_sha256"))},
        "baseline": {"rankings": ranked, "winning": ranked[0] if ranked else None, "median": ranked[(len(ranked) - 1) // 2] if ranked else None, "losing": ranked[-1] if ranked else None},
        "knee": {"status": "unavailable" if dict(selection.get("unselected_roles", {})).get("knee") else "selected", "reason": dict(selection.get("unselected_roles", {})).get("knee")},
        "combined_errors": {level: _statistics(values) for level, values in sorted(by_level.items())},
        "multi_jammer": [_multi_snapshot(row) for row in sorted(tables["multi_jammer"], key=lambda row: (str(row.get("template", "")), _combo(row), str(row.get("mode", ""))))],
        "full_chain": full_chain, "cn0_curves": cn0_by_mode,
        "selection": _strict(selection),
    }
    strict = _strict(output)
    return json.loads(json.dumps(strict, ensure_ascii=False, allow_nan=False))


def render_report(
    summary: Mapping[str, object], manifest: Mapping[str, object], figure_paths: Mapping[str, Path], table_paths: Mapping[str, Path],
) -> str:
    """Render the approved Chinese report sequence with confined relative links."""

    figures = _relative_paths(figure_paths)
    tables = _relative_paths(table_paths)
    study = dict(summary.get("study", {}))
    baseline = dict(summary.get("baseline", {}))
    config_hash = study.get("config_sha256", manifest.get("config_sha256", manifest.get("configuration_sha256", "不可用")))
    lines = [
        "# 空间抗干扰研究报告", "", "## 概览", "",
        "本报告是基于已保存规范化结果的**比较性研究**，用于比较模型内不同组合与算法；不代表真实 Mate 70 产品性能。",
        f"研究状态：`{study.get('status', manifest.get('status', 'unknown'))}`；Git SHA：`{study.get('git_sha', manifest.get('git_sha', 'unknown'))}`；配置哈希：`{config_hash}`。", "",
        "## 基线排名", "", _ranking_text(baseline), _link_line("基线图", figures.get("baseline_rankings")),
        "", "## 空间全景", "", _link_line("方向热力图", figures.get("direction_heatmaps")),
        "", "## 功率", "", _link_line("功率曲线", figures.get("power_curves")),
        "", "## 鲁棒性", "", _knee_text(dict(summary.get("knee", {}))), _link_line("鲁棒性曲线", figures.get("robustness_curves")),
        "", "## 组合误差", "", _combined_text(dict(summary.get("combined_errors", {}))), _link_line("组合误差分布", figures.get("combined_error_distributions")),
        "", "## 多干扰压力", "", _link_line("多干扰比较", figures.get("multi_jammer_comparison")),
        "", "## RF 到跟踪验证", "", "完整链路字段保留 `analytic_output_sinr_db` 与 `final_cn0_db_hz`；捕获或跟踪失败是有效结果，并以不可用原因呈现。", _full_chain_text(dict(summary.get("full_chain", {}))), _link_line("完整链路验证", figures.get("full_chain_validation")), _link_line("C/N₀ 验证", figures.get("cn0_validation")),
        "", "## 结论与限制", "", "模型包括理想角点子阵、窄带各向同性流形、静态 CW 干扰、SINGLE/EGC/MVDR、解析空间指标和 RF/ADC/DFE/捕获/跟踪/C/N₀ 链路。", "", "### 未建模", "", "- 真实 Mate 70 天线位置、方向图、极化、互耦和人体握持效应。", "- 多径、NLOS、动态到达方向和宽带/调制干扰。", "- 多颗卫星、导航电文、定位解算与 PVT，以及实时权值更新和硬件实现限制。",
        "", "## 复现", "", "以下链接指向本研究目录中的固定工件：",
    ]
    lines.extend(_link_line(name, path) for name, path in sorted(tables.items()))
    lines.extend(_link_line(name, path) for name, path in sorted(figures.items()) if name != "baseline_rankings")
    return "\n".join(line for line in lines if line is not None) + "\n"


def verify_report_links(report: str, report_path: Path) -> None:
    """Verify every local Markdown link is relative and exists beside the report."""

    root = Path(report_path).parent.resolve()
    for target in _markdown_targets(report):
        candidate = (root / target).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"report link escapes report directory: {target}") from error
        if not candidate.is_file():
            raise ValueError(f"report link target does not exist: {target}")


def _combo(row: Mapping[str, object]) -> str:
    values = row.get("active_antennas", ())
    return "+".join(str(value) for value in values) if isinstance(values, Sequence) and not isinstance(values, str) else str(values)


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _statistics(values: Sequence[float | None]) -> dict[str, object]:
    finite = [value for value in values if value is not None]
    if not finite:
        return {"count": 0, "p10": None, "median": None, "minimum": None, "maximum": None}
    data = np.asarray(finite, dtype=float)
    return {"count": len(finite), "p10": float(np.percentile(data, 10)), "median": float(np.median(data)), "minimum": float(np.min(data)), "maximum": float(np.max(data))}


def _rate(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    return None if not rows else float(sum(bool(row.get(field)) for row in rows) / len(rows))


def _ranking_snapshot(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "combination": _combo(row), "label": row.get("display_label"), "rank": row.get("rank"),
        "p10_all_db": _finite(row.get("p10_all_db")), "median_all_db": _finite(row.get("median_all_db")),
        "minimum_all_db": _finite(row.get("minimum_all_db")),
        "p10_resolvable_db": _finite(row.get("p10_resolvable_db")),
        "median_resolvable_db": _finite(row.get("median_resolvable_db")),
        "minimum_resolvable_db": _finite(row.get("minimum_resolvable_db")),
    }


def _multi_snapshot(row: Mapping[str, object]) -> dict[str, object]:
    return {"template": row.get("template"), "combination": _combo(row), "mode": row.get("mode"), "output_sinr_db": _finite(row.get("output_sinr_db")), "worst_interferer_gain": _finite(row.get("worst_interferer_gain"))}


def _strict(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _relative_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, raw_path in paths.items():
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"artifact link must be a confined relative path: {name}")
        result[str(name)] = path.as_posix()
    return result


def _markdown_targets(report: str) -> list[str]:
    targets: list[str] = []
    for line in report.splitlines():
        if "](" not in line or not line.endswith(")"):
            continue
        target = line.rsplit("](", 1)[1][:-1]
        if target and not target.startswith(("http://", "https://", "#")):
            targets.append(target)
    return targets


def _link_line(label: str, path: str | None) -> str:
    return f"[{label}]({path})" if path else f"{label}：未提供工件。"


def _ranking_text(baseline: Mapping[str, object]) -> str:
    winner = baseline.get("winning")
    return "无基线排名行。" if not isinstance(winner, Mapping) else f"首选组合：`{winner.get('combination')}`；可分辨区域 p10：`{winner.get('p10_resolvable_db')}` dB；中位数：`{winner.get('median_resolvable_db')}` dB；全方向最差值：`{winner.get('minimum_all_db')}` dB。"


def _knee_text(knee: Mapping[str, object]) -> str:
    return "性能拐点不可用：" + str(knee.get("reason")) if knee.get("status") == "unavailable" else "性能拐点已由选择记录确定。"


def _combined_text(combined: Mapping[str, object]) -> str:
    if not combined:
        return "没有有限组合误差样本。"
    return "；".join(f"{level}：中位数 {dict(stats).get('median')} dB" for level, stats in combined.items() if isinstance(stats, Mapping)) + "。"


def _full_chain_text(full_chain: Mapping[str, object]) -> str:
    if not full_chain:
        return "没有完整链路汇总行。"
    return "；".join(f"{mode.upper()}：捕获率 {dict(stats).get('acquisition_rate')}，跟踪率 {dict(stats).get('tracking_rate')}" for mode, stats in full_chain.items() if isinstance(stats, Mapping)) + "。"
