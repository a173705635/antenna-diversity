"""Strict, evidence-only Markdown reporting for spatial-study artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from antenna_diversity.array.geometry import CORNER_ORDER


_REQUIRED_TABLES = (
    "baseline_rankings", "direction_scan", "power_scan", "robustness_scans",
    "combined_errors", "multi_jammer", "full_chain_validation", "full_chain_cn0",
    "antenna_diagnostics",
)


def build_summary(
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    manifest: Mapping[str, object],
    selection: Mapping[str, object],
    *,
    active_antennas: Sequence[str],
) -> dict[str, object]:
    """Summarize saved tables without recomputing simulation or scientific values."""

    for name in _REQUIRED_TABLES:
        if name not in tables:
            raise ValueError(f"missing required table: {name}")
    rankings = _ranked_rows(tables["baseline_rankings"])
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
    roles = {"best": ranked[0] if ranked else None, "typical": ranked[(len(ranked) - 1) // 2] if ranked else None, "worst": ranked[-1] if ranked else None}
    _validate_selection_roles(roles, selection)
    output: dict[str, object] = {
        "study": {"name": manifest.get("name"), "status": manifest.get("status"), "git_sha": manifest.get("git_sha"), "config_sha256": manifest.get("config_sha256", manifest.get("configuration_sha256"))},
        "baseline": {"rankings": ranked, **roles},
        "knee": _knee_summary(selection),
        "combined_errors": {level: _statistics(values) for level, values in sorted(by_level.items())},
        "multi_jammer": [_multi_snapshot(row) for row in sorted(tables["multi_jammer"], key=lambda row: (str(row.get("template", "")), _combo(row), str(row.get("mode", ""))))],
        "full_chain": full_chain, "cn0_curves": cn0_by_mode,
        "antenna_model": _antenna_summary(
            tables["antenna_diagnostics"], active_antennas
        ),
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
    antenna_model = dict(summary.get("antenna_model", {}))
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
        "", "## 多干扰压力", "", _multi_text(summary.get("multi_jammer", ())), _link_line("多干扰比较", figures.get("multi_jammer_comparison")),
        "", "## 手机天线模型证据", "", _antenna_text(antenna_model),
        _link_line("单阵元方向图", figures.get("element_patterns")),
        _link_line("极化匹配损耗", figures.get("polarization_losses")),
        _link_line("互耦幅相矩阵", figures.get("coupling_matrices")),
        _link_line("遮挡衰减对比", figures.get("blockage_comparison")),
        "", "## RF 到跟踪验证", "", "完整链路字段保留 `analytic_output_sinr_db` 与 `final_cn0_db_hz`；捕获或跟踪失败是有效结果，并以不可用原因呈现。", _full_chain_text(dict(summary.get("full_chain", {}))), _link_line("完整链路验证", figures.get("full_chain_validation")), _link_line("C/N₀ 验证", figures.get("cn0_validation")),
        "", "## 结论与限制", "", "仿真框架可表示四角子阵、窄带复数双极化阵元响应、安装姿态、可配置遮挡与互耦、真实/假设模型失配、静态 CW 干扰、SINGLE/EGC/MVDR、解析空间指标和 RF/ADC/DFE/捕获/跟踪/C/N₀ 链路。本次工件实际启用的天线来源由上节明确标记；参数化模型不代表真实 Mate 70 产品测量。", "", "### 未建模", "", "- 多径、NLOS、运动与动态到达方向，以及非 CW、宽带、扫频、脉冲或调制干扰。", "- 多颗卫星、多 PRN、导航电文、定位解算与 PVT，以及实时权值更新和硬件实现限制。", "- 互耦导致的相关外部天线噪声，以及机壳/人体的全波平台散射和产品级电磁验证。",
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
        "combination": _combo(row), "label": row.get("display_label"), "rank": row["rank"],
        "p10_all_db": _finite(row.get("p10_all_db")), "median_all_db": _finite(row.get("median_all_db")),
        "minimum_all_db": _finite(row.get("minimum_all_db")),
        "p10_resolvable_db": _finite(row.get("p10_resolvable_db")),
        "median_resolvable_db": _finite(row.get("median_resolvable_db")),
        "minimum_resolvable_db": _finite(row.get("minimum_resolvable_db")),
    }


def _multi_snapshot(row: Mapping[str, object]) -> dict[str, object]:
    return {"template": row.get("template"), "combination": _combo(row), "mode": row.get("mode"), "output_sinr_db": _finite(row.get("output_sinr_db")), "worst_interferer_gain": _finite(row.get("worst_interferer_gain"))}


def _antenna_summary(
    rows: Sequence[Mapping[str, object]],
    active_antennas: Sequence[str],
) -> dict[str, object]:
    sources = {str(row["model_source"]) for row in rows if row.get("model_source")}
    labels = {str(row["model_label"]) for row in rows if row.get("model_label")}
    if len(sources) > 1 or len(labels) > 1:
        raise ValueError("antenna diagnostics contain inconsistent model metadata")
    true_digests = {
        str(row["measured_pattern_sha256"])
        for row in rows
        if row.get("model_view") == "true" and row.get("measured_pattern_sha256")
    }
    if len(true_digests) > 1:
        raise ValueError("antenna diagnostics contain inconsistent measured pattern digests")
    configured = tuple(active_antennas)
    if (
        not 2 <= len(configured) <= 4
        or any(not isinstance(corner, str) for corner in configured)
        or len(set(configured)) != len(configured)
        or any(corner not in CORNER_ORDER for corner in configured)
    ):
        raise ValueError("active_antennas must be a unique two-to-four corner subset")
    return {
        "source": next(iter(sources), "unknown"),
        "label": next(iter(labels), "unavailable"),
        "measured_pattern_sha256": next(iter(true_digests), None),
        "active_antennas": list(configured),
    }


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
    best = baseline.get("best")
    if not isinstance(best, Mapping):
        return "无基线排名行。"
    fragments = []
    for label, key in (("最佳组合", "best"), ("典型组合", "typical"), ("最差组合", "worst")):
        snapshot = baseline.get(key)
        if not isinstance(snapshot, Mapping):
            continue
        fragments.append(
            f"{label}：`{snapshot.get('combination')}`；全方向 p10/中位数/最小值："
            f"`{snapshot.get('p10_all_db')}`/`{snapshot.get('median_all_db')}`/`{snapshot.get('minimum_all_db')}` dB；"
            f"可分辨区域 p10/中位数/最小值：`{snapshot.get('p10_resolvable_db')}`/"
            f"`{snapshot.get('median_resolvable_db')}`/`{snapshot.get('minimum_resolvable_db')}` dB。"
        )
    return " ".join(fragments)


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


def _multi_text(outcomes: object) -> str:
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)) or not outcomes:
        return "没有多干扰结果。"
    evidence = []
    for raw in outcomes:
        if not isinstance(raw, Mapping):
            continue
        template, combination, mode = raw.get("template"), raw.get("combination"), raw.get("mode")
        sinr, gain = raw.get("output_sinr_db"), raw.get("worst_interferer_gain")
        if sinr is None or gain is None:
            evidence.append(f"{template}/{combination}/{mode}：不可用。")
        else:
            evidence.append(f"{template}/{combination}/{mode}：输出 SINR {sinr} dB，最差干扰增益 {gain}。")
    return " ".join(evidence) or "没有多干扰结果。"


def _antenna_text(model: Mapping[str, object]) -> str:
    active = model.get("active_antennas", ())
    active_text = "+".join(str(value) for value in active) if isinstance(active, Sequence) and not isinstance(active, str) else str(active)
    digest = model.get("measured_pattern_sha256")
    digest_text = f"；测量文件 SHA-256：`{digest}`" if digest else ""
    return (
        f"模型来源：`{model.get('source', 'unknown')}`；标签："
        f"`{model.get('label', 'unavailable')}`；诊断阵元：`{active_text}`"
        f"{digest_text}。以下图均由基线阶段保存的标量诊断行生成，不在报告阶段重算模型。"
    )


def _ranked_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    ranks: list[int] = []
    for row in rows:
        rank = row.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("baseline ranking requires a positive integer rank")
        ranks.append(rank)
    if sorted(ranks) != list(range(1, len(rows) + 1)):
        raise ValueError("baseline ranking ranks must be contiguous one-based integers")
    return sorted(rows, key=lambda row: int(row["rank"]))


def _selection_reasons(selection: Mapping[str, object]) -> set[str]:
    cases = selection.get("cases", ())
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise ValueError("selection cases must be a sequence")
    reasons: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("selection case must be a mapping")
        raw = case.get("selection_reasons", case.get("role", ()))
        if isinstance(raw, str):
            reasons.add(raw)
        elif isinstance(raw, Sequence):
            reasons.update(str(value) for value in raw)
    return reasons


def _validate_selection_roles(roles: Mapping[str, object], selection: Mapping[str, object]) -> None:
    _selection_reasons(selection)
    cases = selection.get("cases", ())
    for role, snapshot in roles.items():
        if not isinstance(snapshot, Mapping):
            continue
        expected = str(snapshot["combination"])
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            raw = case.get("selection_reasons", case.get("role", ()))
            reasons = {raw} if isinstance(raw, str) else {str(value) for value in raw} if isinstance(raw, Sequence) else set()
            if role in reasons and _combo(case) != expected:
                raise ValueError(f"selection {role} conflicts with ranked baseline")


def _knee_summary(selection: Mapping[str, object]) -> dict[str, object]:
    reasons = _selection_reasons(selection)
    raw_unselected = selection.get("unselected_roles", {})
    if not isinstance(raw_unselected, Mapping):
        raise ValueError("selection unselected_roles must be a mapping")
    unavailable = raw_unselected.get("knee")
    if "knee" in reasons and unavailable is not None:
        raise ValueError("knee selection conflicts with unavailable reason")
    if "knee" in reasons:
        return {"status": "selected", "reason": None}
    return {"status": "unavailable", "reason": str(unavailable) if unavailable is not None else "not selected"}
