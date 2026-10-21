"""按各验证实验的物理问题绘图；只读取逐 seed 接收结果。"""

from collections import Counter
import hashlib
from pathlib import Path
import re

import numpy as np

from experiments.interference.validation_support import ValidationSettings


def _filename_label(value: str) -> str:
    return re.sub(r"[^\w]+", "_", str(value).casefold()).strip("_")


def _distinct_plot_names(base_names: dict) -> dict:
    """扫描标签归一后重名时追加稳定摘要，避免覆盖图片。"""
    counts = Counter(base_names.values())
    return {
        label: (base if counts[base] == 1 else
                f"{base}_{hashlib.sha256(str(label).encode('utf-8')).hexdigest()[:8]}")
        for label, base in base_names.items()
    }


def _mode_label(row: dict) -> str:
    mode = row["mode"]
    if mode == "egc":
        source = row["phase_estimate_source"]
        if source == "preset_assumed_target_response_phase":
            return "EGC (preset phase)"
        if source == "external_input":
            return "EGC (external phase)"
        return f"EGC ({source})"
    if mode == "mvdr":
        return "MVDR (known resp)"
    return mode.upper()


def _case_groups(rows: list[dict]) -> list[list[dict]]:
    """同一物理用例、场景和算法内跨 seed 汇总；绝不按相同横轴合并用例。"""
    groups = {}
    for row in rows:
        key = (row["case"], row["scene"], row["mode"])
        groups.setdefault(key, []).append(row)
    return list(groups.values())


def _mean_metric(group: list[dict], metric: str) -> float:
    if metric == "acquisition_correct_flag":
        return float(np.mean([row[metric] for row in group]))
    available = [row[metric] for row in group if row[metric] is not None]
    return float(np.mean(available)) if available else np.nan


def _scan_points(rows: list[dict], metric: str) -> list[tuple[float, float]]:
    """每个 case 给出一个物理扫描点；只在同一 case 内求指标的 seed 均值。"""
    return sorted((group[0]["scan_value"], _mean_metric(group, metric))
                  for group in _case_groups(rows))


def _plot_points(axis, points: list[tuple], label: str, index: int,
                 connect_flag: bool = True) -> None:
    if not points:
        return
    x_values, y_values = zip(*points)
    axis.plot(x_values, y_values,
              linestyle=("-", "--", ":", "-.")[index % 4] if connect_flag else "None",
              marker=("o", "s", "^", "D")[index % 4], markerfacecolor="none",
              linewidth=1.5, label=label)


def _finish_axis(axis, xlabel: str, metric: str) -> None:
    ylabel = {"tracking_cn0_mean_db_hz": "TRK CN0 (dB-Hz)",
              "acquisition_correct_flag": "ACQ OK rate",
              "acquisition_metric": "ACQ Ratio"}[metric]
    axis.set(xlabel=xlabel, ylabel=ylabel)
    if metric == "tracking_cn0_mean_db_hz":
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        if any(np.isfinite(value) for line in axis.lines for value in line.get_ydata()):
            axis.tick_params(axis="y", labelleft=True)
            axis.text(0.02, 0.02, "Gaps = missing CN0", fontsize=8,
                      transform=axis.transAxes)
        else:
            axis.text(0.5, 0.5, "No valid CN0", ha="center",
                      transform=axis.transAxes)
            axis.tick_params(axis="y", labelleft=False)
    if metric == "acquisition_correct_flag":
        axis.set_ylim(-0.04, 1.04)
    axis.grid(alpha=0.3)
    if axis.lines:
        axis.legend(fontsize=8)


def _save_figure(fig, settings: ValidationSettings, name: str, paths: list[Path]) -> None:
    import matplotlib.pyplot as plt

    figure_path = settings.output_dir / f"{name}.png"
    fig.savefig(figure_path, dpi=150)
    paths.append(figure_path)
    if not settings.show_plots_flag:
        plt.close(fig)


def _phase_direction_gain(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """相位比较补偿三组；方向和增益扫描在同图比较各算法。"""
    import matplotlib.pyplot as plt

    paths = []
    factors = list(dict.fromkeys((row["scan_label"], row["scan_unit"]) for row in rows))
    factor_names = _distinct_plot_names({
        factor: _filename_label("_".join(part for part in factor if part))
        for factor in factors
    })
    for factor in factors:
        factor_rows = [row for row in rows if (row["scan_label"], row["scan_unit"]) == factor]
        for scene in dict.fromkeys(row["scene"] for row in factor_rows):
            scene_rows = [row for row in factor_rows if row["scene"] == scene]
            modes = list(dict.fromkeys(row["mode"] for row in scene_rows))
            phase_flag = ("phase" in factor[0].casefold() or
                          bool({row["series"] for row in scene_rows} &
                               {"uncalibrated", "calibrated"}))
            for metric, metric_name in (("acquisition_metric", "acquisition_peak_ratio"),
                                        ("tracking_cn0_mean_db_hz", "tracking_cn0_mean")):
                if phase_flag:
                    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4.5),
                                             squeeze=False, constrained_layout=True)
                    for mode_index, mode in enumerate(modes):
                        axis = axes[0, mode_index]
                        mode_rows = [row for row in scene_rows if row["mode"] == mode]
                        series_points = {}
                        for series_index, series in enumerate(dict.fromkeys(
                                row["series"] for row in mode_rows)):
                            selected = [row for row in mode_rows if row["series"] == series]
                            points = _scan_points(selected, metric)
                            series_points[series] = points
                            _plot_points(axis, points, series, series_index)
                        axis.set_title(_mode_label(mode_rows[0]))
                        xlabel = {"DFE channel phase": "DFE Phase",
                                  "target true azimuth": "True Az",
                                  "target azimuth": "Target Az"}.get(factor[0], factor[0])
                        _finish_axis(axis, f"{xlabel} ({factor[1]})", metric)
                        if metric == "acquisition_metric":
                            for threshold in sorted({row["acquisition_threshold"] for row in mode_rows}):
                                axis.axhline(threshold, color="#888888", linestyle="-.",
                                             label=f"Thr {threshold:g}")
                            axis.legend(fontsize=8)
                        if "baseline" in series_points and "calibrated" in series_points:
                            baseline = series_points["baseline"]
                            calibrated = series_points["calibrated"]
                            if ([point[0] for point in baseline] ==
                                    [point[0] for point in calibrated]
                                    and np.any(np.isfinite([point[1] for point in baseline]))
                                    and np.allclose([point[1] for point in baseline],
                                                    [point[1] for point in calibrated],
                                                    rtol=0, atol=1e-8, equal_nan=True)):
                                axis.text(0.02, 0.96, "Baseline = calibrated",
                                          va="top", fontsize=8, transform=axis.transAxes)
                    fig.suptitle(f"{scene}: baseline / injected phase / preset correction")
                else:
                    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
                    for mode_index, mode in enumerate(modes):
                        selected = [row for row in scene_rows if row["mode"] == mode]
                        points = _scan_points(selected, metric)
                        _plot_points(axis, points, _mode_label(selected[0]), mode_index,
                                     connect_flag=len({point[0] for point in points}) == len(points))
                    xlabel = {"DFE channel phase": "DFE Phase",
                              "target true azimuth": "True Az",
                              "target azimuth": "Target Az"}.get(factor[0], factor[0])
                    _finish_axis(axis, f"{xlabel} ({factor[1]})", metric)
                    if metric == "acquisition_metric":
                        for threshold in sorted({row["acquisition_threshold"] for row in scene_rows}):
                            axis.axhline(threshold, color="#888888", linestyle="-.",
                                         label=f"Thr {threshold:g}")
                        axis.legend(fontsize=8)
                    axis.set_title(scene)
                _save_figure(fig, settings, f"phase_direction_gain__{_filename_label(scene)}"
                      f"__{factor_names[factor]}__{metric_name}", paths)
    return paths


def _antenna_gain(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """前两根阵元各自衰减，0/0 作为两个面板的共同基准。"""
    import matplotlib.pyplot as plt

    paths = []
    antennas = rows[0]["antennas"].split(",")[:2]
    for metric, metric_name in (("tracking_cn0_mean_db_hz", "tracking_cn0_mean"),
                                ("acquisition_correct_flag", "acquisition_correct_rate")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True,
                                 constrained_layout=True)
        for antenna_index, axis in enumerate(axes):
            other_index = 1 - antenna_index
            panel_rows = [row for row in rows
                          if row["input_gain_db_by_antenna"][other_index] == 0]
            for mode_index, mode in enumerate(dict.fromkeys(row["mode"] for row in panel_rows)):
                selected = [row for row in panel_rows if row["mode"] == mode]
                points = sorted((float(group[0]["input_gain_db_by_antenna"][antenna_index]),
                                 _mean_metric(group, metric)) for group in _case_groups(selected))
                _plot_points(axis, points, _mode_label(selected[0]), mode_index)
            axis.set_title(f"{antennas[antenna_index]} attenuation; {antennas[other_index]} = 0 dB")
            _finish_axis(axis, f"{antennas[antenna_index]} input gain (dB)", metric)
        _save_figure(fig, settings, f"antenna_gain__single_antenna__{metric_name}", paths)
    return paths


def _mutual_coupling(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """真实互耦电压比和相位分面扫描；共同交点在两面板复用。"""
    import matplotlib.pyplot as plt

    paths = []
    for scene in dict.fromkeys(row["scene"] for row in rows):
        scene_rows = [row for row in rows if row["scene"] == scene]
        groups = _case_groups(scene_rows)
        modes = list(dict.fromkeys(row["mode"] for row in scene_rows))
        strength_phases = [row["true_coupling_phase_deg"] for row in scene_rows
                           if row["series"] in ("strength", "strength_phase")
                           and row["true_coupling_phase_deg"] is not None]
        phase_strengths_db = [row["true_coupling_db"] for row in scene_rows
                              if row["series"] in ("phase", "strength_phase")]
        strength_title = (f"Strength; true phase {strength_phases[0]:g}°"
                          if strength_phases else "Strength; coupling off")
        phase_title = (f"Phase; true voltage ratio {10 ** (phase_strengths_db[0] / 20):g} V/V"
                       if phase_strengths_db else "Phase; no phase cases")
        for metric_name, metric in (("tracking_cn0_mean", "tracking_cn0_mean_db_hz"),
                                    ("acquisition_correct_rate", "acquisition_correct_flag")):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True,
                                     constrained_layout=True)
            for axis, series, xlabel, title in (
                (axes[0], ("strength", "strength_phase"), "True coupling voltage ratio (V/V)",
                 strength_title),
                (axes[1], ("phase", "strength_phase"), "True coupling phase (deg)",
                 phase_title),
            ):
                for mode_index, mode in enumerate(modes):
                    selected = [group for group in groups
                                if group[0]["mode"] == mode and group[0]["series"] in series]
                    points = sorted((
                        (group[0]["true_coupling_phase_deg"] if axis is axes[1]
                         else group[0]["scan_value"]), _mean_metric(group, metric)
                    ) for group in selected)
                    if selected:
                        _plot_points(axis, points, _mode_label(selected[0][0]), mode_index)
                axis.set_title(title)
                _finish_axis(axis, xlabel, metric)
            # 0 是真实关闭的电压比，不映射成某个有限 dB 值。
            if any(group[0]["series"] in ("strength", "strength_phase")
                   and group[0]["scan_value"] == 0 for group in groups):
                axes[0].annotate("Off", (0, 0), xycoords=("data", "axes fraction"),
                                 xytext=(0, -34), textcoords="offset points", ha="center", fontsize=8)
            fig.suptitle(f"{scene}: true symmetric coupling; Nom uncoupled")
            _save_figure(fig, settings,
                         f"mutual_coupling__{_filename_label(scene)}__{metric_name}", paths)
    return paths


def _phase_center_offset(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """两根天线各自沿目标视线扫描，零偏移基准在两面板复用。"""
    import matplotlib.pyplot as plt

    paths = []
    for scene in dict.fromkeys(row["scene"] for row in rows):
        scene_rows = [row for row in rows if row["scene"] == scene]
        groups = _case_groups(scene_rows)
        modes = list(dict.fromkeys(row["mode"] for row in scene_rows))
        first, second = scene_rows[0]["antennas"].split(",")[:2]
        for metric_name, metric in (("tracking_cn0_mean", "tracking_cn0_mean_db_hz"),
                                    ("acquisition_correct_rate", "acquisition_correct_flag")):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True,
                                     constrained_layout=True)
            for axis, antenna, other in ((axes[0], first, second), (axes[1], second, first)):
                for mode_index, mode in enumerate(modes):
                    selected = [group for group in groups
                                if group[0]["mode"] == mode
                                and group[0]["series"] in ("baseline", antenna)]
                    points = sorted((group[0]["scan_value"], _mean_metric(group, metric))
                                    for group in selected)
                    if selected:
                        _plot_points(axis, points, _mode_label(selected[0][0]), mode_index)
                axis.set_title(f"{antenna} LOS offset; {other} = 0 mm")
                _finish_axis(axis, f"{antenna} true LOS offset (mm)", metric)
            fig.suptitle(f"{scene}: one antenna moves at a time; Nom baseline")
            _save_figure(fig, settings,
                         f"phase_center_offset__{_filename_label(scene)}__{metric_name}", paths)
    return paths


def plot_validation(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """按显式实验类型分派绘图；保留真实数轴与同用例 seed 统计。"""
    import matplotlib.pyplot as plt

    if not rows:
        return []
    plotters = {
        "phase_direction_gain": _phase_direction_gain,
        "antenna_gain": _antenna_gain,
        "mutual_coupling": _mutual_coupling,
        "phase_center_offset": _phase_center_offset,
    }
    paths = plotters[settings.experiment_type](rows, settings)
    if settings.show_plots_flag and plt.get_backend().lower() != "agg":
        plt.show()
    return paths
