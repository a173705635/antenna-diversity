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
    """真实幅度、真实相位与固定真实条件下的名义失配分别比较。"""
    import matplotlib.pyplot as plt

    paths = []
    for scene in dict.fromkeys(row["scene"] for row in rows):
        scene_rows = [row for row in rows if row["scene"] == scene]
        accurate = [row for row in scene_rows
                    if row["true_coupling_db"] == row["assumed_coupling_db"]
                    and row["true_coupling_phase_deg"] == row["assumed_coupling_phase_deg"]]
        # 从实际条件选择幅度扫描的固定相位、相位扫描的固定幅度。
        phases = [row["true_coupling_phase_deg"] for row in accurate
                  if row["true_coupling_phase_deg"] is not None]
        strength_phase = min(phases) if phases else None
        strength = [row for row in accurate if row["true_coupling_db"] is None
                    or row["true_coupling_phase_deg"] == strength_phase]
        amplitude_phases = {}
        for row in accurate:
            if row["true_coupling_db"] is not None:
                amplitude_phases.setdefault(row["true_coupling_db"], set()).add(
                    row["true_coupling_phase_deg"])
        phase_amplitude = (max(amplitude_phases, key=lambda amplitude:
                               (len(amplitude_phases[amplitude]), amplitude))
                           if amplitude_phases else None)
        phase = [row for row in accurate if row["true_coupling_db"] == phase_amplitude
                 and phase_amplitude is not None]
        nominal_by_true = {}
        for row in scene_rows:
            true_key = (row["true_coupling_db"], row["true_coupling_phase_deg"])
            nominal_by_true.setdefault(true_key, set()).add(
                (row["assumed_coupling_db"], row["assumed_coupling_phase_deg"]))
        mismatch_true = max(nominal_by_true, key=lambda key: len(nominal_by_true[key]))
        mismatch = [row for row in scene_rows
                    if (row["true_coupling_db"], row["true_coupling_phase_deg"]) == mismatch_true
                    and mismatch_true[0] is not None]
        groups = (("true_strength", strength), ("true_phase", phase),
                  ("nominal_mismatch", mismatch))
        for group_name, group_rows in groups:
            if not group_rows:
                continue
            case_groups = _case_groups(group_rows)
            identities = list(dict.fromkeys((group[0]["case"], group[0]["scene"])
                                            for group in case_groups))
            representative = {identity: next(group[0] for group in case_groups
                                             if (group[0]["case"], group[0]["scene"]) == identity)
                              for identity in identities}
            if group_name == "true_strength":
                identities.sort(key=lambda identity: (
                    representative[identity]["true_coupling_db"] is not None,
                    representative[identity]["true_coupling_db"] or -100))
                labels = ["Off" if representative[key]["true_coupling_db"] is None else
                          f"{representative[key]['true_coupling_db']:g} dB" for key in identities]
            elif group_name == "nominal_mismatch":
                labels = [("Off" if representative[key]["assumed_coupling_db"] is None else
                           f"{representative[key]['assumed_coupling_db']:g} dB / "
                           f"{representative[key]['assumed_coupling_phase_deg']:g}°")
                          for key in identities]
            fig, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
            for axis, metric in zip(axes, ("tracking_cn0_mean_db_hz", "acquisition_correct_flag")):
                for mode_index, mode in enumerate(dict.fromkeys(row["mode"] for row in group_rows)):
                    selected = [group for group in case_groups if group[0]["mode"] == mode]
                    points = [((group[0]["true_coupling_phase_deg"] if group_name == "true_phase"
                                else identities.index((group[0]["case"], group[0]["scene"]))),
                               _mean_metric(group, metric)) for group in selected]
                    _plot_points(axis, sorted(points), _mode_label(selected[0][0]), mode_index,
                                 connect_flag=group_name == "true_phase")
                if group_name != "true_phase":
                    axis.set_xticks(range(len(labels)), labels,
                                    rotation=20 if group_name == "nominal_mismatch" else 0)
                _finish_axis(axis, {"true_strength": "True coupling",
                                    "true_phase": "True phase (deg)",
                                    "nominal_mismatch": "Nominal coupling"}[group_name], metric)
            if group_name == "true_strength":
                title = (f"Accurate nominal model; true phase {strength_phase:g}°"
                         if strength_phase is not None else "No coupling")
            elif group_name == "true_phase":
                title = f"Accurate nominal model; true coupling {phase_amplitude:g} dB"
            else:
                title = f"True coupling {mismatch_true[0]:g} dB / {mismatch_true[1]:g}°"
            fig.suptitle(f"{scene}: {title}")
            _save_figure(fig, settings,
                         f"mutual_coupling__{_filename_label(scene)}__{group_name}", paths)
    return paths


def _phase_center_offset(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """总体几何条件只画独立散点；同一真实 XYZ 的名义校准单独成图。"""
    import matplotlib.pyplot as plt

    paths = []
    groups = _case_groups(rows)
    true_keys = {}
    for group in groups:
        true_key = (group[0]["scene"], tuple((antenna, tuple(offset)) for antenna, offset in sorted(
            group[0]["true_offset_xyz_mm_by_antenna"].items())))
        true_keys.setdefault(true_key, []).append(group)
    calibration_groups = [collection for collection in true_keys.values()
                          if len({group[0]["case"] for group in collection}) > 1
                          and len({tuple((antenna, tuple(offset)) for antenna, offset in sorted(
                              group[0]["assumed_offset_xyz_mm_by_antenna"].items()))
                                   for group in collection}) > 1]
    plot_groups = [("geometry", [group for group in groups if group[0]["scene"] == scene])
                   for scene in dict.fromkeys(group[0]["scene"] for group in groups)]
    plot_groups.extend((f"calibration_{index}", collection)
                       for index, collection in enumerate(calibration_groups, 1))
    for group_name, selected_groups in plot_groups:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
        for axis, metric in zip(axes, ("tracking_cn0_mean_db_hz", "acquisition_correct_flag")):
            for mode_index, mode in enumerate(dict.fromkeys(group[0]["mode"]
                                                            for group in selected_groups)):
                mode_groups = [group for group in selected_groups if group[0]["mode"] == mode]
                points = [(group[0]["residual_relative_los_mm"],
                           _mean_metric(group, metric)) for group in mode_groups]
                # 同 x 可属于不同真实 XYZ；保留多个点，不连接成伪扫描。
                _plot_points(axis, sorted(points), _mode_label(mode_groups[0][0]), mode_index,
                             connect_flag=False)
            _finish_axis(axis, "Residual relative LOS (mm)", metric)
        scene = selected_groups[0][0]["scene"]
        fig.suptitle(f"{scene}: one point per physical case" if group_name == "geometry" else
                     f"{scene}: nominal calibration at fixed true XYZ")
        _save_figure(fig, settings,
                     f"phase_center_offset__{_filename_label(scene)}__{group_name}", paths)
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
