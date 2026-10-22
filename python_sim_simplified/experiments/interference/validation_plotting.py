"""按各验证实验的物理问题绘图；只读取逐 seed 接收结果。"""

from pathlib import Path
import re

import numpy as np

from experiments.interference.validation_support import ValidationSettings


def _filename_label(value: str) -> str:
    return re.sub(r"[^\w]+", "_", str(value).casefold()).strip("_")


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
        values = [value for line in axis.lines for value in line.get_ydata()]
        if any(np.isfinite(value) for value in values):
            axis.tick_params(axis="y", labelleft=True)
            if any(not np.isfinite(value) for value in values):
                axis.text(0.02, 0.02, "Gaps = missing CN0", fontsize=8,
                          bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
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


def _phase_offset(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """每场景分面显示真实相位的跟踪 C/N₀；捕获失败时另画状态或比例。"""
    affected = sorted({antenna for row in rows for antenna in row["dfe_phase_deg"]
                       if antenna in row["antennas"].split(",")})
    caption = ("DFE output: whole-channel phase on " + ", ".join(affected)
               + "\nNominal target response fixed")
    return _phase_or_rf_delay(rows, settings, "DFE phase (deg)", caption, "phase_offset")


def _phase_or_rf_delay(rows: list[dict], settings: ValidationSettings, xlabel: str,
                       caption: str, file_prefix: str) -> list[Path]:
    """复用相位与 RF 时延扫描的 C/N₀、条件式 ACQ 绘图。"""
    import matplotlib.pyplot as plt

    paths = []
    scenes = list(dict.fromkeys(row["scene"] for row in rows))
    fig, axes = plt.subplots(1, len(scenes), figsize=(6.2 * len(scenes), 5),
                             squeeze=False, constrained_layout=True)
    for axis, scene in zip(axes[0], scenes):
        scene_rows = [row for row in rows if row["scene"] == scene]
        modes = list(dict.fromkeys(row["mode"] for row in scene_rows))
        curves = {}
        for mode_index, mode in enumerate(modes):
            mode_rows = [row for row in scene_rows if row["mode"] == mode]
            curves[mode] = _scan_points(mode_rows, "tracking_cn0_mean_db_hz")
            _plot_points(axis, curves[mode], _mode_label(mode_rows[0]), mode_index)
        axis.set_title(scene)
        axis.set_xticks(sorted({row["scan_value"] for row in scene_rows}))
        _finish_axis(axis, xlabel, "tracking_cn0_mean_db_hz")
        if all(row["acquisition_correct_flag"] for row in scene_rows):
            axis.text(0.02, 0.97, "ACQ all pass", va="top", fontsize=8,
                      transform=axis.transAxes)
        # 只比较相同横轴的完整有效曲线；接近但不相等的数值不标为重合。
        matching_groups = []
        for mode in modes:
            points = curves[mode]
            if not points or not all(np.isfinite(y) for _, y in points):
                continue
            for group in matching_groups:
                reference = curves[group[0]]
                if ([x for x, _ in points] == [x for x, _ in reference]
                        and all(abs(value - reference_value) <= 1e-8
                                for (_, value), (_, reference_value) in zip(points, reference))):
                    group.append(mode)
                    break
            else:
                matching_groups.append([mode])
        overlaps = [" = ".join(_mode_label(next(row for row in scene_rows
                                               if row["mode"] == mode)) for mode in group)
                    for group in matching_groups if len(group) > 1]
        if overlaps:
            axis.text(0.02, 0.12, "Identical CN0:\n" + "\n".join(overlaps),
                      fontsize=8, transform=axis.transAxes)
    fig.suptitle(caption)
    _save_figure(fig, settings, f"{file_prefix}__tracking_cn0_mean", paths)

    failed_scenes = [scene for scene in scenes if any(
        not row["acquisition_correct_flag"] for row in rows if row["scene"] == scene)]
    if failed_scenes:
        seed_counts = {scene: len({row["seed"] for row in rows if row["scene"] == scene})
                       for scene in failed_scenes}
        fig, axes = plt.subplots(1, len(failed_scenes), figsize=(6.2 * len(failed_scenes), 4.6),
                                 squeeze=False, constrained_layout=True)
        for axis, scene in zip(axes[0], failed_scenes):
            scene_rows = [row for row in rows if row["scene"] == scene]
            modes = list(dict.fromkeys(row["mode"] for row in scene_rows))
            if seed_counts[scene] == 1:
                # 单 seed 的 0/1 是离散状态，按方法分行避免相同曲线遮盖。
                for mode_index, mode in enumerate(modes):
                    for row in (item for item in scene_rows if item["mode"] == mode):
                        axis.scatter(row["scan_value"], mode_index,
                                     marker="o" if row["acquisition_correct_flag"] else "x",
                                     color="#238b45" if row["acquisition_correct_flag"] else "#c53030",
                                     zorder=3)
                axis.set_yticks(range(len(modes)), [_mode_label(next(
                    row for row in scene_rows if row["mode"] == mode)) for mode in modes])
                axis.set_ylabel("Method (○ pass; × fail)")
            else:
                for mode_index, mode in enumerate(modes):
                    selected = [row for row in scene_rows if row["mode"] == mode]
                    _plot_points(axis, _scan_points(selected, "acquisition_correct_flag"),
                                 _mode_label(selected[0]), mode_index)
                axis.set_ylim(-0.04, 1.04)
                axis.set_ylabel("ACQ correct / all seeds")
                axis.legend(fontsize=8)
            axis.set_title(scene)
            axis.set_xticks(sorted({row["scan_value"] for row in scene_rows}))
            axis.set_xlabel(xlabel)
            axis.grid(alpha=0.3)
        fig.suptitle(caption)
        plot_kind = ("status" if all(seed_counts[scene] == 1 for scene in failed_scenes)
                     else "rate" if all(seed_counts[scene] > 1 for scene in failed_scenes)
                     else "status_rate")
        _save_figure(fig, settings, f"{file_prefix}__acquisition_{plot_kind}", paths)
    return paths


def _rf_group_delay(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """绘制接收指标，并单列展示相对参考通道的理想固定时延相位模型。"""
    import matplotlib.pyplot as plt

    metadata = rows[0]["rf_group_delay"]
    affected = metadata["affected_antenna"]
    reference = metadata["reference_antenna"]
    caption = (f"RF group delay on {affected} relative to {reference}"
               "\nNominal target response fixed")
    paths = _phase_or_rf_delay(rows, settings, "RF delay (ns)", caption, "rf_group_delay")

    # 理想传递函数 H_m(ν)=H_0(ν)exp(-j2πνΔτ)；这里只画模型，不把它说成实测响应。
    frequency_mhz = np.linspace(-metadata["dfe_cutoff_hz"], metadata["dfe_cutoff_hz"], 401) / 1e6
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for delay_ns in sorted({row["scan_value"] for row in rows}):
        row = next(item for item in rows if item["scan_value"] == delay_ns)
        offsets_ns = row["rf_group_delay"]["offset_ns_by_antenna"]
        relative_delay_ns = offsets_ns.get(affected, 0.0) - offsets_ns.get(reference, 0.0)
        axis.plot(frequency_mhz, -360.0 * frequency_mhz * 1e6 * relative_delay_ns * 1e-9,
                  label=f"{delay_ns:g} ns")
    axis.set(xlabel="Frequency relative to IF center (MHz)", ylabel="Relative phase (deg)",
             title=f"Ideal model: {affected} minus {reference}; DFE passband cutoff")
    axis.grid(alpha=0.3)
    axis.legend(title="RF delay")
    _save_figure(fig, settings, "rf_group_delay__ideal_phase_model", paths)
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


def _antenna_temperature(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """按真实温度分开显示增益和相位单因素扫描。"""
    import matplotlib.pyplot as plt

    paths = []
    for scene in dict.fromkeys(row["scene"] for row in rows):
        scene_rows = [row for row in rows if row["scene"] == scene]
        modes = list(dict.fromkeys(row["mode"] for row in scene_rows))
        temperature_info = scene_rows[0]["temperature_offset"]
        for metric_name, metric in (("tracking_cn0_mean", "tracking_cn0_mean_db_hz"),
                                    ("acquisition_correct_rate", "acquisition_correct_flag")):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True,
                                     constrained_layout=True)
            for axis, series in zip(axes, ("gain", "phase")):
                panel_rows = [row for row in scene_rows if row["series"] == series]
                for mode_index, mode in enumerate(modes):
                    selected = [row for row in panel_rows if row["mode"] == mode]
                    if selected:
                        points = _scan_points(selected, metric)
                        _plot_points(axis, points, _mode_label(selected[0]), mode_index)
                if panel_rows:
                    series_temperature_info = panel_rows[0]["temperature_offset"]
                    if series == "gain":
                        title = (f"Gain; {series_temperature_info['gain_slope_db_per_c']:g} dB/°C")
                    else:
                        title = (f"Phase; {series_temperature_info['phase_slope_deg_per_c']:g} deg/°C")
                    axis.set_title(title)
                    _finish_axis(axis, "True temperature (°C)", metric)
                    axis.axvline(series_temperature_info["reference_temperature_c"], color="#888888",
                                 linestyle=":", linewidth=1)
            fig.suptitle(
                f"{scene}: {temperature_info['antenna']} illustrative temperature offset; "
                f"Nom at {temperature_info['reference_temperature_c']:g}°C"
            )
            _save_figure(fig, settings,
                         f"antenna_temperature__{_filename_label(scene)}__{metric_name}", paths)
    return paths


def plot_validation(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """按显式实验类型分派绘图；保留真实数轴与同用例 seed 统计。"""
    import matplotlib.pyplot as plt

    if not rows:
        return []
    plotters = {
        "phase_offset": _phase_offset,
        "rf_group_delay": _rf_group_delay,
        "antenna_gain": _antenna_gain,
        "mutual_coupling": _mutual_coupling,
        "phase_center_offset": _phase_center_offset,
        "antenna_temperature": _antenna_temperature,
    }
    paths = plotters[settings.experiment_type](rows, settings)
    if settings.show_plots_flag and plt.get_backend().lower() != "agg":
        plt.show()
    return paths
