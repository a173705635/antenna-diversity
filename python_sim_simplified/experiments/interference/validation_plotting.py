"""通用抗干扰验证结果绘图；只消费标量结果行。"""

from collections import Counter
import hashlib
from pathlib import Path
import re

import numpy as np

from experiments.interference.validation_support import ValidationSettings


def gain_plot_rows(rows: list[dict]) -> list[dict]:
    """将双天线增益批次转为物理横轴，仅复制绘图字段，不改原始记录。

    零增益点同时作为共同衰减和BR衰减的基准；交换图补上交换前条件。
    相同物理条件的重复用例按seed去重，避免统计时重复计入。
    """
    swapped_gains = {
        row["input_gain_db_by_antenna"][0] for row in rows
        if row["mode"] != "single" and row["input_gain_db_by_antenna"][1] == 0
        and row["input_gain_db_by_antenna"][0] != 0
    }
    prepared = []
    seen = set()
    for row in rows:
        tl_gain, br_gain = row["input_gain_db_by_antenna"]
        axes = []
        if row["mode"] == "single":
            axes.append(("TL gain", tl_gain, row["case"]))
        else:
            if tl_gain == br_gain:
                axes.append(("Common antenna gain", tl_gain, row["case"]))
            if tl_gain == 0:
                axes.append(("BR gain (TL = 0 dB)", br_gain, row["case"]))
            if ((tl_gain in swapped_gains and br_gain == 0)
                    or (tl_gain == 0 and br_gain in swapped_gains)):
                axes.append(("TL / BR gain", None, f"{tl_gain:g} / {br_gain:g}"))
        for label, value, name in axes:
            key = (label, row.get("scene"), row["mode"], row.get("seed"),
                   tl_gain, br_gain)
            if key in seen:
                continue
            seen.add(key)
            prepared.append(dict(row, scan_label=label, scan_unit="dB",
                                 scan_value=value, case=name))
    # 交换图按衰减深度成对排列：0/-6、-6/0、0/-20、-20/0。
    swap_rows = [row for row in prepared if row["scan_label"] == "TL / BR gain"]
    swap_rows.sort(key=lambda row: (
        abs(sum(row["input_gain_db_by_antenna"])),
        -row["input_gain_db_by_antenna"][0],
    ))
    return [row for row in prepared if row["scan_label"] != "TL / BR gain"] + swap_rows


def _filename_label(value: str) -> str:
    """保留可读标签，替换斜杠等不适合文件名的字符。"""
    return re.sub(r"[^\w]+", "_", value.casefold()).strip("_")


def _distinct_plot_names(base_names: dict) -> dict:
    """标签归一后同名时，仅给冲突项追加原始标签的稳定摘要。"""
    counts = Counter(base_names.values())
    return {
        label: (base if counts[base] == 1 else
                f"{base}_{hashlib.sha256(str(label).encode('utf-8')).hexdigest()[:8]}")
        for label, base in base_names.items()
    }


def plot_validation(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """从结果行绘图并返回 PNG 路径；不重新仿真，也不修改原始结果。

    每因素、指标、方法独立窗口，场景横排；增益/SINR 为 dB，捕获保留比值。
    同一横轴点跨 seed 按当前显示单位取均值，缺测 C/N₀ 保持断线而非补零。
    """
    import matplotlib.pyplot as plt

    # 只转换具备逐天线增益记录的原子批次；旧相位/方向扫描保持原横轴。
    gain_batch_flag = bool(rows) and all(
        row["case"].startswith("TC-") and "input_gain_db_by_antenna" in row
        for row in rows
    )
    if gain_batch_flag:
        experiment_name = "antenna_gain"
    elif rows and all(row["case"].startswith("MC-") for row in rows):
        experiment_name = "mutual_coupling"
    elif rows and all(row["case"].startswith("PC-") for row in rows):
        experiment_name = "phase_center_offset"
    elif rows and all(any(marker in row["case"] for marker in
                          ("/phase_", "/direction_", "/gain_")) for row in rows):
        experiment_name = "phase_direction_gain"
    else:
        experiment_name = "validation"
    if gain_batch_flag:
        rows = gain_plot_rows(rows)
    factors = list(dict.fromkeys((row["scan_label"], row["scan_unit"]) for row in rows))
    factor_filename_parts = {}
    for factor_label, factor_unit in factors:
        scan_label_name = _filename_label(factor_label)
        scan_unit_name = _filename_label(factor_unit)
        if scan_unit_name and not scan_label_name.endswith(f"_{scan_unit_name}"):
            scan_label_name += f"_{scan_unit_name}"
        factor_filename_parts[(factor_label, factor_unit)] = scan_label_name
    factor_names = _distinct_plot_names(factor_filename_parts)
    figure_paths = []
    for factor_label, factor_unit in factors:
        factor_rows = [row for row in rows
                       if (row["scan_label"], row["scan_unit"]) == (factor_label, factor_unit)]
        sources = list(dict.fromkeys(
            source for row in factor_rows for source in row["interferer_suppression_db"]
        ))
        figures = [
            ("thermal_snr_gain", (("thermal_snr_gain_db", "Thermal SNR gain vs SINGLE (dB)"),)),
            ("target_power_gain", (("target_gain_db", "Target power gain vs SINGLE (dB)"),)),
            ("linear_in_band_sinr", (("linear_in_band_sinr_db", "C / (I + N) (dB)"),)),
            ("acquisition_peak_ratio", (("acquisition_metric", "Acquisition peak ratio"),)),
            ("tracking_cn0_late_median", (("tracking_cn0_late_median_db_hz", "Late tracking C/N0 (dB-Hz)"),)),
        ]
        # 各干扰源独立成图，避免长标签和过多行把目标/接收指标挤在一起。
        source_names = _distinct_plot_names({source: _filename_label(source) for source in sources})
        figures.extend((f"interferer_suppression_{source_names[source]}", ((f"suppression:{source}",
                         f"Source {index}: suppression (dB)"),))
                       for index, source in enumerate(sources))
        scenes = list(dict.fromkeys(row["scene"] for row in factor_rows))
        modes = list(dict.fromkeys(row["mode"] for row in factor_rows))
        series = list(dict.fromkeys(row["series"] for row in factor_rows))
        # 每个指标、每种方法单独一个窗口，仅横排比较场景，不纵向堆叠。
        separate_figures = [(filename, metrics, mode)
                            for filename, metrics in figures for mode in modes]
        for filename, metrics, figure_mode in separate_figures:
            figure_scenes = scenes
            if filename.startswith("interferer_suppression_"):
                source = metrics[0][0].removeprefix("suppression:")
                figure_scenes = [scene for scene in scenes if any(
                    row["scene"] == scene and source in row["interferer_suppression_db"]
                    for row in factor_rows)]
            fig, axes = plt.subplots(1, len(figure_scenes),
                                     figsize=(6 * len(figure_scenes), 4.8),
                                     squeeze=False, constrained_layout=True)
            if filename.startswith("interferer_suppression_"):
                fig.suptitle(source, fontsize=11)
            for metric_index, (metric, ylabel) in enumerate(metrics):
                for mode_index, mode in enumerate([figure_mode]):
                    for scene_index, scene in enumerate(figure_scenes):
                        axis = axes[0, scene_index]
                        plotted = []
                        overlaps = []
                        for series_index, label in enumerate(series):
                            selected = [row for row in factor_rows if row["scene"] == scene
                                        and row["mode"] == mode and row["series"] == label]
                            if not selected:
                                continue
                            numeric_scan_flag = selected[0]["scan_value"] is not None
                            points = (sorted({row["scan_value"] for row in selected})
                                      if numeric_scan_flag else
                                      list(dict.fromkeys(row["case"] for row in selected)))
                            values = []
                            for point in points:
                                matching = [row for row in selected
                                            if (row["scan_value"] if numeric_scan_flag else row["case"]) == point]
                                if metric.startswith("suppression:"):
                                    observations = [row["interferer_suppression_db"].get(
                                        metric.removeprefix("suppression:")) for row in matching]
                                elif metric == "target_gain_db":
                                    observations = [10 * np.log10(row["target_power_retention"])
                                                    for row in matching]
                                else:
                                    observations = [row.get(metric) for row in matching]
                                available = [value for value in observations if value is not None]
                                # 这是跨 seed 的显示统计，不是把线性功率平均后再转 dB。
                                values.append(float(np.mean(available)) if available else np.nan)
                            # 不偏移真实数据：嵌套空心标记、不同线型，并明确注明重合。
                            for previous_label, previous_points, previous_values in plotted:
                                # 只标整条曲线数值一致；局部交点不做人工错位。
                                if points == previous_points and np.allclose(
                                        values, previous_values, atol=1e-8, rtol=0, equal_nan=True
                                ) and np.any(np.isfinite(values)):
                                    overlaps.append(f"{previous_label} = {label}")
                            plotted.append((label, points, values))
                            # 物理数值扫描可连线；TC编号和强弱交换类别只画点。
                            atomic_cases_flag = factor_unit == "TC" or not numeric_scan_flag
                            axis.plot(points, values,
                                      linestyle=("None" if atomic_cases_flag else
                                                 ("-", "--", ":", "-.")[series_index % 4]),
                                      marker=("o", "s", "x", "^")[series_index % 4],
                                      markersize=(10, 7, 5, 8)[series_index % 4],
                                      markerfacecolor="none", linewidth=1.6, label=label)
                        if overlaps:
                            axis.text(0.02, 0.98, "Overlapping (tolerance 1e-8):\n" + "\n".join(overlaps),
                                      va="top", fontsize=9, transform=axis.transAxes,
                                      bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
                        xlabel = factor_label or "condition"
                        if factor_unit:
                            xlabel += f" ({factor_unit})"
                        displayed_rows = [row for row in factor_rows
                                          if row["scene"] == scene and row["mode"] == mode]
                        title = f"{scene} / {mode}"
                        if gain_batch_flag:
                            seed_count = len({row["seed"] for row in displayed_rows})
                            title += f"\nMean across seeds (n={seed_count})"
                            ylabel = ylabel.replace("vs SINGLE", "vs SINGLE-TL")
                            if metric == "tracking_cn0_late_median_db_hz":
                                title += "; available estimates only"
                        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
                        scan_points = [row["scan_value"] for row in factor_rows
                                       if row["scene"] == scene and row["mode"] == mode
                                       and row["scan_value"] is not None]
                        if scan_points:
                            low, high = min(scan_points), max(scan_points)
                            padding = (high - low) * 0.03 if high != low else 0.5
                            axis.set_xlim(low - padding, high + padding)
                        if metric == "acquisition_metric":
                            thresholds = sorted({row["acquisition_threshold"] for row in factor_rows
                                                 if row["scene"] == scene and row["mode"] == mode})
                            for threshold in thresholds:
                                axis.axhline(threshold, color="#888888", linestyle="-.",
                                             label=f"threshold {threshold:g}")
                        if metric == "tracking_cn0_late_median_db_hz":
                            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
                            measured = [row[metric] for row in factor_rows
                                        if row["scene"] == scene and row["mode"] == mode
                                        and row[metric] is not None]
                            if measured:
                                axis.text(0.02, 0.02, "Gaps = no finite tracking estimate",
                                          fontsize=7, transform=axis.transAxes)
                            else:
                                axis.text(0.5, 0.5, "No finite tracking estimate",
                                          ha="center", transform=axis.transAxes)
                                axis.set_yticks([])
                        axis.grid(alpha=0.3)
                        axis.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
            figure_path = settings.output_dir / (
                f"{experiment_name}__{factor_names[(factor_label, factor_unit)]}"
                f"__{filename}__{_filename_label(figure_mode)}.png"
            )
            fig.savefig(figure_path, dpi=150)
            figure_paths.append(figure_path)
            if not settings.show_plots_flag:
                plt.close(fig)
    if settings.show_plots_flag and plt.get_backend().lower() != "agg":
        plt.show()
    return figure_paths
