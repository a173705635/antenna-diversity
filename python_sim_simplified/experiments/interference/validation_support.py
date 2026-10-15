"""通用抗干扰验证：条件分组 → 前端 → 接收 → 独立预算 → 核验/绘图。

合并器根据完整 DFE 数据和配置的名义导向求权。
目标、各干扰及热噪声的分离真值仅用于接收后的固定权值评分。
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import numpy as np

from core.antenna import antenna_response, build_antenna_model
from core.channel_delay import delay_phase
from core.config import SimulationConfig
from core.evaluation import linear_components
from core.diversity import combine_known
from core.metrics import circular_chip_error
from core.pipeline import run_frontend, run_receiver
from core.serialization import serializable


@dataclass
class ValidationCase:
    """一个实验条件：物理/接收配置 + 图上标签 + DFE 注入/补偿 + 验收项目。

    scan_value 是横轴坐标，不会自动修改 config；修改物理参数由配置文件完成。
    reference_case 指向基准条件名，checks 可选 ideal_noise 或 phase_pair。
    """
    name: str
    config: SimulationConfig
    scene_label: str = "scenario"
    series_label: str = "case"
    scan_label: str = ""
    scan_value: float | None = None
    scan_unit: str = ""
    dfe_phase_deg: dict[str, float] = field(default_factory=dict)
    # 估计值由外部给定；只有 compensate_dfe_phase=True 时才执行逆旋转。
    dfe_estimated_phase_deg: dict[str, float] = field(default_factory=dict)
    compensate_dfe_phase: bool = False
    reference_case: str | None = None
    checks: tuple[str, ...] = ()


@dataclass
class ValidationSettings:
    """整个批次的统计与输出设置；功率去边单位为 ms。"""
    power_trim_ms: int = 10
    output_dir: Path = Path("results/interference/validation")
    show_plots: bool = True


def apply_dfe_phase(
    channels: np.ndarray, antennas: tuple[str, ...], phase_deg: dict[str, float],
) -> np.ndarray:
    """按天线名旋转整行 DFE 样本；单位为度，未列出的通道定义为零相位。"""
    factors = np.exp(1j * np.deg2rad([phase_deg.get(name, 0.0) for name in antennas]))
    # channels 为 [天线数, 时间采样数]；列向量广播使整行乘同一个复系数。
    # 目标、干扰、噪声一起旋转，不等同于只给目标导向矢量加相位。
    return channels * factors[:, None]


def _physical_key(cfg: SimulationConfig) -> str:
    """生成物理前端的分组键；名义导向、接收控制和绘图参数不改变采样。"""
    physical = asdict(cfg)
    # 去掉仅影响接收机假设/处理的字段，避免只改名义方向时重复生成物理 IQ。
    # 真实增益、真实方向、seed 等仍留在键中，变化后必须重新生成前端数据。
    for name in ("assumed_direction", "assumed_polarization"):
        physical["signal"].pop(name)
    for name in ("assumed_phase_centers_m", "estimated_channel_amplitude_error_db",
                 "estimated_channel_phase_error_deg", "estimated_channel_delay_ns"):
        physical["phone"].pop(name)
    physical["phone"]["antenna_model"].pop("assumed_state")
    for name in ("diversity", "acquisition", "tracking", "plot"):
        physical.pop(name)
    return json.dumps(serializable(physical), sort_keys=True, default=str)


def _assumed_steering(cfg: SimulationConfig) -> np.ndarray:
    """按本条件的估计方向/天线状态/窄带时延重新计算导向，不沿用别组假设。"""
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    base, _ = antenna_response(
        model, cfg.signal.assumed_direction, cfg.signal.assumed_polarization,
        estimated=True,
    )
    delays_s = 1e-9 * np.array([
        cfg.phone.estimated_channel_delay_ns[name]
        for name in cfg.phone.active_antennas
    ])
    if np.all(delays_s == 0.0):
        return base
    return base * delay_phase(
        # 这里只描述窄带载波相位，不能替代宽带码包络的序列时延对齐。
        delays_s, cfg.signal.carrier_hz + cfg.signal.doppler_hz,
    )


def _power(values: np.ndarray) -> float:
    """复数码值均方功率（码值平方），不是瓦特；同尺度功率比可转换为 dB。"""
    return float(np.mean(np.abs(values) ** 2))


def _score(
    case: ValidationCase, cfg: SimulationConfig, channels: np.ndarray,
    components: dict[str, np.ndarray], branches: dict, fs_out_hz: float,
    trim_ms: int, seed: int, assumed_steering: np.ndarray,
) -> list[dict]:
    """固定实际接收权值，独立投影分量并用 wᴴRw 复算输出功率。"""
    trim = round(trim_ms * 1e-3 * fs_out_hz)
    window = slice(trim, channels.shape[1] - trim)
    # 只裁评分窗口；捕获跟踪已使用完整序列。R 为各通道零时延样本协方差。
    samples = channels[:, window]
    covariance = samples @ samples.conj().T / samples.shape[1]
    rows = []
    for mode, branch in branches.items():
        weights = branch["combining"]["weights"]
        output = branch["combined"][window]
        projected = {
            # 使用接收机实际求出的同一组 w；不能对目标/干扰各自重新优化权值。
            name: weights.conj() @ values[:, window]
            for name, values in components.items()
        }
        powers = {name: _power(values) for name, values in projected.items()}
        interference = {name: power for name, power in powers.items()
                        if name not in ("target", "thermal")}
        linear_sum = sum(projected.values())
        acquisition = branch["acquisition"]
        tracking = branch["tracking"]
        cn0 = tracking["cn0_db_hz"] if tracking["success"] else np.array([])
        # 取跟踪后半段的有限 C/N₀ 中位数；缺测保存 None，不伪造成 0 dB-Hz。
        finite = cn0[len(cn0) // 2:]
        finite = finite[np.isfinite(finite)]
        code_estimate = (acquisition["code_phase_samples"]
                         * cfg.signal.code_rate_hz / fs_out_hz)
        quadratic = float(np.vdot(weights, covariance @ weights).real)
        # 两条独立路径：实际合并序列均方，和输入协方差二次型 wᴴRw。
        actual = _power(output)
        rows.append({
            "case": case.name, "seed": seed, "scene": case.scene_label,
            "series": case.series_label, "scan_label": case.scan_label,
            "scan_value": case.scan_value, "scan_unit": case.scan_unit,
            "reference_case": case.reference_case, "mode": mode,
            "antennas": ",".join(cfg.phone.active_antennas),
            "single_reference_antenna": cfg.phone.active_antennas[0],
            "target_power": powers["target"], "thermal_power": powers["thermal"],
            "interferer_powers": [{"source": name, "power": power}
                                    for name, power in interference.items()],
            "joint_nonlinear_quantization_residual_power": _power(output - linear_sum),
            # 分量可能相关，合成均方不一定等于各均方之和；差值是交叉项。
            "linear_component_cross_power": _power(linear_sum) - sum(powers.values()),
            "actual_power": actual,
            "linear_in_band_sinr_db": float(10 * np.log10(
                # 这里是线性分量预算，不把联合量化残差冒充独立热噪声。
                powers["target"] / (powers["thermal"] + sum(interference.values()))
            )),
            "weight_real_imag": [[float(value.real), float(value.imag)] for value in weights],
            "weight_norm_squared": float(np.vdot(weights, weights).real),
            "combined_direct_max_abs_error": float(np.max(np.abs(
                output - weights.conj() @ samples
            ))),
            "quadratic_power_relative_error": abs(actual - quadratic) / actual,
            "acquisition_metric": float(acquisition["metric"]),
            "acquisition_threshold": cfg.acquisition.threshold,
            "acquired": bool(acquisition["success"]),
            "acquired_code_error_chips": float(circular_chip_error(
                code_estimate, cfg.signal.code_delay_chips,
            )),
            "acquired_doppler_error_hz": float(
                acquisition["doppler_hz"] - cfg.signal.doppler_hz
            ),
            "tracking_success": bool(tracking["success"]),
            "tracking_status": tracking["status"],
            "tracking_cn0_late_median_db_hz": float(np.median(finite)) if finite.size else None,
            "tracking_cn0_late_finite_count": int(finite.size),
        })
    # 即使仅运行 MVDR，也以第一根物理天线计算 SINGLE 的评分参考。
    _, reference_info = combine_known(
        channels, assumed_steering, "single", cfg.diversity.diagonal_loading_factor,
    )
    reference_weights = reference_info["weights"]
    reference_target = _power(reference_weights.conj() @ components["target"][:, window])
    reference_thermal = _power(reference_weights.conj() @ components["thermal"][:, window])
    for row in rows:
        row["target_output_gain_db"] = float(10 * np.log10(
            row["target_power"] / reference_target
        ))
        row["target_response_gain"] = float(np.sqrt(
            row["target_power"] / reference_target
        ))
        row["thermal_snr_gain_db"] = float(10 * np.log10(
            (row["target_power"] / row["thermal_power"])
            / (reference_target / reference_thermal)
        ))
        row["thermal_snr_ratio"] = float(10 ** (row["thermal_snr_gain_db"] / 10))
        row["target_power_retention"] = row["target_power"] / reference_target
        row["interferer_suppression_db"] = {
            # 参考干扰功率 / 合并后干扰功率：正值表示压制，负值表示放大。
            item["source"]: float(10 * np.log10(
                _power(reference_weights.conj() @ components[item["source"]][:, window])
                / item["power"]
            )) for item in row["interferer_powers"]
        }
    return rows


def _ideal_noise_applicability(case: ValidationCase) -> str | None:
    """检查理想 EGC 的 10log10(N) 前提；返回不适用原因，None 表示可检查。

    这是理论结论的适用范围判断，不是为错误配置提供默认回退。
    """
    cfg = case.config
    if any(source.enable for source in cfg.interferers):
        return "external interference is enabled"
    if any(cfg.phone.channel_delay_ns[name] != 0
           for name in cfg.phone.active_antennas):
        return "true sequence delays need a wideband alignment analysis"
    if np.ptp(np.broadcast_to(cfg.rf.amplitude_error_db, len(cfg.phone.active_antennas))) != 0:
        return "RF channel amplitudes differ"
    if np.ptp(np.broadcast_to(cfg.rf.phase_error_deg, len(cfg.phone.active_antennas))) != 0:
        return "RF channel phases differ"
    if any(case.dfe_phase_deg.get(name, 0.0) -
           (case.dfe_estimated_phase_deg.get(name, 0.0)
            if case.compensate_dfe_phase else 0.0) != 0.0
           for name in cfg.phone.active_antennas):
        return "DFE phases are not corrected"
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    true, _ = antenna_response(
        model, cfg.signal.true_direction, cfg.signal.true_polarization,
    )
    assumed = _assumed_steering(cfg)
    if not np.allclose(true, assumed, rtol=1e-12, atol=1e-12):
        return "true and assumed target manifolds differ"
    if not np.allclose(np.abs(true), np.abs(true[0]), rtol=1e-12, atol=1e-12):
        return "element target gains are not equal"
    return None


def verify_validation(rows: list[dict], cases: dict[tuple[str, int], ValidationCase]) -> dict:
    """区分通过、失败、不适用；双方 C/N₀ 缺测时只核对状态，不伪造数值。"""
    checks = {}
    # 第一层：每行检查线性合并恒等式；满足理想前提才检查 N 路热噪声收益。
    for row in rows:
        key = f'math:{row["case"]}:{row["seed"]}:{row["mode"]}'
        direct = row["combined_direct_max_abs_error"] < 1e-8
        quadratic = row["quadratic_power_relative_error"] < 1e-10
        checks[key] = {"status": "passed" if direct and quadratic else "failed",
                       "passed": direct and quadratic,
                       "reason": "fixed-weight direct and covariance identity"}
        case = cases[row["case"], row["seed"]]
        if "ideal_noise" in case.checks and row["mode"] == "egc":
            reason = _ideal_noise_applicability(case)
            if reason is not None:
                checks[f'ideal_noise:{row["case"]}:{row["seed"]}:egc'] = {
                    "status": "not_applicable", "passed": None, "reason": reason,
                }
                continue
            theory = 10 * np.log10(len(case.config.phone.active_antennas))
            error = abs(row["thermal_snr_gain_db"] - theory)
            passed = error < 0.5
            checks[f'ideal_noise:{row["case"]}:{row["seed"]}:egc'] = {
                "status": "passed" if passed else "failed", "passed": passed,
                "reason": f"EGC thermal SNR error {error:.6g} dB; limit 0.5 dB",
            }
    indexed = {(row["case"], row["seed"], row["mode"]): row for row in rows}
    # 第二层：同 seed、同方法与无误差基准配对，避免噪声差异混入补偿误差。
    for row in rows:
        case = cases[row["case"], row["seed"]]
        if "phase_pair" not in case.checks:
            continue
        reference = indexed[case.reference_case, row["seed"], row["mode"]]
        comparisons = {}
        comparisons["physical_reference_available"] = row["phase_reference_available"]
        for name in ("dfe_input_relative_error", "combined_relative_error",
                     "weight_relative_error"):
            comparisons[name] = row[name] is not None and row[name] < 1e-12
        for name in ("target_power", "thermal_power", "actual_power"):
            comparisons[name] = abs(row[name] - reference[name]) / reference[name] < 1e-12
        comparisons["interferer_powers"] = all(
            current["source"] == baseline["source"] and
            abs(current["power"] - baseline["power"]) / baseline["power"] < 1e-12
            for current, baseline in zip(row["interferer_powers"], reference["interferer_powers"])
        ) and len(row["interferer_powers"]) == len(reference["interferer_powers"])
        comparisons["acquisition_metric"] = abs(
            row["acquisition_metric"] - reference["acquisition_metric"]
        ) < 1e-9
        for name in ("acquired", "tracking_success", "tracking_status",
                     "tracking_cn0_late_finite_count"):
            comparisons[name] = row[name] == reference[name]
        for name in ("acquired_code_error_chips", "acquired_doppler_error_hz"):
            comparisons[name] = abs(row[name] - reference[name]) < 1e-9
        current_cn0 = row["tracking_cn0_late_median_db_hz"]
        baseline_cn0 = reference["tracking_cn0_late_median_db_hz"]
        comparisons["cn0_availability"] = (current_cn0 is None) == (baseline_cn0 is None)
        if current_cn0 is None and baseline_cn0 is None:
            # 两者都没跟踪估计只能说明状态一致，不能宣称 C/N₀ 数值验证通过。
            cn0_status = "not_evaluated"
            comparisons["cn0_value"] = None
        elif current_cn0 is None or baseline_cn0 is None:
            cn0_status = "availability_mismatch"
            comparisons["cn0_value"] = False
        else:
            cn0_status = "evaluated"
            comparisons["cn0_value"] = abs(current_cn0 - baseline_cn0) < 1e-7
        passed = all(value is None or bool(value) for value in comparisons.values())
        checks[f'phase_pair:{row["case"]}:{row["seed"]}:{row["mode"]}'] = {
            "status": "passed" if passed else "failed", "passed": passed,
            "reason": "paired restored samples and receiver metrics",
            "cn0_comparison": cn0_status,
            "comparisons": comparisons,
        }
    return {"passed": all(item["status"] != "failed" for item in checks.values()),
            "checks": checks}


def _run_physical_group(group: list[ValidationCase], power_trim_ms: int) -> list[dict]:
    """复用一次物理前端，返回标量结果行；本组 IQ 和参考分支在返回时释放。"""
    front_cfg = group[0].config
    frontend = run_frontend(front_cfg, retain_analysis=True)
    # 真值分量只提供给评分，不交给合并器求权；防止使用真值改善接收结果。
    components = linear_components(frontend)
    remaining_references = {}
    for case in group:
        if "phase_pair" in case.checks:
            remaining_references[case.reference_case] = (
                remaining_references.get(case.reference_case, 0) + 1
            )
    reference_data = {}
    rows = []
    for case in group:
        cfg = case.config
        seed = cfg.random_seed
        antennas = cfg.phone.active_antennas
        # 两次复数乘法与原注入、逆补偿顺序一致；零净相位也保留浮点残差。
        injection_factors = np.exp(1j * np.deg2rad([
            case.dfe_phase_deg.get(name, 0.0) for name in antennas
        ]))[:, None]
        channels = frontend["dfe_channels"] * injection_factors
        if case.compensate_dfe_phase:
            correction_factors = np.exp(1j * np.deg2rad([
                -case.dfe_estimated_phase_deg.get(name, 0.0) for name in antennas
            ]))[:, None]
            channels = channels * correction_factors
        factors = np.exp(1j * np.deg2rad([
            case.dfe_phase_deg.get(name, 0.0) -
            (case.dfe_estimated_phase_deg.get(name, 0.0)
             if case.compensate_dfe_phase else 0.0)
            for name in antennas
        ]))
        # 更新本条件的名义导向与相位后诊断响应；不向导向再补一次相位。
        truth = dict(frontend["truth"])
        truth["desired_steering_assumed"] = _assumed_steering(cfg)
        truth["desired_steering_true"] = truth["desired_steering_true"] * factors
        truth["interferer_steering_effective"] = [
            steering * factors for steering in truth["interferer_steering_effective"]
        ]
        branches = run_receiver(channels, truth, cfg, frontend["dfe_info"]["fs_out_hz"])
        rotated_components = {
            name: values * injection_factors for name, values in components.items()
        }
        if case.compensate_dfe_phase:
            rotated_components = {
                name: values * correction_factors
                for name, values in rotated_components.items()
            }
        case_rows = _score(
            case, cfg, channels, rotated_components, branches,
            frontend["dfe_info"]["fs_out_hz"], power_trim_ms, seed,
            truth["desired_steering_assumed"],
        )
        if case.name in remaining_references:
            reference_data[case.name] = (channels, branches)
        if "phase_pair" in case.checks:
            reference = reference_data.get(case.reference_case)
            if reference is None:
                dfe_input_error = None
            else:
                baseline_channels, baseline_branches = reference
                dfe_input_error = float(
                    np.linalg.norm(channels - baseline_channels)
                    / np.linalg.norm(baseline_channels)
                )
            for row in case_rows:
                row["phase_reference_available"] = reference is not None
                row["dfe_input_relative_error"] = dfe_input_error
                if reference is None:
                    row["combined_relative_error"] = None
                    row["weight_relative_error"] = None
                    continue
                baseline_branch = baseline_branches[row["mode"]]
                current_branch = branches[row["mode"]]
                row["combined_relative_error"] = float(
                    np.linalg.norm(current_branch["combined"] - baseline_branch["combined"])
                    / np.linalg.norm(baseline_branch["combined"])
                )
                row["weight_relative_error"] = float(
                    np.linalg.norm(current_branch["combining"]["weights"]
                                   - baseline_branch["combining"]["weights"])
                    / np.linalg.norm(baseline_branch["combining"]["weights"])
                )
            remaining_references[case.reference_case] -= 1
            if remaining_references[case.reference_case] == 0:
                reference_data.pop(case.reference_case, None)
        rows.extend(case_rows)
    return rows


def run_validation(settings: ValidationSettings, cases: list[ValidationCase]) -> dict:
    """直接执行已准备的条件；同物理条件复用前端，大数组只保留到本组结束。"""
    # seed、真实方向和增益属于物理键；仅名义假设变化可共享 IQ。
    grouped = {}
    for case in cases:
        key = _physical_key(case.config)
        grouped.setdefault(key, []).append(case)
    rows = []
    for group_index, group in enumerate(grouped.values(), start=1):
        print(f"Frontend {group_index}/{len(grouped)}: seed={group[0].config.random_seed}, "
              f"conditions={len(group)}", flush=True)
        rows.extend(_run_physical_group(group, settings.power_trim_ms))
    case_summaries = {}
    for case in cases:
        summary = case_summaries.setdefault(case.name, {
            "name": case.name, "reference_case": case.reference_case,
            "scene_label": case.scene_label, "series_label": case.series_label,
            "scan_label": case.scan_label, "scan_value": case.scan_value,
            "scan_unit": case.scan_unit, "receiver_configs": {},
            "dfe_phase_deg": case.dfe_phase_deg,
            "dfe_estimated_phase_deg": case.dfe_estimated_phase_deg,
            "compensate_dfe_phase": case.compensate_dfe_phase, "checks": case.checks,
        })
        summary["receiver_configs"][str(case.config.random_seed)] = asdict(case.config)
    by_identity = {(case.name, case.config.random_seed): case for case in cases}
    return {"cases": list(case_summaries.values()), "rows": rows,
            "verification": verify_validation(rows, by_identity)}


def plot_validation(rows: list[dict], settings: ValidationSettings) -> list[Path]:
    """从结果行绘图并返回 PNG 路径；不重新仿真，也不修改原始结果。

    每因素、指标、方法独立窗口，场景横排；增益/SINR 为 dB，捕获保留比值。
    同一横轴点跨 seed 按当前显示单位取均值，缺测 C/N₀ 保持断线而非补零。
    """
    import matplotlib.pyplot as plt

    factors = list(dict.fromkeys((row["scan_label"], row["scan_unit"]) for row in rows))
    paths = []
    for factor_label, factor_unit in factors:
        factor_rows = [row for row in rows
                       if (row["scan_label"], row["scan_unit"]) == (factor_label, factor_unit)]
        sources = list(dict.fromkeys(
            source for row in factor_rows for source in row["interferer_suppression_db"]
        ))
        figures = [
            ("thermal_gain", (("thermal_snr_gain_db", "Thermal SNR gain vs SINGLE (dB)"),)),
            ("target_gain", (("target_gain_db", "Target power gain vs SINGLE (dB)"),)),
            ("sinr", (("linear_in_band_sinr_db", "C / (I + N) (dB)"),)),
            ("acquisition", (("acquisition_metric", "Acquisition peak ratio"),)),
            ("cn0", (("tracking_cn0_late_median_db_hz", "Late tracking C/N0 (dB-Hz)"),)),
        ]
        # 各干扰源独立成图，避免长标签和过多行把目标/接收指标挤在一起。
        figures.extend((f"suppression_{index}", ((f"suppression:{source}",
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
            if filename.startswith("suppression_"):
                source = metrics[0][0].removeprefix("suppression:")
                figure_scenes = [scene for scene in scenes if any(
                    row["scene"] == scene and source in row["interferer_suppression_db"]
                    for row in factor_rows)]
            fig, axes = plt.subplots(1, len(figure_scenes),
                                     figsize=(6 * len(figure_scenes), 4.8),
                                     squeeze=False, constrained_layout=True)
            if filename.startswith("suppression_"):
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
                            numeric_scan = selected[0]["scan_value"] is not None
                            points = (sorted({row["scan_value"] for row in selected})
                                      if numeric_scan else
                                      list(dict.fromkeys(row["case"] for row in selected)))
                            values = []
                            for point in points:
                                matching = [row for row in selected
                                            if (row["scan_value"] if numeric_scan else row["case"]) == point]
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
                            axis.plot(points, values,
                                      linestyle=("-", "--", ":", "-.")[series_index % 4],
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
                        axis.set(title=f"{scene} / {mode}", xlabel=xlabel, ylabel=ylabel)
                        scan_points = [row["scan_value"] for row in factor_rows
                                       if row["scene"] == scene and row["scan_value"] is not None]
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
                        axis.legend(fontsize=9, loc="lower right")
            # 序号避免自定义标签中的斜杠或不同单位导致路径/文件名冲突。
            suffix = "" if len(factors) == 1 else f"_factor_{factors.index((factor_label, factor_unit)) + 1}"
            path = settings.output_dir / f"{filename}_{figure_mode}{suffix}.png"
            fig.savefig(path, dpi=150)
            paths.append(path)
            if not settings.show_plots:
                plt.close(fig)
    if settings.show_plots and plt.get_backend().lower() != "agg":
        plt.show()
    return paths
