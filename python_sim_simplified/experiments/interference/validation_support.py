"""通用抗干扰验证：逐条件前端 → 接收 → 独立预算 → 核验/绘图。

合并器根据完整 DFE 数据和配置的名义导向求权。
目标、各干扰及热噪声的分离真值仅用于接收后的固定权值指标评估。
"""

from dataclasses import asdict, dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.stats import t as student_t

from core.antenna import antenna_response, build_antenna_model
from core.channel_delay import delay_phase
from core.config import SimulationConfig
from core.evaluation import linear_components
from core.diversity import combine_known
from core.metrics import circular_chip_error
from core.pipeline import run_frontend, run_receiver
from experiments.interference.coupling_support import (
    coupling_row_fields, model_checks, summarize_coupling,
)
from experiments.interference.phase_center_support import (
    phase_center_row_fields, summarize_phase_center,
)
from experiments.paths import reset_output


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
    # 估计值由外部给定；只有 compensate_dfe_phase_flag=True 时才执行逆旋转。
    dfe_estimated_phase_deg: dict[str, float] = field(default_factory=dict)
    compensate_dfe_phase_flag: bool = False
    reference_case: str | None = None
    checks: tuple[str, ...] = ()


@dataclass
class ValidationSettings:
    """整个批次的统计与输出设置；功率去边单位为 ms。"""
    evaluation_edge_trim_ms: int = 10
    output_dir: Path = Path("results/interference/validation")
    show_plots_flag: bool = True
    workers: int = 1
    generate_plots_flag: bool = False


def pattern_gain_theory(gain_db: tuple[float, ...], mode: str) -> float:
    """独立白噪声下相对首阵元的理论合并 SNR 收益，单位 dB。"""
    voltage = 10.0 ** (np.asarray(gain_db, dtype=float) / 20.0)
    if mode == "egc_strict":
        snr_ratio = np.sum(voltage) ** 2 / (len(voltage) * voltage[0] ** 2)
    else:
        snr_ratio = np.sum(voltage ** 2) / voltage[0] ** 2
    return float(10.0 * np.log10(snr_ratio))


def apply_dfe_phase(
    channels: np.ndarray, antennas: tuple[str, ...], phase_deg: dict[str, float],
) -> np.ndarray:
    """按天线名旋转整行 DFE 样本；单位为度，未列出的通道定义为零相位。"""
    factors = np.exp(1j * np.deg2rad([phase_deg.get(name, 0.0) for name in antennas]))
    # channels 为 [天线数, 时间采样数]；列向量广播使整行乘同一个复系数。
    # 目标、干扰、噪声一起旋转，不等同于只给目标导向矢量加相位。
    return channels * factors[:, None]


def _assumed_steering(cfg: SimulationConfig) -> np.ndarray:
    """为理想噪声适用性检查计算估计导向，包含估计通道时延。"""
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    antenna_steering, _ = antenna_response(
        model, cfg.signal.assumed_direction, cfg.signal.assumed_polarization,
        estimated_flag=True,
    )
    delays_s = 1e-9 * np.array([
        cfg.phone.estimated_channel_delay_ns[name]
        for name in cfg.phone.active_antennas
    ])
    if np.all(delays_s == 0.0):
        return antenna_steering
    return antenna_steering * delay_phase(
        # 这里只描述窄带载波相位，不能替代宽带码包络的序列时延对齐。
        delays_s, cfg.signal.carrier_hz + cfg.signal.doppler_hz,
    )


def _power(values: np.ndarray) -> float:
    """复数码值均方功率（码值平方），不是瓦特；同尺度功率比可转换为 dB。"""
    return float(np.mean(np.abs(values) ** 2))


def evaluate_case(
    case: ValidationCase, result: dict, evaluation_edge_trim_ms: int,
    remaining_reference_uses: dict[tuple[str, int], int],
    reference_data: dict[tuple[str, int], tuple[np.ndarray, dict]],
) -> list[dict]:
    """对一次已完成的接收独立复算分量、指标和相位配对；返回标量行。"""
    case_cfg = case.config
    seed = case_cfg.random_seed
    frontend = result["frontend"]
    channels = result["channels"]
    branches = result["branches"]
    # 重放本次前端保存的热噪声；逐分量施加与实际 DFE 相同的两次相位乘法。
    components = linear_components(frontend)
    components = {
        name: values * result["injection_factors"]
        for name, values in components.items()
    }
    if case.compensate_dfe_phase_flag:
        components = {
            name: values * result["correction_factors"]
            for name, values in components.items()
        }
    fs_out_hz = frontend["dfe_info"]["fs_out_hz"]
    edge_trim_samples = round(evaluation_edge_trim_ms * 1e-3 * fs_out_hz)
    window = slice(edge_trim_samples, channels.shape[1] - edge_trim_samples)
    # 只裁指标窗口；捕获跟踪已使用完整序列。R 为各通道零时延样本协方差。
    samples = channels[:, window]
    covariance = samples @ samples.conj().T / samples.shape[1]
    pattern_fields = {}
    if "pattern_gain" in case.checks:
        # I0 输入端逐通道物理功率：SINGLE 的单位响应归一不能代替天线端功率。
        target_input = [_power(values) for values in components["target"][:, window]]
        thermal_input = [_power(values) for values in components["thermal"][:, window]]
        input_snr = [float(10 * np.log10(signal / noise))
                     for signal, noise in zip(target_input, thermal_input)]
        gain_db = tuple(case_cfg.phone.antenna_model.true_state.amplitude_error_db[name]
                        for name in case_cfg.phone.active_antennas)
        physical_gain_db = 10 * np.log10(
            np.abs(frontend["truth"]["desired_steering_true"]) ** 2
        )
        pattern_fields = {
            "input_gain_db_by_antenna": list(gain_db),
            "input_target_power_by_antenna": target_input,
            "input_thermal_power_by_antenna": thermal_input,
            "input_snr_db_by_antenna": input_snr,
            "best_single_antenna": case_cfg.phone.active_antennas[int(np.argmax(input_snr))],
            "best_single_snr_db": max(input_snr),
            "pattern_expected_gain_db": pattern_gain_theory(gain_db, "egc"),
            "pattern_response_gain_error_db": float(np.max(np.abs(physical_gain_db - gain_db))),
        }
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
        cn0 = tracking["cn0_db_hz"] if tracking["success_flag"] else np.array([])
        # 取跟踪后半段的有限 C/N₀ 中位数；缺测保存 None，不伪造成 0 dB-Hz。
        finite = cn0[len(cn0) // 2:]
        finite = finite[np.isfinite(finite)]
        code_estimate = (acquisition["code_phase_samples"]
                         * case_cfg.signal.code_rate_hz / fs_out_hz)
        quadratic = float(np.vdot(weights, covariance @ weights).real)
        # 两条独立路径：实际合并序列均方，和输入协方差二次型 wᴴRw。
        actual = _power(output)
        row = {
            "case": case.name, "seed": seed, "scene": case.scene_label,
            "series": case.series_label, "scan_label": case.scan_label,
            "scan_value": case.scan_value, "scan_unit": case.scan_unit,
            "reference_case": case.reference_case, "mode": mode,
            "antennas": ",".join(case_cfg.phone.active_antennas),
            "single_reference_antenna": case_cfg.phone.active_antennas[0],
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
            "acquisition_threshold": case_cfg.acquisition.threshold,
            "acquired_flag": bool(acquisition["success_flag"]),
            "acquired_code_error_chips": float(circular_chip_error(
                code_estimate, case_cfg.signal.code_delay_chips,
            )),
            "acquired_doppler_error_hz": float(
                acquisition["doppler_hz"] - case_cfg.signal.doppler_hz
            ),
            "tracking_success_flag": bool(tracking["success_flag"]),
            "tracking_status": tracking["status"],
            "tracking_cn0_late_median_db_hz": float(np.median(finite)) if finite.size else None,
            "tracking_cn0_late_finite_count": int(finite.size),
        }
        if pattern_fields:
            # 同一真实前端只把权值改变；MVDR 权值用独立特征分解复算。
            row.update(pattern_fields)
            row["thermal_snr_db"] = float(10 * np.log10(
                powers["target"] / powers["thermal"]
            ))
            row["thermal_snr_gain_vs_best_db"] = (
                row["thermal_snr_db"] - pattern_fields["best_single_snr_db"]
            )
            if mode == "egc_strict":
                row["pattern_expected_gain_db"] = pattern_gain_theory(
                    tuple(pattern_fields["input_gain_db_by_antenna"]), mode,
                )
            elif mode == "single":
                row["pattern_expected_gain_db"] = 0.0
            desired = frontend["truth"]["desired_steering_assumed"]
            if mode == "mvdr":
                full_covariance = channels @ channels.conj().T / channels.shape[1]
                loading = (case_cfg.diversity.diagonal_loading_factor
                           * np.trace(full_covariance).real / channels.shape[0])
                eigenvalues, eigenvectors = np.linalg.eigh(
                    full_covariance + loading * np.eye(channels.shape[0])
                )
                solved = eigenvectors @ ((eigenvectors.conj().T @ desired) / eigenvalues)
                independent_weights = solved / np.vdot(desired, solved)
            elif mode == "egc_strict":
                independent_weights = (desired / np.abs(desired)) / np.sum(np.abs(desired))
            elif mode == "egc":
                independent_weights = desired / np.sum(np.abs(desired) ** 2)
            else:
                independent_weights = np.zeros_like(desired)
                independent_weights[0] = 1.0 / np.conj(desired[0])
            row["pattern_weight_formula_error"] = float(
                np.linalg.norm(weights - independent_weights)
                / np.linalg.norm(independent_weights)
            )
            row["pattern_response_constraint_error"] = float(
                abs(np.vdot(weights, desired) - 1.0)
            )
            row["pattern_mvdr_weight_error"] = (row["pattern_weight_formula_error"]
                                                if mode == "mvdr" else None)
            row["pattern_mvdr_constraint_error"] = (row["pattern_response_constraint_error"]
                                                    if mode == "mvdr" else None)
        if "mutual_coupling" in case.checks:
            row.update(coupling_row_fields(case, frontend, channels, components,
                                           window, weights))
        if "phase_center_noise_only" in case.checks:
            row.update(phase_center_row_fields(case, frontend, channels, components,
                                               window, weights))
        rows.append(row)
    # 即使仅运行 MVDR，也以第一根物理天线计算 SINGLE 的指标参考。
    _, reference_info = combine_known(
        channels, frontend["truth"]["desired_steering_assumed"], "single",
        case_cfg.diversity.diagonal_loading_factor,
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
    case_key = (case.name, seed)
    if case_key in remaining_reference_uses:
        # 仅保留相位配对需要的 DFE 样本、合并序列与权值。
        reference_data[case_key] = (channels, {
            mode: (branch["combined"], branch["combining"]["weights"])
            for mode, branch in branches.items()
        })
    if "phase_pair" in case.checks:
        reference_key = (case.reference_case, seed)
        reference = reference_data.get(reference_key)
        if reference is None:
            dfe_input_error = None
        else:
            baseline_channels, baseline_branches = reference
            dfe_input_error = float(
                np.linalg.norm(channels - baseline_channels)
                / np.linalg.norm(baseline_channels)
            )
        for row in rows:
            row["phase_reference_available_flag"] = reference is not None
            row["dfe_input_relative_error"] = dfe_input_error
            if reference is None:
                row["combined_relative_error"] = None
                row["weight_relative_error"] = None
                continue
            baseline_combined, baseline_weights = baseline_branches[row["mode"]]
            current_branch = branches[row["mode"]]
            row["combined_relative_error"] = float(
                np.linalg.norm(current_branch["combined"] - baseline_combined)
                / np.linalg.norm(baseline_combined)
            )
            row["weight_relative_error"] = float(
                np.linalg.norm(current_branch["combining"]["weights"] - baseline_weights)
                / np.linalg.norm(baseline_weights)
            )
        remaining_reference_uses[reference_key] -= 1
        if remaining_reference_uses[reference_key] == 0:
            reference_data.pop(reference_key, None)
    return rows


def _ideal_noise_applicability(case: ValidationCase) -> str | None:
    """检查理想 EGC 的 10log10(N) 前提；返回不适用原因，None 表示可检查。

    这是理论结论的适用范围判断，不是为错误配置提供默认回退。
    """
    case_cfg = case.config
    if any(source.enable_flag for source in case_cfg.interferers):
        return "external interference is enabled"
    if any(case_cfg.phone.channel_delay_ns[name] != 0
           for name in case_cfg.phone.active_antennas):
        return "true sequence delays need a wideband alignment analysis"
    if np.ptp(np.broadcast_to(case_cfg.rf.amplitude_error_db, len(case_cfg.phone.active_antennas))) != 0:
        return "RF channel amplitudes differ"
    if np.ptp(np.broadcast_to(case_cfg.rf.phase_error_deg, len(case_cfg.phone.active_antennas))) != 0:
        return "RF channel phases differ"
    if any(case.dfe_phase_deg.get(name, 0.0) -
           (case.dfe_estimated_phase_deg.get(name, 0.0)
            if case.compensate_dfe_phase_flag else 0.0) != 0.0
           for name in case_cfg.phone.active_antennas):
        return "DFE phases are not corrected"
    model = build_antenna_model(case_cfg.phone, case_cfg.signal.carrier_hz)
    true_steering, _ = antenna_response(
        model, case_cfg.signal.true_direction, case_cfg.signal.true_polarization,
    )
    assumed_steering = _assumed_steering(case_cfg)
    if not np.allclose(true_steering, assumed_steering, rtol=1e-12, atol=1e-12):
        return "true and assumed target manifolds differ"
    if not np.allclose(np.abs(true_steering), np.abs(true_steering[0]), rtol=1e-12, atol=1e-12):
        return "element target gains are not equal"
    return None


def _mean_interval(values: list[float]) -> dict:
    """样本均值、样本标准差及双侧 t-95% 均值区间；单次试验不估计区间。"""
    samples = np.asarray(values, dtype=float)
    mean = float(np.mean(samples))
    if len(samples) == 1:
        return {"mean": mean, "std": None, "ci95": None}
    sample_std = float(np.std(samples, ddof=1))
    half_width = float(student_t.ppf(0.975, len(samples) - 1) * sample_std
                       / np.sqrt(len(samples)))
    return {"mean": mean, "std": sample_std,
            "ci95": [mean - half_width, mean + half_width]}


def _wilson_interval(successes: int, total: int) -> list[float]:
    """二项比例 Wilson 95% 区间；不将失捕或失锁样本删除。"""
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / total +
                         z * z / (4 * total * total)) / denominator
    return [float(center - radius), float(center + radius)]


def summarize_pattern_gain(rows: list[dict], references: dict[str, str]) -> dict:
    """按 TC 汇总配对 30 seed 的理论、线性 SNR、接收状态与判定。"""
    grouped = {}
    indexed = {(row["case"], row["seed"]): row for row in rows}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)

    cases = {}
    for name, observations in grouped.items():
        mode = observations[0]["mode"]
        reference_name = references.get(name)
        gain_errors = [row["thermal_snr_gain_db"] - row["pattern_expected_gain_db"]
                       for row in observations]
        absolute_differences = []
        expected_absolute = []
        input_target_errors = []
        input_noise_errors = []
        for row in observations:
            input_baseline = indexed.get(("TC-001", row["seed"]))
            if input_baseline is not None:
                for gain, signal, base_signal, noise, base_noise in zip(
                    row["input_gain_db_by_antenna"],
                    row["input_target_power_by_antenna"],
                    input_baseline["input_target_power_by_antenna"],
                    row["input_thermal_power_by_antenna"],
                    input_baseline["input_thermal_power_by_antenna"],
                ):
                    input_target_errors.append(10 * np.log10(signal / base_signal) - gain)
                    input_noise_errors.append(10 * np.log10(noise / base_noise))
        if reference_name is not None:
            for row in observations:
                reference = indexed[reference_name, row["seed"]]
                if mode == "single":
                    difference = (row["input_snr_db_by_antenna"][0] -
                                  reference["input_snr_db_by_antenna"][0])
                    theory = (row["input_gain_db_by_antenna"][0] -
                              reference["input_gain_db_by_antenna"][0])
                else:
                    difference = row["thermal_snr_db"] - reference["thermal_snr_db"]
                    true_gains = row["input_gain_db_by_antenna"]
                    base_gains = reference["input_gain_db_by_antenna"]
                    theory = (true_gains[0] + pattern_gain_theory(tuple(true_gains), mode)
                              - base_gains[0] - pattern_gain_theory(tuple(base_gains), mode))
                absolute_differences.append(difference)
                expected_absolute.append(theory)

        manifold_error = max(row["pattern_response_gain_error_db"] for row in observations)
        gain_error = _mean_interval(gain_errors)
        absolute_error = (_mean_interval([measured - expected for measured, expected in
                                          zip(absolute_differences, expected_absolute)])
                          if absolute_differences else None)
        mvdr_weight_error = max((row["pattern_mvdr_weight_error"] for row in observations
                                 if row["pattern_mvdr_weight_error"] is not None), default=None)
        mvdr_constraint_error = max((row["pattern_mvdr_constraint_error"] for row in observations
                                     if row["pattern_mvdr_constraint_error"] is not None), default=None)
        weight_error = max(row["pattern_weight_formula_error"] for row in observations)
        response_constraint_error = max(row["pattern_response_constraint_error"]
                                        for row in observations)
        target_input_error = (max(abs(error) for error in input_target_errors)
                              if input_target_errors else None)
        noise_input_error = (max(abs(error) for error in input_noise_errors)
                             if input_noise_errors else None)
        # MVDR 有求权/响应硬判据，性能与理想白噪声 MATCHED 只报告不强求相等。
        passed_flag = (manifold_error <= 1e-8 and weight_error <= 1e-8
                  and response_constraint_error <= 1e-8
                  and target_input_error is not None and target_input_error <= 1e-8
                  and noise_input_error is not None and noise_input_error <= 1e-8)
        if mode == "mvdr":
            passed_flag = passed_flag and mvdr_weight_error <= 1e-8 and mvdr_constraint_error <= 1e-8
        else:
            passed_flag = passed_flag and abs(gain_error["mean"]) <= 0.2
            if absolute_error is not None:
                passed_flag = passed_flag and abs(absolute_error["mean"]) <= 0.2
        acquired = sum(row["acquired_flag"] for row in observations)
        tracked = sum(row["tracking_success_flag"] for row in observations)
        finite_cn0 = [row["tracking_cn0_late_median_db_hz"] for row in observations
                      if row["tracking_cn0_late_median_db_hz"] is not None]
        cases[name] = {
            "status": "passed" if passed_flag else "failed", "mode": mode,
            "seeds": len(observations), "reference_case": reference_name,
            "expected_gain_db": observations[0]["pattern_expected_gain_db"],
            "measured_gain_db": _mean_interval([row["thermal_snr_gain_db"]
                                                for row in observations]),
            "gain_error_db": gain_error, "response_gain_max_error_db": manifold_error,
            "weight_max_relative_error": weight_error,
            "response_constraint_max_error": response_constraint_error,
            "input_target_change_max_error_db": target_input_error,
            "input_thermal_change_max_error_db": noise_input_error,
            "expected_absolute_change_db": expected_absolute[0] if expected_absolute else None,
            "absolute_snr_change_mean_db": (_mean_interval(absolute_differences)["mean"]
                                             if absolute_differences else None),
            "absolute_change_error_db": absolute_error,
            "mvdr_weight_max_relative_error": mvdr_weight_error,
            "mvdr_constraint_max_error": mvdr_constraint_error,
            "acquisition_success_count": acquired,
            "acquisition_success_wilson95": _wilson_interval(acquired, len(observations)),
            "tracking_success_count": tracked,
            "tracking_success_wilson95": _wilson_interval(tracked, len(observations)),
            "tracking_cn0_missing_count": len(observations) - len(finite_cn0),
            "tracking_cn0_observed_db_hz": (_mean_interval(finite_cn0)
                                             if finite_cn0 else None),
        }
    return {"passed_flag": all(item["status"] == "passed" for item in cases.values()),
            "cases": cases}


def verify_validation(rows: list[dict], cases: dict[tuple[str, int], ValidationCase]) -> dict:
    """区分通过、失败、不适用；双方 C/N₀ 缺测时只核对状态，不伪造数值。"""
    checks = {}
    # 第一层：每行检查线性合并恒等式；满足理想前提才检查 N 路热噪声收益。
    for row in rows:
        key = f'math:{row["case"]}:{row["seed"]}:{row["mode"]}'
        direct_flag = row["combined_direct_max_abs_error"] < 1e-8
        quadratic_flag = row["quadratic_power_relative_error"] < 1e-10
        checks[key] = {"status": "passed" if direct_flag and quadratic_flag else "failed",
                       "passed_flag": direct_flag and quadratic_flag,
                       "reason": "fixed-weight direct and covariance identity"}
        case = cases[row["case"], row["seed"]]
        if "ideal_noise" in case.checks and row["mode"] == "egc":
            reason = _ideal_noise_applicability(case)
            if reason is not None:
                checks[f'ideal_noise:{row["case"]}:{row["seed"]}:egc'] = {
                    "status": "not_applicable", "passed_flag": None, "reason": reason,
                }
                continue
            theory = 10 * np.log10(len(case.config.phone.active_antennas))
            error = abs(row["thermal_snr_gain_db"] - theory)
            passed_flag = error < 0.5
            checks[f'ideal_noise:{row["case"]}:{row["seed"]}:egc'] = {
                "status": "passed" if passed_flag else "failed", "passed_flag": passed_flag,
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
        comparisons["physical_reference_available_flag"] = row["phase_reference_available_flag"]
        for name in ("dfe_input_relative_error", "combined_relative_error",
                     "weight_relative_error"):
            comparisons[f"{name}_flag"] = row[name] is not None and row[name] < 1e-12
        for name in ("target_power", "thermal_power", "actual_power"):
            comparisons[f"{name}_flag"] = abs(row[name] - reference[name]) / reference[name] < 1e-12
        comparisons["interferer_powers_flag"] = all(
            current["source"] == baseline["source"] and
            abs(current["power"] - baseline["power"]) / baseline["power"] < 1e-12
            for current, baseline in zip(row["interferer_powers"], reference["interferer_powers"])
        ) and len(row["interferer_powers"]) == len(reference["interferer_powers"])
        comparisons["acquisition_metric_flag"] = abs(
            row["acquisition_metric"] - reference["acquisition_metric"]
        ) < 1e-9
        for comparison_name, source_name in (
            ("acquired_flag", "acquired_flag"),
            ("tracking_success_flag", "tracking_success_flag"),
            ("tracking_status_flag", "tracking_status"),
            ("tracking_cn0_late_finite_count_flag", "tracking_cn0_late_finite_count"),
        ):
            comparisons[comparison_name] = row[source_name] == reference[source_name]
        for name in ("acquired_code_error_chips", "acquired_doppler_error_hz"):
            comparisons[f"{name}_flag"] = abs(row[name] - reference[name]) < 1e-9
        current_cn0 = row["tracking_cn0_late_median_db_hz"]
        baseline_cn0 = reference["tracking_cn0_late_median_db_hz"]
        comparisons["cn0_availability_flag"] = (current_cn0 is None) == (baseline_cn0 is None)
        if current_cn0 is None and baseline_cn0 is None:
            # 两者都没跟踪估计只能说明状态一致，不能宣称 C/N₀ 数值验证通过。
            cn0_status = "not_evaluated"
            comparisons["cn0_value_flag"] = None
        elif current_cn0 is None or baseline_cn0 is None:
            cn0_status = "availability_mismatch"
            comparisons["cn0_value_flag"] = False
        else:
            cn0_status = "evaluated"
            comparisons["cn0_value_flag"] = abs(current_cn0 - baseline_cn0) < 1e-7
        passed_flag = all(value is None or bool(value) for value in comparisons.values())
        checks[f'phase_pair:{row["case"]}:{row["seed"]}:{row["mode"]}'] = {
            "status": "passed" if passed_flag else "failed", "passed_flag": passed_flag,
            "reason": "paired restored samples and receiver metrics",
            "cn0_comparison": cn0_status,
            "comparisons": comparisons,
        }
    return {"passed_flag": all(item["status"] != "failed" for item in checks.values()),
            "checks": checks}


def run_case(case: ValidationCase) -> dict:
    """用本条件运行前端、相位处理与接收，返回本次评估需要的数据引用。"""
    case_cfg = case.config
    antennas = case_cfg.phone.active_antennas
    frontend = run_frontend(case_cfg, retain_analysis_flag=True)
    # 两次复数乘法与原注入、逆补偿顺序一致；零净相位也保留浮点残差。
    injection_factors = np.exp(1j * np.deg2rad([
        case.dfe_phase_deg.get(name, 0.0) for name in antennas
    ]))[:, None]
    channels = frontend["dfe_channels"] * injection_factors
    correction_factors = None
    if case.compensate_dfe_phase_flag:
        correction_factors = np.exp(1j * np.deg2rad([
            -case.dfe_estimated_phase_deg.get(name, 0.0) for name in antennas
        ]))[:, None]
        channels = channels * correction_factors
    factors = np.exp(1j * np.deg2rad([
        case.dfe_phase_deg.get(name, 0.0) -
        (case.dfe_estimated_phase_deg.get(name, 0.0)
         if case.compensate_dfe_phase_flag else 0.0)
        for name in antennas
    ]))
    # 前端已按本条件配置生成名义导向；只修正物理相位后的诊断响应。
    truth = dict(frontend["truth"])
    truth["desired_steering_true"] = truth["desired_steering_true"] * factors
    truth["interferer_steering_effective"] = [
        steering * factors for steering in truth["interferer_steering_effective"]
    ]
    branches = run_receiver(channels, truth, case_cfg, frontend["dfe_info"]["fs_out_hz"])
    return {
        "frontend": frontend, "channels": channels, "branches": branches,
        "injection_factors": injection_factors,
        "correction_factors": correction_factors,
    }


def _run_evaluated_case(case: ValidationCase, evaluation_edge_trim_ms: int) -> list[dict]:
    """进程内完成一个 TC 的前端、接收和评分，只向父进程返回标量。"""
    result = run_case(case)
    return evaluate_case(case, result, evaluation_edge_trim_ms, {}, {})


def run_validation(settings: ValidationSettings, cases: list[ValidationCase]) -> dict:
    """按输入顺序独立运行各条件，仅跨条件保留相位配对的参考数组。"""
    remaining_reference_uses = {}
    for case in cases:
        if "phase_pair" in case.checks:
            key = (case.reference_case, case.config.random_seed)
            remaining_reference_uses[key] = remaining_reference_uses.get(key, 0) + 1
    reference_data = {}
    rows = []
    if settings.workers > 1:
        # TC 相互独立；批次目录在开始时刷新，进度行在长任务中持续落盘。
        reset_output(settings.output_dir)
        ordered_results = [None] * len(cases)
        with (settings.output_dir / "progress.jsonl").open("w", encoding="utf-8") as progress:
            with ProcessPoolExecutor(max_workers=settings.workers) as executor:
                futures = {executor.submit(_run_evaluated_case, case, settings.evaluation_edge_trim_ms): index
                           for index, case in enumerate(cases)}
                for future in as_completed(futures):
                    index = futures[future]
                    case_rows = future.result()
                    ordered_results[index] = case_rows
                    case = cases[index]
                    progress.write(json.dumps({"case": case.name,
                                               "seed": case.config.random_seed,
                                               "rows": case_rows}) + "\n")
                    progress.flush()
                    print(f"Completed {sum(item is not None for item in ordered_results)}"
                          f"/{len(cases)}: seed={case.config.random_seed}, "
                          f"condition={case.name}", flush=True)
        rows = [row for case_rows in ordered_results for row in case_rows]
    else:
        for case_index, case in enumerate(cases, start=1):
            print(f"Frontend {case_index}/{len(cases)}: seed={case.config.random_seed}, "
                  f"condition={case.name}", flush=True)
            result = run_case(case)
            rows.extend(evaluate_case(case, result, settings.evaluation_edge_trim_ms,
                                      remaining_reference_uses, reference_data))
            del result
    case_summaries = {}
    for case in cases:
        summary = case_summaries.setdefault(case.name, {
            "name": case.name, "reference_case": case.reference_case,
            "scene_label": case.scene_label, "series_label": case.series_label,
            "scan_label": case.scan_label, "scan_value": case.scan_value,
            "scan_unit": case.scan_unit, "receiver_configs": {},
            "dfe_phase_deg": case.dfe_phase_deg,
            "dfe_estimated_phase_deg": case.dfe_estimated_phase_deg,
            "compensate_dfe_phase_flag": case.compensate_dfe_phase_flag, "checks": case.checks,
        })
        summary["receiver_configs"][str(case.config.random_seed)] = asdict(case.config)
    by_identity = {(case.name, case.config.random_seed): case for case in cases}
    verification = verify_validation(rows, by_identity)
    summary = {"cases": list(case_summaries.values()), "rows": rows,
               "verification": verification}
    if any("pattern_gain" in case.checks for case in cases):
        references = {case.name: case.reference_case for case in cases
                      if case.reference_case is not None}
        pattern = summarize_pattern_gain(rows, references)
        summary["pattern_gain"] = pattern
        for name, outcome in pattern["cases"].items():
            verification["checks"][f"pattern_gain:{name}"] = {
                "status": outcome["status"], "passed_flag": outcome["status"] == "passed",
                "reason": "independent gain, input power, weight and paired SNR checks",
            }
        verification["passed_flag"] = verification["passed_flag"] and pattern["passed_flag"]
    if any("mutual_coupling" in case.checks for case in cases):
        model = model_checks()
        coupling = summarize_coupling(rows)
        summary["model_checks"] = model
        summary["mutual_coupling"] = coupling
        verification["checks"].update({f"coupling_model:{name}": check
                                       for name, check in model.items()})
        verification["checks"].update(coupling["checks"])
        verification["passed_flag"] = (
            verification["passed_flag"] and coupling["passed_flag"]
            and all(check["passed_flag"] for check in model.values())
        )
    if any("phase_center_noise_only" in case.checks for case in cases):
        phase_center = summarize_phase_center(rows)
        summary["phase_center"] = phase_center
        verification["checks"].update(phase_center["checks"])
        verification["passed_flag"] = verification["passed_flag"] and phase_center["passed_flag"]
    return summary


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
