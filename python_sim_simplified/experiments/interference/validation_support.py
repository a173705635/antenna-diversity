"""通用抗干扰验证：逐条件前端 → 接收 → 独立预算 → 核验/绘图。

合并器根据完整 DFE 数据和配置的名义导向求权。
目标、各干扰及热噪声的分离真值仅用于接收后的固定权值指标评估。
"""

from dataclasses import asdict, dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

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
    coupling_row_fields, independent_weights, model_checks, summarize_coupling,
)
from experiments.interference.phase_center_support import (
    phase_center_row_fields, summarize_phase_center,
)
from experiments.paths import reset_output


@dataclass
class ValidationCase:
    """一个实验条件：物理/接收配置 + 图上标签 + DFE 注入/补偿 + 验收项目。

    scan_value 是横轴坐标，不会自动修改 config；修改物理参数由配置文件完成。
    checks 可选 ideal_noise。
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
    checks: tuple[str, ...] = ()


@dataclass
class ValidationSettings:
    """整个批次的实验类型、统计与输出设置；功率去边单位为 ms。"""
    evaluation_edge_trim_ms: int
    output_dir: Path
    experiment_type: str
    show_plots_flag: bool = True
    workers: int = 1
    generate_plots_flag: bool = False


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


def _pattern_gain_row_fields(
    case_cfg: SimulationConfig, frontend: dict, channels: np.ndarray,
    mode: str, weights: np.ndarray, powers: dict[str, float],
    input_fields: dict,
) -> dict:
    """复算增益用例的固定权值、名义约束和输入端 SNR；不参与接收求权。"""
    fields = dict(input_fields)
    fields["thermal_snr_db"] = float(10 * np.log10(powers["target"] / powers["thermal"]))
    fields["thermal_snr_gain_vs_best_db"] = (
        fields["thermal_snr_db"] - input_fields["best_single_snr_db"]
    )
    desired = frontend["truth"]["desired_steering_assumed"]
    expected_weights, constraint = independent_weights(channels, desired, mode,
                                                       case_cfg.diversity)
    if mode == "single":
        fields["pattern_expected_gain_db"] = 0.0
    elif mode == "mrc":
        # 名义与真实方向可不同：以独立 MRC 权值和真实响应计算相对首通道理论收益。
        actual_steering = frontend["truth"]["desired_steering_true"]
        gain_ratio = (abs(np.vdot(expected_weights, actual_steering)) ** 2 /
                      (np.vdot(expected_weights, expected_weights).real *
                       abs(actual_steering[0]) ** 2))
        fields["pattern_expected_gain_db"] = float(10 * np.log10(gain_ratio))
    else:
        fields["pattern_expected_gain_db"] = None
    constraint_error = abs(np.vdot(weights, constraint) - 1)
    fields["pattern_weight_formula_error"] = float(
        np.linalg.norm(weights - expected_weights) / np.linalg.norm(expected_weights)
    )
    fields["pattern_weight_constraint_error"] = float(constraint_error)
    return fields


def evaluate_case(
    case: ValidationCase, result: dict, settings: ValidationSettings,
) -> list[dict]:
    """对一次已完成的接收独立复算分量和指标；返回标量行。"""
    case_cfg = case.config
    seed = case_cfg.random_seed
    frontend = result["frontend"]
    channels = result["channels"]
    branches = result["branches"]
    # 分量准备：重放本次热噪声，并施加与真实 DFE 相同的两次相位乘法。
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
    edge_trim_samples = round(settings.evaluation_edge_trim_ms * 1e-3 * fs_out_hz)
    window = slice(edge_trim_samples, channels.shape[1] - edge_trim_samples)
    # 只裁指标窗口；捕获跟踪已使用完整序列。R 为各通道零时延样本协方差。
    samples = channels[:, window]
    covariance = samples @ samples.conj().T / samples.shape[1]
    pattern_fields = {}
    if settings.experiment_type == "antenna_gain":
        # I0 输入端逐通道物理功率：SINGLE 的单位响应归一不能代替天线端功率。
        target_input = [_power(values) for values in components["target"][:, window]]
        thermal_input = [_power(values) for values in components["thermal"][:, window]]
        input_snr = [float(10 * np.log10(signal / noise))
                     for signal, noise in zip(target_input, thermal_input)]
        gain_db = tuple(case_cfg.phone.antenna_model.true_state.amplitude_error_db.get(name, 0.0)
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
            "pattern_response_gain_error_db": float(np.max(np.abs(physical_gain_db - gain_db))),
        }
    # 公共指标：每种方法复用同一前端采样，以接收机实际固定权值投影分量。
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
            "mode": mode,
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
        # 专项复算只写入对应批次的附加字段。
        if pattern_fields:
            row.update(_pattern_gain_row_fields(case_cfg, frontend, channels, mode,
                                                weights, powers, pattern_fields))
        if settings.experiment_type == "mutual_coupling":
            row.update(coupling_row_fields(case, frontend, channels, components,
                                           window, weights, mode))
        if settings.experiment_type == "phase_center_offset":
            row.update(phase_center_row_fields(case, frontend, channels, components,
                                               window, weights, mode))
        rows.append(row)
    # SINGLE 公共参照：即使仅运行 MVDR，也用第一根物理天线计算相对收益。
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
    return rows


def _ideal_noise_applicability(case: ValidationCase) -> str | None:
    """检查等幅目标下 MRC 的 10log10(N) 前提；返回不适用原因。

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


def summarize_pattern_gain(rows: list[dict]) -> dict:
    """按物理 TC 和方法分别汇总 seed，避免算法间混行。"""
    grouped = {}
    indexed = {(row["case"], row["mode"], row["seed"]): row for row in rows}
    for row in rows:
        grouped.setdefault((row["case"], row["mode"]), []).append(row)

    cases = {}
    for (case_id, mode), observations in grouped.items():
        name = f"{case_id}-{mode}"
        gain_errors = ([row["thermal_snr_gain_db"] - row["pattern_expected_gain_db"]
                        for row in observations] if mode in ("single", "mrc") else [])
        input_target_errors = []
        input_noise_errors = []
        for row in observations:
            input_baseline = indexed.get(("TC-001", mode, row["seed"]))
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
        manifold_error = max(row["pattern_response_gain_error_db"] for row in observations)
        gain_error = _mean_interval(gain_errors) if gain_errors else None
        weight_error = max(row["pattern_weight_formula_error"] for row in observations)
        weight_constraint_error = max(row["pattern_weight_constraint_error"]
                                      for row in observations)
        target_input_error = (max(abs(error) for error in input_target_errors)
                              if input_target_errors else None)
        noise_input_error = (max(abs(error) for error in input_noise_errors)
                             if input_noise_errors else None)
        # MVDR 有求权/响应硬判据，性能与等噪声 MRC 只报告不强求相等。
        passed_flag = (manifold_error <= 1e-8 and weight_error <= 1e-8
                  and weight_constraint_error <= 1e-8
                  and target_input_error is not None and target_input_error <= 1e-8
                  and noise_input_error is not None and noise_input_error <= 1e-8)
        if mode in ("single", "mrc"):
            passed_flag = passed_flag and abs(gain_error["mean"]) <= 0.2
        acquired = sum(row["acquired_flag"] for row in observations)
        tracked = sum(row["tracking_success_flag"] for row in observations)
        finite_cn0 = [row["tracking_cn0_late_median_db_hz"] for row in observations
                      if row["tracking_cn0_late_median_db_hz"] is not None]
        cases[name] = {
            "case": case_id, "status": "passed" if passed_flag else "failed", "mode": mode,
            "seeds": len(observations),
            "expected_gain_db": observations[0]["pattern_expected_gain_db"],
            "measured_gain_db": _mean_interval([row["thermal_snr_gain_db"]
                                                for row in observations]),
            "gain_error_db": gain_error, "response_gain_max_error_db": manifold_error,
            "weight_max_relative_error": weight_error,
            "weight_constraint_max_error": weight_constraint_error,
            "input_target_change_max_error_db": target_input_error,
            "input_thermal_change_max_error_db": noise_input_error,
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
    """核验公共恒等式及适用条件下的理想噪声收益。"""
    checks = {}
    # 每行检查线性合并恒等式；满足理想前提才检查 N 路热噪声收益。
    for row in rows:
        key = f'math:{row["case"]}:{row["seed"]}:{row["mode"]}'
        direct_flag = row["combined_direct_max_abs_error"] < 1e-8
        quadratic_flag = row["quadratic_power_relative_error"] < 1e-10
        checks[key] = {"status": "passed" if direct_flag and quadratic_flag else "failed",
                       "passed_flag": direct_flag and quadratic_flag,
                       "reason": "fixed-weight direct and covariance identity"}
        case = cases[row["case"], row["seed"]]
        if "ideal_noise" in case.checks and row["mode"] == "mrc":
            reason = _ideal_noise_applicability(case)
            if reason is not None:
                checks[f'ideal_noise:{row["case"]}:{row["seed"]}:mrc'] = {
                    "status": "not_applicable", "passed_flag": None, "reason": reason,
                }
                continue
            theory = 10 * np.log10(len(case.config.phone.active_antennas))
            error = abs(row["thermal_snr_gain_db"] - theory)
            passed_flag = error < 0.5
            checks[f'ideal_noise:{row["case"]}:{row["seed"]}:mrc'] = {
                "status": "passed" if passed_flag else "failed", "passed_flag": passed_flag,
                "reason": f"MRC thermal SNR error {error:.6g} dB; limit 0.5 dB",
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


def _run_evaluated_case(case: ValidationCase, settings: ValidationSettings) -> list[dict]:
    """独立运行一个条件的前端、接收和指标复算，只返回标量行。"""
    result = run_case(case)
    return evaluate_case(case, result, settings)


def run_validation(settings: ValidationSettings, cases: list[ValidationCase]) -> dict:
    """独立运行各条件并按输入顺序汇总。"""
    rows = []
    if settings.workers > 1:
        # 目录在开始时刷新；每个用例单独写一条标量进度，结果按输入索引复原。
        reset_output(settings.output_dir)
        ordered_results = [None] * len(cases)
        completed_count = 0
        with (settings.output_dir / "progress.jsonl").open("w", encoding="utf-8") as progress:
            with ProcessPoolExecutor(max_workers=settings.workers) as executor:
                futures = {executor.submit(_run_evaluated_case, case, settings): index
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
                    completed_count += 1
                    print(f"Completed {completed_count}/{len(cases)}: "
                          f"seed={case.config.random_seed}, "
                          f"condition={case.name}", flush=True)
        rows = [row for case_rows in ordered_results for row in case_rows]
    else:
        for case_index, case in enumerate(cases, start=1):
            print(f"Frontend {case_index}/{len(cases)}: seed={case.config.random_seed}, "
                  f"condition={case.name}", flush=True)
            rows.extend(_run_evaluated_case(case, settings))
    case_summaries = {}
    for case in cases:
        summary = case_summaries.setdefault(case.name, {
            "name": case.name,
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
    if cases and settings.experiment_type == "antenna_gain":
        pattern = summarize_pattern_gain(rows)
        summary["pattern_gain"] = pattern
        for name, outcome in pattern["cases"].items():
            verification["checks"][f"pattern_gain:{name}"] = {
                "status": outcome["status"], "passed_flag": outcome["status"] == "passed",
                "reason": "independent gain, input power and weight checks",
            }
        verification["passed_flag"] = verification["passed_flag"] and pattern["passed_flag"]
    if cases and settings.experiment_type == "mutual_coupling":
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
    if cases and settings.experiment_type == "phase_center_offset":
        phase_center = summarize_phase_center(rows)
        summary["phase_center"] = phase_center
        verification["checks"].update(phase_center["checks"])
        verification["passed_flag"] = verification["passed_flag"] and phase_center["passed_flag"]
    return summary
