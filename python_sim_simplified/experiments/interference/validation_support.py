"""通用抗干扰验证：逐条件前端 → 接收 → 独立核验/绘图。

实际接收的 SINGLE 和 Blind Null 根据 DFE 通道采样求权，不依赖名义目标响应。
分量准备和固定权值的分量功率评估暂时停用。
"""

from dataclasses import asdict, dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np

# from core.antenna import antenna_response, build_antenna_model
# from core.feedline import feedline_response
from core.config import SimulationConfig
# from core.evaluation import linear_components
# from core.diversity import combine_known
from core.metrics import circular_chip_error, tracking_cn0_mean
from core.pipeline import run_frontend, run_receiver
from experiments.interference.coupling_support import (
    coupling_row_fields, sample_statistics, summarize_coupling,
)
from experiments.interference.phase_center_support import (
    phase_center_row_fields, summarize_phase_center,
)
from experiments.paths import reset_output


@dataclass
class ValidationCase:
    """一个实验条件：物理/接收配置 + 图上标签 + DFE 相位注入。

    scan_value 是横轴坐标，不会自动修改 config；修改物理参数由配置文件完成。
    分量功率相关的 ideal_noise 核验暂时停用。
    """
    name: str
    config: SimulationConfig
    scene_label: str = "scenario"
    series_label: str = "case"
    scan_label: str = ""
    scan_value: float | None = None
    scan_unit: str = ""
    dfe_phase_deg: dict[str, float] = field(default_factory=dict)
    # 仅记录温度条件；实际幅相增量已在构造用例时写入 config。
    temperature_offset: dict = field(default_factory=dict)
    # checks: tuple[str, ...] = ()  # 分量功率核验暂停。


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


def rf_group_delay_metadata(case_cfg: SimulationConfig) -> dict:
    """按物理通道名记录注入通道及未注入的参考通道，供结果和理论图使用。"""
    offset_ns_by_antenna = case_cfg.rf.group_delay_offset_ns.copy()
    affected_antenna = next(iter(offset_ns_by_antenna))
    reference_antenna = next(antenna for antenna in case_cfg.phone.active_antennas
                             if antenna != affected_antenna)
    return {"offset_ns_by_antenna": offset_ns_by_antenna,
            "affected_antenna": affected_antenna,
            "reference_antenna": reference_antenna,
            "dfe_cutoff_hz": case_cfg.dfe.cutoff_hz}


# 分量功率核验暂停；保留估计导向计算，恢复 ideal_noise 时可复用。
# def _assumed_steering(cfg: SimulationConfig) -> np.ndarray:
#     """为理想噪声适用性检查计算RF输入目标导向，含名义馈线。"""
#     model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
#     antenna_steering, _ = antenna_response(
#         model, cfg.signal.assumed_direction, cfg.signal.assumed_polarization,
#         use_assumed_state_flag=True,
#     )
#     return antenna_steering * feedline_response(
#         cfg.feedline, cfg.phone.active_antennas, cfg.signal.carrier_hz,
#         cfg.signal.carrier_hz + cfg.signal.doppler_hz,
#         use_assumed_state_flag=True,
#     )


def _power(values: np.ndarray) -> float:
    """复数码值均方功率（码值平方），不是瓦特；同尺度功率比可转换为 dB。"""
    return float(np.mean(np.abs(values) ** 2))


def evaluate_case(
    case: ValidationCase, result: dict, settings: ValidationSettings,
) -> list[dict]:
    """对一次已完成的接收独立复算实际输出与接收指标；返回标量行。"""
    case_cfg = case.config
    seed = case_cfg.random_seed
    frontend = result["frontend"]
    receiver_samples = result["receiver_samples"]
    branches = result["branches"]
    # 分量准备暂停；真实 DFE 与接收仍由 run_case 完整执行。
    # components = linear_components(frontend)
    # components = {
    #     name: values * result["injection_factors"]
    #     for name, values in components.items()
    # }
    fs_out_hz = frontend["dfe_info"]["fs_out_hz"]
    edge_trim_samples = round(settings.evaluation_edge_trim_ms * 1e-3 * fs_out_hz)
    evaluation_window = slice(edge_trim_samples, receiver_samples.shape[1] - edge_trim_samples)
    # 只裁指标窗口；捕获跟踪已使用完整序列。R 为各通道零时延样本协方差。
    evaluation_samples = receiver_samples[:, evaluation_window]
    covariance = evaluation_samples @ evaluation_samples.conj().T / evaluation_samples.shape[1]
    pattern_fields = {}
    if settings.experiment_type == "antenna_gain":
        # 注入增益是物理条件；分量输出收益暂时停用。
        gain_db = tuple(case_cfg.phone.antenna_model.true_state.amplitude_error_db.get(name, 0.0)
                        for name in case_cfg.phone.active_antennas)
        pattern_fields = {"input_gain_db_by_antenna": list(gain_db)}
    # 公共指标：每种方法复用同一前端采样与接收机实际权值。
    rows = []
    for mode, branch_result in branches.items():
        weights = branch_result["combining_info"]["weights"]
        output_samples = branch_result["combined_samples"][evaluation_window]
        # 分量投影、功率、交叉项及量化残差暂时停用。
        # projected = {
        #     name: weights.conj() @ values[:, evaluation_window]
        #     for name, values in components.items()
        # }
        # powers = {name: _power(values) for name, values in projected.items()}
        # interference = {name: power for name, power in powers.items()
        #                 if name not in ("target", "thermal")}
        # linear_sum = sum(projected.values())
        acquisition = branch_result["acquisition"]
        tracking = branch_result["tracking"]
        cn0_mean_db_hz, cn0_valid_count = tracking_cn0_mean(tracking)
        code_estimate = (acquisition["code_phase_samples"]
                         * case_cfg.signal.code_rate_hz / fs_out_hz)
        code_error_chips = float(circular_chip_error(
            code_estimate, case_cfg.signal.code_delay_chips,
        ))
        doppler_error_hz = float(acquisition["doppler_hz"] - case_cfg.signal.doppler_hz)
        # 单频点没有网格间隔，频率正确要求搜索频点恰等于真值。
        doppler_tolerance_hz = (float(np.diff(case_cfg.acquisition.doppler_bins_hz)[0] / 2)
                                if len(case_cfg.acquisition.doppler_bins_hz) > 1 else 0.0)
        covariance_output_power = float(np.vdot(weights, covariance @ weights).real)
        # 两条独立路径：实际合并序列均方，和输入协方差二次型 wᴴRw。
        output_power = _power(output_samples)
        row = {
            "case": case.name, "seed": seed, "scene": case.scene_label,
            "series": case.series_label, "scan_label": case.scan_label,
            "scan_value": case.scan_value, "scan_unit": case.scan_unit,
            "mode": mode,
            "receiver_role": branch_result["receiver_role"],
            "phase_estimate_source": branch_result["phase_estimate_source"],
            "antennas": ",".join(case_cfg.phone.active_antennas),
            "single_reference_antenna": case_cfg.phone.active_antennas[0],
            # "target_power": powers["target"], "thermal_power": powers["thermal"],
            # "interferer_powers": [{"source": name, "power": power}
            #                       for name, power in interference.items()],
            # "joint_nonlinear_quantization_residual_power": _power(output_samples - linear_sum),
            # "linear_component_cross_power": _power(linear_sum) - sum(powers.values()),
            "actual_power": output_power,
            # "linear_in_band_sinr_db": float(10 * np.log10(
            #     powers["target"] / (powers["thermal"] + sum(interference.values()))
            # )),
            # "thermal_snr_db": float(10 * np.log10(powers["target"] / powers["thermal"])),
            "weight_real_imag": [[float(value.real), float(value.imag)] for value in weights],
            "weight_norm_squared": float(np.vdot(weights, weights).real),
            "combined_direct_max_abs_error": float(np.max(np.abs(
                output_samples - weights.conj() @ evaluation_samples
            ))),
            "quadratic_power_relative_error": abs(output_power - covariance_output_power) / output_power,
            "acquisition_metric": float(acquisition["metric"]),
            "acquisition_threshold": case_cfg.acquisition.threshold,
            "acquired_flag": bool(acquisition["success_flag"]),
            "acquired_code_error_chips": code_error_chips,
            "acquired_doppler_error_hz": doppler_error_hz,
            "acquisition_correct_flag": bool(
                acquisition["success_flag"]
                and abs(code_error_chips) <= 0.5
                and abs(doppler_error_hz) <= doppler_tolerance_hz
            ),
            "tracking_success_flag": bool(tracking["success_flag"]),
            "tracking_status": tracking["status"],
            "tracking_cn0_mean_db_hz": cn0_mean_db_hz,
            "tracking_cn0_valid_count": cn0_valid_count,
        }
        # 专项字段只记录实验条件和目标真实响应。
        if pattern_fields:
            row.update(pattern_fields)
        if case.temperature_offset:
            row["temperature_offset"] = case.temperature_offset
        if settings.experiment_type == "phase_offset":
            row["dfe_phase_deg"] = case.dfe_phase_deg
        if settings.experiment_type == "rf_group_delay":
            row["rf_group_delay"] = rf_group_delay_metadata(case_cfg)
        if settings.experiment_type == "mutual_coupling":
            row.update(coupling_row_fields(
                case, frontend["truth"]["desired_steering_true"], weights,
            ))
        if settings.experiment_type == "phase_center_offset":
            row.update(phase_center_row_fields(
                case, frontend["truth"]["desired_steering_true"], weights,
            ))
        rows.append(row)
    # 分量收益暂停；不再额外执行 SINGLE 参考合并。
    # _, reference_info = combine_known(
    #     receiver_samples, frontend["truth"]["desired_steering_assumed"], "single",
    #     case_cfg.diversity.diagonal_loading_factor,
    # )
    # reference_weights = reference_info["weights"]
    # reference_target = _power(reference_weights.conj() @ components["target"][:, evaluation_window])
    # reference_thermal = _power(reference_weights.conj() @ components["thermal"][:, evaluation_window])
    # for row in rows:
    #     row["target_output_gain_db"] = float(10 * np.log10(
    #         row["target_power"] / reference_target
    #     ))
    #     row["thermal_snr_gain_db"] = float(10 * np.log10(
    #         (row["target_power"] / row["thermal_power"])
    #         / (reference_target / reference_thermal)
    #     ))
    #     row["interferer_suppression_db"] = {
    #         item["source"]: float(10 * np.log10(
    #             _power(reference_weights.conj() @ components[item["source"]][:, evaluation_window])
    #             / item["power"]
    #         )) for item in row["interferer_powers"]
    #     }
    return rows


# ideal_noise 适用性只服务已暂停的分量 SNR 核验。
# def _ideal_noise_applicability(case: ValidationCase) -> str | None:
#     """检查等幅目标下 MRC 的 10log10(N) 前提；返回不适用原因。"""
#     case_cfg = case.config
#     if any(source.enable_flag for source in case_cfg.interferers):
#         return "external interference is enabled"
#     if any(case_cfg.feedline.delay_ns.get(name, 0.0) != 0
#            for name in case_cfg.phone.active_antennas):
#         return "true sequence delays need a wideband alignment analysis"
#     if np.ptp(np.broadcast_to(case_cfg.rf.amplitude_error_db, len(case_cfg.phone.active_antennas))) != 0:
#         return "RF channel amplitudes differ"
#     if np.ptp(np.broadcast_to(case_cfg.rf.phase_error_deg, len(case_cfg.phone.active_antennas))) != 0:
#         return "RF channel phases differ"
#     if any(case.dfe_phase_deg.get(name, 0.0) != 0.0
#            for name in case_cfg.phone.active_antennas):
#         return "DFE phases are injected"
#     model = build_antenna_model(case_cfg.phone, case_cfg.signal.carrier_hz)
#     true_steering, _ = antenna_response(
#         model, case_cfg.signal.true_direction, case_cfg.signal.true_polarization,
#     )
#     assumed_steering = _assumed_steering(case_cfg)
#     if not np.allclose(true_steering, assumed_steering, rtol=1e-12, atol=1e-12):
#         return "true and assumed target manifolds differ"
#     if not np.allclose(np.abs(true_steering), np.abs(true_steering[0]), rtol=1e-12, atol=1e-12):
#         return "element target gains are not equal"
#     return None


def summarize_pattern_gain(rows: list[dict]) -> dict:
    """按物理 TC 和方法汇总注入增益及接收统计。"""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["case"], row["receiver_role"], row["mode"]), []).append(row)

    cases = {}
    for (case_id, receiver_role, mode), observations in grouped.items():
        name = f"{case_id}-{mode}"
        acquired = sum(row["acquired_flag"] for row in observations)
        correct = sum(row["acquisition_correct_flag"] for row in observations)
        tracked = sum(row["tracking_success_flag"] for row in observations)
        finite_cn0 = [row["tracking_cn0_mean_db_hz"] for row in observations
                      if row["tracking_cn0_mean_db_hz"] is not None]
        cases[name] = {
            "case": case_id, "mode": mode, "receiver_role": receiver_role,
            "phase_estimate_source": observations[0]["phase_estimate_source"],
            "seeds": len(observations),
            "antennas": observations[0]["antennas"],
            "input_gain_db_by_antenna": observations[0]["input_gain_db_by_antenna"],
            # "target_output_gain_db": sample_statistics(
            #     [row["target_output_gain_db"] for row in observations]),
            # "thermal_snr_gain_db": sample_statistics(
            #     [row["thermal_snr_gain_db"] for row in observations]),
            # "thermal_snr_db": sample_statistics(
            #     [row["thermal_snr_db"] for row in observations]),
            "acquisition_peak_ratio": sample_statistics(
                [row["acquisition_metric"] for row in observations]),
            "acquisition_success_count": acquired,
            "acquisition_correct_count": correct,
            "tracking_success_count": tracked,
            "tracking_cn0_missing_count": len(observations) - len(finite_cn0),
            "tracking_cn0_observed_db_hz": sample_statistics(finite_cn0),
        }
    return {"cases": cases}


def verify_validation(rows: list[dict]) -> dict:
    """核验实际合并序列与固定权值的公共恒等式。"""
    checks = {}
    # 每行检查线性合并恒等式；分量热噪声收益核验暂时停用。
    for row in rows:
        key = f'math:{row["case"]}:{row["seed"]}:{row["receiver_role"]}:{row["mode"]}'
        direct_flag = row["combined_direct_max_abs_error"] < 1e-8
        quadratic_flag = row["quadratic_power_relative_error"] < 1e-10
        checks[key] = {"status": "passed" if direct_flag and quadratic_flag else "failed",
                       "passed_flag": direct_flag and quadratic_flag,
                       "reason": "fixed-weight direct and covariance identity"}
        # case = cases[row["case"], row["seed"]]
        # if "ideal_noise" in case.checks and row["mode"] == "mrc":
        #     reason = _ideal_noise_applicability(case)
        #     if reason is not None:
        #         checks[f'ideal_noise:{row["case"]}:{row["seed"]}:mrc'] = {
        #             "status": "not_applicable", "passed_flag": None, "reason": reason,
        #         }
        #         continue
        #     theory = 10 * np.log10(len(case.config.phone.active_antennas))
        #     error = abs(row["thermal_snr_gain_db"] - theory)
        #     passed_flag = error < 0.5
        #     checks[f'ideal_noise:{row["case"]}:{row["seed"]}:mrc'] = {
        #         "status": "passed" if passed_flag else "failed", "passed_flag": passed_flag,
        #         "reason": f"MRC thermal SNR error {error:.6g} dB; limit 0.5 dB",
        #     }
    return {"passed_flag": all(item["status"] != "failed" for item in checks.values()),
            "checks": checks}


def run_case(case: ValidationCase) -> dict:
    """返回前端结果、相位处理后送接收机的 [天线, 时间] 采样及各分支。"""
    case_cfg = case.config
    antennas = case_cfg.phone.active_antennas
    # frontend = run_frontend(case_cfg, retain_analysis_flag=True)  # 分量重放暂停。
    frontend = run_frontend(case_cfg)
    # 在 DFE 输出处对整通道只旋转一次；零度也走相同路径。
    receiver_samples = apply_dfe_phase(frontend["dfe_channels"], antennas, case.dfe_phase_deg)
    # 对照输入仅取原始名义目标响应，不把 DFE 注入相位当作目标相位估计。
    assumed_target_response = frontend["truth"]["desired_steering_assumed"]
    phase_estimates_deg = np.rad2deg(np.angle(assumed_target_response))
    branches = run_receiver(
        receiver_samples, case_cfg, frontend["dfe_info"]["fs_out_hz"],
        phase_estimates_deg=phase_estimates_deg,
        assumed_target_response=assumed_target_response,
        phase_estimate_source="preset_assumed_target_response_phase",
    )
    return {
        "frontend": frontend, "receiver_samples": receiver_samples, "branches": branches,
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
            # "checks": case.checks,  # ideal_noise 标签暂停。
        })
        if case.temperature_offset:
            summary["temperature_offset"] = case.temperature_offset
        summary["receiver_configs"][str(case.config.random_seed)] = asdict(case.config)
    # by_identity = {(case.name, case.config.random_seed): case for case in cases}
    verification = verify_validation(rows)
    summary = {"cases": list(case_summaries.values()), "rows": rows,
               "verification": verification}
    if cases and settings.experiment_type == "antenna_gain":
        summary["pattern_gain"] = summarize_pattern_gain(rows)
    if cases and settings.experiment_type == "mutual_coupling":
        summary["mutual_coupling"] = summarize_coupling(rows)
    if cases and settings.experiment_type == "phase_center_offset":
        summary["phase_center"] = summarize_phase_center(rows)
    return summary
