"""实际接收链：天线端电压 → RF/ADC → DFE → 合并 → 捕获跟踪。"""

import numpy as np

from .acquisition import acquire
from .config import SimulationConfig
from .dfe import process_dfe
from .diversity import combine_egc, combine_mvdr, combine_single
from .interference.combining import combine_blind_null
from .gnss import generate_scenario
from .metrics import compute_metrics
from .models import (
    BranchResult,
    CombiningInfo,
    ScenarioTruth,
    SimulationResults,
    TrackingResult,
)
from .presets import default_config
from .rf import process_rf
from .tracking import track


def _run_branch(
    mode: str,
    dfe_channels: np.ndarray,
    cfg: SimulationConfig,
    fs_out_hz: float,
) -> BranchResult:
    """运行不读取目标响应的实际接收分支。"""
    if mode == "single":
        combined_samples, combining_info = combine_single(dfe_channels)
    elif mode == "blind_null":
        combined_samples, combining_info = combine_blind_null(
            dfe_channels,
            cfg.diversity.blind_null,
        )
    else:
        raise ValueError(f"Unsupported actual receiver mode: {mode}")

    return _finish_branch(
        combined_samples,
        combining_info,
        cfg,
        fs_out_hz,
    )


def _finish_branch(
    combined_samples: np.ndarray,
    combining_info: CombiningInfo,
    cfg: SimulationConfig,
    fs_out_hz: float,
) -> BranchResult:
    """以一维合并复采样完成捕获跟踪；求权信息不混入真值评分。"""
    acquisition = acquire(
        combined_samples,
        fs_out_hz,
        cfg.signal.prn,
        cfg.signal.code_rate_hz,
        cfg.acquisition,
    )
    if acquisition["success_flag"]:
        tracking = track(
            combined_samples,
            fs_out_hz,
            cfg.signal.prn,
            acquisition,
            cfg.tracking,
            cfg.signal.carrier_hz,
            cfg.signal.code_rate_hz,
        )
    else:
        tracking: TrackingResult = {
            "success_flag": False,
            "status": "acquisition_failed",
            "prompt": np.array([], dtype=complex),
        }
    return {
        "combined_samples": combined_samples,
        "combining_info": combining_info,
        "acquisition": acquisition,
        "tracking": tracking,
    }


def run_end_to_end(
    cfg: SimulationConfig | None = None,
    *,
    scene: tuple[np.ndarray, ScenarioTruth] | None = None,
    retain_analysis_flag: bool = False,
) -> SimulationResults:
    """运行完整真实链并返回各方法结果。

    scene 为可选的天线端复电压 [通道, RF采样] 与真值；未给定时按 cfg 生成。
    仅在 retain_analysis_flag=True 时保存本次热噪声供事后线性评分，
    接收分支不会读取评分分量。
    """

    results = run_frontend(cfg, scene=scene, retain_analysis_flag=retain_analysis_flag)
    results["branches"] = run_receiver(
        results["dfe_channels"], results["config"],
        results["dfe_info"]["fs_out_hz"],
    )
    results["metrics"] = compute_metrics(results)
    return results


def run_frontend(
    cfg: SimulationConfig | None = None,
    *,
    scene: tuple[np.ndarray, ScenarioTruth] | None = None,
    retain_analysis_flag: bool = False,
) -> SimulationResults:
    """生成一次场景并完成 RF/DFE，返回完整时间序列与各阶段信息。

    输入天线端电压单位 V；RF 输出为复数 ADC 码值，DFE 输出为
    [通道, 采样] 复数码值。默认不保留热噪声评分数组。
    """
    cfg = default_config() if cfg is None else cfg
    scene_input_v, truth = generate_scenario(cfg) if scene is None else scene
    adc_codes, rf_info = process_rf(
        scene_input_v,
        cfg.rf,
        cfg.reference_impedance_ohm,
        cfg.random_seed,
        fs_hz=cfg.signal.fs_rf_hz,
        if_hz=cfg.signal.if_hz,
        channel_ids=cfg.phone.active_antennas,
        retain_analysis_flag=retain_analysis_flag,
    )
    dfe_channels, dfe_info = process_dfe(
        adc_codes,
        cfg.dfe,
        cfg.signal.fs_rf_hz,
        cfg.signal.if_hz,
    )
    results: SimulationResults = {
        "config": cfg,
        "truth": truth,
        "scene_input_v": scene_input_v,
        "adc_codes": adc_codes,
        "rf_info": rf_info,
        "dfe_channels": dfe_channels,
        "dfe_info": dfe_info,
        "branches": {},
        "metrics": [],
    }
    if retain_analysis_flag:
        results["analysis_inputs"] = {
            "thermal_noise_v": rf_info.pop("thermal_noise_v"),
        }
    return results


def run_receiver(
    dfe_channels: np.ndarray,
    cfg: SimulationConfig,
    fs_out_hz: float,
) -> dict[str, BranchResult]:
    """只运行 SINGLE 和 Blind Null 的实际接收；其他模式显式报错。

    输入是 [通道, DFE采样] 复数码值；不传入目标真值或名义响应。
    """
    return {
        mode: _run_branch(
            mode,
            dfe_channels,
            cfg,
            fs_out_hz,
        )
        for mode in cfg.diversity.modes
    }


def run_known_response_comparison(
    dfe_channels: np.ndarray,
    cfg: SimulationConfig,
    fs_out_hz: float,
    *,
    phase_estimates_deg: np.ndarray,
    assumed_target_response: np.ndarray,
    modes: tuple[str, ...],
) -> dict[str, BranchResult]:
    """显式对照入口：EGC 只读外部相位估计，MVDR 读取名义目标复响应。"""
    comparison_branches = {}
    for mode in modes:
        if mode == "egc":
            combined_samples, combining_info = combine_egc(
                dfe_channels, phase_estimates_deg,
            )
        elif mode == "mvdr":
            combined_samples, combining_info = combine_mvdr(
                dfe_channels, assumed_target_response,
                cfg.diversity.diagonal_loading_factor,
            )
        else:
            raise ValueError(f"Unsupported known-response comparison mode: {mode}")
        comparison_branches[mode] = _finish_branch(
            combined_samples, combining_info, cfg, fs_out_hz,
        )
    return comparison_branches
