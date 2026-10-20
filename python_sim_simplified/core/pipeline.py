"""实际接收链：天线端电压 → RF/ADC → DFE → 合并 → 捕获跟踪。"""

import numpy as np

from .acquisition import acquire
from .antenna import process_antenna
from .config import SimulationConfig
from .dfe import process_dfe
from .diversity import combine_egc, combine_mvdr, combine_single
from .interference.combining import combine_blind_null
from .gnss import generate_signal_sources
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


def _finish_branch(
    combined_samples: np.ndarray,
    combining_info: CombiningInfo,
    cfg: SimulationConfig,
    fs_out_hz: float,
    receiver_role: str,
    phase_estimate_source: str | None,
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
        "receiver_role": receiver_role,
        "phase_estimate_source": phase_estimate_source,
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
    phase_estimates_deg: np.ndarray | None = None,
    assumed_target_response: np.ndarray | None = None,
    phase_estimate_source: str = "external_input",
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
        phase_estimates_deg=phase_estimates_deg,
        assumed_target_response=assumed_target_response,
        phase_estimate_source=phase_estimate_source,
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
    if scene is None:
        signal_sources = generate_signal_sources(cfg)
        scene_input_v, truth = process_antenna(signal_sources, cfg)
        del signal_sources
    else:
        scene_input_v, truth = scene
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
    *,
    phase_estimates_deg: np.ndarray | None = None,
    assumed_target_response: np.ndarray | None = None,
    phase_estimate_source: str = "external_input",
) -> dict[str, BranchResult]:
    """按配置顺序运行所有模式；对照输入必须由调用方显式提供。

    DFE 输入形状为 [通道, 时间采样]；EGC 只读相位估计（度），
    MVDR 只读名义目标复响应，不从真值或配置隐式推断。
    """
    branches = {}
    for mode in cfg.diversity.modes:
        if mode == "single":
            combined_samples, combining_info = combine_single(dfe_channels)
        elif mode == "blind_null":
            combined_samples, combining_info = combine_blind_null(
                dfe_channels, cfg.diversity.blind_null,
            )
        elif mode == "egc":
            if phase_estimates_deg is None:
                raise ValueError("EGC requires phase_estimates_deg")
            combined_samples, combining_info = combine_egc(dfe_channels, phase_estimates_deg)
        elif mode == "mvdr":
            if assumed_target_response is None:
                raise ValueError("MVDR requires assumed_target_response")
            combined_samples, combining_info = combine_mvdr(
                dfe_channels, assumed_target_response,
                cfg.diversity.diagonal_loading_factor,
            )
        else:
            raise ValueError(f"Unsupported receiver mode: {mode}")
        receiver_role = ("actual" if mode in ("single", "blind_null")
                         else "known_response_comparison")
        branches[mode] = _finish_branch(
            combined_samples, combining_info, cfg, fs_out_hz, receiver_role,
            phase_estimate_source if mode == "egc" else None,
        )
    return branches
