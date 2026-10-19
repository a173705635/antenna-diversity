"""实际接收链：天线端电压 → RF/ADC → DFE → 合并 → 捕获跟踪。"""

import numpy as np

from .acquisition import acquire
from .config import SimulationConfig
from .dfe import process_dfe
from .diversity import combine_known
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
    truth: ScenarioTruth,
    cfg: SimulationConfig,
    fs_out_hz: float,
) -> BranchResult:
    """一组固定权值处理完整 DFE 序列；结果含合并码值和接收状态。"""
    if mode == "blind_null":
        combined_samples, combining_info = combine_blind_null(
            dfe_channels,
            cfg.diversity.blind_null,
        )
    else:
        combined_samples, combining_info = combine_known(
            dfe_channels,
            truth["desired_steering_assumed"],
            mode,
            cfg.diversity.diagonal_loading_factor,
        )

    return _finish_branch(
        combined_samples,
        combining_info,
        truth,
        cfg,
        fs_out_hz,
        dfe_channels.shape[1],
    )


def _finish_branch(
    combined_samples: np.ndarray,
    combining_info: CombiningInfo,
    truth: ScenarioTruth,
    cfg: SimulationConfig,
    fs_out_hz: float,
    covariance_snapshots: int,
) -> BranchResult:
    """接收一维合并复采样和求权信息，补齐指标并完成捕获跟踪。"""
    weights = combining_info["weights"]
    interferer_gains = [
        float(abs(np.vdot(weights, steering)))
        for steering in truth["interferer_steering_effective"]
    ]
    combining_info.update(
        {
            "target_direction_gain": float(
                abs(np.vdot(weights, truth["desired_steering_true"]))
            ),
            "interferer_direction_gains": interferer_gains,
            "worst_interferer_gain": max(interferer_gains, default=0.0),
            "covariance_snapshots": covariance_snapshots,
            "weights_are_fixed_flag": True,
        }
    )
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
        results["dfe_channels"], results["truth"], results["config"],
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
    truth: ScenarioTruth,
    cfg: SimulationConfig,
    fs_out_hz: float,
) -> dict[str, BranchResult]:
    """对同一份 [通道, DFE采样] 复数码值执行配置中的各合并接收分支。

    fs_out_hz 决定捕获和跟踪的采样时间；合并导向来自配置给出的名义阵列响应。
    """
    return {
        mode: _run_branch(
            mode,
            dfe_channels,
            truth,
            cfg,
            fs_out_hz,
        )
        for mode in cfg.diversity.modes
    }
