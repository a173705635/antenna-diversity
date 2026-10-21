"""接收完成后的线性分量评分；源真值仅供评估使用。"""

import numpy as np

from .dfe import linear_dfe
from .antenna import antenna_transfer_matrix
from .feedline import apply_feedline_transfer
from .models import SimulationResults
from .rf import filter_rf_input


def linear_components(frontend: SimulationResults) -> dict[str, np.ndarray]:
    """把各干净源与三段总热噪声映射到真实 DFE 码值尺度。

    输入须由 retain_analysis_flag=True 的前端产生。返回目标、各干扰和热噪声
    [通道, DFE采样] 复数码值；各源输入已含自身来波幅相，随后经 A 和 B，
    噪声已合成于 RF 输入。
    分量不经过限幅和量化，其和与实际输出的差值由实验层记作联合残差；
    线性分量只用于固定权值功率预算，不反馈到接收链。
    """
    cfg = frontend["config"]
    truth = frontend["truth"]
    rf_info = frontend["rf_info"]
    dfe_info = frontend["dfe_info"]
    thermal_noise_v = frontend["analysis_inputs"]["thermal_noise_v"]
    gain = (10.0 ** (rf_info["channel_total_gain_db"] / 20.0) *
            np.exp(1j * np.deg2rad(rf_info["phase_error_deg"])))
    adc_scale = gain[:, None] / rf_info["adc_lsb_v"]
    output_scale = dfe_info["output_scale"]

    def replay(voltage: np.ndarray) -> np.ndarray:
        # 所有源和三段噪声在 RF 输入处汇合，统一滤波恰一次。
        filtered = filter_rf_input(
            voltage, rf_info["filter_coefficients"], rf_info["fs_hz"], rf_info["if_hz"],
        )
        return linear_dfe(filtered * adc_scale, dfe_info) / output_scale

    transfer = antenna_transfer_matrix(
        cfg.phone.antenna_model.true_state, cfg.phone.active_antennas,
    )

    def source_at_rf_input(input_component_v: np.ndarray) -> np.ndarray:
        return apply_feedline_transfer(
            transfer @ input_component_v, cfg.feedline, cfg.phone.active_antennas,
            cfg.signal.fs_rf_hz, cfg.signal.if_hz,
        )

    components = {"target": replay(source_at_rf_input(truth["desired_input_component_v"]))}
    enabled = [source for source in cfg.interferers if source.enable_flag]
    for index, (source, voltage) in enumerate(zip(enabled, truth["interferer_input_components_v"])):
        components[f"interferer_{index}_{type(source).__name__}"] = replay(
            source_at_rf_input(voltage),
        )
    components["thermal"] = replay(thermal_noise_v)
    return components
