"""Typed descriptions of dictionaries exchanged between simulation stages."""

from typing import NotRequired, TypedDict

import numpy as np

from .config import Interferer, SimulationConfig


class AntennaDiagnostics(TypedDict):
    positions_m: np.ndarray
    spatial_phase: np.ndarray
    pattern_polarization_voltage: np.ndarray
    pcv_path_length_mm: np.ndarray
    pcv_response: np.ndarray
    element_error_response: np.ndarray
    blockage_voltage: np.ndarray
    open_circuit_response: np.ndarray
    coupling_matrix: np.ndarray
    coupled_response: np.ndarray
    antenna_transfer_matrix: np.ndarray
    effective_manifold: np.ndarray


class AntennaModelDiagnostics(TypedDict):
    desired_true: AntennaDiagnostics
    desired_assumed: AntennaDiagnostics
    interferers: list[AntennaDiagnostics]


class InterfererSignal(TypedDict):
    """一个干扰源的配置与已门控一维参考电压，尚未经过天线。"""
    interferer: Interferer
    waveform_v: np.ndarray
    center_offset_hz: float


class SignalSources(TypedDict):
    """目标及各干扰源的参考电压；waveform 均为一维时间采样。"""
    time_s: np.ndarray
    code: np.ndarray
    desired_waveform: np.ndarray
    received_code_rate_hz: float
    interferers: list[InterfererSignal]


class ScenarioTruth(TypedDict):
    time_s: np.ndarray
    code: np.ndarray
    desired_waveform: np.ndarray
    desired_input_component_v: np.ndarray
    desired_steering_true: np.ndarray
    desired_steering_assumed: np.ndarray
    desired_open_circuit_response_true: np.ndarray
    desired_open_circuit_response_assumed: np.ndarray
    interferer_steering_effective: list[np.ndarray]
    interferer_input_components_v: list[np.ndarray]
    active_antennas: tuple[str, ...]
    positions_m: np.ndarray
    doppler_hz: float
    code_rate_hz: float
    code_delay_chips: float
    fs_rf_hz: float
    if_hz: float
    antenna_model_diagnostics: AntennaModelDiagnostics


class RFInfo(TypedDict):
    added_noise_v: NotRequired[np.ndarray]
    fs_hz: float
    if_hz: float
    channel_ids: tuple[str, ...]
    filter_coefficients: np.ndarray
    noise_bandwidth_hz: float
    measured_noise_bandwidth_hz: float
    group_delay_compensation_samples: int
    noise_reference_temperature_k: float
    added_noise_temperature_k: float
    added_noise_psd_w_per_hz: float
    total_gain_db: float
    channel_total_gain_db: np.ndarray
    amplitude_error_db: np.ndarray
    phase_error_deg: np.ndarray
    noise_figure_db: float
    added_noise_power_w: float
    added_noise_rms_v: float
    adc_lsb_v: float
    saturation_fraction: np.ndarray
    input_rms_v: np.ndarray
    adc_input_rms_v: np.ndarray
    adc_output_rms_code: np.ndarray
    adc_bits: int
    adc_vpp: float


class DFEInfo(TypedDict):
    fs_in_hz: float
    fs_out_hz: float
    decimation: int
    cutoff_hz: float
    num_taps: int
    group_delay_input_samples: int
    if_hz: float
    output_bits: int
    output_scale: int
    output_length: int
    filter_coefficients: np.ndarray


class CombiningInfo(TypedDict):
    mode: str
    weights: np.ndarray
    weights_are_fixed_flag: bool
    sample_covariance: NotRequired[np.ndarray]
    condition_number: NotRequired[float]
    covariance_snapshots: NotRequired[int]
    constraint_steering: NotRequired[np.ndarray]
    desired_response: NotRequired[complex]
    loaded_covariance: NotRequired[np.ndarray]
    diagonal_loading: NotRequired[float]
    weight_norm: NotRequired[float]
    channel_powers: NotRequired[np.ndarray]
    normalized_correlation: NotRequired[np.ndarray]
    correlation_eigenvalues: NotRequired[np.ndarray]
    detection_score_db: NotRequired[float]
    detection_threshold_db: NotRequired[float]
    estimated_interference_vector: NotRequired[np.ndarray]
    reference_channel: NotRequired[int]
    interference_detected_flag: NotRequired[bool]
    nulling_enabled_flag: NotRequired[bool]
    reference_projection_power: NotRequired[float]
    estimated_interference_response: NotRequired[complex]
    constraint_matrix: NotRequired[np.ndarray]
    constraint_response: NotRequired[np.ndarray]


class AcquisitionResult(TypedDict):
    success_flag: bool
    prn: int
    doppler_hz: float
    code_phase_samples: int
    peak: float
    second_peak: float
    second_peak_code_phase_samples: int
    exclusion_samples: int
    metric: float
    doppler_bins_hz: np.ndarray
    search_power: np.ndarray
    samples_per_ms: int
    coherent_samples: int
    input_end_sample: int  # 捕获已使用输入的右开端点，相对本次捕获输入


class TrackingInitialization(TypedDict):
    code_phase_offset_chips: float
    carrier_phase_rad: float
    at_search_boundary_flag: bool
    samples_used: int
    uses_truth_flag: bool


class TrackingResult(TypedDict):
    success_flag: bool
    status: str
    prompt: np.ndarray
    first_epoch_sample: NotRequired[int]
    epoch_start_samples: NotRequired[np.ndarray]
    early: NotRequired[np.ndarray]
    late: NotRequired[np.ndarray]
    dll_error: NotRequired[np.ndarray]
    pll_error_rad: NotRequired[np.ndarray]
    fll_error_hz: NotRequired[np.ndarray]
    carrier_frequency_hz: NotRequired[np.ndarray]
    carrier_nco_phase_rad: NotRequired[np.ndarray]
    carrier_nco_frequency_hz: NotRequired[np.ndarray]
    code_nco_phase_chips: NotRequired[np.ndarray]
    code_nco_rate_hz: NotRequired[np.ndarray]
    code_rate_hz: NotRequired[np.ndarray]
    code_phase_chips: NotRequired[np.ndarray]
    prompt_magnitude: NotRequired[np.ndarray]
    phase_lock_metric: NotRequired[np.ndarray]
    cn0_db_hz: NotRequired[np.ndarray]
    initialization: NotRequired[TrackingInitialization]


class BranchResult(TypedDict):
    # 合并输出为一维接收机复采样；求权信息另存，供复算和绘图读取。
    receiver_role: str
    phase_estimate_source: str | None
    combined_samples: np.ndarray
    combining_info: CombiningInfo
    acquisition: AcquisitionResult
    tracking: TrackingResult


class MetricRecord(TypedDict):
    mode: str
    receiver_role: str
    phase_estimate_source: str | None
    acquired_flag: bool
    acquisition_metric: float
    acquired_doppler_hz: float
    target_direction_gain: float
    worst_interferer_gain: float
    tracking_flag: bool
    final_doppler_hz: float | None
    pll_rms_rad: float | None
    dll_rms: float | None
    mean_prompt_magnitude: float | None
    final_cn0_db_hz: float | None
    acquisition_code_bias_chips: float
    tracking_code_bias_rms_chips: float | None
    tracking_code_bias_rms_m: float | None


class SimulationResults(TypedDict):
    analysis_inputs: NotRequired[dict[str, np.ndarray]]
    config: SimulationConfig
    truth: ScenarioTruth
    scene_input_v: np.ndarray
    noise_info: dict[str, object]
    adc_codes: np.ndarray
    rf_info: RFInfo
    dfe_channels: np.ndarray
    dfe_info: DFEInfo
    branches: dict[str, BranchResult]
    metrics: list[MetricRecord]
