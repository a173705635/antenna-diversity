"""Typed descriptions of dictionaries exchanged between simulation stages."""

from typing import NotRequired, TypedDict

import numpy as np

from .config import Direction, SimulationConfig


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
    channel_response: np.ndarray
    effective_manifold: np.ndarray


class AntennaModelDiagnostics(TypedDict):
    desired_true: AntennaDiagnostics
    desired_assumed: AntennaDiagnostics
    interferers: list[AntennaDiagnostics]
    multipath_paths: list[AntennaDiagnostics]


class MultipathTruth(TypedDict):
    component: np.ndarray
    steering_true: np.ndarray
    delay_phase_true: np.ndarray
    excess_delay_chips: float
    excess_delay_s: float
    relative_power_db: float
    reflection_phase_rad: float
    doppler_offset_hz: float
    direction: Direction


class MultipathCandidate(TypedDict):
    excess_delay_chips: float
    residual_power: float
    detection_score_db: float
    direction: Direction
    steering_estimated: np.ndarray
    steering_correlation_with_los: float
    spatial_status: str
    azimuths_deg: np.ndarray
    elevations_deg: np.ndarray
    angle_power: np.ndarray


class MultipathDetectionResult(TypedDict):
    status: str
    pilot_mode: str | None
    delay_axis_chips: np.ndarray
    correlation_vectors: np.ndarray
    raw_power: np.ndarray
    residual_power: np.ndarray
    threshold_power: float
    candidates: list[MultipathCandidate]


class MultipathMatch(TypedDict):
    detected_index: int
    truth_index: int
    delay_error_chips: float
    direction_error_deg: float


class MultipathEvaluation(TypedDict):
    detected_count: int
    missed_count: int
    false_alarm_count: int
    matches: list[MultipathMatch]


class ScenarioTruth(TypedDict):
    source_response_convention: NotRequired[str]
    source_response_coefficients: NotRequired[dict[str, object]]
    time_s: np.ndarray
    code: np.ndarray
    desired_waveform: np.ndarray
    desired_component: np.ndarray
    desired_steering_true: np.ndarray
    desired_steering_assumed: np.ndarray
    desired_open_circuit_response_true: np.ndarray
    desired_open_circuit_response_assumed: np.ndarray
    interferer_steering_effective: list[np.ndarray]
    interferer_components: list[np.ndarray]
    multipath_paths: list[MultipathTruth]
    active_antennas: tuple[str, ...]
    positions_m: np.ndarray
    channel_delay_true_ns: np.ndarray
    channel_delay_estimated_ns: np.ndarray
    channel_delay_true_samples: np.ndarray
    channel_delay_estimated_samples: np.ndarray
    desired_delay_phase_true: np.ndarray
    desired_delay_phase_assumed: np.ndarray
    interferer_delay_phases_true: list[np.ndarray]
    doppler_hz: float
    code_rate_hz: float
    code_delay_chips: float
    fs_rf_hz: float
    if_hz: float
    antenna_model_diagnostics: AntennaModelDiagnostics


class RFInfo(TypedDict):
    fs_hz: float
    if_hz: float
    channel_ids: tuple[str, ...]
    filter_coefficients: np.ndarray
    noise_bandwidth_hz: float
    measured_noise_bandwidth_hz: float
    group_delay_compensation_samples: int
    noise_psd_w_per_hz: float
    total_gain_db: float
    channel_total_gain_db: np.ndarray
    amplitude_error_db: np.ndarray
    phase_error_deg: np.ndarray
    noise_figure_db: float
    input_noise_power_w: float
    input_noise_rms_v: float
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
    weights_are_fixed: bool
    sample_covariance: np.ndarray
    condition_number: float
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
    interference_detected: NotRequired[bool]
    nulling_enabled: NotRequired[bool]
    reference_projection_power: NotRequired[float]
    estimated_interference_response: NotRequired[complex]
    constraint_matrix: NotRequired[np.ndarray]
    constraint_response: NotRequired[np.ndarray]
    multipath_constraint_steerings: NotRequired[list[np.ndarray]]
    multipath_constraint_responses: NotRequired[list[complex]]
    suppression_status: NotRequired[str]
    rejected_constraint_indices: NotRequired[list[int]]
    target_direction_gain: NotRequired[float]
    interferer_direction_gains: NotRequired[list[float]]
    worst_interferer_gain: NotRequired[float]
    multipath_direction_gains: NotRequired[list[float]]
    worst_multipath_gain: NotRequired[float]


class AcquisitionResult(TypedDict):
    success: bool
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


class TrackingInitialization(TypedDict):
    code_phase_offset_chips: float
    carrier_phase_rad: float
    at_search_boundary: bool
    samples_used: int
    uses_truth: bool


class TrackingResult(TypedDict):
    success: bool
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
    combined: np.ndarray
    combining: CombiningInfo
    acquisition: AcquisitionResult
    tracking: TrackingResult


class MetricRecord(TypedDict):
    mode: str
    acquired: bool
    acquisition_metric: float
    acquired_doppler_hz: float
    target_direction_gain: float
    worst_interferer_gain: float
    tracking: bool
    final_doppler_hz: float | None
    pll_rms_rad: float | None
    dll_rms: float | None
    mean_prompt_magnitude: float | None
    final_cn0_db_hz: float | None
    multipath_direction_gains: list[float]
    worst_multipath_gain: float
    acquisition_code_bias_chips: float
    tracking_code_bias_rms_chips: float | None
    tracking_code_bias_rms_m: float | None


class SimulationResults(TypedDict):
    config: SimulationConfig
    truth: ScenarioTruth
    scene_input_v: np.ndarray
    adc_codes: np.ndarray
    rf_info: RFInfo
    dfe_channels: np.ndarray
    dfe_info: DFEInfo
    branches: dict[str, BranchResult]
    multipath_detection: MultipathDetectionResult
    multipath_evaluation: MultipathEvaluation | None
    metrics: list[MetricRecord]
