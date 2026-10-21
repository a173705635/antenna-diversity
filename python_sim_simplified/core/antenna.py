"""方向源阵元输入及固定天线传递：几何、极化、方向图、阵元损耗和互耦。"""

from dataclasses import dataclass

import numpy as np

from .config import (
    AntennaState,
    Direction,
    ElementOrientation,
    ParametricPattern,
    PhoneConfig,
    SimulationConfig,
    SPEED_OF_LIGHT_MPS,
    WavePolarization,
)
from .feedline import feedline_response
from .noise import generate_complex_noise
from .measured_pattern import (
    MeasuredGrid,
    interpolate_measured,
    load_measured_pattern,
)
from .models import AntennaDiagnostics, ScenarioTruth, SignalSources


@dataclass
class LocalDirection:
    azimuth_deg: float
    elevation_deg: float


@dataclass
class AntennaModel:
    active_antennas: tuple[str, ...]
    carrier_hz: float
    true_positions_m: dict[str, tuple[float, float, float]]
    assumed_positions_m: dict[str, tuple[float, float, float]]
    true_state: AntennaState
    assumed_state: AntennaState
    true_measured: dict[str, MeasuredGrid] | None
    assumed_measured: dict[str, MeasuredGrid] | None


def direction_vector(direction: Direction) -> np.ndarray:
    azimuth_rad, elevation_rad = np.deg2rad(
        [direction.azimuth_deg, direction.elevation_deg]
    )
    return np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ]
    )


def jones_vector(polarization: WavePolarization) -> np.ndarray:
    """按 [方位, 仰角] 分量构造单位范数的复 Jones 矢量。"""
    if polarization.kind == "rhcp":
        polarization_vector = np.array([1.0, -1.0j])
    elif polarization.kind == "lhcp":
        polarization_vector = np.array([1.0, 1.0j])
    elif polarization.kind == "linear":
        tilt_rad = np.deg2rad(polarization.tilt_deg)
        polarization_vector = np.array([np.cos(tilt_rad), np.sin(tilt_rad)], dtype=complex)
    else:
        polarization_vector = np.array(
            [
                complex(polarization.az_real, polarization.az_imag),
                complex(polarization.el_real, polarization.el_imag),
            ]
        )
    return polarization_vector / np.linalg.norm(polarization_vector)


def rotation_matrix_ypr(orientation: ElementOrientation) -> np.ndarray:
    """将阵元局部列向量转换到手机坐标；Rz @ Ry @ Rx 依次施加 roll、pitch、yaw。"""
    yaw_rad, pitch_rad, roll_rad = np.deg2rad(
        [orientation.yaw_deg, orientation.pitch_deg, orientation.roll_deg]
    )
    cz, sz = np.cos(yaw_rad), np.sin(yaw_rad)
    cy, sy = np.cos(pitch_rad), np.sin(pitch_rad)
    cx, sx = np.cos(roll_rad), np.sin(roll_rad)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def spherical_frame(azimuth_deg: float, elevation_deg: float):
    """返回该方向的径向、方位和仰角单位基向量，使用输入角所在坐标系。"""
    azimuth_rad, elevation_rad = np.deg2rad([azimuth_deg, elevation_deg])
    radial = np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ]
    )
    azimuth_basis = np.array([-np.sin(azimuth_rad), np.cos(azimuth_rad), 0.0])
    elevation_basis = np.array(
        [
            -np.sin(elevation_rad) * np.cos(azimuth_rad),
            -np.sin(elevation_rad) * np.sin(azimuth_rad),
            np.cos(elevation_rad),
        ]
    )
    return radial, azimuth_basis, elevation_basis


def wave_in_element_frame(
    direction: Direction,
    polarization: WavePolarization,
    element_to_phone_rotation: np.ndarray,
) -> tuple[LocalDirection, np.ndarray]:
    """把入射方向及复 Jones 场从手机坐标投影到阵元局部球面基。"""
    incident_direction_phone, azimuth_basis_phone, elevation_basis_phone = spherical_frame(
        direction.azimuth_deg, direction.elevation_deg
    )
    jones_phone = jones_vector(polarization)
    electric_field_phone = jones_phone[0] * azimuth_basis_phone + jones_phone[1] * elevation_basis_phone
    # element_to_phone_rotation 的转置将手机坐标转换回阵元局部坐标。
    incident_direction_local = element_to_phone_rotation.T @ incident_direction_phone
    electric_field_local = element_to_phone_rotation.T @ electric_field_phone
    local_azimuth_deg = np.rad2deg(np.arctan2(incident_direction_local[1], incident_direction_local[0])) % 360.0
    local_elevation_deg = np.rad2deg(np.arcsin(np.clip(incident_direction_local[2], -1.0, 1.0)))
    _, azimuth_basis_local, elevation_basis_local = spherical_frame(local_azimuth_deg, local_elevation_deg)
    # vdot 对基向量取共轭；此处基向量为实值，仍按复内积统一计算。
    local_jones = np.array(
        [np.vdot(azimuth_basis_local, electric_field_local), np.vdot(elevation_basis_local, electric_field_local)]
    )
    local_jones /= np.linalg.norm(local_jones)
    return LocalDirection(local_azimuth_deg, local_elevation_deg), local_jones


def parametric_height(
    pattern_parameters: ParametricPattern,
    local_elevation_deg: float,
    local_azimuth_deg: float = 0.0,
) -> np.ndarray:
    """返回无量纲接收 Jones 系数 [h_az, h_el]；不表示几何高度或米制有效高度。"""
    front_power = max(np.sin(np.deg2rad(local_elevation_deg)), 0.0) ** (
        pattern_parameters.cosine_exponent
    )
    relative_db = -np.inf if front_power == 0.0 else 10.0 * np.log10(front_power)
    gain_db = pattern_parameters.peak_gain_db + max(
        relative_db,
        pattern_parameters.back_lobe_gain_db - pattern_parameters.peak_gain_db,
    )
    azimuth_offset_rad = np.deg2rad(local_azimuth_deg - pattern_parameters.azimuth_peak_deg)
    zenith_factor = np.cos(np.deg2rad(local_elevation_deg)) ** 2
    gain_db -= pattern_parameters.azimuth_attenuation_db * zenith_factor * (
        1.0 - np.cos(azimuth_offset_rad)
    ) / 2.0
    nominal_polarization_vector = jones_vector(pattern_parameters.nominal_polarization)
    orthogonal_polarization_vector = np.array([-np.conj(nominal_polarization_vector[1]), np.conj(nominal_polarization_vector[0])])
    cross_polar_amplitude_ratio = 10.0 ** (-pattern_parameters.cross_polar_discrimination_db / 20.0)
    mixed_polarization_vector = nominal_polarization_vector + cross_polar_amplitude_ratio * np.exp(
        1j * np.deg2rad(pattern_parameters.cross_phase_deg)
    ) * orthogonal_polarization_vector
    return 10.0 ** (gain_db / 20.0) * mixed_polarization_vector / np.linalg.norm(mixed_polarization_vector)


def attenuation_voltage(
    antenna_state: AntennaState,
    active_antenna_names: tuple[str, ...],
    direction: Direction,
) -> np.ndarray:
    """按遮挡区域给出每阵元的无量纲复响应系数，并非实际端口电压。"""
    direction_unit = direction_vector(direction)
    attenuation_db = np.zeros(len(active_antenna_names))
    phase_shift_deg = np.zeros(len(active_antenna_names))
    for attenuation_region in antenna_state.attenuation_regions:
        region_center_direction = direction_vector(
            Direction(attenuation_region.center_azimuth_deg, attenuation_region.center_elevation_deg)
        )
        separation_deg = np.rad2deg(
            np.arccos(np.clip(np.dot(direction_unit, region_center_direction), -1.0, 1.0))
        )
        if attenuation_region.transition_width_deg == 0.0:
            attenuation_fraction = float(separation_deg <= attenuation_region.radius_deg)
        else:
            transition_inner_radius_deg = (
                attenuation_region.radius_deg - attenuation_region.transition_width_deg / 2.0
            )
            transition_outer_radius_deg = (
                attenuation_region.radius_deg + attenuation_region.transition_width_deg / 2.0
            )
            if separation_deg <= transition_inner_radius_deg:
                attenuation_fraction = 1.0
            elif separation_deg >= transition_outer_radius_deg:
                attenuation_fraction = 0.0
            else:
                transition_phase_rad = (
                    np.pi * (separation_deg - transition_inner_radius_deg)
                    / attenuation_region.transition_width_deg
                )
                attenuation_fraction = 0.5 * (1.0 + np.cos(transition_phase_rad))
        affected_antenna_names = attenuation_region.elements or active_antenna_names
        for antenna_index, antenna_name in enumerate(active_antenna_names):
            if antenna_name in affected_antenna_names:
                attenuation_db[antenna_index] += attenuation_fraction * attenuation_region.attenuation_db
                phase_shift_deg[antenna_index] += attenuation_fraction * attenuation_region.phase_shift_deg
    return 10.0 ** (-attenuation_db / 20.0) * np.exp(
        1j * np.deg2rad(phase_shift_deg)
    )


def coupling_matrix(
    antenna_state: AntennaState, active_antenna_names: tuple[str, ...],
) -> np.ndarray:
    """互耦矩阵的行是接收端口、列是来源端口；dB 按电压比用 20 换算。"""
    coupling_coefficients = np.eye(len(active_antenna_names), dtype=complex)
    antenna_index_by_name = {
        antenna_name: antenna_index
        for antenna_index, antenna_name in enumerate(active_antenna_names)
    }
    for coupling_path in antenna_state.coupling_paths:
        if coupling_path.source in antenna_index_by_name and coupling_path.receiver in antenna_index_by_name:
            coupling_coefficients[
                antenna_index_by_name[coupling_path.receiver],
                antenna_index_by_name[coupling_path.source],
            ] = 10.0 ** (
                coupling_path.magnitude_db / 20.0
            ) * np.exp(1j * np.deg2rad(coupling_path.phase_deg))
    return coupling_coefficients


def build_antenna_model(phone_cfg: PhoneConfig, carrier_hz: float) -> AntennaModel:
    """加载真实/名义实测方向图，供同一载波频率复用。"""
    true_state = phone_cfg.antenna_model.true_state
    assumed_state = phone_cfg.antenna_model.assumed_state
    true_measured = (
        load_measured_pattern(true_state.measured_pattern_csv_path, carrier_hz)
        if true_state.pattern_kind == "measured"
        else None
    )
    assumed_measured = (
        load_measured_pattern(
            assumed_state.measured_pattern_csv_path, carrier_hz
        )
        if assumed_state.pattern_kind == "measured"
        else None
    )
    return AntennaModel(
        active_antennas=phone_cfg.active_antennas,
        carrier_hz=carrier_hz,
        true_positions_m=phone_cfg.phase_centers_m,
        assumed_positions_m=phone_cfg.assumed_phase_centers_m,
        true_state=true_state,
        assumed_state=assumed_state,
        true_measured=true_measured,
        assumed_measured=assumed_measured,
    )


def _element_pattern_and_pcv(
    active_antenna_names: tuple[str, ...],
    antenna_state: AntennaState,
    measured_patterns: dict[str, MeasuredGrid] | None,
    direction: Direction,
    polarization: WavePolarization,
) -> tuple[np.ndarray, np.ndarray]:
    """逐阵元计算无量纲极化复响应与 PCV 路径长度（mm）。"""
    pattern_polarization_response = []
    pcv_path_length_mm = []
    for antenna_name in active_antenna_names:
        if antenna_state.pattern_kind == "ideal":
            polarization_response = 1.0 + 0.0j
            pcv_mm = 0.0
        else:
            local_direction, local_jones = wave_in_element_frame(
                direction,
                polarization,
                rotation_matrix_ypr(antenna_state.orientations[antenna_name]),
            )
            if antenna_state.pattern_kind == "parametric":
                pattern_parameters = antenna_state.parametric_patterns[antenna_name]
                element_jones_response = parametric_height(
                    pattern_parameters,
                    local_direction.elevation_deg,
                    local_direction.azimuth_deg,
                )
                local_azimuth_rad = np.deg2rad(local_direction.azimuth_deg)
                horizontal_direction_magnitude = np.cos(np.deg2rad(local_direction.elevation_deg))
                ux = horizontal_direction_magnitude * np.cos(local_azimuth_rad)
                uy = horizontal_direction_magnitude * np.sin(local_azimuth_rad)
                # ux/uy 是阵元局部方向余弦；三个毫米系数给出随方向变化的附加路径。
                qx, qy, qxy = pattern_parameters.phase_variation_mm
                pcv_mm = qx * ux**2 + qy * uy**2 + qxy * ux * uy
            else:
                element_jones_response = interpolate_measured(
                    measured_patterns[antenna_name],
                    local_direction.azimuth_deg,
                    local_direction.elevation_deg,
                )
                # 实测复方向图已含相位，不再另叠 PCV 路径相位。
                pcv_mm = 0.0
            # vdot 对阵元 Jones 响应取共轭，再与入射场的局部 Jones 矢量投影。
            polarization_response = np.vdot(element_jones_response, local_jones)
        pattern_polarization_response.append(polarization_response)
        pcv_path_length_mm.append(pcv_mm)
    return np.asarray(pattern_polarization_response), np.asarray(pcv_path_length_mm)


def antenna_transfer_matrix(
    antenna_state: AntennaState, active_antenna_names: tuple[str, ...],
) -> np.ndarray:
    """返回 A=C diag(sqrt(η))；只含互耦和阵元无源损耗，作用于源和输入噪声。"""
    passive_loss_db = np.array([
        antenna_state.passive_loss_db.get(name, 0.0)
        for name in active_antenna_names
    ])
    voltage_transmission = 10.0 ** (-passive_loss_db / 20.0)
    return coupling_matrix(antenna_state, active_antenna_names) * voltage_transmission[None, :]


def antenna_response(
    antenna_model: AntennaModel,
    direction: Direction,
    polarization: WavePolarization,
    use_assumed_state_flag: bool = False,
) -> tuple[np.ndarray, AntennaDiagnostics]:
    """返回各阵元的复响应系数及分阶段诊断；不是时间采样序列。"""
    active_antenna_names = antenna_model.active_antennas
    antenna_state = (
        antenna_model.assumed_state if use_assumed_state_flag else antenna_model.true_state
    )
    phase_center_positions_by_antenna_m = (
        antenna_model.assumed_positions_m if use_assumed_state_flag else antenna_model.true_positions_m
    )
    measured_patterns = (
        antenna_model.assumed_measured if use_assumed_state_flag else antenna_model.true_measured
    )

    # 阵元相位中心和入射方向确定几何相位。
    phase_center_positions_m = np.array([
        phase_center_positions_by_antenna_m[antenna_name]
        for antenna_name in active_antenna_names
    ])
    spatial_phase_response = np.exp(
        -1j
        * 2.0
        * np.pi
        * antenna_model.carrier_hz
        / SPEED_OF_LIGHT_MPS
        * (phase_center_positions_m @ direction_vector(direction))
    )

    pattern_polarization_response, pcv_path_length_mm = _element_pattern_and_pcv(
        active_antenna_names, antenna_state, measured_patterns, direction, polarization,
    )
    # PCV 路径由毫米转米；正附加路径按接收相位约定产生负相位。
    pcv_response = np.exp(
        -1j * 2.0 * np.pi * antenna_model.carrier_hz / SPEED_OF_LIGHT_MPS
        * pcv_path_length_mm * 1e-3
    )

    blockage_response = attenuation_voltage(antenna_state, active_antenna_names, direction)
    # 幅相偏置属于每个离散来波的阵元响应，不是端口器件误差或损耗。
    amplitude_error_db = np.array([
        antenna_state.amplitude_error_db.get(name, 0.0)
        for name in active_antenna_names
    ])
    phase_error_deg = np.array([
        antenna_state.phase_error_deg.get(name, 0.0)
        for name in active_antenna_names
    ])
    source_gain_phase_response = (
        10.0 ** (amplitude_error_db / 20.0)
        * np.exp(1j * np.deg2rad(phase_error_deg))
    )
    open_circuit_response = (
        spatial_phase_response * pattern_polarization_response * pcv_response
        * blockage_response * source_gain_phase_response
    )
    coupling_coefficients = coupling_matrix(antenna_state, active_antenna_names)
    transfer_matrix = antenna_transfer_matrix(antenna_state, active_antenna_names)
    effective_response = transfer_matrix @ open_circuit_response
    coupled_response = effective_response
    diagnostics: AntennaDiagnostics = {
        "positions_m": phase_center_positions_m,
        "spatial_phase": spatial_phase_response,
        "pattern_polarization_voltage": pattern_polarization_response,
        "pcv_path_length_mm": pcv_path_length_mm,
        "pcv_response": pcv_response,
        "source_gain_phase_response": source_gain_phase_response,
        "blockage_voltage": blockage_response,
        "open_circuit_response": open_circuit_response,
        "coupling_matrix": coupling_coefficients,
        "coupled_response": coupled_response,
        "antenna_transfer_matrix": transfer_matrix,
        "effective_manifold": effective_response,
    }
    return effective_response, diagnostics


def build_antenna_input(
    signal_sources: SignalSources,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, ScenarioTruth]:
    """逐方向构造 b_q，并在阵元等效输入参考面汇总干净源。"""
    antenna_model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)

    desired_true_base, desired_true_stages = antenna_response(
        antenna_model,
        cfg.signal.true_direction,
        cfg.signal.true_polarization,
    )
    desired_assumed_base, desired_assumed_stages = antenna_response(
        antenna_model,
        cfg.signal.assumed_direction,
        cfg.signal.assumed_polarization,
        use_assumed_state_flag=True,
    )
    desired_input_component_v = (
        desired_true_stages["open_circuit_response"][:, None]
        * signal_sources["desired_waveform"][None, :]
    )
    desired_rf_frequency_hz = cfg.signal.carrier_hz + cfg.signal.doppler_hz
    desired_true = desired_true_base * feedline_response(
        cfg.feedline, cfg.phone.active_antennas, cfg.signal.carrier_hz,
        desired_rf_frequency_hz,
    )
    desired_assumed = desired_assumed_base * feedline_response(
        cfg.feedline, cfg.phone.active_antennas, cfg.signal.carrier_hz,
        desired_rf_frequency_hz, use_assumed_state_flag=True,
    )

    # 每个入射源按自身方向过天线；宽带分量的实际序列仍用通道时延处理。
    interferer_responses = []
    interferer_components = []
    interferer_diagnostics = []
    for interferer_signal in signal_sources["interferers"]:
        interferer = interferer_signal["interferer"]
        response_base, diagnostics = antenna_response(
            antenna_model,
            interferer.direction,
            interferer.polarization,
        )
        response = response_base * feedline_response(
            cfg.feedline, cfg.phone.active_antennas, cfg.signal.carrier_hz,
            cfg.signal.carrier_hz + interferer_signal["center_offset_hz"],
        )
        component = (
            diagnostics["open_circuit_response"][:, None]
            * interferer_signal["waveform_v"][None, :]
        )
        interferer_responses.append(response)
        interferer_components.append(component)
        interferer_diagnostics.append(diagnostics)

    scene = desired_input_component_v.copy()
    for component in interferer_components:
        scene += component

    truth: ScenarioTruth = {
        "time_s": signal_sources["time_s"],
        "code": signal_sources["code"],
        "desired_waveform": signal_sources["desired_waveform"],
        "desired_input_component_v": desired_input_component_v,
        "desired_steering_true": desired_true,
        "desired_steering_assumed": desired_assumed,
        "desired_open_circuit_response_true": desired_true_stages[
            "open_circuit_response"
        ],
        "desired_open_circuit_response_assumed": desired_assumed_stages[
            "open_circuit_response"
        ],
        "interferer_steering_effective": interferer_responses,
        "interferer_input_components_v": interferer_components,
        "active_antennas": cfg.phone.active_antennas,
        "positions_m": desired_true_stages["positions_m"],
        "doppler_hz": cfg.signal.doppler_hz,
        "code_rate_hz": signal_sources["received_code_rate_hz"],
        "code_delay_chips": cfg.signal.code_delay_chips,
        "fs_rf_hz": cfg.signal.fs_rf_hz,
        "if_hz": cfg.signal.if_hz,
        "antenna_model_diagnostics": {
            "desired_true": desired_true_stages,
            "desired_assumed": desired_assumed_stages,
            "interferers": interferer_diagnostics,
        },
    }
    return scene, truth


def process_antenna(
    clean_input_v: np.ndarray, cfg: SimulationConfig, *,
    retain_analysis_flag: bool = False,
) -> tuple[np.ndarray, dict]:
    """独立阵元130 K输入噪声与所有干净源共同经过一次 A=C diag(sqrt(η))。"""
    state = cfg.phone.antenna_model.true_state
    transfer = antenna_transfer_matrix(state, cfg.phone.active_antennas)
    coupling = coupling_matrix(state, cfg.phone.active_antennas)
    element_transmission = np.array([
        10.0 ** (-state.passive_loss_db.get(name, 0.0) / 10.0)
        for name in cfg.phone.active_antennas
    ])
    input_noise_v = generate_complex_noise(
        clean_input_v.shape[1], cfg.phone.antenna_noise_temperature_k,
        cfg.reference_impedance_ohm, cfg.signal.fs_rf_hz,
        cfg.phone.active_antennas, cfg.random_seed, "antenna_input",
    )
    info = {"input_temperature_k": cfg.phone.antenna_noise_temperature_k,
            "transfer_matrix": transfer,
            "coupling_matrix": coupling,
            "element_loss_temperature_k": state.temperature_k * (1.0 - element_transmission)}
    if retain_analysis_flag:
        info["propagated_noise_v"] = transfer @ input_noise_v
    return transfer @ (clean_input_v + input_noise_v), info
