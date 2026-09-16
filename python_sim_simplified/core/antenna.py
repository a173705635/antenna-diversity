"""Readable composition of the complete phone antenna response."""

from dataclasses import dataclass

import numpy as np

from .config import (
    AntennaState,
    Direction,
    ElementOrientation,
    ParametricPattern,
    PhoneConfig,
    SPEED_OF_LIGHT_MPS,
    WavePolarization,
)
from .measured_pattern import (
    MeasuredGrid,
    interpolate_measured,
    load_measured_pattern,
)
from .models import AntennaDiagnostics


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
    true_channel_response: np.ndarray
    assumed_channel_response: np.ndarray
    true_measured: dict[str, MeasuredGrid] | None
    assumed_measured: dict[str, MeasuredGrid] | None


def direction_vector(direction: Direction) -> np.ndarray:
    azimuth, elevation = np.deg2rad(
        [direction.azimuth_deg, direction.elevation_deg]
    )
    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def jones_vector(polarization: WavePolarization) -> np.ndarray:
    if polarization.kind == "rhcp":
        vector = np.array([1.0, -1.0j])
    elif polarization.kind == "lhcp":
        vector = np.array([1.0, 1.0j])
    elif polarization.kind == "linear":
        tilt = np.deg2rad(polarization.tilt_deg)
        vector = np.array([np.cos(tilt), np.sin(tilt)], dtype=complex)
    else:
        vector = np.array(
            [
                complex(polarization.az_real, polarization.az_imag),
                complex(polarization.el_real, polarization.el_imag),
            ]
        )
    return vector / np.linalg.norm(vector)


def rotation_matrix_ypr(orientation: ElementOrientation) -> np.ndarray:
    yaw, pitch, roll = np.deg2rad(
        [orientation.yaw_deg, orientation.pitch_deg, orientation.roll_deg]
    )
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def spherical_frame(azimuth_deg: float, elevation_deg: float):
    azimuth, elevation = np.deg2rad([azimuth_deg, elevation_deg])
    radial = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    azimuth_basis = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    elevation_basis = np.array(
        [
            -np.sin(elevation) * np.cos(azimuth),
            -np.sin(elevation) * np.sin(azimuth),
            np.cos(elevation),
        ]
    )
    return radial, azimuth_basis, elevation_basis


def wave_in_element_frame(
    direction: Direction,
    polarization: WavePolarization,
    rotation: np.ndarray,
) -> tuple[LocalDirection, np.ndarray]:
    unit_phone, az_phone, el_phone = spherical_frame(
        direction.azimuth_deg, direction.elevation_deg
    )
    jones_phone = jones_vector(polarization)
    field_phone = jones_phone[0] * az_phone + jones_phone[1] * el_phone
    unit_local = rotation.T @ unit_phone
    field_local = rotation.T @ field_phone
    local_azimuth = np.rad2deg(np.arctan2(unit_local[1], unit_local[0])) % 360.0
    local_elevation = np.rad2deg(np.arcsin(np.clip(unit_local[2], -1.0, 1.0)))
    _, az_local, el_local = spherical_frame(local_azimuth, local_elevation)
    local_jones = np.array(
        [np.vdot(az_local, field_local), np.vdot(el_local, field_local)]
    )
    local_jones /= np.linalg.norm(local_jones)
    return LocalDirection(local_azimuth, local_elevation), local_jones


def parametric_height(
    parameters: ParametricPattern,
    local_elevation_deg: float,
) -> np.ndarray:
    front_power = max(np.sin(np.deg2rad(local_elevation_deg)), 0.0) ** (
        parameters.cosine_exponent
    )
    relative_db = -np.inf if front_power == 0.0 else 10.0 * np.log10(front_power)
    gain_db = parameters.peak_gain_db + max(
        relative_db,
        parameters.back_lobe_gain_db - parameters.peak_gain_db,
    )
    nominal = jones_vector(parameters.nominal_polarization)
    orthogonal = np.array([-np.conj(nominal[1]), np.conj(nominal[0])])
    cross = 10.0 ** (-parameters.cross_polar_discrimination_db / 20.0)
    mixture = nominal + cross * np.exp(
        1j * np.deg2rad(parameters.cross_phase_deg)
    ) * orthogonal
    return 10.0 ** (gain_db / 20.0) * mixture / np.linalg.norm(mixture)


def attenuation_voltage(
    state: AntennaState,
    names: tuple[str, ...],
    direction: Direction,
) -> np.ndarray:
    direction_unit = direction_vector(direction)
    attenuation_db = np.zeros(len(names))
    for region in state.attenuation_regions:
        center = direction_vector(
            Direction(region.center_azimuth_deg, region.center_elevation_deg)
        )
        separation = np.rad2deg(
            np.arccos(np.clip(np.dot(direction_unit, center), -1.0, 1.0))
        )
        if region.transition_width_deg == 0.0:
            fraction = float(separation <= region.radius_deg)
        else:
            inner = region.radius_deg - region.transition_width_deg / 2.0
            outer = region.radius_deg + region.transition_width_deg / 2.0
            if separation <= inner:
                fraction = 1.0
            elif separation >= outer:
                fraction = 0.0
            else:
                phase = np.pi * (separation - inner) / region.transition_width_deg
                fraction = 0.5 * (1.0 + np.cos(phase))
        selected = region.elements or names
        for index, name in enumerate(names):
            if name in selected:
                attenuation_db[index] += fraction * region.attenuation_db
    return 10.0 ** (-attenuation_db / 20.0)


def coupling_matrix(state: AntennaState, names: tuple[str, ...]) -> np.ndarray:
    matrix = np.eye(len(names), dtype=complex)
    indices = {name: index for index, name in enumerate(names)}
    for path in state.coupling_paths:
        if path.source in indices and path.receiver in indices:
            matrix[indices[path.receiver], indices[path.source]] = 10.0 ** (
                path.magnitude_db / 20.0
            ) * np.exp(1j * np.deg2rad(path.phase_deg))
    return matrix


def channel_response(phone: PhoneConfig, estimated: bool = False) -> np.ndarray:
    amplitude = (
        phone.estimated_channel_amplitude_error_db
        if estimated
        else phone.channel_amplitude_error_db
    )
    phase = (
        phone.estimated_channel_phase_error_deg
        if estimated
        else phone.channel_phase_error_deg
    )
    amplitude_db = np.array([amplitude[name] for name in phone.active_antennas])
    phase_deg = np.array([phase[name] for name in phone.active_antennas])
    return 10.0 ** (amplitude_db / 20.0) * np.exp(1j * np.deg2rad(phase_deg))


def build_antenna_model(phone: PhoneConfig, carrier_hz: float) -> AntennaModel:
    true_state = phone.antenna_model.true_state
    assumed_state = phone.antenna_model.assumed_state
    true_measured = (
        load_measured_pattern(true_state.measured_pattern_csv, carrier_hz)
        if true_state.pattern_kind == "measured"
        else None
    )
    assumed_measured = (
        load_measured_pattern(
            assumed_state.measured_pattern_csv, carrier_hz
        )
        if assumed_state.pattern_kind == "measured"
        else None
    )
    return AntennaModel(
        active_antennas=phone.active_antennas,
        carrier_hz=carrier_hz,
        true_positions_m=phone.phase_centers_m,
        assumed_positions_m=phone.assumed_phase_centers_m,
        true_state=true_state,
        assumed_state=assumed_state,
        true_channel_response=channel_response(phone, estimated=False),
        assumed_channel_response=channel_response(phone, estimated=True),
        true_measured=true_measured,
        assumed_measured=assumed_measured,
    )


def antenna_response(
    model: AntennaModel,
    direction: Direction,
    polarization: WavePolarization,
    estimated: bool = False,
) -> tuple[np.ndarray, AntennaDiagnostics]:
    names = model.active_antennas
    state = (
        model.assumed_state
        if estimated
        else model.true_state
    )
    position_map = (
        model.assumed_positions_m
        if estimated
        else model.true_positions_m
    )
    measured = model.assumed_measured if estimated else model.true_measured
    positions = np.array([position_map[name] for name in names])
    spatial = np.exp(
        -1j
        * 2.0
        * np.pi
        * model.carrier_hz
        / SPEED_OF_LIGHT_MPS
        * (positions @ direction_vector(direction))
    )

    pattern_voltage = []
    for name in names:
        if state.pattern_kind == "ideal":
            voltage = 1.0 + 0.0j
        else:
            local_direction, local_jones = wave_in_element_frame(
                direction,
                polarization,
                rotation_matrix_ypr(state.orientations[name]),
            )
            if state.pattern_kind == "parametric":
                height = parametric_height(
                    state.parametric_patterns[name],
                    local_direction.elevation_deg,
                )
            else:
                height = interpolate_measured(
                    measured[name],
                    local_direction.azimuth_deg,
                    local_direction.elevation_deg,
                )
            voltage = np.vdot(height, local_jones)
        pattern_voltage.append(voltage)
    pattern_voltage = np.asarray(pattern_voltage)

    if (set(state.amplitude_error_db) | set(state.phase_error_deg)) - set(position_map):
        raise ValueError('antenna error maps contain unknown element names')
    amplitude_db = np.array([state.amplitude_error_db.get(name, 0.) for name in names])
    phase_deg = np.array([state.phase_error_deg.get(name, 0.) for name in names])
    if not np.all(np.isfinite(amplitude_db)) or not np.all(np.isfinite(phase_deg)):
        raise ValueError('antenna amplitude/phase errors must be finite')
    with np.errstate(over='raise', invalid='raise'):
        element_error = 10.**(amplitude_db/20.) * np.exp(1j*np.deg2rad(phase_deg))

    blockage = attenuation_voltage(state, names, direction)
    open_circuit = spatial * pattern_voltage * element_error * blockage
    coupling = coupling_matrix(state, names)
    coupled = coupling @ open_circuit
    channel = (
        model.assumed_channel_response
        if estimated
        else model.true_channel_response
    )
    effective = channel * coupled
    diagnostics: AntennaDiagnostics = {
        "positions_m": positions,
        "spatial_phase": spatial,
        "pattern_polarization_voltage": pattern_voltage,
        "element_error_response": element_error,
        "blockage_voltage": blockage,
        "open_circuit_response": open_circuit,
        "coupling_matrix": coupling,
        "coupled_response": coupled,
        "channel_response": channel,
        "effective_manifold": effective,
    }
    return effective, diagnostics
