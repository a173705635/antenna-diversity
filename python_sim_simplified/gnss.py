"""GPS L1 C/A code and RF-input scene generation."""

from dataclasses import dataclass

import numpy as np

from antenna import AntennaModel, antenna_response, build_antenna_model
from channel_delay import apply_channel_delays, delay_phase
from code_replica import bandlimited_code_replica, _sample_code
from config import (
    CWInterferer,
    Interferer,
    MultipathPath,
    PhoneConfig,
    SimulationConfig,
    WidebandInterferer,
)
from interference import generate_wideband_waveform
from models import AntennaDiagnostics, MultipathTruth, ScenarioTruth


G2_TAPS = np.array(
    [
        [2, 6], [3, 7], [4, 8], [5, 9], [1, 9], [2, 10], [1, 8], [2, 9],
        [3, 10], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
        [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [1, 3], [4, 6],
        [5, 7], [6, 8], [7, 9], [8, 10], [1, 6], [2, 7], [3, 8], [4, 9],
    ]
)


@dataclass
class _InterfererScene:
    response: np.ndarray
    component: np.ndarray
    delay_phase: np.ndarray
    diagnostics: AntennaDiagnostics


@dataclass
class _MultipathScene:
    truth: MultipathTruth
    diagnostics: AntennaDiagnostics


def ca_code(prn: int) -> np.ndarray:
    """Return one 1023-chip bipolar GPS L1 C/A sequence."""

    tap1, tap2 = G2_TAPS[prn - 1] - 1
    g1 = np.ones(10, dtype=np.uint8)
    g2 = np.ones(10, dtype=np.uint8)
    code = np.empty(1023)
    for chip in range(1023):
        code[chip] = 1.0 - 2.0 * float(g1[9] ^ g2[tap1] ^ g2[tap2])
        g1_feedback = g1[2] ^ g1[9]
        g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1[1:] = g1[:-1]
        g2[1:] = g2[:-1]
        g1[0] = g1_feedback
        g2[0] = g2_feedback
    return code


def _channel_delays_s(phone: PhoneConfig, estimated: bool) -> np.ndarray:
    delay_map = (
        phone.estimated_channel_delay_ns
        if estimated
        else phone.channel_delay_ns
    )
    return 1e-9 * np.asarray(
        [delay_map[name] for name in phone.active_antennas]
    )


def _desired_waveform(
    cfg: SimulationConfig,
    sample_index: np.ndarray,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    received_code_rate_hz = cfg.signal.code_rate_hz * (
        1.0 + cfg.signal.doppler_hz / cfg.signal.carrier_hz
    )
    from l5 import l5q_code, NH20
    if cfg.signal.signal_kind not in ('L1CA', 'L5Q'):
        raise ValueError('unsupported signal kind')
    code = l5q_code(cfg.signal.prn) if cfg.signal.signal_kind == 'L5Q' else ca_code(cfg.signal.prn)
    code_phase = np.mod(
        sample_index * received_code_rate_hz / cfg.signal.fs_rf_hz
        - cfg.signal.code_delay_chips,
        float(code.size),
    )
    sampled_code = code[np.floor(code_phase).astype(int)]
    if cfg.signal.code_bandwidth_hz is not None:
        if abs(cfg.signal.if_hz + cfg.signal.doppler_hz) + cfg.signal.code_bandwidth_hz >= cfg.signal.fs_rf_hz / 2:
            raise ValueError("bandlimited source must fit inside RF sampling Nyquist")
        replica = bandlimited_code_replica(
            code, received_code_rate_hz, cfg.signal.code_bandwidth_hz
        )
        # Preserve configured total signal power, not the unfiltered chip amplitude.
        replica /= np.sqrt(np.mean(replica**2))
        sampled_code = _sample_code(replica, code_phase, code.size)
    if cfg.signal.signal_kind == 'L5Q':
        transmit_chips = sample_index * received_code_rate_hz / cfg.signal.fs_rf_hz - cfg.signal.code_delay_chips
        sampled_code = 1j * sampled_code * NH20[np.floor(transmit_chips / 10230).astype(np.int64) % 20]
    signal_power_w = 10.0 ** ((cfg.signal.power_dbm - 30.0) / 10.0)
    signal_amplitude_v = np.sqrt(
        signal_power_w * cfg.reference_impedance_ohm
    )
    carrier = np.exp(
        1j
        * (
            2.0
            * np.pi
            * (cfg.signal.if_hz + cfg.signal.doppler_hz)
            * time_s
            + cfg.signal.initial_phase_rad
        )
    )
    return signal_amplitude_v * sampled_code * carrier, code, received_code_rate_hz


def _interferer_scene(
    interferer: Interferer,
    cfg: SimulationConfig,
    antenna_model: AntennaModel,
    time_s: np.ndarray,
    true_delays_s: np.ndarray,
) -> _InterfererScene:
    response_base, diagnostics = antenna_response(
        antenna_model,
        interferer.direction,
        interferer.polarization,
    )
    if isinstance(interferer, CWInterferer):
        center_offset_hz = interferer.offset_hz
        amplitude_v = np.sqrt(
            10.0 ** ((interferer.power_dbm - 30.0) / 10.0)
            * cfg.reference_impedance_ohm
        )
        waveform = amplitude_v * np.exp(
            1j
            * (
                2.0
                * np.pi
                * (cfg.signal.if_hz + center_offset_hz)
                * time_s
                + interferer.initial_phase_rad
            )
        )
    elif isinstance(interferer, WidebandInterferer):
        center_offset_hz = interferer.center_offset_hz
        waveform = generate_wideband_waveform(
            interferer,
            time_s.size,
            cfg.signal.fs_rf_hz,
            cfg.signal.if_hz,
            cfg.reference_impedance_ohm,
        )
    first = round(interferer.start_ms * 1e-3 * cfg.signal.fs_rf_hz)
    stop = (
        time_s.size
        if interferer.stop_ms is None
        else round(interferer.stop_ms * 1e-3 * cfg.signal.fs_rf_hz)
    )
    gate = np.zeros(time_s.size)
    gate[first:stop] = 1.0
    interferer_delay_phase = delay_phase(
        true_delays_s,
        cfg.signal.carrier_hz + center_offset_hz,
    )
    response = (
        response_base
        if np.all(true_delays_s == 0.0)
        else response_base * interferer_delay_phase
    )
    component = apply_channel_delays(
        response_base[:, None] * (waveform * gate)[None, :],
        true_delays_s,
        cfg.signal.fs_rf_hz,
        cfg.signal.carrier_hz,
        cfg.signal.if_hz,
    )
    return _InterfererScene(
        response,
        component,
        interferer_delay_phase,
        diagnostics,
    )


def _multipath_scene(
    path: MultipathPath,
    cfg: SimulationConfig,
    antenna_model: AntennaModel,
    desired_waveform: np.ndarray,
    time_s: np.ndarray,
    true_delays_s: np.ndarray,
) -> _MultipathScene:
    excess_delay_s = path.excess_delay_chips / cfg.signal.code_rate_hz
    total_delays_s = true_delays_s + excess_delay_s
    relative_voltage = 10.0 ** (path.relative_power_db / 20.0)
    path_waveform = (
        desired_waveform
        * relative_voltage
        * np.exp(1j * path.reflection_phase_rad)
        * np.exp(1j * 2.0 * np.pi * path.doppler_offset_hz * time_s)
    )
    response_base, diagnostics = antenna_response(
        antenna_model,
        path.direction,
        path.polarization,
    )
    component = apply_channel_delays(
        response_base[:, None] * path_waveform[None, :],
        total_delays_s,
        cfg.signal.fs_rf_hz,
        cfg.signal.carrier_hz,
        cfg.signal.if_hz,
    )
    path_rf_frequency_hz = (
        cfg.signal.carrier_hz
        + cfg.signal.doppler_hz
        + path.doppler_offset_hz
    )
    path_delay_phase = delay_phase(total_delays_s, path_rf_frequency_hz)
    truth: MultipathTruth = {
        "component": component,
        "steering_true": response_base * path_delay_phase,
        "delay_phase_true": path_delay_phase,
        "excess_delay_chips": path.excess_delay_chips,
        "excess_delay_s": excess_delay_s,
        "relative_power_db": path.relative_power_db,
        "reflection_phase_rad": path.reflection_phase_rad,
        "doppler_offset_hz": path.doppler_offset_hz,
        "direction": path.direction,
    }
    return _MultipathScene(truth, diagnostics)


def generate_scenario(
    cfg: SimulationConfig,
) -> tuple[np.ndarray, ScenarioTruth]:
    """Generate target and CW-interference voltages at the antenna ports."""

    n_samples = round(cfg.signal.fs_rf_hz * cfg.signal.duration_ms * 1e-3)
    true_delays_s = _channel_delays_s(cfg.phone, estimated=False)
    estimated_delays_s = _channel_delays_s(cfg.phone, estimated=True)
    sample_index = np.arange(n_samples)
    time_s = sample_index / cfg.signal.fs_rf_hz
    desired_waveform, code, received_code_rate_hz = _desired_waveform(
        cfg,
        sample_index,
        time_s,
    )

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
        estimated=True,
    )
    desired_component = apply_channel_delays(
        desired_true_base[:, None] * desired_waveform[None, :],
        true_delays_s,
        cfg.signal.fs_rf_hz,
        cfg.signal.carrier_hz,
        cfg.signal.if_hz,
    )
    desired_rf_frequency_hz = cfg.signal.carrier_hz + cfg.signal.doppler_hz
    desired_delay_phase_true = delay_phase(
        true_delays_s,
        desired_rf_frequency_hz,
    )
    desired_delay_phase_assumed = delay_phase(
        estimated_delays_s,
        desired_rf_frequency_hz,
    )
    desired_true = (
        desired_true_base
        if np.all(true_delays_s == 0.0)
        else desired_true_base * desired_delay_phase_true
    )
    desired_assumed = (
        desired_assumed_base
        if np.all(estimated_delays_s == 0.0)
        else desired_assumed_base * desired_delay_phase_assumed
    )

    interferer_scenes = [
        _interferer_scene(
            interferer,
            cfg,
            antenna_model,
            time_s,
            true_delays_s,
        )
        for interferer in cfg.interferers
        if interferer.enable
    ]
    multipath_scenes = [
        _multipath_scene(
            path,
            cfg,
            antenna_model,
            desired_waveform,
            time_s,
            true_delays_s,
        )
        for path in cfg.signal.multipath_paths
        if path.enable
    ]
    scene = desired_component.copy()
    for path in multipath_scenes:
        scene += path.truth["component"]
    for interferer in interferer_scenes:
        scene += interferer.component

    truth: ScenarioTruth = {
        "time_s": time_s,
        "code": code,
        "desired_waveform": desired_waveform,
        "desired_component": desired_component,
        "desired_steering_true": desired_true,
        "desired_steering_assumed": desired_assumed,
        "desired_open_circuit_response_true": desired_true_stages[
            "open_circuit_response"
        ],
        "desired_open_circuit_response_assumed": desired_assumed_stages[
            "open_circuit_response"
        ],
        "interferer_steering_effective": [
            interferer.response for interferer in interferer_scenes
        ],
        "interferer_components": [
            interferer.component for interferer in interferer_scenes
        ],
        "multipath_paths": [path.truth for path in multipath_scenes],
        "active_antennas": cfg.phone.active_antennas,
        "positions_m": desired_true_stages["positions_m"],
        "channel_delay_true_ns": true_delays_s * 1e9,
        "channel_delay_estimated_ns": estimated_delays_s * 1e9,
        "channel_delay_true_samples": true_delays_s * cfg.signal.fs_rf_hz,
        "channel_delay_estimated_samples": (
            estimated_delays_s * cfg.signal.fs_rf_hz
        ),
        "desired_delay_phase_true": desired_delay_phase_true,
        "desired_delay_phase_assumed": desired_delay_phase_assumed,
        "interferer_delay_phases_true": [
            interferer.delay_phase for interferer in interferer_scenes
        ],
        "doppler_hz": cfg.signal.doppler_hz,
        "code_rate_hz": received_code_rate_hz,
        "code_delay_chips": cfg.signal.code_delay_chips,
        "fs_rf_hz": cfg.signal.fs_rf_hz,
        "if_hz": cfg.signal.if_hz,
        "antenna_model_diagnostics": {
            "desired_true": desired_true_stages,
            "desired_assumed": desired_assumed_stages,
            "interferers": [
                interferer.diagnostics for interferer in interferer_scenes
            ],
            "multipath_paths": [
                path.diagnostics for path in multipath_scenes
            ],
        },
    }
    return scene, truth
