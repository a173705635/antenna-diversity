"""GPS L1 C/A code and RF-input scene generation."""

import numpy as np

from antenna import antenna_response, build_antenna_model
from config import SimulationConfig


G2_TAPS = np.array(
    [
        [2, 6], [3, 7], [4, 8], [5, 9], [1, 9], [2, 10], [1, 8], [2, 9],
        [3, 10], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
        [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [1, 3], [4, 6],
        [5, 7], [6, 8], [7, 9], [8, 10], [1, 6], [2, 7], [3, 8], [4, 9],
    ]
)


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


def generate_scenario(
    cfg: SimulationConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate target and CW-interference voltages at the antenna ports."""

    n_samples = round(cfg.signal.fs_rf_hz * cfg.signal.duration_ms * 1e-3)
    n = np.arange(n_samples)
    time_s = n / cfg.signal.fs_rf_hz
    received_code_rate = cfg.signal.code_rate_hz * (
        1.0 + cfg.signal.doppler_hz / cfg.signal.carrier_hz
    )
    code = ca_code(cfg.signal.prn)
    code_phase = np.mod(
        n * received_code_rate / cfg.signal.fs_rf_hz
        - cfg.signal.code_delay_chips,
        1023.0,
    )
    sampled_code = code[np.floor(code_phase).astype(int)]
    signal_power_w = 10.0 ** ((cfg.signal.power_dbm - 30.0) / 10.0)
    signal_amplitude_v = np.sqrt(signal_power_w * cfg.signal.resistance_ohm)
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
    desired_waveform = signal_amplitude_v * sampled_code * carrier

    antenna_model = build_antenna_model(cfg)
    desired_true, desired_true_stages = antenna_response(
        antenna_model,
        cfg.signal.true_direction,
        cfg.signal.true_polarization,
    )
    desired_assumed, desired_assumed_stages = antenna_response(
        antenna_model,
        cfg.signal.assumed_direction,
        cfg.signal.assumed_polarization,
        estimated=True,
    )
    desired_component = desired_true[:, None] * desired_waveform[None, :]

    interferer_responses = []
    interferer_components = []
    interferer_stages = []
    for interferer in cfg.interferers:
        if not interferer.enable:
            continue
        response, stages = antenna_response(
            antenna_model,
            interferer.direction,
            interferer.polarization,
        )
        amplitude_v = np.sqrt(
            10.0 ** ((interferer.power_dbm - 30.0) / 10.0)
            * cfg.signal.resistance_ohm
        )
        waveform = amplitude_v * np.exp(
            1j
            * (
                2.0
                * np.pi
                * (cfg.signal.if_hz + interferer.offset_hz)
                * time_s
                + interferer.initial_phase_rad
            )
        )
        first = round(interferer.start_ms * 1e-3 * cfg.signal.fs_rf_hz)
        stop = (
            n_samples
            if interferer.stop_ms is None
            else round(interferer.stop_ms * 1e-3 * cfg.signal.fs_rf_hz)
        )
        gate = np.zeros(n_samples)
        gate[first:stop] = 1.0
        component = response[:, None] * (waveform * gate)[None, :]
        interferer_responses.append(response)
        interferer_components.append(component)
        interferer_stages.append(stages)

    scene = desired_component.copy()
    for component in interferer_components:
        scene += component

    truth = {
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
        "interferer_steering_effective": interferer_responses,
        "interferer_components": interferer_components,
        "active_antennas": cfg.phone.active_antennas,
        "positions_m": desired_true_stages["positions_m"],
        "doppler_hz": cfg.signal.doppler_hz,
        "code_rate_hz": received_code_rate,
        "code_delay_chips": cfg.signal.code_delay_chips,
        "fs_rf_hz": cfg.signal.fs_rf_hz,
        "if_hz": cfg.signal.if_hz,
        "antenna_model_diagnostics": {
            "desired_true": desired_true_stages,
            "desired_assumed": desired_assumed_stages,
            "interferers": interferer_stages,
        },
    }
    return scene, truth
