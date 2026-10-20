"""GPS L1 C/A 码、目标波形与各入射源的一维参考电压生成。"""

import numpy as np

from .code_replica import bandlimited_code_replica, _sample_code
from .config import SimulationConfig
from .interference.interference import generate_interferer_signal
from .models import SignalSources


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


def _desired_waveform(
    cfg: SimulationConfig,
    sample_index: np.ndarray,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """返回目标一维参考电压 (V)、码序列及含多普勒的接收码率 (chip/s)。"""
    received_code_rate_hz = cfg.signal.code_rate_hz * (
        1.0 + cfg.signal.doppler_hz / cfg.signal.carrier_hz
    )
    from .l5 import l5q_code, NH20
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


def generate_signal_sources(cfg: SimulationConfig) -> SignalSources:
    """生成目标及启用干扰的一维参考电压，尚未经过方向响应或通道时延。"""
    n_samples = round(cfg.signal.fs_rf_hz * cfg.signal.duration_ms * 1e-3)
    sample_index = np.arange(n_samples)
    time_s = sample_index / cfg.signal.fs_rf_hz
    desired_waveform, code, received_code_rate_hz = _desired_waveform(
        cfg, sample_index, time_s,
    )
    interferers = [
        generate_interferer_signal(interferer, cfg, time_s)
        for interferer in cfg.interferers
        if interferer.enable_flag
    ]
    return {
        "time_s": time_s,
        "code": code,
        "desired_waveform": desired_waveform,
        "received_code_rate_hz": received_code_rate_hz,
        "interferers": interferers,
    }
