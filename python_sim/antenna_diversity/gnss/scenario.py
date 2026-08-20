"""GPS L1 C/A and continuous-wave interference scene generation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from antenna_diversity.array.phone_model import PhoneAntennaModel
from antenna_diversity.array.polarization import WavePolarization
from antenna_diversity.array.steering import steering_vector
from antenna_diversity.config import (
    ArrayConfig,
    CWInterferer,
    Direction,
    PhoneArrayConfig,
    SimulationConfig,
)

from .ca_code import ca_code


def _legacy_channel_response(cfg: SimulationConfig) -> NDArray[np.complex128]:
    amplitude_errors_db = np.asarray(
        cfg.array.channel_amplitude_error_db, dtype=np.float64
    ).reshape(-1)
    phase_errors_deg = np.asarray(
        cfg.array.channel_phase_error_deg, dtype=np.float64
    ).reshape(-1)
    return 10.0 ** (amplitude_errors_db / 20.0) * np.exp(
        1j * np.deg2rad(phase_errors_deg)
    )


def _desired_responses(
    cfg: SimulationConfig,
    phone_model: PhoneAntennaModel | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object] | None,
    dict[str, object] | None,
]:
    """Return true nominal, assumed nominal, and true effective steering."""

    if isinstance(cfg.array, ArrayConfig):
        nominal = steering_vector(
            cfg.array.positions_m,
            cfg.array.desired_angle_deg,
            cfg.signal.wavelength_m,
        )
        effective = _legacy_channel_response(cfg) * nominal
        return nominal, nominal, effective, None, None

    if not isinstance(cfg.array, PhoneArrayConfig) or cfg.phone is None:
        raise ValueError("phone array configuration is incomplete")
    if cfg.signal.true_direction is None or cfg.signal.assumed_direction is None:
        raise ValueError("phone array directions are incomplete")
    if phone_model is None:
        raise ValueError("phone antenna model is required")

    true_diagnostics = _phone_response_diagnostics(
        phone_model,
        cfg.signal.true_direction,
        cfg.signal.true_polarization,
        cfg.signal.carrier_hz,
        estimated=False,
    )
    assumed_diagnostics = _phone_response_diagnostics(
        phone_model,
        cfg.signal.assumed_direction,
        cfg.signal.assumed_polarization,
        cfg.signal.carrier_hz,
        estimated=True,
    )
    return (
        np.asarray(true_diagnostics["open_circuit_response"]).copy(),
        np.asarray(assumed_diagnostics["effective_manifold"]).copy(),
        np.asarray(true_diagnostics["effective_manifold"]).copy(),
        true_diagnostics,
        assumed_diagnostics,
    )


def _active_interferers(cfg: SimulationConfig) -> tuple[CWInterferer, ...]:
    """Return phone interferers or a CWInterferer view of the legacy jammer."""

    if isinstance(cfg.array, PhoneArrayConfig):
        return tuple(interferer for interferer in cfg.interferers if interferer.enable)
    if not cfg.jammer.enable:
        return ()
    return (
        CWInterferer(
            direction=Direction(cfg.jammer.angle_deg, 0.0),
            power_dbm=cfg.jammer.power_dbm,
            offset_hz=cfg.jammer.offset_hz,
            initial_phase_rad=cfg.jammer.initial_phase_rad,
        ),
    )


def _time_gate(
    n_samples: int,
    fs_hz: float,
    start_ms: float,
    stop_ms: float | None,
) -> NDArray[np.float64]:
    gate = np.zeros(n_samples, dtype=np.float64)
    if not np.isfinite(start_ms) or start_ms < 0.0:
        raise ValueError("interferer start_ms must be finite and nonnegative")
    if stop_ms is not None and (not np.isfinite(stop_ms) or stop_ms < start_ms):
        raise ValueError("interferer stop_ms must be finite and not precede start_ms")

    duration_ms = n_samples / fs_hz * 1e3
    if start_ms > duration_ms or (
        stop_ms is not None and stop_ms > duration_ms
    ):
        raise ValueError("interferer time gate must lie within signal duration")

    first = round(start_ms * 1e-3 * fs_hz)
    stop = n_samples if stop_ms is None else round(stop_ms * 1e-3 * fs_hz)
    if first > n_samples or stop > n_samples:
        raise ValueError("interferer time gate must lie within signal duration")
    gate[first:stop] = 1.0
    return gate


def _interferer_response(
    cfg: SimulationConfig,
    interferer: CWInterferer,
    phone_model: PhoneAntennaModel | None,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    dict[str, object] | None,
]:
    if isinstance(cfg.array, ArrayConfig):
        nominal = steering_vector(
            cfg.array.positions_m,
            interferer.direction.azimuth_deg,
            cfg.signal.wavelength_m,
        )
        return nominal, _legacy_channel_response(cfg) * nominal, None

    if not isinstance(cfg.array, PhoneArrayConfig) or cfg.phone is None:
        raise ValueError("phone array configuration is incomplete")
    if phone_model is None:
        raise ValueError("phone antenna model is required")
    diagnostics = _phone_response_diagnostics(
        phone_model,
        interferer.direction,
        interferer.polarization,
        cfg.signal.carrier_hz,
        estimated=False,
    )
    return (
        np.asarray(diagnostics["open_circuit_response"]).copy(),
        np.asarray(diagnostics["effective_manifold"]).copy(),
        diagnostics,
    )


def _phone_response_diagnostics(
    model: PhoneAntennaModel,
    direction: Direction,
    polarization: WavePolarization,
    frequency_hz: float,
    *,
    estimated: bool,
) -> dict[str, object]:
    diagnostics = model.diagnostics(
        direction,
        polarization,
        frequency_hz,
        estimated=estimated,
    )
    diagnostics["polarization_match_loss_db"] = _polarization_match_loss_db(
        model,
        direction,
        polarization,
        frequency_hz,
        diagnostics,
        estimated=estimated,
    )
    return diagnostics


def _polarization_match_loss_db(
    model: PhoneAntennaModel,
    direction: Direction,
    polarization: WavePolarization,
    frequency_hz: float,
    diagnostics: dict[str, object],
    *,
    estimated: bool,
) -> NDArray[np.float64]:
    pattern_kind = str(diagnostics["pattern_kind"])
    pattern_voltage = np.asarray(
        diagnostics["pattern_polarization_voltage"], dtype=np.complex128
    )
    if pattern_kind == "isotropic":
        return np.zeros(pattern_voltage.shape, dtype=np.float64)

    jones = polarization.jones_vector()
    orthogonal = np.array(
        [-np.conj(jones[1]), np.conj(jones[0])], dtype=np.complex128
    )
    orthogonal_polarization = WavePolarization(
        "jones",
        az_real=float(orthogonal[0].real),
        az_imag=float(orthogonal[0].imag),
        el_real=float(orthogonal[1].real),
        el_imag=float(orthogonal[1].imag),
    )
    orthogonal_diagnostics = model.diagnostics(
        direction,
        orthogonal_polarization,
        frequency_hz,
        estimated=estimated,
    )
    orthogonal_voltage = np.asarray(
        orthogonal_diagnostics["pattern_polarization_voltage"],
        dtype=np.complex128,
    )
    reference_voltage = np.hypot(np.abs(pattern_voltage), np.abs(orthogonal_voltage))
    match = np.divide(
        np.abs(pattern_voltage),
        reference_voltage,
        out=np.zeros(pattern_voltage.shape, dtype=np.float64),
        where=reference_voltage > 0.0,
    )
    match = np.clip(match, np.finfo(np.float64).tiny, 1.0)
    return np.asarray(-20.0 * np.log10(match), dtype=np.float64)


def _strict_diagnostic_snapshot(diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        key: _strict_diagnostic_value(value) for key, value in diagnostics.items()
    }


def _strict_diagnostic_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "real": np.asarray(value.real, dtype=np.float64).copy(),
                "imag": np.asarray(value.imag, dtype=np.float64).copy(),
            }
        return value.copy()
    if isinstance(value, tuple):
        return tuple(value)
    return value


def generate_scenario(
    cfg: SimulationConfig,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Generate complex voltages at every antenna before RF noise and gain."""

    cfg.validate()
    n_antennas = cfg.array.num_elements
    n_samples = round(cfg.signal.fs_rf_hz * cfg.signal.duration_ms * 1e-3)
    if n_samples <= 0:
        raise ValueError("signal duration must produce at least one sample")

    code = ca_code(cfg.signal.prn)
    n = np.arange(n_samples, dtype=np.float64)
    time_s = n / cfg.signal.fs_rf_hz
    received_code_rate_hz = cfg.signal.code_rate_hz * (
        1.0 + cfg.signal.doppler_hz / cfg.signal.carrier_hz
    )
    code_phase = np.mod(
        n * received_code_rate_hz / cfg.signal.fs_rf_hz
        - cfg.signal.code_delay_chips,
        1023.0,
    )
    sampled_code = code[np.floor(code_phase).astype(np.int64)]

    signal_power_w = 10.0 ** ((cfg.signal.power_dbm - 30.0) / 10.0)
    signal_amplitude_v = np.sqrt(signal_power_w * cfg.signal.resistance_ohm)
    signal_carrier = np.exp(
        1j
        * (
            2.0
            * np.pi
            * (cfg.signal.if_hz + cfg.signal.doppler_hz)
            * time_s
            + cfg.signal.initial_phase_rad
        )
    )
    desired_waveform = signal_amplitude_v * sampled_code * signal_carrier

    phone_model = (
        PhoneAntennaModel.from_config(cfg)
        if isinstance(cfg.array, PhoneArrayConfig)
        else None
    )
    (
        desired_nominal,
        desired_assumed,
        desired_true,
        desired_true_diagnostics,
        desired_assumed_diagnostics,
    ) = _desired_responses(cfg, phone_model)
    desired_component = desired_true[:, np.newaxis] * desired_waveform[np.newaxis, :]

    interferer_steering_nominal: list[NDArray[np.complex128]] = []
    interferer_steering_effective: list[NDArray[np.complex128]] = []
    interferer_components: list[NDArray[np.complex128]] = []
    interferer_diagnostics: list[dict[str, object]] = []
    for interferer in _active_interferers(cfg):
        nominal, effective, diagnostics = _interferer_response(
            cfg, interferer, phone_model
        )
        power_w = 10.0 ** ((interferer.power_dbm - 30.0) / 10.0)
        amplitude_v = np.sqrt(power_w * cfg.signal.resistance_ohm)
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
        waveform *= _time_gate(
            n_samples,
            cfg.signal.fs_rf_hz,
            interferer.start_ms,
            interferer.stop_ms,
        )
        interferer_steering_nominal.append(nominal)
        interferer_steering_effective.append(effective)
        interferer_components.append(effective[:, np.newaxis] * waveform[np.newaxis, :])
        if diagnostics is not None:
            interferer_diagnostics.append(diagnostics)

    zero_steering = np.zeros(n_antennas, dtype=np.complex128)
    zero_component = np.zeros((n_antennas, n_samples), dtype=np.complex128)
    jammer_steering_nominal = (
        interferer_steering_nominal[0]
        if interferer_steering_nominal
        else zero_steering
    )
    jammer_steering_effective = (
        interferer_steering_effective[0]
        if interferer_steering_effective
        else zero_steering
    )
    jammer_component = (
        interferer_components[0] if interferer_components else zero_component
    )
    x_antenna_v = desired_component + sum(interferer_components, zero_component)

    if isinstance(cfg.array, PhoneArrayConfig):
        if phone_model is None:
            raise ValueError("phone antenna model is required")
        active_antennas = phone_model.active_antennas
        positions_m = phone_model.true_positions.copy()
        channel_response = phone_model.channel_response(estimated=False)
    else:
        active_antennas = tuple(str(index) for index in range(n_antennas))
        positions_m = np.asarray(cfg.array.positions_m, dtype=np.float64)
        channel_response = _legacy_channel_response(cfg)

    truth: dict[str, Any] = {
        "time_s": time_s,
        "code": code,
        "desired_waveform": desired_waveform,
        "desired_component": desired_component,
        "desired_steering_true": desired_true,
        "desired_steering_assumed": desired_assumed,
        "interferer_steering_nominal": interferer_steering_nominal,
        "interferer_steering_effective": interferer_steering_effective,
        "interferer_components": interferer_components,
        "active_antennas": active_antennas,
        "positions_m": positions_m,
        "desired_steering_nominal": desired_nominal,
        "desired_steering_effective": desired_true,
        "jammer_steering_nominal": jammer_steering_nominal,
        "jammer_steering_effective": jammer_steering_effective,
        "jammer_component": jammer_component,
        "channel_response": channel_response,
        "doppler_hz": cfg.signal.doppler_hz,
        "code_rate_hz": received_code_rate_hz,
        "code_delay_chips": cfg.signal.code_delay_chips,
        "code_delay_rf_samples": (
            cfg.signal.code_delay_chips
            * cfg.signal.fs_rf_hz
            / cfg.signal.code_rate_hz
        ),
        "fs_rf_hz": cfg.signal.fs_rf_hz,
        "if_hz": cfg.signal.if_hz,
    }
    if phone_model is not None:
        if desired_true_diagnostics is None or desired_assumed_diagnostics is None:
            raise ValueError("phone antenna diagnostics are incomplete")
        truth.update(
            {
                "antenna_model_source": phone_model.model_source,
                "antenna_model_label": phone_model.model_label,
                "desired_open_circuit_response_true": _strict_diagnostic_value(
                    desired_true_diagnostics["open_circuit_response"]
                ),
                "desired_open_circuit_response_assumed": _strict_diagnostic_value(
                    desired_assumed_diagnostics["open_circuit_response"]
                ),
                "interferer_open_circuit_responses": [
                    _strict_diagnostic_value(item["open_circuit_response"])
                    for item in interferer_diagnostics
                ],
                "desired_blockage_voltage_true": np.asarray(
                    desired_true_diagnostics["blockage_voltage"]
                ).copy(),
                "desired_blockage_voltage_assumed": np.asarray(
                    desired_assumed_diagnostics["blockage_voltage"]
                ).copy(),
                "interferer_blockage_voltages": [
                    np.asarray(item["blockage_voltage"]).copy()
                    for item in interferer_diagnostics
                ],
                "coupling_matrix_true": _strict_diagnostic_value(
                    desired_true_diagnostics["coupling_matrix"]
                ),
                "coupling_matrix_assumed": _strict_diagnostic_value(
                    desired_assumed_diagnostics["coupling_matrix"]
                ),
                "desired_polarization_match_loss_db_true": np.asarray(
                    desired_true_diagnostics["polarization_match_loss_db"]
                ).copy(),
                "desired_polarization_match_loss_db_assumed": np.asarray(
                    desired_assumed_diagnostics["polarization_match_loss_db"]
                ).copy(),
                "interferer_polarization_match_losses_db": [
                    np.asarray(item["polarization_match_loss_db"]).copy()
                    for item in interferer_diagnostics
                ],
                "antenna_model_diagnostics": {
                    "desired_true": _strict_diagnostic_snapshot(
                        desired_true_diagnostics
                    ),
                    "desired_assumed": _strict_diagnostic_snapshot(
                        desired_assumed_diagnostics
                    ),
                    "interferers": [
                        _strict_diagnostic_snapshot(item)
                        for item in interferer_diagnostics
                    ],
                },
            }
        )
    return x_antenna_v, truth
