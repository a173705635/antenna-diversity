"""Composition service for true and assumed phone-antenna responses."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .antenna_config import AntennaStateConfig, resolved_phone_antenna_config
from .effects import AttenuationRegion, attenuation_voltage, coupling_matrix
from .geometry import select_phase_centers
from .manifold import IsotropicElementPattern, array_manifold
from .patterns import (
    LocalDirection,
    ParametricElementPattern,
    PolarizedElementPattern,
    ScalarPatternAdapter,
    TabulatedElementPattern,
)
from .polarization import WavePolarization, rotation_matrix_ypr, wave_in_element_frame

if TYPE_CHECKING:
    from antenna_diversity.config import Direction, SimulationConfig


@dataclass(frozen=True)
class _RuntimeState:
    pattern_kind: str
    pattern: PolarizedElementPattern
    scalar_pattern: IsotropicElementPattern | None
    rotations: Mapping[str, NDArray[np.float64]]
    attenuation_regions: tuple[AttenuationRegion, ...]
    coupling: NDArray[np.complex128]


@dataclass(frozen=True)
class _OpenCircuitStages:
    positions_m: NDArray[np.float64]
    polarization_jones: NDArray[np.complex128]
    spatial_phase: NDArray[np.complex128]
    pattern_polarization_voltage: NDArray[np.complex128]
    blockage_voltage: NDArray[np.float64]
    open_circuit_response: NDArray[np.complex128]


@dataclass(frozen=True)
class PhoneAntennaModel:
    """Detached true/assumed antenna states composed in active-element order."""

    active_antennas: tuple[str, ...]
    true_positions: NDArray[np.float64]
    assumed_positions: NDArray[np.float64]
    true_state: _RuntimeState
    assumed_state: _RuntimeState
    true_channel: NDArray[np.complex128]
    assumed_channel: NDArray[np.complex128]
    model_label: str
    model_source: str

    @classmethod
    def from_config(cls, cfg: SimulationConfig) -> PhoneAntennaModel:
        """Validate and detach a phone-array configuration."""
        from antenna_diversity.config import PhoneArrayConfig

        candidate = copy.deepcopy(cfg)
        candidate.validate()
        if not isinstance(candidate.array, PhoneArrayConfig) or candidate.phone is None:
            raise ValueError("PhoneAntennaModel requires a phone-array configuration")
        antenna = resolved_phone_antenna_config(candidate.array)
        names = tuple(candidate.array.active_antennas)
        return cls(
            active_antennas=names,
            true_positions=_frozen_array(
                select_phase_centers(candidate.phone.phase_centers_m, names),
                np.float64,
                "true positions",
            ),
            assumed_positions=_frozen_array(
                select_phase_centers(candidate.phone.assumed_phase_centers_m, names),
                np.float64,
                "assumed positions",
            ),
            true_state=_build_runtime_state(
                antenna.true_state, names, candidate.signal.carrier_hz
            ),
            assumed_state=_build_runtime_state(
                antenna.assumed_state, names, candidate.signal.carrier_hz
            ),
            true_channel=_build_channel_response(
                candidate.array, estimated=False, label="true channel response"
            ),
            assumed_channel=_build_channel_response(
                candidate.array, estimated=True, label="assumed channel response"
            ),
            model_label=antenna.model_label,
            model_source=antenna.model_source,
        )

    def open_circuit_response(
        self,
        direction: Direction,
        polarization: WavePolarization,
        frequency_hz: float,
        *,
        estimated: bool = False,
    ) -> NDArray[np.complex128]:
        """Return pattern/position/polarization response before coupling."""
        return self._open_circuit_stages(
            direction, polarization, frequency_hz, estimated=estimated
        ).open_circuit_response.copy()

    def _open_circuit_stages(
        self,
        direction: Direction,
        polarization: WavePolarization,
        frequency_hz: float,
        *,
        estimated: bool,
    ) -> _OpenCircuitStages:
        validated_frequency, polarization_jones = _validated_source_inputs(
            direction, polarization, frequency_hz
        )
        positions, state = self._selected_geometry_state(estimated)
        try:
            with np.errstate(all="ignore"):
                spatial = array_manifold(
                    positions,
                    direction,
                    validated_frequency,
                    self.active_antennas,
                )
        except (ArithmeticError, FloatingPointError) as error:
            raise ValueError("spatial phase must contain finite values") from error
        spatial = _finite_array(spatial, np.complex128, "spatial phase")
        try:
            with np.errstate(all="ignore"):
                if state.scalar_pattern is not None:
                    pattern_voltage = np.asarray(
                        [
                            state.scalar_pattern.complex_gain(
                                name, direction, validated_frequency
                            )
                            for name in self.active_antennas
                        ],
                        dtype=np.complex128,
                    )
                else:
                    voltages = []
                    for name in self.active_antennas:
                        local_wave = wave_in_element_frame(
                            direction.azimuth_deg,
                            direction.elevation_deg,
                            polarization,
                            state.rotations[name],
                        )
                        height = state.pattern.effective_height(
                            name,
                            LocalDirection(
                                local_wave.azimuth_deg, local_wave.elevation_deg
                            ),
                            validated_frequency,
                        )
                        voltages.append(np.vdot(height, local_wave.jones))
                    pattern_voltage = np.asarray(voltages, dtype=np.complex128)
        except (ArithmeticError, FloatingPointError) as error:
            raise ValueError(
                "pattern/polarization voltage must contain finite values"
            ) from error
        pattern_voltage = _finite_array(
            pattern_voltage,
            np.complex128,
            "pattern/polarization voltage",
        )
        try:
            with np.errstate(all="ignore"):
                blockage = attenuation_voltage(
                    state.attenuation_regions, self.active_antennas, direction
                )
        except (ArithmeticError, FloatingPointError) as error:
            raise ValueError("blockage voltage must contain finite values") from error
        blockage = _finite_array(blockage, np.float64, "blockage voltage")
        with np.errstate(all="ignore"):
            open_circuit = np.asarray(
                spatial * pattern_voltage * blockage,
                dtype=np.complex128,
            )
        open_circuit = _finite_array(
            open_circuit, np.complex128, "open-circuit response"
        )
        return _OpenCircuitStages(
            positions_m=positions,
            polarization_jones=polarization_jones,
            spatial_phase=spatial,
            pattern_polarization_voltage=pattern_voltage,
            blockage_voltage=blockage,
            open_circuit_response=open_circuit,
        )

    def coupling_matrix(self, estimated: bool = False) -> NDArray[np.complex128]:
        """Return a detached ordered coupling matrix."""
        state = self.assumed_state if estimated else self.true_state
        return state.coupling.copy()

    def channel_response(self, estimated: bool = False) -> NDArray[np.complex128]:
        """Return a detached per-channel complex response vector."""
        channel = self.assumed_channel if estimated else self.true_channel
        return channel.copy()

    def effective_manifold(
        self,
        direction: Direction,
        polarization: WavePolarization,
        frequency_hz: float,
        *,
        estimated: bool = False,
    ) -> NDArray[np.complex128]:
        """Apply coupling and then channel response to the open-circuit vector."""
        open_circuit = self.open_circuit_response(
            direction, polarization, frequency_hz, estimated=estimated
        )
        _, _, _, effective = self._composed_stages(
            open_circuit, estimated=estimated
        )
        return effective.copy()

    def diagnostics(
        self,
        direction: Direction,
        polarization: WavePolarization,
        frequency_hz: float,
        *,
        estimated: bool = False,
    ) -> dict[str, object]:
        """Return a detached snapshot of every numeric composition stage."""
        _, state = self._selected_geometry_state(estimated)
        stages = self._open_circuit_stages(
            direction, polarization, frequency_hz, estimated=estimated
        )
        coupling, coupled, channel, effective = self._composed_stages(
            stages.open_circuit_response, estimated=estimated
        )
        return {
            "active_antennas": self.active_antennas,
            "estimated": bool(estimated),
            "model_label": self.model_label,
            "model_source": self.model_source,
            "pattern_kind": state.pattern_kind,
            "frequency_hz": _validated_frequency(frequency_hz),
            "direction_azimuth_deg": float(direction.azimuth_deg),
            "direction_elevation_deg": float(direction.elevation_deg),
            "polarization_jones": stages.polarization_jones.copy(),
            "positions_m": stages.positions_m.copy(),
            "spatial_phase": stages.spatial_phase.copy(),
            "pattern_polarization_voltage": (
                stages.pattern_polarization_voltage.copy()
            ),
            "blockage_voltage": stages.blockage_voltage.copy(),
            "open_circuit_response": stages.open_circuit_response.copy(),
            "coupling_matrix": coupling.copy(),
            "coupled_response": coupled.copy(),
            "channel_response": channel.copy(),
            "effective_manifold": effective.copy(),
        }

    def _composed_stages(
        self,
        open_circuit: NDArray[np.complex128],
        *,
        estimated: bool,
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
    ]:
        coupling = self.coupling_matrix(estimated=estimated)
        channel = self.channel_response(estimated=estimated)
        with np.errstate(all="ignore"):
            coupled = coupling @ open_circuit
        coupled = _finite_array(coupled, np.complex128, "coupled response")
        with np.errstate(all="ignore"):
            effective = channel * coupled
        effective = _finite_array(
            effective, np.complex128, "effective manifold"
        )
        return coupling, coupled, channel, effective

    def _selected_geometry_state(
        self, estimated: bool
    ) -> tuple[NDArray[np.float64], _RuntimeState]:
        if estimated:
            return self.assumed_positions, self.assumed_state
        return self.true_positions, self.true_state


def _build_runtime_state(
    state: AntennaStateConfig,
    active_antennas: tuple[str, ...],
    frequency_hz: float,
) -> _RuntimeState:
    if state.pattern_kind == "isotropic":
        scalar_pattern = IsotropicElementPattern()
        pattern: PolarizedElementPattern = ScalarPatternAdapter(scalar_pattern)
    elif state.pattern_kind == "parametric":
        scalar_pattern = None
        pattern = ParametricElementPattern(state.parametric_patterns)
    elif state.pattern_kind == "measured":
        scalar_pattern = None
        if state.measured_pattern_csv is None:
            raise ValueError("measured phone pattern requires a CSV path")
        pattern = TabulatedElementPattern.from_csv(
            state.measured_pattern_csv,
            frequency_hz,
            state.measured_pattern_sha256,
        )
    else:
        raise ValueError(f"unsupported phone pattern kind: {state.pattern_kind!r}")
    rotations = {
        name: _frozen_array(
            rotation_matrix_ypr(
                orientation.yaw_deg,
                orientation.pitch_deg,
                orientation.roll_deg,
            ),
            np.float64,
            f"{name} rotation",
        )
        for name, orientation in state.orientations.items()
    }
    return _RuntimeState(
        pattern_kind=state.pattern_kind,
        pattern=pattern,
        scalar_pattern=scalar_pattern,
        rotations=MappingProxyType(rotations),
        attenuation_regions=tuple(state.attenuation_regions),
        coupling=_build_coupling(state, active_antennas),
    )


def _build_coupling(
    state: AntennaStateConfig, active_antennas: tuple[str, ...]
) -> NDArray[np.complex128]:
    try:
        with np.errstate(all="ignore"):
            values = coupling_matrix(state.coupling_paths, active_antennas)
    except (ArithmeticError, FloatingPointError, TypeError, ValueError) as error:
        raise ValueError("coupling matrix must contain finite values") from error
    return _frozen_array(values, np.complex128, "coupling matrix")


def _build_channel_response(
    array_config: object, *, estimated: bool, label: str
) -> NDArray[np.complex128]:
    try:
        with np.errstate(all="ignore"):
            values = array_config.channel_response(estimated)
    except (ArithmeticError, FloatingPointError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite values") from error
    return _frozen_array(values, np.complex128, label)


def _frozen_array(values: object, dtype: object, label: str) -> NDArray:
    source = _finite_array(values, dtype, label, order="C")
    return np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(
        source.shape
    )


def _finite_array(
    values: object, dtype: object, label: str, *, order: str = "K"
) -> NDArray:
    try:
        result = np.array(values, dtype=dtype, copy=True, order=order)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must contain finite values") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain finite values")
    return result


def _validated_source_inputs(
    direction: object,
    polarization: object,
    frequency_hz: object,
) -> tuple[float, NDArray[np.complex128]]:
    from antenna_diversity.config import Direction

    if not isinstance(direction, Direction):
        raise ValueError("direction must be a Direction")
    direction.validate()
    if not isinstance(polarization, WavePolarization):
        raise ValueError("polarization must be a WavePolarization")
    return _validated_frequency(frequency_hz), polarization.jones_vector()


def _validated_frequency(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("frequency_hz must be a positive finite real scalar")
    try:
        candidate = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "frequency_hz must be a positive finite real scalar"
        ) from error
    if (
        candidate.shape != ()
        or np.iscomplexobj(candidate)
        or not np.issubdtype(candidate.dtype, np.number)
    ):
        raise ValueError("frequency_hz must be a positive finite real scalar")
    numeric = float(candidate)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("frequency_hz must be a positive finite real scalar")
    return numeric
