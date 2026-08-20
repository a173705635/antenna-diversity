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
            ),
            assumed_positions=_frozen_array(
                select_phase_centers(candidate.phone.assumed_phase_centers_m, names),
                np.float64,
            ),
            true_state=_build_runtime_state(
                antenna.true_state, names, candidate.signal.carrier_hz
            ),
            assumed_state=_build_runtime_state(
                antenna.assumed_state, names, candidate.signal.carrier_hz
            ),
            true_channel=_frozen_array(
                candidate.array.channel_response(False), np.complex128
            ),
            assumed_channel=_frozen_array(
                candidate.array.channel_response(True), np.complex128
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
        if not isinstance(polarization, WavePolarization):
            raise ValueError("polarization must be a WavePolarization")
        polarization.jones_vector()
        positions, state = self._selected_geometry_state(estimated)
        spatial = array_manifold(
            positions,
            direction,
            frequency_hz,
            self.active_antennas,
            state.scalar_pattern,
        )
        if state.scalar_pattern is not None:
            nominal = spatial
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
                    frequency_hz,
                )
                voltages.append(np.vdot(height, local_wave.jones))
            nominal = spatial * np.asarray(voltages, dtype=np.complex128)
        blockage = attenuation_voltage(
            state.attenuation_regions, self.active_antennas, direction
        )
        return np.asarray(nominal * blockage, dtype=np.complex128)

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
        coupled = self.coupling_matrix(estimated=estimated) @ open_circuit
        return self.channel_response(estimated=estimated) * coupled

    def diagnostics(
        self,
        direction: Direction,
        polarization: WavePolarization,
        frequency_hz: float,
        *,
        estimated: bool = False,
    ) -> dict[str, object]:
        """Return a detached snapshot of every numeric composition stage."""
        positions, state = self._selected_geometry_state(estimated)
        open_circuit = self.open_circuit_response(
            direction, polarization, frequency_hz, estimated=estimated
        )
        attenuation = attenuation_voltage(
            state.attenuation_regions, self.active_antennas, direction
        )
        coupling = self.coupling_matrix(estimated=estimated)
        coupled = coupling @ open_circuit
        channel = self.channel_response(estimated=estimated)
        effective = channel * coupled
        return {
            "active_antennas": self.active_antennas,
            "estimated": bool(estimated),
            "model_label": self.model_label,
            "model_source": self.model_source,
            "pattern_kind": state.pattern_kind,
            "positions_m": positions.copy(),
            "attenuation_voltage": attenuation.copy(),
            "open_circuit_response": open_circuit.copy(),
            "coupling_matrix": coupling.copy(),
            "coupled_response": coupled.copy(),
            "channel_response": channel.copy(),
            "effective_manifold": effective.copy(),
        }

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
        )
        for name, orientation in state.orientations.items()
    }
    return _RuntimeState(
        pattern_kind=state.pattern_kind,
        pattern=pattern,
        scalar_pattern=scalar_pattern,
        rotations=MappingProxyType(rotations),
        attenuation_regions=tuple(state.attenuation_regions),
        coupling=_frozen_array(
            coupling_matrix(state.coupling_paths, active_antennas), np.complex128
        ),
    )


def _frozen_array(values: object, dtype: object) -> NDArray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result
