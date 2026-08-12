"""Top-level result containers for the end-to-end pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SimulationConfig


@dataclass
class BranchResult:
    combined: np.ndarray
    combining: dict[str, Any]
    acquisition: dict[str, Any]
    tracking: dict[str, Any]


@dataclass
class SimulationResults:
    config: SimulationConfig
    truth: dict[str, Any]
    scene_input_v: np.ndarray
    adc_codes: np.ndarray
    rf_info: dict[str, Any]
    dfe_channels: np.ndarray
    dfe_info: dict[str, Any]
    branches: dict[str, BranchResult]
    metrics: list[dict[str, Any]]


@dataclass(frozen=True)
class SpatialCombinationResult:
    active_antennas: tuple[str, ...]
    positions_m: tuple[tuple[float, float, float], ...]
    maximum_baseline_m: float
    mode: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_antennas": list(self.active_antennas),
            "positions_m": [list(position) for position in self.positions_m],
            "maximum_baseline_m": self.maximum_baseline_m,
            "mode": self.mode,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class SpatialEvaluationResult:
    mode: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "metrics": self.metrics}


@dataclass(frozen=True)
class SpatialGridScanResult:
    azimuths_deg: np.ndarray
    elevations_deg: np.ndarray
    mode: str
    metrics: dict[str, np.ndarray]
    interferer_gain: np.ndarray

    @property
    def output_sinr_db(self) -> np.ndarray:
        return self.metrics["output_sinr_db"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "azimuths_deg": self.azimuths_deg.tolist(),
            "elevations_deg": self.elevations_deg.tolist(),
            "mode": self.mode,
            "metrics": {
                name: values.tolist() for name, values in self.metrics.items()
            },
            "interferer_gain": self.interferer_gain.tolist(),
        }


@dataclass(frozen=True)
class SpatialParameterScanResult:
    parameter_name: str
    parameter_values: np.ndarray
    mode: str
    metrics: dict[str, np.ndarray]

    @property
    def output_sinr_db(self) -> np.ndarray:
        return self.metrics["output_sinr_db"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "parameter_values": self.parameter_values.tolist(),
            "mode": self.mode,
            "metrics": {
                name: values.tolist() for name, values in self.metrics.items()
            },
        }
