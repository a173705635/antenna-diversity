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
