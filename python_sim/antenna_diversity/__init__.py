"""GPS L1 C/A antenna-diversity simulation package."""

from .config import SimulationConfig, default_config, parametric_phone_config
from .models import BranchResult, SimulationResults

__all__ = [
    "BranchResult",
    "SimulationConfig",
    "SimulationResults",
    "default_config",
    "parametric_phone_config",
]
