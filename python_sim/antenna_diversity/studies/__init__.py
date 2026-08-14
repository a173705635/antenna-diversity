"""Reproducible spatial-study configuration and orchestration interfaces."""

from .config import PRESET_NAMES, resolve_study_config
from .models import ErrorLevel, InterfererTemplate, ResolvedStudyConfig, StudyConfig
from .runner import run_research

__all__ = [
    "ErrorLevel",
    "InterfererTemplate",
    "PRESET_NAMES",
    "ResolvedStudyConfig",
    "StudyConfig",
    "resolve_study_config",
    "run_research",
]
