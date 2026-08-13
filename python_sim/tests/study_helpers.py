from dataclasses import replace
from pathlib import Path

from antenna_diversity.studies.config import resolve_study_config
from antenna_diversity.studies.models import ResolvedStudyConfig


def resolved_for_test(*, preset: str = "quick", output_dir: Path | None = None, **study_changes: object) -> ResolvedStudyConfig:
    base = resolve_study_config(preset=preset)
    changes = dict(study_changes)
    if output_dir is not None:
        changes["output_dir"] = output_dir
    study = replace(base.study, **changes)
    return ResolvedStudyConfig.from_parts(study, base.simulation)
