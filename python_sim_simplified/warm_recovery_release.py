"""Publication boundary around the unchanged frozen research estimator.

Initialization completion is a scope requirement, never a selectable gate.
"""
import importlib.util
from pathlib import Path

FROZEN_PATH=Path(__file__).parent/'results/research-warm-hundred/recovery/warm_recovery_frozen.py'
_spec=importlib.util.spec_from_file_location('warm_recovery_frozen_release',FROZEN_PATH)
RESEARCH=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RESEARCH)

def estimate(obs):
    result=RESEARCH.estimate(obs,RESEARCH.SELECTED_METHODS)
    research=result['accepted'].copy()
    complete=result['init_completed_s'] is not None and bool(result['init_ok'].all())
    publish=research & result['init_ok'] & complete
    return {**result,'research_accepted':research,'publish_accepted':publish,'accepted':publish}
