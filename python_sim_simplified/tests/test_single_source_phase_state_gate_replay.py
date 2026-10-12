"""Only completed past projection blocks can authorize internal anchors."""
import numpy as np
import pytest

from core.spoofing.recovery_geometry import FS
from core.spoofing.single_source import navigate
from experiments.spoofing.single_source_phase_state_gate import (
    projection_eligibility, replay_triplet,
)
from test_single_source_phase_state import fixed_code_geometry, observations
from test_single_source_phase_state_replay import saved_short


BLOCK=round(.02*FS)


def test_five_completed_blocks_change_mask_at_the_exact_sample_boundary():
    trace=dict(trace_dimensions=np.array([3,2,2,2,2,2,3]),
               trace_projectors=np.zeros((7,3,3),complex))
    samples=np.array([5*BLOCK-1,5*BLOCK,6*BLOCK-1,6*BLOCK,
                      7*BLOCK-1,7*BLOCK])
    mask,latest,recent=projection_eligibility(samples/FS,trace)
    np.testing.assert_array_equal(mask,[False,False,False,True,True,False])
    np.testing.assert_array_equal(latest,[3,4,4,5,5,6])
    np.testing.assert_array_equal(recent[3],[2,2,2,2,2])
    np.testing.assert_array_equal(recent[5],[2,2,2,2,3])


def test_future_block_dimension_cannot_rewrite_earlier_eligibility():
    times=np.array([6*BLOCK/FS,7*BLOCK/FS])
    trace=dict(trace_dimensions=np.array([3,2,2,2,2,2,3]),
               trace_projectors=np.zeros((7,3,3),complex))
    prefix=projection_eligibility(times[:1],trace)
    trace['trace_dimensions'][6]=2
    full=projection_eligibility(times,trace)
    for old,new in zip(prefix,full):
        np.testing.assert_array_equal(old,new[:1])


def test_triplet_checks_both_frozen_paths_and_runs_masked_state_independently(monkeypatch):
    fixed_code_geometry(monkeypatch)
    record=observations()
    baseline=navigate(record,diagnostics=True,code_blend=0.,
                      phase_leave_one_out=True)
    ungated=navigate(record,diagnostics=True,code_blend=0.,
                    phase_leave_one_out=True,phase_state_continuity=True)
    saved={**{key.replace('short_','baseline_',1):value
              for key,value in saved_short(baseline).items()},
           **{key.replace('short_','candidate_',1):value
              for key,value in saved_short(ungated).items()}}
    mask=np.ones(len(record['time']),bool)
    mask[24]=False
    old,open_state,gated=replay_triplet(record,saved,mask)
    assert np.isnan(old['position'][27]).all()
    assert np.isfinite(open_state['position'][27]).all()
    assert np.isnan(gated['position'][27]).all()
    assert 'phase_state_eligible' not in open_state['diagnostics']
    np.testing.assert_array_equal(gated['diagnostics']['phase_state_eligible'],mask)
    saved['candidate_diagnostic_reason'][3]='wrong'
    with pytest.raises(AssertionError):
        replay_triplet(record,saved,mask)
