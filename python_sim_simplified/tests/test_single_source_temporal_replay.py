"""Saved-observation temporal-trigger comparison uses matched publication epochs."""
import json
import numpy as np
import pytest

from core.serialization import serializable
from experiments.single_source_temporal_trigger import (
    block_summary, compare_publications, current_projector_diagnostics,
    navigation_input,
)


def test_common_errors_exclude_changed_publication_sets():
    observations=dict(time=np.array([0.,.01,.02]),
                      available_time=np.array([.30,.31,.32]))
    case=dict(attack_start=.3,velocity_east=0.)
    baseline=dict(position=np.array([[1.,0.,0.],[20.,0.,0.],[np.nan]*3]))
    candidate=dict(position=np.array([[2.,0.,0.],[np.nan]*3,[50.,0.,0.]]))
    score,base_error,candidate_error,common=compare_publications(
        observations,baseline,candidate,case)
    np.testing.assert_array_equal(common,[True,False,False])
    assert score['common_published']==1
    assert score['candidate_only_published']==1
    assert score['baseline_only_published']==1
    assert score['baseline_common']['max_error_m']==1.
    assert score['candidate_common']['max_error_m']==2.
    assert base_error[1]==20. and candidate_error[2]==50.


def test_navigation_input_keeps_observation_fields_and_uses_selected_support():
    fields=('time','available_time','satellites','ranges','lock','cn0',
            'gain','support','arc','carrier')
    tracking={key:np.arange(3) for key in fields}
    tracking['beam_trigger_effective']=np.arange(2)
    replacement=np.full(3,4.)
    result=navigation_input(tracking,replacement)
    assert set(result)==set(fields)
    assert result['support'] is replacement
    for key in fields:
        if key!='support':assert result[key] is tracking[key]


def test_block_audit_recomputes_history_mean_independently():
    nominal=np.array([[1.,1.],[1.,-1.],[1.,1.]],complex)
    projector=np.diag([1.,0.,1.]).astype(complex)
    gain=np.sum(nominal.conj()*(projector@nominal),axis=0).real/3.
    tracking=dict(time=np.array([.02,.04]),gain=np.broadcast_to(gain,(2,2)),
                  ratio=np.array([20.,1.]),
                  beam_trigger_raw=np.array([np.nan,20.,1.]),
                  beam_trigger_history_mean=np.array([np.nan,20.,10.5]),
                  beam_trigger_effective=np.array([np.nan,20.,10.5]),
                  beam_trigger_history_only=np.array([False,False,True]))
    trace=dict(trace_nominal=nominal,
               trace_projectors=np.array([np.eye(3),projector,projector]),
               trace_dimensions=np.array([3,2,2]))
    result=current_projector_diagnostics(tracking,trace)
    np.testing.assert_array_equal(result['trigger_on'],[False,True,True])
    tracking['beam_trigger_history_mean']=np.array([np.nan,20.,1.])
    with pytest.raises(AssertionError):
        current_projector_diagnostics(tracking,trace)


def test_block_summary_is_json_safe_and_points_to_saved_complex_trace():
    block=dict(raw_trigger=np.array([2.]),
               history_mean_trigger=np.array([4.]),
               effective_trigger=np.array([4.]),
               trigger_on=np.array([True]),
               projector_dimension=np.array([2]),
               current_projector_gain=np.array([[.1,.2]]))
    trace=dict(trace_projectors=np.array([np.diag([1.+0j,0.+0j,1.+0j])]))
    item=block_summary(block,trace,0)
    assert item['projector_trace_index']==0
    assert 'projector' not in item
    assert json.loads(json.dumps(serializable(item),allow_nan=False))['on']
