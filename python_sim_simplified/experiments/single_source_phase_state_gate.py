"""Gate unpublished phase anchors using five completed projection blocks.

This replays saved observations only. The gate is a fixed causal diagnostic,
not a calibrated authentication decision or an RF receiver change.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.recovery_geometry import FS
from core.serialization import serializable
from core.single_source import navigate
from experiments.paths import RESULTS
from experiments.single_source_code_blend import (
    assert_saved_navigation, published_metrics,
)
from experiments.single_source_coherent_navigation import (
    quality_support_counts, score_mode,
)
from experiments.single_source_long_coherent import (
    navigation_prefix, save_navigation, source_snapshot,
)
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_phase_state import (
    CASES, OUTPUT as STAGE36, STAGES, preflight as preflight_old,
)
from experiments.single_source_ramp_takeover_long import long_tail_scores
from experiments.single_source_suppression_study import OUTPUT as STUDY_OUTPUT, digest
from experiments.single_source_temporal_trigger import (
    compare_publications, navigation_input,
)


OUTPUT=RESULTS/STUDY_OUTPUT/'phase_state_gate'
BLOCK_SAMPLES=round(.02*FS)
HISTORY_BLOCKS=5


def projection_eligibility(available_time,trace):
    """Use only 20 ms blocks fully ended by each output's availability."""
    dimensions=trace['trace_dimensions']
    antennas=trace['trace_projectors'].shape[1]
    completed=np.floor(np.asarray(available_time)*FS+1e-6).astype(np.int64)//BLOCK_SAMPLES
    latest=completed-1
    recent=np.full((len(latest),HISTORY_BLOCKS),-1,int)
    eligible=np.zeros(len(latest),bool)
    for k,last in enumerate(latest):
        if last>=0:
            block=dimensions[max(0,last-HISTORY_BLOCKS+1):last+1]
            recent[k,-len(block):]=block
            eligible[k]=(last>=HISTORY_BLOCKS-1
                         and np.all(block==antennas-1)
                         and np.all(block>=2))
    return eligible,latest,recent


def preflight():
    historic,current=preflight_old()
    stage36=json.loads((STAGE36/'summary.json').read_text(encoding='utf-8'))
    assert stage36['source_snapshot_sha256']==digest(STAGE36/'source_snapshot.zip')
    verify_source_snapshot(STAGE36/'source_snapshot.zip',stage36['source_sha256'])
    assert stage36['manifest_sha256']==digest(STAGE36/'manifest.json')
    assert stage36['figure_sha256']==digest(STAGE36/'overview.png')
    for name,stage,scene,mode in CASES:
        assert stage36['cases'][name]['result_sha256']==digest(STAGE36/f'{name}.npz')
        old=historic[stage]['cases'][scene]
        expected=(old['output'][mode]['output_sha256'] if stage=='long'
                  else old['output_sha256'][mode])
        assert digest(STAGES[stage]/scene/mode/'trace.npz')==expected['trace.npz']
    for name,sha in stage36['source_sha256'].items():
        if name.startswith('core/') and name!='core/single_source.py':
            assert current[name]==sha
    return historic,stage36,current


def replay_triplet(observations,saved,mask):
    baseline=navigate(observations,diagnostics=True,code_blend=0.,
                      phase_leave_one_out=True)
    assert_saved_navigation(baseline,saved,prefix='baseline_')
    open_state=navigate(observations,diagnostics=True,code_blend=0.,
                        phase_leave_one_out=True,phase_state_continuity=True)
    assert_saved_navigation(open_state,saved,prefix='candidate_')
    gated=navigate(observations,diagnostics=True,code_blend=0.,
                   phase_leave_one_out=True,phase_state_continuity=True,
                   phase_state_eligible=mask)
    np.testing.assert_array_equal(baseline['selected'],gated['selected'])
    return baseline,open_state,gated


def check_prefix(observations,mask,gated):
    count=len(observations['time'])-50
    prefix=navigate(navigation_prefix(observations,count),diagnostics=True,
                    code_blend=0.,phase_leave_one_out=True,
                    phase_state_continuity=True,phase_state_eligible=mask[:count])
    for key,value in prefix.items():
        if key=='diagnostics':
            for name,array in value.items():
                np.testing.assert_array_equal(array,gated[key][name][:count],err_msg=name)
        elif key=='initialization_time':
            np.testing.assert_array_equal(value,gated[key])
        else:
            np.testing.assert_array_equal(value,gated[key][:count],err_msg=key)


def pair_statistics(tracking,case,left,right):
    comparison,left_error,right_error,common=compare_publications(
        tracking,left,right,case)
    post=tracking['available_time']>=case['attack_start']
    left_published=np.isfinite(left['position']).all(axis=1)
    right_published=np.isfinite(right['position']).all(axis=1)
    comparison['left_only_error']=published_metrics(
        left_error,post&left_published&~right_published)
    comparison['right_only_error']=published_metrics(
        right_error,post&right_published&~left_published)
    return comparison,common


def replay_case(name,stage,scene,mode,historic,stage36):
    source=STAGES[stage]/scene/mode
    with np.load(source/'tracking.npz') as saved:
        tracking=dict(saved)
    with np.load(source/'support.npz') as saved:
        support=saved['coherent_support'].copy()
    with np.load(source/'trace.npz') as saved:
        trace=dict(saved)
    with np.load(STAGE36/f'{name}.npz') as saved:
        old_result=dict(saved)
    metadata=json.loads((source.parent/'input'/'metadata.json').read_text(encoding='utf-8'))
    mask,latest,recent=projection_eligibility(tracking['available_time'],trace)
    observations=navigation_input(tracking,support)
    baseline,open_state,gated=replay_triplet(observations,old_result,mask)
    check_prefix(observations,mask,gated)
    np.testing.assert_array_equal(gated['diagnostics']['phase_state_eligible'],mask)
    navigations=dict(baseline=baseline,open_state=open_state,gated=gated)
    counts=quality_support_counts(tracking,support)
    scores={label:score_mode(nav,tracking,metadata['case'],counts,support)
            for label,nav in navigations.items()}
    tails={label:long_tail_scores(tracking,nav,metadata['case'])
           for label,nav in navigations.items()}
    pairs={}
    for label,left,right in (('baseline_open',baseline,open_state),
                             ('baseline_gated',baseline,gated),
                             ('open_gated',open_state,gated)):
        pairs[label],_=pair_statistics(tracking,metadata['case'],left,right)
    published={label:np.isfinite(nav['position']).all(axis=1)
               for label,nav in navigations.items()}
    post=tracking['available_time']>=metadata['case']['attack_start']
    triple=post&published['baseline']&published['open_state']&published['gated']
    truth=np.column_stack((metadata['case']['velocity_east']*tracking['time'],
                           np.zeros(len(tracking['time']))))
    errors={label:np.linalg.norm(nav['position'][:,:2]-truth,axis=1)
            for label,nav in navigations.items()}
    three_way={label:published_metrics(error,triple)
               for label,error in errors.items()}
    diagnostic=gated['diagnostics']
    internal=((diagnostic['phase_state_short_anchor_index']>=0)
              & ~diagnostic['phase_state_short_anchor_published'])
    blocked=np.any(diagnostic['phase_state_blocked_internal_anchor_index']>=0,
                   axis=1)
    state=dict(eligible_post_epochs=int((post&mask).sum()),
               output_gated_epochs=int((post&(
                   diagnostic['phase_state_source']=='output_gate')).sum()),
               internal_short_anchor_epochs=int((post&internal).sum()),
               blocked_internal_anchor_epochs=int((post&blocked).sum()),
               max_unpublished_run_epochs=int(
                   diagnostic['phase_state_unpublished_run_epochs'][post].max()),
               recovery_confirmed_epochs=int((post&diagnostic['recovery_confirmed']).sum()),
               published_recovery_confirmed_epochs=int((
                   post&published['gated']&diagnostic['recovery_confirmed']).sum()))
    arrays=dict(observation_time=tracking['time'],
                available_time=tracking['available_time'],
                eligible=mask,last_completed_block_index=latest,
                recent_completed_dimensions=recent,
                three_way_common_published=triple)
    arrays.update({f'{label}_horizontal_error_m':error
                   for label,error in errors.items()})
    save_navigation(arrays,navigations)
    path=OUTPUT/f'{name}.npz'
    np.savez_compressed(path,**arrays)
    report=dict(stage36_result_sha256=stage36['cases'][name]['result_sha256'],
                input_trace_sha256=digest(source/'trace.npz'),
                result_sha256=digest(path),scores=scores,
                observed_time=tails,pairs=pairs,
                three_way_common=three_way,state=state)
    return report,(tracking['available_time'],errors)


def plot_cases(plots):
    figure,axes=plt.subplots(4,2,figsize=(12,13),constrained_layout=True)
    for axis,(name,(time,errors)) in zip(axes.flat,plots.items()):
        for label,value in errors.items():
            axis.plot(time,value,label=label,linewidth=.7,
                      marker='.',markersize=1)
        axis.axhline(20.,color='gray',linestyle=':',linewidth=1)
        axis.set(title=name,xlabel='Available time (s)',
                 ylabel='Published horizontal error (m)')
        axis.grid(alpha=.25)
    axes[0,0].legend()
    axes.flat[-1].axis('off')
    figure.savefig(OUTPUT/'overview.png',dpi=150)
    plt.close(figure)


def main():
    historic,stage36,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip',sources)
    manifest=dict(scope=__doc__,source_sha256=sources,
                  source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
                  stage36_summary_sha256=digest(STAGE36/'summary.json'),
                  cases=CASES,completed_blocks=HISTORY_BLOCKS,
                  block_samples=BLOCK_SAMPLES,required_dimension='M-1 and >=2',
                  baseline_published_only=True,open_state_stage36=True,
                  gated_state_only_new_factor=True)
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(manifest),
        indent=2,allow_nan=False),encoding='utf-8')
    reports={};plots={}
    for name,stage,scene,mode in CASES:
        reports[name],plots[name]=replay_case(name,stage,scene,mode,
                                              historic[stage],stage36)
        print(name,'saved',flush=True)
    plot_cases(plots)
    assert source_hashes()==sources
    summary=dict(scope=__doc__,source_sha256=sources,
                 source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
                 manifest_sha256=digest(OUTPUT/'manifest.json'),
                 figure_sha256=digest(OUTPUT/'overview.png'),cases=reports)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),
        indent=2,allow_nan=False),encoding='utf-8')
    return summary


if __name__=='__main__':
    main()
