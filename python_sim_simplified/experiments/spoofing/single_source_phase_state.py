"""Replay an unpublished short-phase state on seven frozen RF observation sets.

Only the phase position used by the existing short carrier fit changes. Code
admission, support, recovery rules, and publication criteria stay intact; the
published trajectory can change.
"""
from core.provenance import current_path_for_historical_source, ROOT
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from core.spoofing.single_source import navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_code_blend import (
    assert_saved_navigation, published_metrics,
)
from experiments.spoofing.single_source_coherent_navigation import (
    quality_support_counts, score_mode,
)
from experiments.spoofing.single_source_long_coherent import (
    navigation_prefix, save_navigation, source_snapshot,
)
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_ramp_takeover_long import long_tail_scores
from experiments.spoofing.single_source_suppression_study import OUTPUT as STUDY_OUTPUT, digest
from experiments.spoofing.single_source_temporal_trigger import (
    compare_publications, navigation_input,
)


ROOT=RESULTS/STUDY_OUTPUT
OUTPUT=ROOT/'phase_state'
STAGES=dict(long=ROOT/'ramp_takeover_long',
            ramp=ROOT/'ramp_takeover',
            power=ROOT/'spoof_power_zero')
CASES=(('long_weak','long','weak','aided_five_ms'),
       ('long_control','long','no_attack_control','aided_five_ms'),
       ('ramp_weak','ramp','weak','aided_five_ms'),
       ('ramp_absent','ramp','genuine_absent','aided_five_ms'),
       ('ramp_absent_default','ramp','genuine_absent','default'),
       ('power_weak','power','weak','off'),
       ('power_absent','power','genuine_absent','off'))


def preflight():
    summaries={stage:json.loads((folder/'summary.json').read_text(encoding='utf-8'))
               for stage,folder in STAGES.items()}
    current=source_hashes()
    for stage,folder in STAGES.items():
        summary=summaries[stage]
        assert summary['source_snapshot_sha256']==digest(folder/'source_snapshot.zip')
        verify_source_snapshot(folder/'source_snapshot.zip',summary['source_sha256'])
        for key,name in (('plan_sha256','plan.json'),('manifest_sha256','manifest.json')):
            assert summary[key]==digest(folder/name)
    for name,sha in summaries['long']['source_sha256'].items():
        if name.startswith('core/') and name not in (
                'core/single_source.py','core/mechanism_phase.py'):
            assert current[current_path_for_historical_source(ROOT, name).relative_to(ROOT).as_posix()]==sha
    checked_inputs=set()
    for _,stage,scene,mode in CASES:
        folder=STAGES[stage]/scene
        case=summaries[stage]['cases'][scene]
        if (stage,scene) not in checked_inputs:
            for filename,sha in case['input_sha256'].items():
                assert digest(folder/'input'/filename)==sha
            checked_inputs.add((stage,scene))
        outputs=(case['output'][mode]['output_sha256'] if stage=='long'
                 else case['output_sha256'][mode])
        for filename in ('tracking.npz','support.npz','navigation.npz'):
            assert digest(folder/mode/filename)==outputs[filename]
    return summaries,current


def replay_pair(observations,saved):
    """Check the complete old short run, then start the candidate afresh."""
    baseline=navigate(observations,diagnostics=True,code_blend=0.,
                      phase_leave_one_out=True)
    assert_saved_navigation(baseline,saved,prefix='short_')
    candidate=navigate(observations,diagnostics=True,code_blend=0.,
                       phase_leave_one_out=True,
                       phase_state_continuity=True)
    np.testing.assert_array_equal(baseline['selected'],candidate['selected'])
    return baseline,candidate


def check_prefix(observations,candidate):
    count=len(observations['time'])-50
    prefix=navigate(navigation_prefix(observations,count),diagnostics=True,
                    code_blend=0.,phase_leave_one_out=True,
                    phase_state_continuity=True)
    for key,value in prefix.items():
        if key=='diagnostics':
            for name,array in value.items():
                np.testing.assert_array_equal(array,candidate[key][name][:count],
                                              err_msg=name)
        elif key=='initialization_time':
            np.testing.assert_array_equal(value,candidate[key])
        else:
            np.testing.assert_array_equal(value,candidate[key][:count],err_msg=key)


def replay_case(name,stage,scene,mode,summary):
    folder=STAGES[stage]/scene
    source=folder/mode
    with np.load(source/'tracking.npz') as saved:
        tracking=dict(saved)
    with np.load(source/'support.npz') as saved:
        support=saved['coherent_support'].copy()
    with np.load(source/'navigation.npz') as saved:
        reference=dict(saved)
    metadata=json.loads((folder/'input'/'metadata.json').read_text(encoding='utf-8'))
    observations=navigation_input(tracking,support)
    baseline,candidate=replay_pair(observations,reference)
    check_prefix(observations,candidate)
    counts=quality_support_counts(tracking,support)
    scores={label:score_mode(nav,tracking,metadata['case'],counts,support)
            for label,nav in (('baseline',baseline),('candidate',candidate))}
    comparison,old_error,new_error,common=compare_publications(
        tracking,baseline,candidate,metadata['case'])
    post=tracking['available_time']>=metadata['case']['attack_start']
    old_published=np.isfinite(baseline['position']).all(axis=1)
    new_published=np.isfinite(candidate['position']).all(axis=1)
    comparison['baseline_only_error']=published_metrics(
        old_error,post & old_published & ~new_published)
    comparison['candidate_only_error']=published_metrics(
        new_error,post & new_published & ~old_published)
    tail={label:long_tail_scores(tracking,nav,metadata['case'])
          for label,nav in (('baseline',baseline),('candidate',candidate))}
    diagnostic=candidate['diagnostics']
    internal=((diagnostic['phase_state_short_anchor_index']>=0)
              & ~diagnostic['phase_state_short_anchor_published'])
    state=dict(output_gated_epochs=int((post & (diagnostic['phase_state_source']
                                               =='output_gate')).sum()),
               internal_short_anchor_epochs=int((post & internal).sum()),
               max_unpublished_run_epochs=int(
                   diagnostic['phase_state_unpublished_run_epochs'][post].max()),
               phase_ok_epochs=int((post & diagnostic['phase_ok']).sum()),
               recovery_confirmed_epochs=int((post & diagnostic['recovery_confirmed']).sum()),
               published_recovery_confirmed_epochs=int((
                   post & new_published & diagnostic['recovery_confirmed']).sum()))
    arrays=dict(observation_time=tracking['time'],
                available_time=tracking['available_time'],
                common_published=common,
                baseline_horizontal_error_m=old_error,
                candidate_horizontal_error_m=new_error)
    save_navigation(arrays,dict(baseline=baseline,candidate=candidate))
    path=OUTPUT/f'{name}.npz'
    np.savez_compressed(path,**arrays)
    return (dict(reference_sha256=summary['cases'][scene]['output'][mode][
                     'output_sha256']['navigation.npz'] if stage=='long' else
                     summary['cases'][scene]['output_sha256'][mode]['navigation.npz'],
                 tracking_sha256=digest(source/'tracking.npz'),
                 support_sha256=digest(source/'support.npz'),
                 metadata_sha256=digest(folder/'input'/'metadata.json'),
                 result_sha256=digest(path),scores=scores,
                 publication_comparison=comparison,observed_time=tail,
                 state=state),
            (tracking['available_time'],old_error,new_error))


def plot_cases(plots):
    figure,axes=plt.subplots(4,2,figsize=(12,13),constrained_layout=True)
    for axis,(name,(time,old,new)) in zip(axes.flat,plots.items()):
        axis.plot(time,old,label='published-only',linewidth=.7,marker='.',markersize=1)
        axis.plot(time,new,label='phase state',linewidth=.7,marker='.',markersize=1)
        axis.axhline(20.,color='gray',linestyle=':',linewidth=1)
        axis.set(title=name,xlabel='Available time (s)',
                 ylabel='Published horizontal error (m)')
        axis.grid(alpha=.25)
    axes[0,0].legend()
    axes.flat[-1].axis('off')
    figure.savefig(OUTPUT/'overview.png',dpi=150)
    plt.close(figure)


def main():
    historic,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip',sources)
    manifest=dict(scope=__doc__,source_sha256=sources,
                  source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
                  historical_summary_sha256={stage:digest(folder/'summary.json')
                                             for stage,folder in STAGES.items()},
                  cases=CASES,baseline_code_blend=0.,
                  baseline_phase_leave_one_out=True,
                  candidate_phase_state_continuity=True)
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(manifest),
        indent=2,allow_nan=False),encoding='utf-8')
    reports={};plots={}
    for name,stage,scene,mode in CASES:
        reports[name],plots[name]=replay_case(name,stage,scene,mode,historic[stage])
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
