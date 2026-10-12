"""One 20 s genuine-absent RF paired to the saved weak/control velocity step.

Only the aided 5 ms frontend is rerun. Three short-navigation states replay its
saved observations independently; no code, phase, or support parameter changes.
"""
from core.provenance import current_path_for_historical_source, ROOT
import gc
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.spoofing.recovery_geometry import FS
from core.serialization import serializable
from core.spoofing.single_source import navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_code_blend import assert_saved_navigation, published_metrics
from experiments.spoofing.single_source_coherent_navigation import quality_support_counts, score_mode
from experiments.spoofing.single_source_history_study import generate_new_record
from experiments.spoofing.single_source_long_coherent import (
    navigation_prefix, save_navigation, source_snapshot,
)
from experiments.spoofing.single_source_phase_state_gate import (
    OUTPUT as STAGE37, pair_statistics, preflight as preflight_stage37,
    projection_eligibility,
)
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_ramp_takeover_long import (
    OUTPUT as STAGE35, PLAN, audit_fit, long_tail_scores, run_receiver,
    save_new_input,
)
from experiments.spoofing.single_source_spoof_power import observation_quality
from experiments.spoofing.single_source_suppression_study import OUTPUT as STUDY_OUTPUT, digest
from experiments.spoofing.single_source_temporal_trigger import navigation_input


OUTPUT=RESULTS/STUDY_OUTPUT/'long_absent'
SCENE='genuine_absent'
MODE='aided_five_ms'
MODES=('published_only','open_state','gated_state')
OUTPUT_FILES=('tracking.npz','trace.npz','history.npz','groups.npz',
              'support.npz','navigation.npz')


def verify_pair(record,identity,references,onset_sample):
    """Only true-signal removal may differ from the attacked weak record."""
    assert identity['scenario']==SCENE and identity['attacked']
    assert not identity['case']['genuine']
    weak_case=references[0][1]['case']
    assert references[1][1]['case']==weak_case
    assert {key:value for key,value in identity['case'].items()
            if key!='genuine'}=={
            key:value for key,value in weak_case.items() if key!='genuine'}
    for index,(old,old_identity) in enumerate(references):
        assert old_identity['scenario']==('weak','no_attack_control')[index]
        assert old_identity['attacked']==(index==0)
        for key,value in identity.items():
            if key not in ('scenario','attacked','case','rf_sha256'):
                assert old_identity[key]==value
        np.testing.assert_array_equal(record['samples'][:,:onset_sample],
                                      old['samples'][:,:onset_sample])
        for key,value in record.items():
            if key!='samples':
                np.testing.assert_array_equal(value,old[key],err_msg=key)


def preflight():
    historic,stage36,current=preflight_stage37()
    stage35=historic['long']
    stage37=json.loads((STAGE37/'summary.json').read_text(encoding='utf-8'))
    assert stage37['manifest_sha256']==digest(STAGE37/'manifest.json')
    assert stage37['source_snapshot_sha256']==digest(STAGE37/'source_snapshot.zip')
    verify_source_snapshot(STAGE37/'source_snapshot.zip',stage37['source_sha256'])
    assert stage37['figure_sha256']==digest(STAGE37/'overview.png')
    for name,sha in stage37['source_sha256'].items():
        if name.startswith('core/'):
            assert current[current_path_for_historical_source(ROOT, name).relative_to(ROOT).as_posix()]==sha
    for scene in ('weak','no_attack_control'):
        for filename,sha in stage35['cases'][scene]['input_sha256'].items():
            assert digest(STAGE35/scene/'input'/filename)==sha
    return stage35,stage37,current


def reference_pair():
    references=[]
    for scene in ('weak','no_attack_control'):
        folder=STAGE35/scene/'input'
        with np.load(folder/'observations.npz') as saved:
            old=dict(saved)
        old['samples']=np.load(folder/'samples.npy',mmap_mode='r')
        identity=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        references.append((old,identity))
    return references


def replay_three(observations,saved,mask):
    baseline=navigate(observations,diagnostics=True,code_blend=0.,
                      phase_leave_one_out=True)
    assert_saved_navigation(baseline,saved,prefix='short_')
    open_state=navigate(observations,diagnostics=True,code_blend=0.,
                        phase_leave_one_out=True,phase_state_continuity=True)
    gated=navigate(observations,diagnostics=True,code_blend=0.,
                   phase_leave_one_out=True,phase_state_continuity=True,
                   phase_state_eligible=mask)
    np.testing.assert_array_equal(baseline['selected'],open_state['selected'])
    np.testing.assert_array_equal(baseline['selected'],gated['selected'])
    return baseline,open_state,gated


def check_prefix(observations,mask,navigations):
    count=len(observations['time'])-50
    prefix_record=navigation_prefix(observations,count)
    for label,navigation in navigations.items():
        options=dict(phase_state_continuity=label!='published_only')
        if label=='gated_state':options['phase_state_eligible']=mask[:count]
        prefix=navigate(prefix_record,diagnostics=True,code_blend=0.,
                        phase_leave_one_out=True,**options)
        for key,value in prefix.items():
            if key=='diagnostics':
                for name,array in value.items():
                    np.testing.assert_array_equal(array,
                        navigation[key][name][:count],err_msg=f'{label}.{name}')
            elif key=='initialization_time':
                np.testing.assert_array_equal(value,navigation[key])
            else:
                np.testing.assert_array_equal(value,navigation[key][:count],err_msg=label)


def replay_navigation(folder,case,output):
    with np.load(folder/'tracking.npz') as saved:
        tracking=dict(saved)
    with np.load(folder/'support.npz') as saved:
        support=saved['coherent_support'].copy()
    with np.load(folder/'trace.npz') as saved:
        trace=dict(saved)
    with np.load(folder/'navigation.npz') as saved:
        receiver_nav=dict(saved)
    mask,latest,recent=projection_eligibility(tracking['available_time'],trace)
    observations=navigation_input(tracking,support)
    baseline,open_state,gated=replay_three(observations,receiver_nav,mask)
    navigations=dict(zip(MODES,(baseline,open_state,gated)))
    check_prefix(observations,mask,navigations)
    counts=quality_support_counts(tracking,support)
    scores={label:dict(navigation=score_mode(nav,tracking,case,counts,support),
                       observation_quality=observation_quality(
                           observations,nav,case['attack_start']),
                       observed_time=long_tail_scores(tracking,nav,case))
            for label,nav in navigations.items()}
    pairs={}
    for label,left,right in (('published_open',baseline,open_state),
                             ('published_gated',baseline,gated),
                             ('open_gated',open_state,gated)):
        pairs[label],_=pair_statistics(tracking,case,left,right)
    published={label:np.isfinite(nav['position']).all(axis=1)
               for label,nav in navigations.items()}
    post=tracking['available_time']>=case['attack_start']
    triple=post&published['published_only']&published['open_state']&published['gated_state']
    truth=np.column_stack((case['velocity_east']*tracking['time'],
                           np.zeros(len(tracking['time']))))
    errors={label:np.linalg.norm(nav['position'][:,:2]-truth,axis=1)
            for label,nav in navigations.items()}
    common={label:published_metrics(error,triple) for label,error in errors.items()}
    states={}
    for label in ('open_state','gated_state'):
        diagnostic=navigations[label]['diagnostics']
        internal=((diagnostic['phase_state_short_anchor_index']>=0)
                  & ~diagnostic['phase_state_short_anchor_published'])
        states[label]=dict(output_gated_epochs=int((post&(
            diagnostic['phase_state_source']=='output_gate')).sum()),
            internal_short_anchor_epochs=int((post&internal).sum()),
            max_unpublished_run_epochs=int(
                diagnostic['phase_state_unpublished_run_epochs'][post].max()),
            recovery_confirmed_epochs=int((post&diagnostic['recovery_confirmed']).sum()),
            published_recovery_confirmed_epochs=int((post&published[label]
                &diagnostic['recovery_confirmed']).sum()))
    states['gated_state']['eligible_post_epochs']=int((post&mask).sum())
    states['gated_state']['blocked_internal_anchor_epochs']=int((post&np.any(
        gated['diagnostics']['phase_state_blocked_internal_anchor_index']>=0,
        axis=1)).sum())
    arrays=dict(observation_time=tracking['time'],
                available_time=tracking['available_time'],
                eligible=mask,last_completed_block_index=latest,
                recent_completed_dimensions=recent,
                three_way_common_published=triple)
    arrays.update({f'{label}_horizontal_error_m':error
                   for label,error in errors.items()})
    save_navigation(arrays,navigations)
    np.savez_compressed(output/'navigation_replay.npz',**arrays)
    return (dict(result_sha256=digest(output/'navigation_replay.npz'),
                 scores=scores,pairs=pairs,three_way_common=common,state=states),
            (tracking['available_time'],counts,errors))


def plot_result(view,output):
    time,counts,errors=view
    figure,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].plot(time,counts,linewidth=.7)
    axes[0].set(title='Genuine absent: quality-gated coherent support',
                xlabel='Available time (s)',ylabel='Stars')
    for label,error in errors.items():
        axes[1].plot(time,error,label=label,linewidth=.7,
                     marker='.',markersize=1)
    axes[1].axhline(20.,color='gray',linestyle=':',linewidth=1)
    axes[1].set(xlabel='Available time (s)',ylabel='Published true error (m)')
    axes[1].legend(fontsize=7)
    for axis in axes:axis.grid(alpha=.25)
    figure.savefig(output/'overview.png',dpi=150)
    plt.close(figure)


def main():
    stage35,stage37,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip',sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(dict(
        **PLAN,scenario=SCENE,receiver_mode=MODE,
        navigation_modes=MODES,source_sha256=sources)),
        indent=2,allow_nan=False),encoding='utf-8')
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        stage35_summary_sha256=digest(STAGE35/'summary.json'),
        stage37_summary_sha256=digest(STAGE37/'summary.json'))),
        indent=2,allow_nan=False),encoding='utf-8')
    record,identity=generate_new_record(PLAN,SCENE,fake_rate=PLAN['fake_rate_mps'])
    references=reference_pair()
    onset=round(PLAN['attack_start_s']*FS)
    assert (onset-1)/FS<PLAN['attack_start_s']<=onset/FS
    verify_pair(record,identity,references,onset)
    weak=references[0][0]
    prefix=weak['samples'][:,:onset].copy()
    geometry={name:value.copy() for name,value in weak.items() if name!='samples'}
    del references
    gc.collect()
    input_sha,_,_=save_new_input(OUTPUT,record,identity,prefix,geometry)
    del record,prefix,geometry
    gc.collect()
    print('20 s genuine-absent IQ saved',flush=True)
    with np.load(OUTPUT/'input'/'observations.npz') as saved:
        replay=dict(saved)
    replay['samples']=np.load(OUTPUT/'input'/'samples.npy',mmap_mode='r')
    folder=OUTPUT/MODE
    receiver=run_receiver(replay,identity['case'],folder,MODE)
    fit_sha=audit_fit(replay,OUTPUT,False)
    del replay
    gc.collect()
    print('aided 5 ms frontend and independent 1000-block fit saved',flush=True)
    navigation,view=replay_navigation(folder,identity['case'],OUTPUT)
    plot_result(view,OUTPUT)
    assert source_hashes()==sources
    summary=dict(scope=__doc__,source_sha256=sources,
                 source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
                 plan_sha256=digest(OUTPUT/'plan.json'),
                 manifest_sha256=digest(OUTPUT/'manifest.json'),
                 figure_sha256=digest(OUTPUT/'overview.png'),
                 stage35_source_snapshot_sha256=stage35['source_snapshot_sha256'],
                 stage37_source_snapshot_sha256=stage37['source_snapshot_sha256'],
                 input_sha256=input_sha,receiver=receiver,
                 fit_sha256=fit_sha,navigation=navigation)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),
        indent=2,allow_nan=False),encoding='utf-8')
    return summary


if __name__=='__main__':
    main()
