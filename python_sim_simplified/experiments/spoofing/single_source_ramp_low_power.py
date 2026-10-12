"""Paired 20 s, -5 dB velocity-step RF with weak and genuine-absent truth.

The sole change from the saved 0 dB weak/absent pair is forged PRN power.
Each new IQ receives one aided-5 ms frontend; three short navigation states
replay its saved observations. Truth-directed spatial scores are offline only.
"""
import gc
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.spoofing.recovery_geometry import FS
from core.serialization import serializable
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_code_blend import published_metrics
from experiments.spoofing.single_source_history_study import generate_new_record
from experiments.spoofing.single_source_long_absent import (
    OUTPUT as STAGE38, MODE, MODES, preflight as preflight_stage38,
    replay_navigation,
)
from experiments.spoofing.single_source_long_coherent import source_snapshot
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_phase_state_gate import OUTPUT as STAGE37, pair_statistics
from experiments.spoofing.single_source_ramp_takeover_long import (
    OUTPUT as STAGE35, PLAN, audit_fit, run_receiver, save_new_input,
)
from experiments.spoofing.single_source_spoof_power import fake_east_at_observation
from experiments.spoofing.single_source_suppression_study import OUTPUT as STUDY_OUTPUT, digest
from experiments.spoofing.single_source_template_coherence import truth_audit


OUTPUT=RESULTS/STUDY_OUTPUT/'ramp_low_power'
LOW_PLAN={**PLAN,'spoof_db':-5.}
SCENES=('weak','genuine_absent')
REFERENCES={'weak':STAGE35/'weak','genuine_absent':STAGE38}
OLD_NAV={'weak':STAGE37/'long_weak.npz',
         'genuine_absent':STAGE38/'navigation_replay.npz'}
OLD_LABEL={'weak':{'published_only':'baseline','open_state':'open_state',
                   'gated_state':'gated'},
           'genuine_absent':{name:name for name in MODES}}


def verify_power_input(record,identity,old,old_identity,onset_sample):
    """Verify a new IQ against its own 0 dB scene before receiver replay."""
    assert identity['scenario']==old_identity['scenario']
    assert identity['attacked']==old_identity['attacked']
    for name,value in identity.items():
        if name not in ('case','rf_sha256'):
            assert old_identity[name]==value
    case=identity['case']
    reference=old_identity['case']
    assert case['spoof_db']==-5. and reference['spoof_db']==0.
    assert {name:value for name,value in case.items() if name!='spoof_db'}=={
        name:value for name,value in reference.items() if name!='spoof_db'}
    np.testing.assert_array_equal(record['samples'][:,:onset_sample],
                                  old['samples'][:,:onset_sample])
    for name,value in record.items():
        if name!='samples':
            np.testing.assert_array_equal(value,old[name],err_msg=name)


def compare_power_pair(weak,absent,onset_sample):
    """Only true-signal presence may differ between the new paired scenes."""
    weak_record,weak_identity=weak
    absent_record,absent_identity=absent
    assert weak_identity['scenario']=='weak'
    assert absent_identity['scenario']=='genuine_absent'
    assert weak_identity['attacked'] and absent_identity['attacked']
    assert weak_identity['case']['genuine']
    assert not absent_identity['case']['genuine']
    assert {name:value for name,value in weak_identity['case'].items()
            if name!='genuine'}=={
            name:value for name,value in absent_identity['case'].items()
            if name!='genuine'}
    for name,value in weak_identity.items():
        if name not in ('scenario','case','rf_sha256','samples_sha256',
                        'observations_sha256'):
            assert absent_identity[name]==value
    np.testing.assert_array_equal(weak_record['samples'][:,:onset_sample],
                                  absent_record['samples'][:,:onset_sample])
    for name,value in weak_record.items():
        if name!='samples':
            np.testing.assert_array_equal(value,absent_record[name],err_msg=name)


def compare_iq_components(low_weak,low_absent,high_weak,high_absent,
                          block_samples):
    """The true component is unchanged by fake power, across every sample."""
    max_abs=0.
    for start in range(0,low_weak.shape[1],block_samples):
        stop=min(start+block_samples,low_weak.shape[1])
        low=low_weak[:,start:stop]-low_absent[:,start:stop]
        high=high_weak[:,start:stop]-high_absent[:,start:stop]
        residual=np.abs(low-high)
        max_abs=max(max_abs,float(residual.max()))
        np.testing.assert_allclose(low,high,rtol=0,atol=1e-11)
    return max_abs


def reference_input(scene):
    folder=REFERENCES[scene]/'input'
    with np.load(folder/'observations.npz') as saved:
        record=dict(saved)
    record['samples']=np.load(folder/'samples.npy',mmap_mode='r')
    identity=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    return record,identity


def preflight():
    stage35,stage37,sources=preflight_stage38()
    stage38=json.loads((STAGE38/'summary.json').read_text(encoding='utf-8'))
    for name,folder,summary in (('stage35',STAGE35,stage35),
                                ('stage37',STAGE37,stage37),
                                ('stage38',STAGE38,stage38)):
        assert summary['source_snapshot_sha256']==digest(folder/'source_snapshot.zip')
        verify_source_snapshot(folder/'source_snapshot.zip',summary['source_sha256'])
        assert summary['manifest_sha256']==digest(folder/'manifest.json')
        assert summary['figure_sha256']==digest(folder/'overview.png')
    assert stage38['plan_sha256']==digest(STAGE38/'plan.json')
    for name,sha in stage38['source_sha256'].items():
        if name.startswith('core/'):
            assert sources[name]==sha
    for scene in SCENES:
        folder=REFERENCES[scene]
        old=stage35['cases']['weak'] if scene=='weak' else stage38
        for name,sha in old['input_sha256'].items():
            assert digest(folder/'input'/name)==sha
        receivers=old['output'] if scene=='weak' else {MODE:old['receiver']}
        for mode,receiver in receivers.items():
            for name,sha in receiver['output_sha256'].items():
                assert digest(folder/mode/name)==sha
        assert old['fit_sha256']==digest(folder/'fit.npz')
    assert stage38['navigation']['result_sha256']==digest(
        STAGE38/'navigation_replay.npz')
    for name,entry in stage37['cases'].items():
        assert entry['result_sha256']==digest(STAGE37/f'{name}.npz')
    control=stage35['cases']['no_attack_control']
    for name,sha in control['input_sha256'].items():
        assert digest(STAGE35/'no_attack_control'/'input'/name)==sha
    for mode,receiver in control['output'].items():
        for name,sha in receiver['output_sha256'].items():
            assert digest(STAGE35/'no_attack_control'/mode/name)==sha
    return stage35,stage37,stage38,sources


def cross_power(tracking,case,navigation,reference,scene):
    """Compare the same navigation mode across power on identical epochs."""
    np.testing.assert_array_equal(tracking['time'],reference['observation_time'])
    np.testing.assert_array_equal(tracking['available_time'],reference['available_time'])
    time=tracking['time']
    fake_offset=fake_east_at_observation(case,time)
    masks=dict(available_post=tracking['available_time']>=case['attack_start'],
               observed_attack=time>=case['attack_start'],
               final_observed_second=time>=time[-1]-1.,
               fake_offset_ge_20=fake_offset>=20.,
               fake_offset_ge_30=fake_offset>=30.,
               fake_offset_ge_50=fake_offset>=50.)
    truth=np.column_stack((case['velocity_east']*time,np.zeros(len(time))))
    result={}
    for mode,nav in navigation.items():
        old_label=OLD_LABEL[scene][mode]
        old=reference[f'{old_label}_position']
        new=nav['position']
        old_valid=np.isfinite(old).all(axis=1)
        new_valid=np.isfinite(new).all(axis=1)
        first_old=int(np.flatnonzero(old_valid)[0])
        first_new=int(np.flatnonzero(new_valid)[0])
        assert first_old==first_new
        np.testing.assert_array_equal(old[first_old],new[first_new])
        np.testing.assert_array_equal(reference[f'{old_label}_initialization_time'],
                                      nav['initialization_time'])
        old_error=np.linalg.norm(old[:,:2]-truth,axis=1)
        new_error=np.linalg.norm(new[:,:2]-truth,axis=1)
        segments={}
        for label,mask in masks.items():
            segments[label]=dict(opportunities=int(mask.sum()),
                old=published_metrics(old_error,mask&old_valid),
                new=published_metrics(new_error,mask&new_valid),
                common_old=published_metrics(old_error,mask&old_valid&new_valid),
                common_new=published_metrics(new_error,mask&old_valid&new_valid),
                lost=published_metrics(old_error,mask&old_valid&~new_valid),
                added=published_metrics(new_error,mask&new_valid&~old_valid))
        available,_=pair_statistics(tracking,case,dict(position=old),nav)
        result[mode]=dict(segments=segments,available_pair=available)
    return result


def projection_audit(record,identity,fit,trace,folder,filename):
    """Post-fit truth scoring of the projector that actually entered RF."""
    np.testing.assert_allclose(trace['trace_projectors'][5:],
        fit['would_projector'][3,5:],rtol=0,atol=1e-9)
    audit=truth_audit(record,identity['case'],identity['seed'],fit,True)
    starts=fit['block_start_s']
    post=starts>=identity['case']['attack_start']
    genuine=audit['would_true_power_retention'][3]
    forged=audit['would_fake_power_retention'][3]
    np.savez_compressed(folder/filename,
        block_start_s=starts,true_power_retention=genuine,
        fake_power_retention=forged)
    loss=1.-genuine[post]
    return dict(result_sha256=digest(folder/filename),
        meaning=('counterfactual true direction' if not identity['case']['genuine']
                 else 'present true direction'),
        post_blocks=int(post.sum()),
        fake_retention_median=float(np.median(forged[post])),
        fake_retention_worst=float(np.max(forged[post])),
        true_direction_loss_median_by_prn=np.median(loss,axis=0).tolist(),
        true_direction_loss_worst_by_prn=np.max(loss,axis=0).tolist())


def plot_results(views):
    figure,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for row,scene in enumerate(SCENES):
        time,counts,errors,reference=views[scene]
        axes[row,0].plot(time,counts,color='black',linewidth=.7)
        axes[row,0].set(title=scene,xlabel='Available time (s)',
                        ylabel='Quality-supported stars')
        for mode,error in errors.items():
            old=reference[f'{OLD_LABEL[scene][mode]}_position']
            truth=np.column_stack((2*reference['observation_time'],
                                   np.zeros(len(time))))
            old_error=np.linalg.norm(old[:,:2]-truth,axis=1)
            axes[row,1].plot(time,old_error,linewidth=.6,linestyle='--',
                             label=f'0 dB {mode}')
            axes[row,1].plot(time,error,linewidth=.7,marker='.',markersize=1,
                             label=f'-5 dB {mode}')
        axes[row,1].set(xlabel='Available time (s)',
                        ylabel='Published horizontal error (m)')
        for axis in axes[row]:axis.grid(alpha=.25)
        axes[row,1].legend(fontsize=6)
    figure.savefig(OUTPUT/'overview.png',dpi=150)
    plt.close(figure)


def main():
    stage35,stage37,stage38,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip',sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(dict(
        **LOW_PLAN,scenes=SCENES,receiver_mode=MODE,navigation_modes=MODES,
        paired_reference_power_db=0.,source_sha256=sources)),
        indent=2,allow_nan=False),encoding='utf-8')
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        stage35_summary_sha256=digest(STAGE35/'summary.json'),
        stage37_summary_sha256=digest(STAGE37/'summary.json'),
        stage38_summary_sha256=digest(STAGE38/'summary.json'))),
        indent=2,allow_nan=False),encoding='utf-8')
    onset=round(LOW_PLAN['attack_start_s']*FS)
    assert (onset-1)/FS<LOW_PLAN['attack_start_s']<=onset/FS
    cases={};views={}
    for scene in SCENES:
        folder=OUTPUT/scene
        folder.mkdir(exist_ok=True)
        record,identity=generate_new_record(LOW_PLAN,scene,
                                             fake_rate=LOW_PLAN['fake_rate_mps'])
        old,old_identity=reference_input(scene)
        verify_power_input(record,identity,old,old_identity,onset)
        input_sha,_,_=save_new_input(folder,record,identity,
            old['samples'][:,:onset],
            {name:value for name,value in old.items() if name!='samples'})
        del record,old
        gc.collect()
        print(scene,'-5 dB IQ saved',flush=True)
        with np.load(folder/'input'/'observations.npz') as saved:
            replay=dict(saved)
        replay['samples']=np.load(folder/'input'/'samples.npy',mmap_mode='r')
        receiver=run_receiver(replay,identity['case'],folder/MODE,MODE)
        fit_sha=audit_fit(replay,folder,False)
        navigation,view=replay_navigation(folder/MODE,identity['case'],folder)
        with np.load(folder/'navigation_replay.npz') as saved:
            new=dict(saved)
        with np.load(OLD_NAV[scene]) as saved:
            reference=dict(saved)
        with np.load(folder/MODE/'tracking.npz') as saved:
            tracking=dict(saved)
        paths={name:dict(position=new[f'{name}_position'],
                         initialization_time=new[f'{name}_initialization_time'])
               for name in MODES}
        paired=cross_power(tracking,identity['case'],paths,reference,scene)
        with np.load(folder/'fit.npz') as saved:
            fit=dict(saved)
        with np.load(folder/MODE/'trace.npz') as saved:
            trace=dict(saved)
        projection=projection_audit(replay,identity,fit,trace,folder,
                                    'projection_truth_minus5.npz')
        old_record,old_identity=reference_input(scene)
        with np.load(REFERENCES[scene]/'fit.npz') as saved:
            old_fit=dict(saved)
        with np.load(REFERENCES[scene]/MODE/'trace.npz') as saved:
            old_trace=dict(saved)
        reference_projection=projection_audit(old_record,old_identity,old_fit,
            old_trace,folder,'projection_truth_zero.npz')
        del old_record,old_fit,old_trace,replay,fit,trace
        gc.collect()
        cases[scene]=dict(input_sha256=input_sha,receiver=receiver,
            fit_sha256=fit_sha,projection=projection,
            reference_projection=reference_projection,navigation=navigation,
            cross_power=paired,reference_input_sha256=(
                stage35['cases']['weak']['input_sha256'] if scene=='weak'
                else stage38['input_sha256']),
            reference_navigation_sha256=digest(OLD_NAV[scene]))
        views[scene]=(*view,reference)
        print(scene,'receiver, navigation, and truth audit saved',flush=True)
    with np.load(OUTPUT/'weak'/'input'/'observations.npz') as saved:
        weak=dict(saved)
    with np.load(OUTPUT/'genuine_absent'/'input'/'observations.npz') as saved:
        absent=dict(saved)
    weak['samples']=np.load(OUTPUT/'weak'/'input'/'samples.npy',mmap_mode='r')
    absent['samples']=np.load(OUTPUT/'genuine_absent'/'input'/'samples.npy',mmap_mode='r')
    compare_power_pair((weak,json.loads((OUTPUT/'weak'/'input'/'metadata.json').read_text(
        encoding='utf-8'))),(absent,json.loads((OUTPUT/'genuine_absent'/'input'/'metadata.json').read_text(
        encoding='utf-8'))),onset)
    with np.load(OUTPUT/'weak'/'navigation_replay.npz') as saved:
        weak_nav=dict(saved)
    with np.load(OUTPUT/'genuine_absent'/'navigation_replay.npz') as saved:
        absent_nav=dict(saved)
    for mode in MODES:
        left=weak_nav[f'{mode}_position']
        right=absent_nav[f'{mode}_position']
        first=int(np.flatnonzero(np.isfinite(left).all(axis=1))[0])
        assert first==int(np.flatnonzero(np.isfinite(right).all(axis=1))[0])
        np.testing.assert_array_equal(left[first],right[first])
        np.testing.assert_array_equal(weak_nav[f'{mode}_initialization_time'],
                                      absent_nav[f'{mode}_initialization_time'])
    old_weak,_=reference_input('weak')
    old_absent,_=reference_input('genuine_absent')
    component_residual=compare_iq_components(weak['samples'],absent['samples'],
        old_weak['samples'],old_absent['samples'],round(.1*FS))
    plot_results(views)
    assert source_hashes()==sources
    summary=dict(scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        manifest_sha256=digest(OUTPUT/'manifest.json'),
        figure_sha256=digest(OUTPUT/'overview.png'),
        reference_summary_sha256=dict(stage35=digest(STAGE35/'summary.json'),
            stage37=digest(STAGE37/'summary.json'),stage38=digest(STAGE38/'summary.json')),
        shared_control_reference=dict(
            input_sha256=stage35['cases']['no_attack_control']['input_sha256'],
            output_sha256=stage35['cases']['no_attack_control']['output'],
            three_navigation_sha256=stage37['cases']['long_control']['result_sha256']),
        paired_genuine_component_maxabs=component_residual,cases=cases)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),
        indent=2,allow_nan=False),encoding='utf-8')
    return summary


if __name__=='__main__':
    main()
