"""Fixed minus-five-dB five-degree spoof-power pair with independent RF paths.

Only per-PRN spoof power changes from the zero-dB IQ. Each new weak/absent IQ
is processed once by the default beam and once by the startup-aided 5 ms beam.
The same-seed no-attack control is referenced from the saved Stage 28 result.
"""
import gc
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.recovery_geometry import FS
from core.serialization import serializable
from experiments.paths import RESULTS
from experiments.single_source_aided_coherent_rf import (
    OUTPUT as STAGE28,candidate_frontend,sources_and_inputs,
)
from experiments.single_source_coherence_diagnosis import analyze_groups
from experiments.single_source_coherent_navigation import build_coherent_support
from experiments.single_source_evidence_history import analyze_trace
from experiments.single_source_history_study import generate_new_record
from experiments.single_source_long_coherent import OUTPUT as STAGE21,source_snapshot
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_spoof_power import (
    OUTPUT as STAGE26,PLAN as ZERO_PLAN,compare_power,process_frontend,
    run_navigation,
)
from experiments.single_source_suppression_study import OUTPUT as STUDY_OUTPUT,digest
from experiments.single_source_template_coherence import fit_saved_iq
from experiments.single_source_temporal_trigger import (
    OUTPUT as STAGE25,compare_publications,
)


OUTPUT=RESULTS / STUDY_OUTPUT / 'spoof_power_minus_five'
PLAN={**ZERO_PLAN,'spoof_db':-5.}
SCENES=('weak','genuine_absent')
MODES=('original','short')


def verify_power_pair(record,identity,older,older_identity,onset,db,older_db):
    """The paired records differ only in spoof power after the onset sample."""
    assert (identity['case']['spoof_db'],older_identity['case']['spoof_db'])==(
        db,older_db)
    assert {key:value for key,value in identity.items()
            if key not in ('case','rf_sha256')}=={
                key:value for key,value in older_identity.items()
                if key in identity and key not in ('case','rf_sha256')}
    assert {key:value for key,value in identity['case'].items()
            if key!='spoof_db'}=={
                key:value for key,value in older_identity['case'].items()
                if key!='spoof_db'}
    np.testing.assert_array_equal(record['samples'][:,:onset],
                                  older['samples'][:,:onset])
    for key in record:
        if key!='samples':np.testing.assert_array_equal(record[key],older[key],
                                                        err_msg=key)


def preflight():
    historic,current=sources_and_inputs()
    saved=json.loads((STAGE28 / 'summary.json').read_text(encoding='utf-8'))
    assert saved['source_snapshot_sha256']==digest(STAGE28 / 'source_snapshot.zip')
    verify_source_snapshot(STAGE28 / 'source_snapshot.zip',saved['source_sha256'])
    assert saved['plan_sha256']==digest(STAGE28 / 'plan.json')
    assert saved['manifest_sha256']==digest(STAGE28 / 'manifest.json')
    assert {key:value for key,value in current.items() if key.startswith('core/')}=={
        key:value for key,value in saved['source_sha256'].items()
        if key.startswith('core/')}
    for scene,case in saved['cases'].items():
        input_folder=(STAGE26 / scene / 'input' if scene!='shared_control'
                      else STAGE21 / 'no_attack_control' / 'input')
        for name,sha in case['input_sha256'].items():
            assert digest(input_folder / name)==sha
        for name,sha in case['output_sha256'].items():
            assert digest(STAGE28 / scene / name)==sha
    assert saved['figure_sha256']==digest(STAGE28 / 'overview.png')
    return historic,saved,current


def audit_new_fit(fit,default,default_trace,candidate,candidate_trace):
    """Match both saved RF paths to a fresh four-variant fit on this IQ."""
    np.testing.assert_array_equal(candidate['startup_prns'],default['startup_prns'])
    np.testing.assert_array_equal(candidate['calibration'],default['calibration'])
    np.testing.assert_array_equal(candidate['calibration_rate'],
                                  default['calibration_rate'])
    for variant,tracking,trace in ((0,default,default_trace),
                                   (3,candidate,candidate_trace)):
        np.testing.assert_allclose(trace['trace_projectors'][5:],
                                   fit['would_projector'][variant,5:],
                                   rtol=0,atol=1e-9)
        indices=(tracking['time']/.02).astype(int)
        np.testing.assert_allclose(tracking['gain'],
                                   fit['would_gain'][variant,indices],
                                   rtol=0,atol=1e-9)
        np.testing.assert_array_equal(trace['trace_dimensions'][5:],
                                      np.where(fit['raw'][variant,5:]>=3.,2,3))
        np.testing.assert_array_equal(trace['trace_projectors'][:5],
                                      np.broadcast_to(np.eye(3),
                                                      trace['trace_projectors'][:5].shape))
    np.testing.assert_allclose(candidate_trace['trace_raw_trigger'][5:],
                               fit['raw'][3,5:],rtol=0,atol=1e-10)


def plot_results(views):
    figure,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    colors=dict(original='#65717b',short='#cf5738')
    for row,scene in enumerate((*SCENES,'shared_control')):
        for path,view in views[scene].items():
            time=view['available']
            truth=np.column_stack((2.*view['observation'],np.zeros(len(time))))
            for mode in MODES:
                position=view['position'][mode]
                error=np.linalg.norm(position[:,:2]-truth,axis=1)
                label=f'{mode} {path}'
                style=':' if path=='default' else '-'
                axes[row,0].plot(time,view['support'][mode],color=colors[mode],
                                 linestyle=style,linewidth=.8,label=label)
                axes[row,1].plot(time,error,color=colors[mode],
                                 linestyle=style,linewidth=.8,label=label)
                finite=np.isfinite(error)
                axes[row,1].scatter(time[finite],error[finite],
                                    color=colors[mode],s=4)
        axes[row,0].set(title=scene,ylabel='Quality-gated support count')
        axes[row,1].set(ylabel='Published horizontal error (m)')
        for axis in axes[row]:
            axis.set(xlabel='Output available time (s)')
            axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    axes[0,1].legend(fontsize=7)
    figure.savefig(OUTPUT / 'overview.png',dpi=150)
    plt.close(figure)


def main():
    historic,stage28,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip',sources)
    onset=round(PLAN['attack_start_s']*FS)
    assert (onset-1)/FS<PLAN['attack_start_s']<=onset/FS
    plan=dict(**PLAN,first_attack_sample=onset,
              first_attack_sample_time_s=onset/FS,
              spoof_power_definition='per fake PRN relative to nominal genuine PRN',
              control_reference=str(STAGE28 / 'shared_control'),
              source_sha256=sources,
              source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'))
    (OUTPUT / 'plan.json').write_text(json.dumps(serializable(plan),
        indent=2,allow_nan=False),encoding='utf-8')
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT / 'plan.json'),
        stage26_summary_sha256=digest(STAGE26 / 'summary.json'),
        stage28_summary_sha256=digest(STAGE28 / 'summary.json'),
        scenes=SCENES,control='unchanged Stage 28 saved IQ and outputs')),
        indent=2,allow_nan=False),encoding='utf-8')
    cases={};views={};shared_prefix=None
    for scene in SCENES:
        folder=OUTPUT / scene
        folder.mkdir(exist_ok=True)
        print(scene,'minus-five-dB IQ generation start',flush=True)
        record,identity=generate_new_record(PLAN,scene)
        zero_folder=STAGE26 / scene / 'input'
        zero_identity=json.loads((zero_folder / 'metadata.json').read_text(
            encoding='utf-8'))
        with np.load(zero_folder / 'observations.npz') as saved:
            zero={key:saved[key] for key in saved.files}
        zero['samples']=np.load(zero_folder / 'samples.npy',mmap_mode='r')
        verify_power_pair(record,identity,zero,zero_identity,onset,-5.,0.)
        assert np.any(record['samples'][:,onset:]!=zero['samples'][:,onset:])
        if shared_prefix is None:shared_prefix=record['samples'][:,:onset].copy()
        else:np.testing.assert_array_equal(record['samples'][:,:onset],shared_prefix)
        input_folder=folder / 'input'
        input_folder.mkdir(exist_ok=True)
        np.save(input_folder / 'samples.npy',record['samples'])
        np.savez_compressed(input_folder / 'observations.npz',**{
            key:value for key,value in record.items() if key!='samples'})
        inputs={name:digest(input_folder / name)
                for name in ('samples.npy','observations.npz')}
        metadata=dict(**identity,samples_sha256=inputs['samples.npy'],
                      observations_sha256=inputs['observations.npz'],
                      zero_db_input_sha256=stage28['cases'][scene]['input_sha256'],
                      pre_attack_exact_prefix_samples=onset)
        (input_folder / 'metadata.json').write_text(json.dumps(
            serializable(metadata),indent=2,allow_nan=False),encoding='utf-8')
        inputs['metadata.json']=digest(input_folder / 'metadata.json')
        del record,zero,zero_identity
        gc.collect()
        print(scene,'minus-five-dB IQ saved',flush=True)
        with np.load(input_folder / 'observations.npz') as saved:
            replay=dict(saved)
        replay['samples']=np.load(input_folder / 'samples.npy',mmap_mode='r')
        scores={};navs={};views[scene]={};outputs={}
        default_folder=folder / 'default';default_folder.mkdir(exist_ok=True)
        print(scene,'default frontend start',flush=True)
        default,default_support,_=process_frontend(replay,default_folder,False)
        navs['default'],scores['default'],arrays=run_navigation(
            default,default_support,metadata['case'],default_folder)
        views[scene]['default']=dict(available=default['available_time'].copy(),
            observation=default['time'].copy(),
            support={mode:arrays[f'{mode}_support_count'].copy() for mode in MODES},
            position={mode:navs['default'][mode]['position'].copy() for mode in MODES})
        outputs['default']={name:digest(default_folder / name) for name in (
            'tracking.npz','trace.npz','history.npz','groups.npz',
            'support.npz','navigation.npz')}
        del default_support,arrays
        gc.collect()
        candidate_folder=folder / 'candidate';candidate_folder.mkdir(exist_ok=True)
        print(scene,'aided-five-ms frontend start',flush=True)
        observations,candidate,trace=candidate_frontend(replay,candidate_folder)
        fit=fit_saved_iq(replay,default)
        np.savez_compressed(folder / 'fit.npz',**fit)
        with np.load(default_folder / 'trace.npz') as saved:
            default_trace=dict(saved)
        audit_new_fit(fit,default,default_trace,candidate,trace)
        with np.load(STAGE26 / scene / 'off' / 'tracking.npz') as saved:
            zero_default_time=saved['time'].copy()
            zero_default_available=saved['available_time'].copy()
        np.testing.assert_array_equal(default['time'],zero_default_time)
        np.testing.assert_array_equal(default['available_time'],
                                      zero_default_available)
        with np.load(STAGE28 / scene / 'tracking.npz') as saved:
            zero_candidate_time=saved['time'].copy()
            zero_candidate_available=saved['available_time'].copy()
        np.testing.assert_array_equal(candidate['time'],zero_candidate_time)
        np.testing.assert_array_equal(candidate['available_time'],
                                      zero_candidate_available)
        history=analyze_trace(observations)
        np.savez_compressed(candidate_folder / 'history.npz',**history)
        groups=analyze_groups(trace)
        np.savez_compressed(candidate_folder / 'groups.npz',**groups)
        coherent=build_coherent_support(candidate,groups)
        np.savez_compressed(candidate_folder / 'support.npz',**{
            f'coherent_{key}':value for key,value in coherent.items()})
        navs['candidate'],scores['candidate'],arrays=run_navigation(
            candidate,coherent['support'],metadata['case'],candidate_folder)
        views[scene]['candidate']=dict(available=candidate['available_time'].copy(),
            observation=candidate['time'].copy(),
            support={mode:arrays[f'{mode}_support_count'].copy() for mode in MODES},
            position={mode:navs['candidate'][mode]['position'].copy()
                      for mode in MODES})
        outputs['candidate']={name:digest(candidate_folder / name) for name in (
            'tracking.npz','trace.npz','history.npz','groups.npz',
            'support.npz','navigation.npz')}
        with np.load(STAGE26 / scene / 'off' / 'navigation.npz') as saved:
            zero_default={mode:saved[f'{mode}_position'] for mode in MODES}
        with np.load(STAGE28 / scene / 'navigation.npz') as saved:
            zero_candidate={mode:saved[f'{mode}_position'] for mode in MODES}
        paired={mode:compare_publications(candidate,
            navs['default'][mode],navs['candidate'][mode],metadata['case'])[0]
            for mode in MODES}
        power=dict(default=compare_power(navs['default'],zero_default,
                                         default,metadata['case']),
                   candidate=compare_power(navs['candidate'],zero_candidate,
                                           candidate,metadata['case']))
        cases[scene]=dict(input_sha256=inputs,output_sha256=outputs,
                          fit_sha256=digest(folder / 'fit.npz'),
                          trigger_clean=int((trace['trace_raw_trigger'][5:15]>=3.).sum()),
                          trigger_post=int((trace['trace_raw_trigger'][15:]>=3.).sum()),
                          scores=scores,default_candidate=paired,
                          minus_five_zero_db_same_path=power)
        del replay,default,candidate,trace,observations,fit,history,groups
        del coherent,navs,arrays,zero_default,zero_candidate
        gc.collect()
        print(scene,'analysis done',flush=True)
    with np.load(STAGE28 / 'shared_control' / 'navigation.npz') as saved:
        control=dict(saved)
    with np.load(STAGE25 / 'shared_control' / 'navigation.npz') as saved:
        old_control=dict(saved)
    views['shared_control']={
        'default':dict(available=old_control['available_time'],
            observation=old_control['observation_time'],
            support={mode:old_control[f'{mode}_baseline_support_count']
                     for mode in MODES},
            position={mode:old_control[f'baseline_{mode}_position']
                      for mode in MODES}),
        'candidate':dict(available=control['available_time'],
            observation=control['observation_time'],
            support={mode:control[f'{mode}_support_count'] for mode in MODES},
            position={mode:control[f'{mode}_position'] for mode in MODES})}
    cases['shared_control']=dict(
        source='Stage 21 IQ, Stage 25 default, Stage 28 candidate; no new IQ or RF',
        input_sha256=stage28['cases']['shared_control']['input_sha256'],
        default_output_sha256=historic['stage25']['cases']['shared_control'][
            'output_sha256'],
        candidate_output_sha256=stage28['cases']['shared_control']['output_sha256'],
        default_scores={mode:historic['stage25']['cases']['shared_control'][
            'modes'][mode]['baseline'] for mode in MODES},
        candidate_scores=stage28['cases']['shared_control']['candidate_scores'])
    plot_results(views)
    assert source_hashes()==sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT / 'plan.json'),
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        figure_sha256=digest(OUTPUT / 'overview.png'),cases=cases)),
        indent=2,allow_nan=False),encoding='utf-8')
    return cases


if __name__=='__main__':
    main()
