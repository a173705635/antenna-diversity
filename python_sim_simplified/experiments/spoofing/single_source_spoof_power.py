"""Fixed zero-dB-per-PRN spoof-power stress test at the saved five-degree angle.

Only spoof_db changes from the saved +10 dB development scene. Weak and
genuine-absent IQ are newly generated once each; both default and temporal
beam frontends independently process each IQ. Control is the saved same-seed
input and cached off/on receiver outputs, with no new control metadata.
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
from core.spoofing.single_source import frontend, navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_close_angle import OUTPUT as STAGE24, PLAN as OLD_PLAN
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import (
    build_coherent_support, quality_support_counts, score_mode,
)
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_history_study import generate_new_record
from experiments.spoofing.single_source_long_coherent import (
    OUTPUT as STAGE21, save_navigation, source_snapshot,
)
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, RECEIVER_PLL, digest,
)
from experiments.spoofing.single_source_temporal_trigger import (
    OUTPUT as STAGE25, assert_prefix, compare_publications,
    current_projector_diagnostics, navigation_input,
    preflight_inputs as preflight_stage25,
)


OUTPUT=RESULTS / STUDY_OUTPUT / 'spoof_power_zero'
PLAN={**OLD_PLAN,'spoof_db':0.}
SCENES=('weak','genuine_absent')
MODES=('original','short')


def observation_quality(tracking,navigation,attack_start):
    """Count observation, support, and publication separately after onset."""
    finite=np.isfinite(tracking['ranges'])
    quality=(finite&(tracking['lock']>=.5)&(tracking['cn0']>=25.)
             &(tracking['gain']>=.05))
    supported=quality&(tracking['support']>=3.)
    selected=navigation['selected']
    published=np.isfinite(navigation['position']).all(axis=1)
    phase_ok=navigation['diagnostics']['phase_ok']
    available=tracking['available_time']
    result={}
    for label,mask in (('post',available>=attack_start),
                       ('steady',available>=.5)):
        result[f'{label}_opportunities']=int(mask.sum())
        for field,values in (('finite',finite),('quality',quality),
                             ('support',supported),('selected',selected)):
            counts=values.sum(axis=1)
            result[f'{label}_{field}_ge4']=int((mask&(counts>=4)).sum())
            result[f'{label}_{field}_ge6']=int((mask&(counts>=6)).sum())
        result[f'{label}_published']=int((mask&published).sum())
        result[f'{label}_published_phase_ok']=int((mask&published&phase_ok).sum())
    return result


def fake_east_at_observation(case,observation_time):
    """East offset of the modeled fake trajectory at signal observation time."""
    return (case['fake_east']+case['fake_rate']*
            np.maximum(observation_time-case['attack_start'],0.))


def preflight_inputs():
    stage21,stage24=preflight_stage25()
    stage25=json.loads((STAGE25 / 'summary.json').read_text(encoding='utf-8'))
    current_sources=source_hashes()
    assert {name:value for name,value in current_sources.items()
            if name.startswith('core/')}=={
                name:value for name,value in stage25['rf_source_sha256'].items()
                if name.startswith('core/')}
    assert stage25['rf_manifest_sha256']==digest(STAGE25 / 'manifest.json')
    assert stage25['rf_source_snapshot_sha256']==digest(STAGE25 / 'source_snapshot.zip')
    assert stage25['finalizer_source_snapshot_sha256']==digest(
        STAGE25 / 'finalizer_source_snapshot.zip')
    verify_source_snapshot(STAGE25 / 'source_snapshot.zip',
                           stage25['rf_source_sha256'])
    verify_source_snapshot(STAGE25 / 'finalizer_source_snapshot.zip',
                           stage25['finalizer_source_sha256'])
    for scene,case in stage25['cases'].items():
        for name,value in case['output_sha256'].items():
            assert value==digest(STAGE25 / scene / name)
    return stage21,stage24,stage25


def process_frontend(record,folder,temporal):
    observations=frontend(record,ANTENNAS,split_support=True,
                          receiver_pll=RECEIVER_PLL,evidence_trace=True,
                          temporal_beam_trigger_enabled=temporal)
    trace_keys=[key for key in observations if key.startswith('trace_')]
    trace={key:value for key,value in observations.items() if key in trace_keys}
    tracking={key:value for key,value in observations.items() if key not in trace_keys}
    np.savez_compressed(folder / 'tracking.npz',**tracking)
    np.savez_compressed(folder / 'trace.npz',**trace)
    print(folder.name,'frontend saved',flush=True)
    if temporal:
        block=current_projector_diagnostics(tracking,trace)
        np.savez_compressed(folder / 'blocks.npz',**block)
    history=analyze_trace(observations)
    np.savez_compressed(folder / 'history.npz',**history)
    groups=analyze_groups(trace)
    np.savez_compressed(folder / 'groups.npz',**groups)
    coherent=build_coherent_support(tracking,groups)
    np.savez_compressed(folder / 'support.npz',**{
        f'coherent_{key}':value for key,value in coherent.items()})
    del observations,history,groups
    gc.collect()
    return tracking,coherent['support'],block if temporal else None


def run_navigation(tracking,coherent,case,folder):
    supports=dict(original=tracking['support'],short=coherent)
    navigations={}
    scores={}
    arrays=dict(observation_time=tracking['time'],
                available_time=tracking['available_time'])
    for mode,blend,loo in (('original',.1,False),('short',0.,True)):
        inputs=navigation_input(tracking,supports[mode])
        navigation=navigate(inputs,diagnostics=True,code_blend=blend,
                            phase_leave_one_out=loo)
        assert_prefix(inputs,navigation,blend,loo)
        navigations[mode]=navigation
        counts=quality_support_counts(tracking,supports[mode])
        scores[mode]=score_mode(navigation,tracking,case,counts,supports[mode])
        scores[mode]['observation_quality']=observation_quality(
            {**tracking,'support':supports[mode]},navigation,case['attack_start'])
        truth_east=case['velocity_east']*tracking['time']
        arrays[f'{mode}_east_minus_true_m']=navigation['position'][:,0]-truth_east
        arrays[f'{mode}_east_minus_fake_m']=(navigation['position'][:,0]
            -truth_east-fake_east_at_observation(case,tracking['time']))
        arrays[f'{mode}_support_count']=counts
    save_navigation(arrays,navigations)
    np.savez_compressed(folder / 'navigation.npz',**arrays)
    return navigations,scores,arrays


def compare_power(fresh,older,tracking,case):
    comparisons={}
    for mode in MODES:
        details,_,_,_=compare_publications(tracking,
            dict(position=older[mode]),fresh[mode],case)
        comparisons[mode]=details
    return comparisons


def cached_positions(saved,prefix):
    """Read Stage 25 navigation's path-first position field names."""
    return {mode:saved[f'{prefix}_{mode}_position'] for mode in MODES}


def main():
    stage21,stage24,stage25=preflight_inputs()
    sources=source_hashes()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip',sources)
    onset_sample=round(PLAN['attack_start_s']*FS)
    assert (onset_sample-1)/FS<PLAN['attack_start_s']<=onset_sample/FS
    plan=dict(**PLAN,first_attack_sample=onset_sample,
              first_attack_sample_time_s=onset_sample/FS,
              spoof_power_definition='per fake PRN relative to its nominal genuine PRN; amplitude=sqrt(10**(spoof_db/10))',
              control_reference=str(STAGE21 / 'no_attack_control'),
              source_sha256=sources,
              source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'))
    (OUTPUT / 'plan.json').write_text(json.dumps(serializable(plan),
        indent=2,allow_nan=False),encoding='utf-8')
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,plan_sha256=digest(OUTPUT / 'plan.json'),
        source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        stage24_summary_sha256=digest(STAGE24 / 'summary.json'),
        stage25_summary_sha256=digest(STAGE25 / 'summary.json'),
        scenarios=SCENES,control='Stage 21 input, Stage 25 cached off/on')),
        indent=2,allow_nan=False),encoding='utf-8')
    cases={}
    plots={}
    prefix=None
    for scene in SCENES:
        print(scene,'zero-dB IQ generation start',flush=True)
        folder=OUTPUT / scene
        folder.mkdir(exist_ok=True)
        record,identity=generate_new_record(PLAN,scene)
        old=STAGE24 / f'angle5_{scene}' / 'input'
        old_metadata=json.loads((old / 'metadata.json').read_text(encoding='utf-8'))
        assert {key:value for key,value in identity.items()
                if key not in ('case','rf_sha256')}=={
            key:value for key,value in old_metadata.items()
            if key in identity and key not in ('case','rf_sha256')}
        assert {key:value for key,value in identity['case'].items()
                if key!='spoof_db'}=={
            key:value for key,value in old_metadata['case'].items()
            if key!='spoof_db'}
        assert identity['case']['spoof_db']==0.
        assert old_metadata['case']['spoof_db']==10.
        old_samples=np.load(old / 'samples.npy',mmap_mode='r')
        np.testing.assert_array_equal(record['samples'][:,:onset_sample],
                                      old_samples[:,:onset_sample])
        assert np.any(record['samples'][:,onset_sample:]!=
                      old_samples[:,onset_sample:])
        if prefix is None:prefix=record['samples'][:,:onset_sample].copy()
        else:np.testing.assert_array_equal(record['samples'][:,:onset_sample],prefix)
        with np.load(old / 'observations.npz') as saved:
            for key in saved.files:
                np.testing.assert_array_equal(record[key],saved[key],err_msg=key)
        input_folder=folder / 'input'
        input_folder.mkdir(exist_ok=True)
        np.save(input_folder / 'samples.npy',record['samples'])
        np.savez_compressed(input_folder / 'observations.npz',**{
            key:value for key,value in record.items() if key!='samples'})
        inputs={name:digest(input_folder / name)
                for name in ('samples.npy','observations.npz')}
        metadata=dict(**identity,samples_sha256=inputs['samples.npy'],
                      observations_sha256=inputs['observations.npz'],
                      old_ten_db_input_sha256=stage24['cases'][f'angle5_{scene}']['input_sha256'],
                      pre_attack_exact_prefix_samples=onset_sample)
        (input_folder / 'metadata.json').write_text(json.dumps(
            serializable(metadata),indent=2,allow_nan=False),encoding='utf-8')
        inputs['metadata.json']=digest(input_folder / 'metadata.json')
        del record,identity,old_samples
        gc.collect()
        print(scene,'zero-dB IQ saved',flush=True)
        with np.load(input_folder / 'observations.npz') as saved:
            replay=dict(saved)
        replay['samples']=np.load(input_folder / 'samples.npy',mmap_mode='r')
        outputs={};navs={};scores={};plots[scene]={}
        for mode,temporal in (('off',False),('on',True)):
            part=folder / mode
            part.mkdir(exist_ok=True)
            print(scene,mode,'frontend start',flush=True)
            tracking,coherent,block=process_frontend(replay,part,temporal)
            navigation,score,arrays=run_navigation(
                tracking,coherent,metadata['case'],part)
            outputs[mode]={name:digest(part / name) for name in
                ('tracking.npz','trace.npz','history.npz','groups.npz',
                 'support.npz','navigation.npz')+
                (('blocks.npz',) if temporal else ())}
            navs[mode]=navigation
            scores[mode]=score
            plots[scene][mode]=dict(available=tracking['available_time'].copy(),
                observation=tracking['time'].copy(),
                counts={name:arrays[f'{name}_support_count'].copy() for name in MODES},
                positions={name:navigation[name]['position'].copy() for name in MODES})
            if temporal:
                scores[mode]['history_only_blocks']=np.flatnonzero(block['history_only'])
            del tracking,coherent,block,navigation,arrays
            gc.collect()
            print(scene,mode,'analysis done',flush=True)
        np.testing.assert_array_equal(plots[scene]['off']['available'],
                                      plots[scene]['on']['available'])
        with np.load(folder / 'off' / 'tracking.npz') as saved:off_tracking=dict(saved)
        with np.load(folder / 'on' / 'tracking.npz') as saved:on_tracking=dict(saved)
        comparisons={mode:compare_publications(on_tracking,
            navs['off'][mode],navs['on'][mode],metadata['case'])[0]
            for mode in MODES}
        with np.load(STAGE25 / f'angle5_{scene}' / 'navigation.npz') as saved:
            ten_off=cached_positions(saved,'baseline')
            ten_on=cached_positions(saved,'candidate')
        power_comparisons=dict(
            off=compare_power(navs['off'],ten_off,off_tracking,metadata['case']),
            on=compare_power(navs['on'],ten_on,on_tracking,metadata['case']))
        cases[scene]=dict(input_sha256=inputs,output_sha256=outputs,
                          off_on_same_power=comparisons,
                          zero_ten_db_same_mode=power_comparisons,
                          paths=scores)
        del replay,off_tracking,on_tracking,navs,ten_off,ten_on
        gc.collect()

    control=stage25['cases']['shared_control']
    cases['shared_control']=dict(
        source='unchanged Stage 21 no-attack IQ and Stage 25 off/on cache',
        input_sha256=stage21['cases']['no_attack_control']['input_sha256'],
        stage25_output_sha256=control['output_sha256'],
        paths=control['modes'])
    with np.load(STAGE25 / 'shared_control' / 'navigation.npz') as saved:
        control_nav=dict(saved)
    control_available=control_nav['available_time']
    truth=2.*control_nav['observation_time']
    plots['shared_control']={}
    for mode,prefix in (('off','baseline'),('on','candidate')):
        positions=cached_positions(control_nav,prefix)
        plots['shared_control'][mode]=dict(available=control_available,
            observation=control_nav['observation_time'],
            counts={name:control_nav[f'{name}_{prefix}_support_count']
                    for name in MODES},
            positions=positions)

    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for row,scene in enumerate((*SCENES,'shared_control')):
        for mode,style in (('off',':'),('on','-')):
            view=plots[scene][mode]
            time=view['available']
            observation=view['observation']
            truth_east=2.*observation
            for path,color in (('original','#777777'),('short','#cc5735')):
                axes[row,0].plot(time,view['counts'][path],color=color,
                                 linestyle=style,label=f'{path} {mode}')
                position=view['positions'][path]
                error=np.linalg.norm(position[:,:2]-np.column_stack(
                    (truth_east,np.zeros(len(time)))),axis=1)
                axes[row,1].plot(time,error,color=color,linestyle=style,
                                 linewidth=.8,label=f'{path} {mode}')
                finite=np.isfinite(error)
                axes[row,1].scatter(time[finite],error[finite],color=color,s=5)
        axes[row,0].set(title=scene,ylabel='Quality-gated support count')
        axes[row,1].set(ylabel='Published horizontal error (m)')
        for axis in axes[row]:
            axis.set(xlabel='Output available time (s)')
            axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    axes[0,1].legend(fontsize=7)
    fig.savefig(OUTPUT / 'overview.png',dpi=150)
    plt.close(fig)
    assert source_hashes()==sources
    summary=dict(
        scope=__doc__,plan_sha256=digest(OUTPUT / 'plan.json'),
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        figure_sha256=digest(OUTPUT / 'overview.png'),cases=cases)
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(summary),
        indent=2,allow_nan=False),encoding='utf-8')
    return cases


if __name__=='__main__':
    main()
