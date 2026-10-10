"""Fixed 20 s velocity-step takeover on new paired weak/control IQ.

The genuine-absent case is outside this run; no long-duration authentication
claim follows from the weak and no-attack control pair.
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
from core.single_source import frontend
from experiments.paths import RESULTS
from experiments.single_source_coherence_diagnosis import analyze_groups
from experiments.single_source_coherent_navigation import build_coherent_support
from experiments.single_source_evidence_history import analyze_trace
from experiments.single_source_history_study import generate_new_record
from experiments.single_source_long_coherent import source_snapshot
from experiments.single_source_nav_bit_epoch_shift import anchor_transport
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_ramp_takeover import (
    OUTPUT as STAGE33,PLAN as OLD_PLAN,segment_scores,
)
from experiments.single_source_spoof_power import (
    fake_east_at_observation,run_navigation,
)
from experiments.single_source_spoof_power_minus_five import audit_new_fit
from experiments.single_source_stream_storage import OUTPUT as STAGE34
from experiments.single_source_suppression_study import (
    ANTENNAS,OUTPUT as STUDY_OUTPUT,RECEIVER_PLL,digest,
)
from experiments.single_source_template_coherence import fit_saved_iq
from experiments.single_source_temporal_trigger import compare_publications


OUTPUT=RESULTS/STUDY_OUTPUT/'ramp_takeover_long'
PLAN={**OLD_PLAN,'duration_s':20.}
SCENES=('weak','no_attack_control')
MODES=('original','short')
OUTPUT_FILES=('tracking.npz','trace.npz','history.npz','groups.npz',
              'support.npz','navigation.npz')


def long_tail_scores(tracking,navigation,case):
    """Evaluate true-position errors on observed-time masks and broken runs."""
    time=tracking['time']
    fake_offset=fake_east_at_observation(case,time)
    masks=dict(observed_attack=time>=case['attack_start'],
               final_observed_second=time>=time[-1]-1.,
               fake_offset_ge_20=fake_offset>=20.,
               fake_offset_ge_30=fake_offset>=30.,
               fake_offset_ge_50=fake_offset>=50.)
    truth=np.column_stack((case['velocity_east']*time,np.zeros(len(time))))
    position=navigation['position']
    published=np.isfinite(position).all(axis=1)
    error=np.linalg.norm(position[:,:2]-truth,axis=1)
    result={}
    for name,mask in masks.items():
        accepted=mask&published
        values=error[accepted]
        longest={}
        for limit in (20,30,50):
            current=maximum=0
            for bad in accepted&(error>=limit):
                current=current+1 if bad else 0
                maximum=max(maximum,current)
            longest[f'ge{limit}']=maximum
        result[name]=dict(opportunities=int(mask.sum()),
            published=int(accepted.sum()),
            error_ge_20=int((values>=20.).sum()),
            error_ge_30=int((values>=30.).sum()),
            error_ge_50=int((values>=50.).sum()),
            median_true_error_m=None if not len(values) else float(np.median(values)),
            p95_true_error_m=None if not len(values) else float(np.percentile(values,95)),
            max_true_error_m=None if not len(values) else float(values.max()),
            longest_error_run_epochs=longest)
    return result


def preflight():
    summaries={name:json.loads((folder/'summary.json').read_text(
        encoding='utf-8')) for name,folder in (('stage33',STAGE33),
                                                ('stage34',STAGE34))}
    current=source_hashes()
    for name,folder in (('stage33',STAGE33),('stage34',STAGE34)):
        summary=summaries[name]
        assert summary['source_snapshot_sha256']==digest(folder/'source_snapshot.zip')
        verify_source_snapshot(folder/'source_snapshot.zip',summary['source_sha256'])
        for key,filename in (('plan_sha256','plan.json'),
                             ('manifest_sha256','manifest.json')):
            assert summary[key]==digest(folder/filename)
    for name,sha in summaries['stage34']['source_sha256'].items():
        if name.startswith('core/'):
            assert current[name]==sha
    for name,sha in summaries['stage34']['output_sha256'].items():
        assert digest(STAGE34/name)==sha
    assert summaries['stage34']['reference_output_sha256']==summaries[
        'stage33']['cases']['weak']['output_sha256']['aided_five_ms']
    return summaries,current


def save_new_input(folder,record,identity,first_prefix,first_geometry):
    onset=round(PLAN['attack_start_s']*FS)
    assert (onset-1)/FS<PLAN['attack_start_s']<=onset/FS
    if first_prefix is not None:
        np.testing.assert_array_equal(record['samples'][:,:onset],first_prefix)
        for name,value in first_geometry.items():
            np.testing.assert_array_equal(record[name],value,err_msg=name)
    else:
        first_prefix=record['samples'][:,:onset].copy()
        first_geometry={name:value.copy() for name,value in record.items()
                        if name!='samples'}
    input_folder=folder/'input';input_folder.mkdir(exist_ok=True)
    np.save(input_folder/'samples.npy',record['samples'])
    np.savez_compressed(input_folder/'observations.npz',**{
        name:value for name,value in record.items() if name!='samples'})
    hashes={name:digest(input_folder/name) for name in (
        'samples.npy','observations.npz')}
    metadata=dict(**identity,samples_sha256=hashes['samples.npy'],
                  observations_sha256=hashes['observations.npz'],
                  pre_attack_exact_prefix_samples=onset)
    (input_folder/'metadata.json').write_text(json.dumps(
        serializable(metadata),indent=2,allow_nan=False),encoding='utf-8')
    hashes['metadata.json']=digest(input_folder/'metadata.json')
    return hashes,first_prefix,first_geometry


def run_receiver(record,case,folder,mode):
    folder.mkdir(exist_ok=True)
    aided=mode=='aided_five_ms'
    scratch=folder/'beam_stream.dat'
    observations=frontend(record,ANTENNAS,split_support=True,
        receiver_pll=RECEIVER_PLL,evidence_trace=True,
        temporal_beam_trigger_enabled=False,
        beam_template_carrier_aiding=aided,beam_coherent_ms=5 if aided else 1,
        stream_memmap_path=scratch)
    scratch_bytes=scratch.stat().st_size
    assert scratch_bytes==len(record['codes'])*record['samples'].shape[1]*np.dtype(
        np.complex128).itemsize
    trace={name:value for name,value in observations.items()
           if name.startswith('trace_')}
    tracking={name:value for name,value in observations.items()
              if not name.startswith('trace_')}
    np.savez_compressed(folder/'tracking.npz',**tracking)
    np.savez_compressed(folder/'trace.npz',**trace)
    scratch.unlink()
    assert not scratch.exists()
    print(folder.parent.name,mode,'frontend saved; scratch removed',flush=True)
    history=analyze_trace(observations)
    np.savez_compressed(folder/'history.npz',**history)
    groups=analyze_groups(trace)
    np.savez_compressed(folder/'groups.npz',**groups)
    coherent=build_coherent_support(tracking,groups)
    np.savez_compressed(folder/'support.npz',**{
        f'coherent_{name}':value for name,value in coherent.items()})
    navigation,scores,arrays=run_navigation(tracking,coherent['support'],case,folder)
    segments={name:segment_scores(tracking,navigation[name],case,
              arrays[f'{name}_support_count']) for name in MODES}
    long_scores={name:long_tail_scores(tracking,navigation[name],case)
                 for name in MODES}
    anchors={name:anchor_transport(navigation[name],tracking['time'],
             case['velocity_east']) for name in MODES}
    hashes={name:digest(folder/name) for name in OUTPUT_FILES}
    del observations,history,groups,coherent,navigation,scores,arrays
    gc.collect()
    print(folder.parent.name,mode,'navigation saved',flush=True)
    return dict(output_sha256=hashes,scratch_bytes=scratch_bytes,
                segments=segments,long_tail=long_scores,anchor_transport=anchors)


def audit_fit(record,folder,compare_default):
    aided=folder/'aided_five_ms'
    with np.load(aided/'tracking.npz') as saved:
        candidate=dict(saved)
    with np.load(aided/'trace.npz') as saved:
        candidate_trace=dict(saved)
    if compare_default:
        default=folder/'default'
        with np.load(default/'tracking.npz') as saved:
            original=dict(saved)
        with np.load(default/'trace.npz') as saved:
            original_trace=dict(saved)
        fit=fit_saved_iq(record,original)
        audit_new_fit(fit,original,original_trace,candidate,candidate_trace)
    else:
        fit=fit_saved_iq(record,candidate,reference_variant=3)
        np.testing.assert_allclose(candidate_trace['trace_raw_trigger'][5:],
                                   fit['raw'][3,5:],rtol=0,atol=1e-10)
        np.testing.assert_allclose(candidate_trace['trace_projectors'][5:],
                                   fit['would_projector'][3,5:],rtol=0,atol=1e-9)
        np.testing.assert_array_equal(candidate_trace['trace_dimensions'][5:],
                                      np.where(fit['raw'][3,5:]>=3.,2,3))
        indices=(candidate['time']/.02).astype(int)
        np.testing.assert_allclose(candidate['gain'],
                                   fit['would_gain'][3,indices],rtol=0,atol=1e-9)
    np.savez_compressed(folder/'fit.npz',**fit)
    return digest(folder/'fit.npz')


def plot_results(views):
    figure,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for row,scene in enumerate(SCENES):
        for path,view in views[scene].items():
            time=view['available']
            truth=np.column_stack((2*view['observation'],np.zeros(len(time))))
            for mode in MODES:
                position=view['position'][mode]
                error=np.linalg.norm(position[:,:2]-truth,axis=1)
                label=f'{path} {mode}'
                axes[row,0].plot(time,view['support'][mode],linewidth=.7,label=label)
                axes[row,1].plot(time,error,linewidth=.7,label=label)
                good=np.isfinite(error)
                axes[row,1].scatter(time[good],error[good],s=2)
        axes[row,0].set(title=scene,ylabel='Direction-supported stars',
                        xlabel='Available time (s)')
        axes[row,1].set(ylabel='Published horizontal error (m)',
                        xlabel='Available time (s)')
        for axis in axes[row]:axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=6)
    axes[0,1].legend(fontsize=6)
    figure.savefig(OUTPUT/'overview.png',dpi=150)
    plt.close(figure)


def main():
    historic,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip',sources)
    plan=dict(**PLAN,scenes=SCENES,weak_modes=('default','aided_five_ms'),
              control_modes=('aided_five_ms',),
              beam_stream_storage='complex128 C memmap, deleted after frontend',
              source_sha256=sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(plan),indent=2),
                                    encoding='utf-8')
    manifest=dict(source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
                  plan_sha256=digest(OUTPUT/'plan.json'),
                  stage33_summary_sha256=digest(STAGE33/'summary.json'),
                  stage34_summary_sha256=digest(STAGE34/'summary.json'))
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(manifest),indent=2),
                                        encoding='utf-8')
    results={};views={};first_prefix=None;first_geometry=None
    for scene in SCENES:
        folder=OUTPUT/scene;folder.mkdir(exist_ok=True)
        record,identity=generate_new_record(PLAN,scene,fake_rate=PLAN['fake_rate_mps'])
        hashes,first_prefix,first_geometry=save_new_input(
            folder,record,identity,first_prefix,first_geometry)
        del record
        gc.collect()
        print(scene,'20 s IQ saved',flush=True)
        with np.load(folder/'input'/'observations.npz') as saved:
            replay=dict(saved)
        replay['samples']=np.load(folder/'input'/'samples.npy',mmap_mode='r')
        modes=('default','aided_five_ms') if scene=='weak' else ('aided_five_ms',)
        outputs={mode:run_receiver(replay,identity['case'],folder/mode,mode)
                 for mode in modes}
        fit_sha=audit_fit(replay,folder,scene=='weak')
        del replay
        gc.collect()
        with np.load(folder/'aided_five_ms'/'navigation.npz') as saved:
            aided_navigation=dict(saved)
        with np.load(folder/'aided_five_ms'/'tracking.npz') as saved:
            aided_tracking=dict(saved)
        if scene=='weak':
            with np.load(folder/'default'/'navigation.npz') as saved:
                default_navigation=dict(saved)
            paired={mode:compare_publications(aided_tracking,
                dict(position=default_navigation[f'{mode}_position']),
                dict(position=aided_navigation[f'{mode}_position']),
                identity['case'])[0] for mode in MODES}
        else:paired=None
        results[scene]=dict(input_sha256=hashes,output=outputs,
                            fit_sha256=fit_sha,
                            paired_publications_available_post=paired)
        views[scene]={}
        for mode in modes:
            with np.load(folder/mode/'tracking.npz') as saved:
                tracking=dict(saved)
            with np.load(folder/mode/'navigation.npz') as saved:
                navigation=dict(saved)
            views[scene][mode]=dict(available=tracking['available_time'],
                observation=tracking['time'],
                support={name:navigation[f'{name}_support_count'] for name in MODES},
                position={name:navigation[f'{name}_position'] for name in MODES})
        print(scene,'independent fit and navigation complete',flush=True)
    plot_results(views)
    assert source_hashes()==sources
    summary=dict(scope=__doc__,source_sha256=sources,
                 source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
                 plan_sha256=digest(OUTPUT/'plan.json'),
                 manifest_sha256=digest(OUTPUT/'manifest.json'),
                 figure_sha256=digest(OUTPUT/'overview.png'),cases=results)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),indent=2),
                                       encoding='utf-8')
    return summary


if __name__=='__main__':
    main()
