"""Fixed continuous-position, velocity-step fake takeover on paired IQ.

Only weak and genuine-absent IQ are new. Each gets independent default 1 ms
and aided 5 ms receiver paths; the same-seed no-attack control is frozen.
"""
import gc
import json

import numpy as np

from core.provenance import source_hashes
from core.spoofing.recovery_geometry import FS
from core.serialization import serializable
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_aided_coherent_rf import (
    OUTPUT as STAGE28,candidate_frontend,plot_cases,
)
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import build_coherent_support
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_history_study import generate_new_record
from experiments.spoofing.single_source_long_coherent import OUTPUT as STAGE21,source_snapshot
from experiments.spoofing.single_source_nav_bit_epoch_shift import anchor_transport
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_spoof_power import (
    PLAN as OLD_PLAN,fake_east_at_observation,process_frontend,run_navigation,
)
from experiments.spoofing.single_source_spoof_power_minus_five import audit_new_fit
from experiments.spoofing.single_source_suppression_study import OUTPUT as STUDY_OUTPUT,digest
from experiments.spoofing.single_source_template_coherence import fit_saved_iq
from experiments.spoofing.single_source_temporal_trigger import (
    OUTPUT as STAGE25,compare_publications,
)
from experiments.spoofing.single_source_nav_bit_epoch_shift import OUTPUT as STAGE32


OUTPUT=RESULTS / STUDY_OUTPUT / 'ramp_takeover'
PLAN={**OLD_PLAN,'spoof_db':0.,'fake_east_m':0.,'fake_rate_mps':3.}
SCENES=('weak','genuine_absent')
MODES=('original','short')


def preflight():
    """Bind frozen no-attack control and preceding model to saved hashes."""
    current=source_hashes()
    summaries={name:json.loads((folder / 'summary.json').read_text(
        encoding='utf-8')) for name,folder in (
            ('stage21',STAGE21),('stage25',STAGE25),
            ('stage28',STAGE28),('stage32',STAGE32))}
    for name,folder in (('stage21',STAGE21),('stage28',STAGE28),
                        ('stage32',STAGE32)):
        summary=summaries[name]
        assert summary['source_snapshot_sha256']==digest(folder / 'source_snapshot.zip')
        verify_source_snapshot(folder / 'source_snapshot.zip',summary['source_sha256'])
        for key,filename in (('plan_sha256','plan.json'),
                             ('manifest_sha256','manifest.json'),
                             ('figure_sha256','overview.png')):
            assert summary[key]==digest(folder / filename)
    older=summaries['stage25']
    assert older['rf_source_snapshot_sha256']==digest(STAGE25 / 'source_snapshot.zip')
    assert older['finalizer_source_snapshot_sha256']==digest(
        STAGE25 / 'finalizer_source_snapshot.zip')
    verify_source_snapshot(STAGE25 / 'source_snapshot.zip',older['rf_source_sha256'])
    verify_source_snapshot(STAGE25 / 'finalizer_source_snapshot.zip',
                           older['finalizer_source_sha256'])
    assert older['rf_manifest_sha256']==digest(STAGE25 / 'manifest.json')
    assert older['figure_sha256']==digest(STAGE25 / 'overview.png')
    assert {key:sha for key,sha in current.items() if key.startswith('core/')}=={
        key:sha for key,sha in summaries['stage32']['source_sha256'].items()
        if key.startswith('core/')}
    control_input=STAGE21 / 'no_attack_control' / 'input'
    for name,sha in summaries['stage21']['cases'][
            'no_attack_control']['input_sha256'].items():
        assert sha==digest(control_input / name)
    assert summaries['stage21']['cases']['no_attack_control'][
        'output_sha256']['tracking.npz']==digest(
            STAGE21 / 'no_attack_control' / 'tracking.npz')
    for name,sha in older['cases']['shared_control']['output_sha256'].items():
        assert sha==digest(STAGE25 / 'shared_control' / name)
    for name,sha in summaries['stage28']['cases'][
            'shared_control']['output_sha256'].items():
        assert sha==digest(STAGE28 / 'shared_control' / name)
    return summaries,current


def segment_scores(tracking,navigation,case,support_count):
    """Score observed attack, final second, and actual fake offset at >=20 m."""
    observation=tracking['time']
    available=tracking['available_time']
    fake_offset=fake_east_at_observation(case,observation)
    truth=np.column_stack((case['velocity_east']*observation,
                           np.zeros(len(observation))))
    fake=truth.copy();fake[:,0]+=fake_offset
    position=navigation['position'][:,:2]
    true_error=np.linalg.norm(position-truth,axis=1)
    fake_error=np.linalg.norm(position-fake,axis=1)
    published=np.isfinite(navigation['position']).all(axis=1)
    phase_ok=navigation['diagnostics']['phase_ok']
    recovery=navigation['diagnostics']['recovery_confirmed']
    finite=np.isfinite(tracking['ranges'])
    quality=(finite&(tracking['lock']>=.5)&(tracking['cn0']>=25.)
             &(tracking['gain']>=.05))
    finite_count=finite.sum(axis=1)
    quality_count=quality.sum(axis=1)
    selected_count=navigation['selected'].sum(axis=1)
    masks=dict(available_post=available>=case['attack_start'],
               observed_attack=observation>=case['attack_start'],
               final_observed_second=observation>=observation[-1]-1.,
               fake_offset_ge_20=fake_offset>=20.)
    scores={}
    for name,mask in masks.items():
        accepted=mask&published
        errors=true_error[accepted]
        scores[name]=dict(opportunities=int(mask.sum()),
            published=int(accepted.sum()),
            error_ge_20=int((errors>=20.).sum()),
            error_ge_30=int((errors>=30.).sum()),
            error_ge_50=int((errors>=50.).sum()),
            max_true_error_m=None if not len(errors) else float(errors.max()),
            p95_true_error_m=None if not len(errors) else float(
                np.percentile(errors,95)),
            max_fake_error_m=None if not len(errors) else float(
                fake_error[accepted].max()),
            median_fake_error_m=None if not len(errors) else float(
                np.median(fake_error[accepted])),
            last_fake_error_m=None if not len(errors) else float(
                fake_error[np.flatnonzero(accepted)[-1]]),
            published_phase_ok=int((accepted&phase_ok).sum()),
            published_recovery_confirmed=int((accepted&recovery).sum()),
            finite_ge4=int((mask&(finite_count>=4)).sum()),
            quality_ge4=int((mask&(quality_count>=4)).sum()),
            support_ge4=int((mask&(support_count>=4)).sum()),
            selected_ge4=int((mask&(selected_count>=4)).sum()),
            finite_ge6=int((mask&(finite_count>=6)).sum()),
            quality_ge6=int((mask&(quality_count>=6)).sum()),
            support_ge6=int((mask&(support_count>=6)).sum()),
            selected_ge6=int((mask&(selected_count>=6)).sum()),
            last_published_available_s=None if not len(errors) else float(
                available[np.flatnonzero(accepted)[-1]]))
    return scores


def verify_record_pair(record,identity,reference,first_prefix,first_geometry):
    """New weak/absent share all samples before attack and old clean IQ."""
    onset=round(PLAN['attack_start_s']*FS)
    assert (onset-1)/FS<PLAN['attack_start_s']<=onset/FS
    assert identity['case']['fake_east']==0.
    assert identity['case']['fake_rate']==PLAN['fake_rate_mps']
    assert identity['case']['nav_bit_epoch_s']==0.
    np.testing.assert_array_equal(record['samples'][:,:onset],
                                  reference['samples'][:,:onset])
    for name in record:
        if name!='samples':
            np.testing.assert_array_equal(record[name],reference[name],
                                          err_msg=name)
    if first_prefix is not None:
        np.testing.assert_array_equal(record['samples'][:,:onset],first_prefix)
        for name in first_geometry:
            np.testing.assert_array_equal(record[name],first_geometry[name],
                                          err_msg=name)
    return record['samples'][:,:onset].copy(),{
        name:value.copy() for name,value in record.items() if name!='samples'}


def main():
    summaries,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip',sources)
    plan=dict(**PLAN,scenes=SCENES,control='frozen Stage 21/25/28',
              source_sha256=sources,
              source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'))
    (OUTPUT / 'plan.json').write_text(json.dumps(serializable(plan),
        indent=2,allow_nan=False),encoding='utf-8')
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT / 'plan.json'),
        reference_sha256={name:digest(folder / 'summary.json')
            for name,folder in (('stage21',STAGE21),('stage25',STAGE25),
                                ('stage28',STAGE28),('stage32',STAGE32))})),
        indent=2,allow_nan=False),encoding='utf-8')
    control_input=STAGE21 / 'no_attack_control' / 'input'
    with np.load(control_input / 'observations.npz') as saved:
        reference=dict(saved)
    reference['samples']=np.load(control_input / 'samples.npy',mmap_mode='r')
    results={};views={};first_prefix=None;first_geometry=None
    for scene in SCENES:
        folder=OUTPUT / scene;folder.mkdir(exist_ok=True)
        print(scene,'continuous-position IQ generation start',flush=True)
        record,identity=generate_new_record(PLAN,scene,
                                            fake_rate=PLAN['fake_rate_mps'])
        first_prefix,first_geometry=verify_record_pair(
            record,identity,reference,first_prefix,first_geometry)
        input_folder=folder / 'input';input_folder.mkdir(exist_ok=True)
        np.save(input_folder / 'samples.npy',record['samples'])
        np.savez_compressed(input_folder / 'observations.npz',**{
            name:value for name,value in record.items() if name!='samples'})
        inputs={name:digest(input_folder / name) for name in (
            'samples.npy','observations.npz')}
        metadata=dict(**identity,samples_sha256=inputs['samples.npy'],
                      observations_sha256=inputs['observations.npz'],
                      pre_attack_exact_prefix_samples=round(
                          PLAN['attack_start_s']*FS))
        (input_folder / 'metadata.json').write_text(json.dumps(
            serializable(metadata),indent=2,allow_nan=False),encoding='utf-8')
        inputs['metadata.json']=digest(input_folder / 'metadata.json')
        del record
        gc.collect()
        print(scene,'continuous-position IQ saved',flush=True)
        with np.load(input_folder / 'observations.npz') as saved:
            replay=dict(saved)
        replay['samples']=np.load(input_folder / 'samples.npy',mmap_mode='r')
        default_folder=folder / 'default';default_folder.mkdir(exist_ok=True)
        default,default_support,_=process_frontend(replay,default_folder,False)
        nav_default,score_default,arrays_default=run_navigation(
            default,default_support,metadata['case'],default_folder)
        candidate_folder=folder / 'aided_five_ms';candidate_folder.mkdir(exist_ok=True)
        observations,candidate,trace=candidate_frontend(replay,candidate_folder)
        fit=fit_saved_iq(replay,default)
        np.savez_compressed(folder / 'fit.npz',**fit)
        with np.load(default_folder / 'trace.npz') as saved:
            default_trace=dict(saved)
        audit_new_fit(fit,default,default_trace,candidate,trace)
        np.testing.assert_array_equal(default['time'],candidate['time'])
        np.testing.assert_array_equal(default['available_time'],
                                      candidate['available_time'])
        history=analyze_trace(observations)
        np.savez_compressed(candidate_folder / 'history.npz',**history)
        groups=analyze_groups(trace)
        np.savez_compressed(candidate_folder / 'groups.npz',**groups)
        coherent=build_coherent_support(candidate,groups)
        np.savez_compressed(candidate_folder / 'support.npz',**{
            f'coherent_{name}':value for name,value in coherent.items()})
        nav_candidate,score_candidate,arrays_candidate=run_navigation(
            candidate,coherent['support'],metadata['case'],candidate_folder)
        matched={mode:compare_publications(candidate,nav_default[mode],
            nav_candidate[mode],metadata['case'])[0] for mode in MODES}
        outputs={path:{name:digest(folder / path / name) for name in (
            'tracking.npz','trace.npz','history.npz','groups.npz',
            'support.npz','navigation.npz')} for path in ('default','aided_five_ms')}
        results[scene]=dict(input_sha256=inputs,output_sha256=outputs,
            fit_sha256=digest(folder / 'fit.npz'),
            default_scores=score_default,candidate_scores=score_candidate,
            raw_trigger_clean=int((fit['raw'][3,5:15]>=3.).sum()),
            raw_trigger_post=int((fit['raw'][3,15:]>=3.).sum()),
            paired_publications=matched,
            segments={path:{mode:segment_scores(tracking,navigation[mode],
                metadata['case'],arrays[f'{mode}_support_count'])
                for mode in MODES}
                for path,tracking,navigation,arrays in (
                    ('default',default,nav_default,arrays_default),
                    ('aided_five_ms',candidate,nav_candidate,arrays_candidate))},
            anchor_transport={path:{mode:anchor_transport(navigation[mode],
                candidate['time'],metadata['case']['velocity_east'])
                for mode in MODES} for path,navigation in (
                    ('default',nav_default),('aided_five_ms',nav_candidate))})
        views[scene]=dict(available=candidate['available_time'].copy(),
            observation=candidate['time'].copy(),
            velocity_east=metadata['case']['velocity_east'],
            baseline=nav_default,candidate=nav_candidate)
        del replay,default,default_support,nav_default,arrays_default
        del observations,candidate,trace,fit,default_trace,history,groups
        del coherent,nav_candidate,arrays_candidate
        gc.collect()
        print(scene,'paired receiver analysis done',flush=True)
    with np.load(STAGE25 / 'shared_control' / 'navigation.npz') as saved:
        default_control=dict(saved)
    with np.load(STAGE28 / 'shared_control' / 'navigation.npz') as saved:
        aided_control=dict(saved)
    with np.load(STAGE21 / 'no_attack_control' / 'tracking.npz') as saved:
        control_tracking=dict(saved)
    views['shared_control']=dict(
        available=control_tracking['available_time'],
        observation=control_tracking['time'],velocity_east=2.,
        baseline={mode:dict(position=default_control[f'baseline_{mode}_position'],
            selected=default_control[f'baseline_{mode}_selected']) for mode in MODES},
        candidate={mode:dict(position=aided_control[f'{mode}_position'],
            selected=aided_control[f'{mode}_selected']) for mode in MODES})
    results['shared_control']=dict(
        source='unchanged no-attack Stage 21 IQ, Stage 25 default and Stage 28 aided navigation',
        input_sha256=summaries['stage21']['cases'][
            'no_attack_control']['input_sha256'],
        default_navigation_sha256=digest(STAGE25 / 'shared_control' / 'navigation.npz'),
        aided_navigation_sha256=digest(STAGE28 / 'shared_control' / 'navigation.npz'))
    plot_cases(views,output=OUTPUT)
    assert source_hashes()==sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT / 'plan.json'),
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        figure_sha256=digest(OUTPUT / 'overview.png'),cases=results)),
        indent=2,allow_nan=False),encoding='utf-8')
    return results


if __name__=='__main__':
    main()
