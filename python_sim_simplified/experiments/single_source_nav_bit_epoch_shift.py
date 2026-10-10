"""Fixed 5 ms navigation-bit epoch shift on paired five-degree IQ.

Three new scenes share one seed and bit epoch. Each IQ gets independent aided
5 and 10 ms full receiver runs; offline block fits audit both actual beams.
"""
import gc
import json

import numpy as np

from core.provenance import source_hashes
from core.recovery_geometry import FS
from core.serialization import serializable
from experiments.paths import RESULTS
from experiments.single_source_aided_coherent_rf import candidate_frontend,plot_cases
from experiments.single_source_aided_ten_ms_rf import ten_ms_frontend
from experiments.single_source_beam_coherence_lengths import fit_saved_iq_lengths
from experiments.single_source_coherence_diagnosis import analyze_groups
from experiments.single_source_coherent_navigation import build_coherent_support
from experiments.single_source_evidence_history import analyze_trace
from experiments.single_source_history_study import generate_new_record
from experiments.single_source_long_coherent import OUTPUT as STAGE21,source_snapshot
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_spoof_power import run_navigation
from experiments.single_source_spoof_power_minus_five import (
    OUTPUT as STAGE29,PLAN as OLD_PLAN,
)
from experiments.single_source_aided_coherent_rf import OUTPUT as STAGE28
from experiments.single_source_suppression_study import OUTPUT as STUDY_OUTPUT,digest
from experiments.single_source_template_coherence import truth_code_phases
from experiments.single_source_temporal_trigger import compare_publications
from experiments.single_source_aided_ten_ms_rf import OUTPUT as STAGE31


OUTPUT=RESULTS / STUDY_OUTPUT / 'nav_bit_epoch_shift'
PLAN={**OLD_PLAN,'nav_bit_epoch_s':.005}
SCENES=('weak','genuine_absent','shared_control')
MODES=('original','short')


def preflight():
    """Check frozen sources and the original IQ used to define this pair."""
    current=source_hashes()
    historic={name:json.loads((folder / 'summary.json').read_text(
        encoding='utf-8')) for name,folder in (('stage28',STAGE28),
                                                ('stage29',STAGE29),
                                                ('stage31',STAGE31))}
    for name,folder in (('stage28',STAGE28),('stage29',STAGE29),
                        ('stage31',STAGE31)):
        summary=historic[name]
        assert summary['source_snapshot_sha256']==digest(folder / 'source_snapshot.zip')
        verify_source_snapshot(folder / 'source_snapshot.zip',summary['source_sha256'])
        for key,filename in (('plan_sha256','plan.json'),
                             ('manifest_sha256','manifest.json'),
                             ('figure_sha256','overview.png')):
            assert summary[key]==digest(folder / filename)
    assert {name:sha for name,sha in current.items()
            if name.startswith('core/') and name!='core/single_source_scene.py'}=={
                name:sha for name,sha in historic['stage31']['source_sha256'].items()
                if name.startswith('core/') and name!='core/single_source_scene.py'}
    for scene in SCENES:
        prior=historic['stage31']['cases'][scene]
        folder=STAGE31 / scene
        for name,sha in prior['output_sha256'].items():
            assert sha==digest(folder / name)
        for name,sha in prior['input_sha256'].items():
            assert sha==digest(historic_input_folder(scene) / name)
        if scene=='shared_control':
            old_five=historic['stage28']['cases'][scene][
                'output_sha256']['navigation.npz']
        else:
            earlier=historic['stage29']['cases'][scene]
            for name,sha in earlier['input_sha256'].items():
                assert sha==digest(STAGE29 / scene / 'input' / name)
            old_five=earlier['output_sha256']['candidate']['navigation.npz']
        assert old_five==digest(historic_navigation_folder(
            scene,5) / 'navigation.npz')
    return historic,current


def historic_input_folder(scene):
    if scene=='shared_control':
        return STAGE21 / 'no_attack_control' / 'input'
    return STAGE29 / scene / 'input'


def historic_navigation_folder(scene,length):
    if length==10:
        return STAGE31 / scene
    if scene=='shared_control':
        return STAGE28 / scene
    return STAGE29 / scene / 'candidate'


def verify_historic_single_factor(record,identity,metadata,observations):
    """Old/new case differs only in the declared bit epoch, not geometry."""
    assert {key:value for key,value in identity.items() if key not in (
        'rf_sha256','case')}=={key:metadata[key] for key in identity
        if key not in ('rf_sha256','case')}
    assert {key:value for key,value in identity['case'].items()
            if key!='nav_bit_epoch_s'}==metadata['case']
    for key,value in observations.items():
        np.testing.assert_array_equal(record[key],value,err_msg=key)


def verify_epoch_pair(record,identity,first_prefix,first_record):
    """Only the attack state may separate paired scenes before 0.3 s."""
    onset=round(PLAN['attack_start_s']*FS)
    assert (onset-1)/FS<PLAN['attack_start_s']<=onset/FS
    assert identity['case']['nav_bit_epoch_s']==PLAN['nav_bit_epoch_s']
    if first_prefix is not None:
        np.testing.assert_array_equal(record['samples'][:,:onset],first_prefix)
        for key in record:
            if key not in ('samples',):
                np.testing.assert_array_equal(record[key],first_record[key],
                                              err_msg=key)
    return record['samples'][:,:onset].copy(),{
        key:value.copy() for key,value in record.items() if key!='samples'}


def bit_boundaries(record,metadata,fit):
    """Approximate arrival bit-boundary phase from the current-time delay."""
    times=fit['block_start_s']+.01
    true=np.full((len(times),record['nominal'].shape[1]),np.nan)
    fake=np.full_like(true,np.nan)
    epoch=metadata['case']['nav_bit_epoch_s']
    for block in range(5,len(times)):
        true_phase,fake_phase=truth_code_phases(metadata['case'],
                                               metadata['seed'],times[block])
        true[block]=times[block]-true_phase/1.023e6+epoch
        if metadata['attacked'] and times[block]>=metadata['case']['attack_start']:
            fake[block]=times[block]-fake_phase/1.023e6+epoch
    return {f'{source}_arrival_bit_phase_mod_{length}_ms':(
        delay%(length/1000.))*1000. for source,delay in (
            ('true',true),('fake',fake)) for length in (5,10,20)}


def audit_actual_beam(tracking,trace,fit,index):
    """Match every real block's raw score, used P, dimension, and gain."""
    np.testing.assert_array_equal(tracking['startup_prns'],fit['startup_prns'])
    np.testing.assert_allclose(tracking['calibration'],fit['calibration'],
                               rtol=0,atol=1e-12)
    np.testing.assert_allclose(tracking['calibration_rate'],fit['calibration_rate'],
                               rtol=0,atol=1e-12)
    np.testing.assert_allclose(trace['trace_raw_trigger'][5:],fit['raw'][index,5:],
                               rtol=0,atol=1e-10)
    np.testing.assert_allclose(trace['trace_projectors'][5:],
                               fit['would_projector'][index,5:],rtol=0,atol=1e-9)
    np.testing.assert_array_equal(trace['trace_dimensions'][5:],
                                  np.where(fit['raw'][index,5:]>=3.,2,3))
    indices=(tracking['time']/.02).astype(int)
    np.testing.assert_allclose(tracking['gain'],
                               fit['would_gain'][index,indices],rtol=0,atol=1e-9)


def complete_navigation(observations,tracking,trace,case,folder):
    history=analyze_trace(observations)
    np.savez_compressed(folder / 'history.npz',**history)
    groups=analyze_groups(trace)
    np.savez_compressed(folder / 'groups.npz',**groups)
    coherent=build_coherent_support(tracking,groups)
    np.savez_compressed(folder / 'support.npz',**{
        f'coherent_{key}':value for key,value in coherent.items()})
    navigation,scores,arrays=run_navigation(
        tracking,coherent['support'],case,folder)
    del observations,history,groups,coherent
    gc.collect()
    return navigation,scores,arrays


def anchor_transport(navigation,observation_time,velocity_east):
    """Separate initial absolute bias from relative motion after first fix."""
    position=navigation['position']
    published=np.isfinite(position[:,0])
    if not published.any():
        return dict(first_observation_time_s=None,initial_horizontal_error_m=None,
                    max_relative_motion_error_m=None)
    first=np.flatnonzero(published)[0]
    truth=np.column_stack((velocity_east*observation_time,
                           np.zeros(len(observation_time))))
    absolute=np.linalg.norm(position[:,:2]-truth,axis=1)
    moved=position[:,:2]-position[first,:2]
    true_moved=truth-truth[first]
    relative=np.linalg.norm(moved-true_moved,axis=1)
    return dict(first_observation_time_s=float(observation_time[first]),
                initial_horizontal_error_m=float(absolute[first]),
                max_relative_motion_error_m=float(np.max(relative[published])))


def main():
    historic,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip',sources)
    plan=dict(**PLAN,scenes=SCENES,beam_lengths_ms=(5,10),
              old_epoch_s=0.,new_epoch_s=PLAN['nav_bit_epoch_s'],
              source_sha256=sources,
              source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'))
    (OUTPUT / 'plan.json').write_text(json.dumps(serializable(plan),
        indent=2,allow_nan=False),encoding='utf-8')
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT / 'plan.json'),
        stage29_summary_sha256=digest(STAGE29 / 'summary.json'),
        stage31_summary_sha256=digest(STAGE31 / 'summary.json'))),
        indent=2,allow_nan=False),encoding='utf-8')
    results={};views={};first_prefix=None;first_record=None
    for scene in SCENES:
        folder=OUTPUT / scene
        folder.mkdir(exist_ok=True)
        print(scene,'new bit epoch IQ generation start',flush=True)
        old_input=historic_input_folder(scene)
        old_metadata=json.loads((old_input / 'metadata.json').read_text(
            encoding='utf-8'))
        scene_plan=PLAN if scene!='shared_control' else {
            **PLAN,'spoof_db':old_metadata['case']['spoof_db'],
            'angle_deg':old_metadata['case']['angle']}
        record,identity=generate_new_record(
            scene_plan,'no_attack_control' if scene=='shared_control' else scene,
            nav_bit_epoch_s=PLAN['nav_bit_epoch_s'])
        with np.load(old_input / 'observations.npz') as saved:
            old_observations=dict(saved)
        verify_historic_single_factor(record,identity,old_metadata,
                                      old_observations)
        first_prefix,first_record=verify_epoch_pair(
            record,identity,first_prefix,first_record)
        input_folder=folder / 'input'
        input_folder.mkdir(exist_ok=True)
        np.save(input_folder / 'samples.npy',record['samples'])
        np.savez_compressed(input_folder / 'observations.npz',**{
            key:value for key,value in record.items() if key!='samples'})
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
        print(scene,'new bit epoch IQ saved',flush=True)
        with np.load(input_folder / 'observations.npz') as saved:
            replay=dict(saved)
        replay['samples']=np.load(input_folder / 'samples.npy',mmap_mode='r')
        five_folder=folder / 'five_ms';five_folder.mkdir(exist_ok=True)
        observations5,tracking5,trace5=candidate_frontend(replay,five_folder)
        fit=fit_saved_iq_lengths(replay,tracking5)
        fit.update(bit_boundaries(replay,metadata,fit))
        np.savez_compressed(folder / 'fit.npz',**fit)
        audit_actual_beam(tracking5,trace5,fit,1)
        nav5,scores5,arrays5=complete_navigation(
            observations5,tracking5,trace5,metadata['case'],five_folder)
        ten_folder=folder / 'ten_ms';ten_folder.mkdir(exist_ok=True)
        observations10,tracking10,trace10=ten_ms_frontend(replay,ten_folder)
        audit_actual_beam(tracking10,trace10,fit,2)
        np.testing.assert_array_equal(tracking5['time'],tracking10['time'])
        np.testing.assert_array_equal(tracking5['available_time'],
                                      tracking10['available_time'])
        nav10,scores10,arrays10=complete_navigation(
            observations10,tracking10,trace10,metadata['case'],ten_folder)
        comparisons={mode:compare_publications(
            tracking10,nav5[mode],nav10[mode],metadata['case'])[0]
            for mode in MODES}
        transport={length:{mode:anchor_transport(navigation[mode],
            tracking10['time'],metadata['case']['velocity_east'])
            for mode in MODES} for length,navigation in (
                ('five_ms',nav5),('ten_ms',nav10))}
        historic_epoch={}
        for label,length,navigation in (('five_ms',5,nav5),
                                        ('ten_ms',10,nav10)):
            with np.load(historic_navigation_folder(
                    scene,length) / 'navigation.npz') as saved:
                old_navigation=dict(saved)
            np.testing.assert_array_equal(old_navigation['observation_time'],
                                          tracking10['time'])
            np.testing.assert_array_equal(old_navigation['available_time'],
                                          tracking10['available_time'])
            historic_epoch[label]={mode:dict(
                publications=compare_publications(tracking10,
                    dict(position=old_navigation[f'{mode}_position']),
                    navigation[mode],metadata['case'])[0],
                old_anchor_transport=anchor_transport(
                    dict(position=old_navigation[f'{mode}_position']),
                    tracking10['time'],metadata['case']['velocity_east']),
                new_anchor_transport=transport[label][mode])
                for mode in MODES}
        results[scene]=dict(input_sha256=inputs,
            fit_sha256=digest(folder / 'fit.npz'),
            output_sha256={length:{name:digest(folder / length / name)
                for name in ('tracking.npz','trace.npz','history.npz',
                             'groups.npz','support.npz','navigation.npz')}
                for length in ('five_ms','ten_ms')},
            raw_trigger_clean={str(length):int((fit['raw'][index,5:15]>=3.).sum())
                for index,length in ((1,5),(2,10))},
            raw_trigger_post={str(length):int((fit['raw'][index,15:]>=3.).sum())
                for index,length in ((1,5),(2,10))},
            scores=dict(five_ms=scores5,ten_ms=scores10),
            matched_publications=comparisons,anchor_transport=transport,
            old_epoch_new_epoch=historic_epoch)
        views[scene]=dict(available=tracking10['available_time'].copy(),
            observation=tracking10['time'].copy(),
            velocity_east=metadata['case']['velocity_east'],
            baseline=nav5,candidate=nav10)
        del replay,observations5,tracking5,trace5,nav5,arrays5
        del observations10,tracking10,trace10,nav10,arrays10,fit
        gc.collect()
        print(scene,'five/ten-ms analysis done',flush=True)
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
