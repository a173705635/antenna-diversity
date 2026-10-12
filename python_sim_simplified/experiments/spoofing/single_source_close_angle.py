"""Fixed five-degree spoof-direction RF against the saved sixty-degree scene.

Only the physical spoof steering angle changes. Weak and genuine-absent RF are
new; the no-attack control is the exact saved eight-second input. Each scene
replays original support, the short bridge, and the fixed one-second bridge.
"""
import gc
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.gps_simulation import receiver_reference
from core.presets import mate70_phase_centers
from core.provenance import source_hashes
from core.spoofing.recovery_geometry import FS, sky, steering
from core.serialization import serializable
from core.spoofing.single_source import frontend, navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_code_blend import (
    assert_saved_navigation, published_metrics,
)
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import (
    build_coherent_support, quality_support_counts, score_mode,
)
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_history_study import generate_new_record
from experiments.spoofing.single_source_long_coherent import (
    OUTPUT as STAGE21, PLAN as BASE_PLAN, comparative_metrics,
    navigation_prefix, save_navigation, source_snapshot,
)
from experiments.spoofing.single_source_long_phase_replay import (
    OUTPUT as STAGE23, preflight_stage22,
)
from experiments.spoofing.single_source_phase_exclusion import (
    OUTPUT as STAGE22, verify_source_snapshot,
)
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, RECEIVER_PLL, digest,
)


OUTPUT = RESULTS / STUDY_OUTPUT / 'close_angle'
PLAN = {**BASE_PLAN, 'angle_deg':5.,
        'scenarios':('weak','genuine_absent')}
SCENES = ('weak','genuine_absent','no_attack_control')
MODES = ('original','short','long')


def scene_geometry(angle_deg, seed, antennas):
    """All PRN angular separations and processed-array response overlap."""
    directions=sky(seed,24)
    tangent=np.cross(directions[0],[0,0,1.])
    tangent/=np.linalg.norm(tangent)
    source=directions[0]*np.cos(np.radians(angle_deg)) \
           +tangent*np.sin(np.radians(angle_deg))
    positions=np.array(list(mate70_phase_centers().values()))[:antennas]
    nominal=steering(directions,positions,0.)
    spoof=steering(source[None],positions,0.)[:,0]
    angles=np.degrees(np.arccos(np.clip(directions@source,-1.,1.)))
    response=abs(nominal.conj().T@spoof)/antennas
    # Coordinates are (east,north,up); azimuth is north-clockwise here.
    azimuth=np.degrees(np.arctan2(directions[:,0],directions[:,1]))%360.
    source_azimuth=float(np.degrees(np.arctan2(source[0],source[1]))%360.)
    return dict(directions=directions,source_direction=source,
                angles_deg=angles,response_similarity=response,
                theoretical_ideal_power_retention=1.-response**2,
                prn_azimuth_north_clockwise_deg=azimuth,
                prn_elevation_deg=np.degrees(np.arcsin(directions[:,2])),
                source_azimuth_north_clockwise_deg=source_azimuth,
                source_elevation_deg=float(np.degrees(np.arcsin(source[2]))))


def remaining_geometry(observations, selected):
    """Actual selected geometry, also evaluated after removing PRN 1 for audit."""
    quality=(np.isfinite(observations['ranges'])
             &(observations['lock']>=.5)&(observations['cn0']>=25.)
             &(observations['gain']>=.05))
    remaining=selected.copy()
    remaining[:,0]=False
    reference,_=receiver_reference()
    full_condition=np.full(len(selected),np.nan)
    remaining_condition=np.full(len(selected),np.nan)
    for k in range(len(selected)):
        for mask,condition in ((selected,full_condition),
                               (remaining,remaining_condition)):
            stars=mask[k]
            if stars.sum()<4:continue
            direction=observations['satellites'][k,stars]-reference
            design=np.column_stack((-direction/np.linalg.norm(direction,axis=1)[:,None],
                                    np.ones(stars.sum())))
            condition[k]=np.linalg.cond(design)
    return dict(quality=quality,prn1_quality=quality[:,0],
                remaining_quality_count=quality[:,1:].sum(axis=1),
                remaining_selected_count=remaining.sum(axis=1),
                selected_condition=full_condition,
                remaining_condition=remaining_condition)


def replay_modes(tracking, coherent_support, references=None):
    """Run three navigation states; saved 60-degree modes must match exactly."""
    inputs=dict(original=tracking['support'],
                short=coherent_support,long=coherent_support)
    result={}
    for name in MODES:
        observations={**tracking,'support':inputs[name]}
        kwargs={} if name=='original' else dict(code_blend=0.,
                 phase_leave_one_out=True,phase_long_anchor=name=='long')
        navigation=navigate(observations,diagnostics=True,**kwargs)
        if references is not None:
            saved,prefix=references[name]
            assert_saved_navigation(navigation,saved,prefix=prefix)
        result[name]=navigation
    return result


def preflight_references():
    _,stage21,stage22=preflight_stage22()
    stage23=json.loads((STAGE23 / 'summary.json').read_text(encoding='utf-8'))
    assert stage23['manifest_sha256']==digest(STAGE23 / 'manifest.json')
    assert stage23['source_snapshot_sha256']==digest(STAGE23 / 'source_snapshot.zip')
    verify_source_snapshot(STAGE23 / 'source_snapshot.zip',stage23['source_sha256'])
    for scene in SCENES:
        name=f'long_{scene}'
        assert stage23['cases'][name]['result_sha256']==digest(STAGE23 / f'{name}.npz')
        folder=STAGE21 / scene
        for file,expected in stage21['cases'][scene]['input_sha256'].items():
            assert expected==digest(folder / 'input' / file)
        for file,expected in stage21['cases'][scene]['output_sha256'].items():
            assert expected==digest(folder / file)
    return stage21,stage22,stage23


def old_case(scene):
    folder=STAGE21 / scene
    with np.load(folder / 'tracking.npz') as saved:tracking=dict(saved)
    with np.load(folder / 'support.npz') as saved:support=saved['coherent_support'].copy()
    metadata=json.loads((folder / 'input' / 'metadata.json').read_text(encoding='utf-8'))
    with np.load(folder / 'navigation.npz') as saved:original=dict(saved)
    name=f'long_{scene}'
    with np.load(STAGE22 / f'{name}.npz') as saved:short=dict(saved)
    with np.load(STAGE23 / f'{name}.npz') as saved:long=dict(saved)
    references=dict(original=(original,'original_'),short=(short,'on_'),
                    long=(long,'long_'))
    return tracking,support,metadata,references


def case_scores(tracking,support,navigations,metadata):
    # comparative_metrics expects Stage 21's keys; both aliases use blend zero here.
    mapping=dict(original=navigations['original'],
                 **{'coherent_.1':navigations['short'],
                    'coherent_0':navigations['long']})
    comparisons,relative,availability=comparative_metrics(
        tracking,mapping,metadata['case'])
    scores={}
    arrays={}
    post=tracking['available_time']>=metadata['case']['attack_start']
    steady=tracking['available_time']>=.5
    for name,key in (('original','original'),('short','coherent_.1'),
                     ('long','coherent_0')):
        values=tracking['support'] if name=='original' else support
        counts=quality_support_counts(tracking,values)
        score=score_mode(navigations[name],tracking,metadata['case'],counts,values)
        score.update(comparisons[key])
        geometry=remaining_geometry(tracking,navigations[name]['selected'])
        remaining=geometry['remaining_selected_count']
        condition=geometry['remaining_condition'][steady]
        finite_condition=condition[np.isfinite(condition)]
        score['per_prn']=dict(
            quality_post=(geometry['quality']&post[:,None]).sum(axis=0),
            quality_steady=(geometry['quality']&steady[:,None]).sum(axis=0),
            selected_post=(navigations[name]['selected']&post[:,None]).sum(axis=0),
            selected_steady=(navigations[name]['selected']&steady[:,None]).sum(axis=0))
        score['prn1']=dict(quality_post=int((post&geometry['prn1_quality']).sum()),
                           support_post=int((post&geometry['prn1_quality']
                                             &(values[:,0]>=3.)).sum()),
                           selected_post=int((post&navigations[name]['selected'][:,0]).sum()),
                           selected_steady=int((steady&navigations[name]['selected'][:,0]).sum()))
        score['remaining23']=dict(selected_ge4_steady=int((steady&(remaining>=4)).sum()),
                                  selected_ge6_steady=int((steady&(remaining>=6)).sum()),
                                  condition_finite_steady=int(len(finite_condition)),
                                  condition_median_steady=None if not len(finite_condition)
                                  else float(np.median(finite_condition)),
                                  condition_max_steady=None if not len(finite_condition)
                                  else float(finite_condition.max()))
        scores[name]=score
        arrays[f'{name}_relative_motion_error_m']=relative[key]
        arrays[f'{name}_support_count']=counts
        arrays[f'{name}_prn1_support']=values[:,0]
        arrays[f'{name}_prn1_selected']=navigations[name]['selected'][:,0]
        arrays[f'{name}_remaining_selected_count']=remaining
        arrays[f'{name}_selected_condition']=geometry['selected_condition']
        arrays[f'{name}_remaining_condition']=geometry['remaining_condition']
        truth=np.column_stack((metadata['case']['velocity_east']*tracking['time'],
                               np.zeros(len(tracking['time']))))
        arrays[f'{name}_horizontal_error_m']=np.linalg.norm(
            navigations[name]['position'][:,:2]-truth,axis=1)
    arrays['prn1_range_finite']=np.isfinite(tracking['ranges'][:,0])
    for key in ('lock','cn0','gain'):
        arrays[f'prn1_{key}']=tracking[key][:,0]
    arrays['remaining_quality_count']=geometry['remaining_quality_count']
    return scores,arrays,availability


def save_case(folder,tracking,support,metadata,references,input_hashes):
    navigations=replay_modes(tracking,support,references)
    prefix_count=len(tracking['time'])-20
    for name,navigation in navigations.items():
        values=tracking['support'] if name=='original' else support
        prefix=navigate(navigation_prefix({**tracking,'support':values},prefix_count),
                        diagnostics=True,**({} if name=='original' else
                        dict(code_blend=0.,phase_leave_one_out=True,
                             phase_long_anchor=name=='long')))
        for key,data in prefix.items():
            if key=='diagnostics':
                for diagnostic,value in data.items():
                    np.testing.assert_array_equal(value,
                                                  navigation[key][diagnostic][:prefix_count])
            elif key=='initialization_time':
                np.testing.assert_array_equal(data,navigation[key])
            else:
                np.testing.assert_array_equal(data,navigation[key][:prefix_count])
    scores,arrays,availability=case_scores(tracking,support,navigations,metadata)
    arrays.update(observation_time=tracking['time'],available_time=tracking['available_time'])
    save_navigation(arrays,navigations)
    path=folder / 'navigation.npz'
    np.savez_compressed(path,**arrays)
    return dict(input_sha256=input_hashes,navigation_sha256=digest(path),
                observation_availability=availability,modes=scores),arrays


def angle_pair_metrics(five,sixty,attack_start):
    """Compare each same-mode error on the same published observation epochs."""
    np.testing.assert_array_equal(five['observation_time'],sixty['observation_time'])
    np.testing.assert_array_equal(five['available_time'],sixty['available_time'])
    post=five['available_time']>=attack_start
    pairs={}
    for mode in MODES:
        five_published=np.isfinite(five[f'{mode}_position']).all(axis=1)
        sixty_published=np.isfinite(sixty[f'{mode}_position']).all(axis=1)
        common=post&five_published&sixty_published
        pairs[mode]=dict(common_published=int(common.sum()),
                         five_degree=published_metrics(
                             five[f'{mode}_horizontal_error_m'],common),
                         sixty_degree=published_metrics(
                             sixty[f'{mode}_horizontal_error_m'],common),
                         five_only=int((post&five_published&~sixty_published).sum()),
                         sixty_only=int((post&sixty_published&~five_published).sum()))
    return pairs


def main():
    stage21,stage22,stage23=preflight_references()
    sources=source_hashes()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip',sources)
    first_sample=round(PLAN['attack_start_s']*FS)
    assert (first_sample-1)/FS<PLAN['attack_start_s']<=first_sample/FS
    geometries={str(angle):scene_geometry(angle,PLAN['seed'],len(ANTENNAS))
                for angle in (5.,60.)}
    (OUTPUT / 'geometry.json').write_text(json.dumps(serializable(dict(
        model='nominal static array; ideal 3-antenna single-source power projection',
        prn_numbers=list(range(1,25)),angles=geometries)),indent=2,
        allow_nan=False),encoding='utf-8')
    plan=dict(**PLAN,first_attack_sample=first_sample,
              first_attack_sample_time_s=first_sample/FS,
              stage21_control_reference=str(STAGE21 / 'no_attack_control'),
              geometry_sha256=digest(OUTPUT / 'geometry.json'),source_sha256=sources,
              source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'))
    (OUTPUT / 'plan.json').write_text(json.dumps(serializable(plan),indent=2,
                                           allow_nan=False),encoding='utf-8')
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,plan_sha256=digest(OUTPUT / 'plan.json'),
        geometry_sha256=digest(OUTPUT / 'geometry.json'),
        source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        stage21_summary_sha256=digest(STAGE21 / 'summary.json'),
        stage22_summary_sha256=digest(STAGE22 / 'summary.json'),
        stage23_summary_sha256=digest(STAGE23 / 'summary.json'))),
        indent=2,allow_nan=False),encoding='utf-8')

    reports={}
    plots={}
    first_prefix=None
    for scene in PLAN['scenarios']:
        folder=OUTPUT / f'angle5_{scene}'
        folder.mkdir(exist_ok=True)
        print(scene,'five-degree RF generation start',flush=True)
        record,identity=generate_new_record(PLAN,scene)
        samples=record['samples']
        old_input=STAGE21 / scene / 'input'
        old_samples=np.load(old_input / 'samples.npy',mmap_mode='r')
        np.testing.assert_array_equal(samples[:,:first_sample],
                                      old_samples[:,:first_sample])
        assert np.any(samples[:,first_sample:]!=old_samples[:,first_sample:])
        if first_prefix is None:first_prefix=samples[:,:first_sample].copy()
        else:np.testing.assert_array_equal(samples[:,:first_sample],first_prefix)
        with np.load(old_input / 'observations.npz') as old_observations:
            for key in old_observations.files:
                np.testing.assert_array_equal(record[key],old_observations[key])
        input_folder=folder / 'input'
        input_folder.mkdir(exist_ok=True)
        np.save(input_folder / 'samples.npy',samples)
        np.savez_compressed(input_folder / 'observations.npz',
                            **{key:value for key,value in record.items()
                               if key!='samples'})
        input_hashes={name:digest(input_folder / name)
                      for name in ('samples.npy','observations.npz')}
        metadata=dict(**identity,samples_sha256=input_hashes['samples.npy'],
                      observations_sha256=input_hashes['observations.npz'],
                      pre_attack_exact_prefix_samples=first_sample,
                      old_angle_input_sha256=stage21['cases'][scene]['input_sha256'])
        (input_folder / 'metadata.json').write_text(json.dumps(serializable(metadata),
            indent=2,allow_nan=False),encoding='utf-8')
        input_hashes['metadata.json']=digest(input_folder / 'metadata.json')
        del record,samples,identity,old_samples
        gc.collect()
        print(scene,'five-degree IQ saved; frontend start',flush=True)

        with np.load(input_folder / 'observations.npz') as saved:replay=dict(saved)
        replay['samples']=np.load(input_folder / 'samples.npy',mmap_mode='r')
        observations=frontend(replay,ANTENNAS,split_support=True,
                              receiver_pll=RECEIVER_PLL,evidence_trace=True)
        del replay
        gc.collect()
        trace_keys=[key for key in observations if key.startswith('trace_')]
        trace={key:observations[key] for key in trace_keys}
        tracking={key:value for key,value in observations.items()
                  if key not in trace_keys}
        np.savez_compressed(folder / 'tracking.npz',**tracking)
        np.savez_compressed(folder / 'trace.npz',**trace)
        print(scene,'five-degree frontend saved',flush=True)

        history=analyze_trace(observations)
        np.savez_compressed(folder / 'history.npz',**history)
        groups=analyze_groups(trace)
        np.savez_compressed(folder / 'groups.npz',**groups)
        coherent=build_coherent_support(tracking,groups)
        np.savez_compressed(folder / 'support.npz',**{
            f'coherent_{key}':value for key,value in coherent.items()})
        result,arrays=save_case(folder,tracking,coherent['support'],
                                     metadata,None,input_hashes)
        result['output_sha256']={name:digest(folder / name) for name in
                                 ('tracking.npz','trace.npz','history.npz',
                                  'groups.npz','support.npz','navigation.npz')}
        reports[f'angle5_{scene}']=result
        plots[(scene,5.)]=(tracking['available_time'].copy(),
                           arrays['long_remaining_selected_count'].copy(),
                           arrays['long_horizontal_error_m'].copy())
        del observations,trace,tracking,history,groups,coherent,arrays
        gc.collect()
        print(scene,'five-degree done',flush=True)

    for scene in SCENES:
        folder=OUTPUT / (f'angle60_{scene}' if scene!='no_attack_control' else
                         'shared_control')
        folder.mkdir(exist_ok=True)
        tracking,support,metadata,references=old_case(scene)
        input_hashes=stage21['cases'][scene]['input_sha256']
        result,arrays=save_case(folder,tracking,support,metadata,
                                     references,input_hashes)
        result['source']='Stage 21 saved input/tracking/support; no new RF'
        result['stage21_output_sha256']=stage21['cases'][scene]['output_sha256']
        result['stage22_reference_sha256']=stage22['cases'][f'long_{scene}']['result_sha256']
        result['stage23_reference_sha256']=stage23['cases'][f'long_{scene}']['result_sha256']
        name='shared_control' if scene=='no_attack_control' else f'angle60_{scene}'
        reports[name]=result
        plots[(scene,60.)]=(tracking['available_time'].copy(),
                            arrays['long_remaining_selected_count'].copy(),
                            arrays['long_horizontal_error_m'].copy())
        print(name,'saved-observation replay done',flush=True)

    comparisons={}
    for scene in PLAN['scenarios']:
        with np.load(OUTPUT / f'angle5_{scene}' / 'navigation.npz') as saved:
            five=dict(saved)
        with np.load(OUTPUT / f'angle60_{scene}' / 'navigation.npz') as saved:
            sixty=dict(saved)
        comparisons[scene]=angle_pair_metrics(five,sixty,PLAN['attack_start_s'])

    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for row,scene in enumerate(SCENES):
        for angle in ((5.,60.) if scene!='no_attack_control' else (60.,)):
            available,count,error=plots[(scene,angle)]
            label='shared control' if scene=='no_attack_control' else f'{angle:g} deg'
            axes[row,0].plot(available,count,label=label)
            axes[row,1].plot(available,error,label=label)
        axes[row,0].set(title=scene,ylabel='Selected among PRNs 2–24')
        axes[row,1].set(ylabel='Long-bridge published error (m)')
        for axis in axes[row]:
            axis.set(xlabel='Output available time (s)')
            axis.grid(alpha=.3)
            axis.legend()
    fig.savefig(OUTPUT / 'overview.png',dpi=150)
    plt.close(fig)
    assert source_hashes()==sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,plan_sha256=digest(OUTPUT / 'plan.json'),
        geometry_sha256=digest(OUTPUT / 'geometry.json'),
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        source_sha256=sources,figure_sha256=digest(OUTPUT / 'overview.png'),
        cases=reports,angle_pair_common_published=comparisons)),
        indent=2,allow_nan=False),encoding='utf-8')
    return reports


if __name__=='__main__':
    main()
