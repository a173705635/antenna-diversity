"""Fixed -4 dB transition on the saved 20 s velocity-step development geometry.

Two paired IQs each receive amplitude-steering off/on aided-5 ms frontends.
The receiver never reads source truth or the saved power-interpolation oracle.
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
from core.spoofing.single_source import LOOP_HISTORY_FIELDS,frontend
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import (
    build_coherent_support,quality_support_counts,
)
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_history_study import generate_new_record
from experiments.spoofing.single_source_long_coherent import source_snapshot
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_ramp_low_power import (
    OUTPUT as STAGE39,compare_iq_components,compare_power_pair,
)
from experiments.spoofing.single_source_ramp_takeover_long import (
    OUTPUT as STAGE35,PLAN,save_new_input,long_tail_scores,
)
from experiments.spoofing.single_source_long_absent import OUTPUT as STAGE38
from experiments.spoofing.single_source_spoof_power import observation_quality
from experiments.spoofing.single_source_startup_amplitude import (
    OUTPUT as STAGE40,audit_saved_raw,estimate_amplitude,fit_blocks,
    startup_prompts,
)
from experiments.spoofing.single_source_startup_amplitude_rf import (
    OUTPUT as STAGE41,audit_alignment,compare_paths,replay_navigation,
)
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS,OUTPUT as STUDY_OUTPUT,RECEIVER_PLL,digest,
)


OUTPUT=RESULTS/STUDY_OUTPUT/'ramp_transition'
SCENES=('weak','genuine_absent')
PATHS=('off','on')
MODES=('original','short')
TRANSITION_PLAN={**PLAN,'spoof_db':-4.}
OUTPUT_FILES=('tracking.npz','trace.npz','loops.npz','history.npz',
              'groups.npz','support.npz','navigation.npz')
REFERENCES={
    'weak':(STAGE35/'weak',STAGE39/'weak'),
    'genuine_absent':(STAGE38,STAGE39/'genuine_absent'),
}


def interpolation_alpha():
    """Amplitude interpolation from -5 and 0 dB to fixed -4 dB."""
    low=10**(-5/20)
    return (10**(-4/20)-low)/(1-low)


def trigger_counts(raw,attack_start_s):
    """Separate fitted blocks after startup from blocks starting after attack."""
    post_attack=(np.arange(len(raw))*round(.02*FS)>=round(attack_start_s*FS))
    return dict(post_startup=dict(triggered=int((raw[5:]>=3.).sum()),
                                   opportunities=len(raw)-5),
                post_attack=dict(triggered=int((post_attack&(raw>=3.)).sum()),
                                 opportunities=int(post_attack.sum())))


def verify_interpolated_samples(middle,low,high,block_samples):
    """Independent input-only oracle over every channel and sample."""
    alpha=interpolation_alpha()
    maximum=0.
    for start in range(0,middle.shape[1],block_samples):
        stop=min(start+block_samples,middle.shape[1])
        expected=low[:,start:stop]+alpha*(high[:,start:stop]-low[:,start:stop])
        difference=np.abs(middle[:,start:stop]-expected)
        maximum=max(maximum,float(difference.max()))
        np.testing.assert_allclose(middle[:,start:stop],expected,rtol=0,atol=1e-11)
    return maximum


def saved_record(folder):
    with np.load(folder/'input'/'observations.npz') as saved:
        record=dict(saved)
    record['samples']=np.load(folder/'input'/'samples.npy',mmap_mode='r')
    identity=json.loads((folder/'input'/'metadata.json').read_text(encoding='utf-8'))
    return record,identity


def verify_power_metadata(record,identity,reference,reference_identity,onset):
    """New factor is only spoof power; startup IQ and geometry stay paired."""
    assert identity['scenario']==reference_identity['scenario']
    assert identity['attacked']==reference_identity['attacked']
    for key,value in identity.items():
        if key not in ('case','rf_sha256'):
            assert reference_identity[key]==value
    assert identity['case']['spoof_db']==-4.
    assert reference_identity['case']['spoof_db'] in (0.,-5.)
    assert {key:value for key,value in identity['case'].items()
            if key!='spoof_db'}=={
            key:value for key,value in reference_identity['case'].items()
            if key!='spoof_db'}
    np.testing.assert_array_equal(record['samples'][:,:onset],
                                  reference['samples'][:,:onset])
    for key,value in record.items():
        if key!='samples':
            np.testing.assert_array_equal(value,reference[key],err_msg=key)


def preflight():
    """Bind old saved inputs and source snapshots without treating old core as current."""
    stages={name:json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            for name,folder in (('stage35',STAGE35),('stage38',STAGE38),
                                ('stage39',STAGE39),('stage40',STAGE40),
                                ('stage41',STAGE41))}
    folders=dict(stage35=STAGE35,stage38=STAGE38,stage39=STAGE39,
                 stage40=STAGE40,stage41=STAGE41)
    for name,summary in stages.items():
        folder=folders[name]
        assert summary['source_snapshot_sha256']==digest(folder/'source_snapshot.zip')
        verify_source_snapshot(folder/'source_snapshot.zip',summary['source_sha256'])
        for field,file in (('plan_sha256','plan.json'),
                           ('manifest_sha256','manifest.json'),
                           ('figure_sha256','overview.png')):
            assert summary[field]==digest(folder/file)
    current=source_hashes()
    for name,sha in stages['stage41']['source_sha256'].items():
        if name.startswith('core/') and name!='core/single_source.py':
            assert current[current_path_for_historical_source(ROOT, name).relative_to(ROOT).as_posix()]==sha
    for scene in SCENES:
        high,low=REFERENCES[scene]
        high_hashes=(stages['stage35']['cases']['weak']['input_sha256']
                     if scene=='weak' else stages['stage38']['input_sha256'])
        low_hashes=stages['stage39']['cases'][scene]['input_sha256']
        for folder,hashes in ((high,high_hashes),(low,low_hashes)):
            for file,sha in hashes.items():
                assert digest(folder/'input'/file)==sha
        fit=stages['stage40']['cases'][
            'weak_0' if scene=='weak' else 'absent_0']
        for key,file in (('startup_sha256','startup.npz'),('fit_sha256','fit.npz')):
            assert fit[key]==digest(STAGE40/('weak_0' if scene=='weak'
                                            else 'absent_0')/file)
    return stages,current


def startup_reference(scene):
    name='weak_0' if scene=='weak' else 'absent_0'
    with np.load(STAGE40/name/'startup.npz') as saved:
        return dict(saved)


def fit_input(record,startup,folder):
    prns=startup['prns']
    prompts=startup_prompts(record,prns,startup['templates'],
        startup['calibration'],startup['calibration_rate'])
    amplitude,_=estimate_amplitude(prompts,record['nominal'][ANTENNAS],prns)
    np.testing.assert_allclose(amplitude,startup['relative_amplitude'],rtol=0,
                               atol=1e-12)
    fit=fit_blocks(record,prns,startup['templates'],startup['calibration'],
                   startup['calibration_rate'],amplitude)
    audit_saved_raw(fit,record['nominal'][ANTENNAS],prns)
    np.savez_compressed(folder/'fit.npz',**fit)
    return fit,digest(folder/'fit.npz')


def save_loops(folder,segments):
    """Keep the actual accepted track histories, indexed by 1-based PRN and arc."""
    arrays={f'prn{segment["prn"]}_arc{segment["arc"]}_{field}':segment[field]
            for segment in segments for field in LOOP_HISTORY_FIELDS}
    np.savez_compressed(folder/'loops.npz',**arrays)
    return dict(accepted_segments=len(segments),
                accepted_epochs=sum(len(segment['epoch_start_samples'])
                                    for segment in segments))


def path_segments(tracking,folder,case):
    with np.load(folder/'navigation.npz') as saved:
        navigation=dict(saved)
    with np.load(folder/'support.npz') as saved:
        support=saved['coherent_support'].copy()
    result={}
    for mode in MODES:
        nav=dict(position=navigation[f'{mode}_position'],
                 selected=navigation[f'{mode}_selected'],
                 diagnostics={key[len(f'{mode}_diagnostic_'):]:value
                     for key,value in navigation.items()
                     if key.startswith(f'{mode}_diagnostic_')})
        actual_support=tracking['support'] if mode=='original' else support
        observation=dict(tracking,support=actual_support)
        published=np.isfinite(nav['position']).all(axis=1)
        post=tracking['available_time']>=case['attack_start']
        result[mode]=dict(
            observed_time=long_tail_scores(tracking,nav,case),
            observation_quality=observation_quality(observation,nav,
                                                    case['attack_start']),
            published_phase_ok=int((post&published&nav['diagnostics']['phase_ok']).sum()),
            published_recovery_confirmed=int((post&published&
                nav['diagnostics']['recovery_confirmed']).sum()))
    return result


def run_path(record,case,scene,path,fit,startup,folder):
    folder.mkdir(exist_ok=True)
    scratch=folder/'beam_stream.dat'
    loops=[]
    observations=frontend(record,ANTENNAS,split_support=True,
        receiver_pll=RECEIVER_PLL,evidence_trace=True,
        temporal_beam_trigger_enabled=False,beam_template_carrier_aiding=True,
        beam_coherent_ms=5,beam_startup_amplitude=path=='on',
        stream_memmap_path=scratch,loop_history=loops)
    scratch_bytes=scratch.stat().st_size
    assert scratch_bytes==len(record['codes'])*record['samples'].shape[1]*16
    trace={key:value for key,value in observations.items() if key.startswith('trace_')}
    tracking={key:value for key,value in observations.items()
              if not key.startswith('trace_')}
    np.savez_compressed(folder/'tracking.npz',**tracking)
    np.savez_compressed(folder/'trace.npz',**trace)
    loop_counts=save_loops(folder,loops)
    del loops
    scratch.unlink()
    assert not scratch.exists()
    print(scene,path,'frontend and loops saved; scratch removed',flush=True)
    audit_alignment(tracking,trace,fit,startup,record['nominal'][ANTENNAS],
                    mode=int(path=='on'))
    history=analyze_trace(observations)
    np.savez_compressed(folder/'history.npz',**history)
    groups=analyze_groups(trace)
    np.savez_compressed(folder/'groups.npz',**groups)
    coherent=build_coherent_support(tracking,groups)
    np.savez_compressed(folder/'support.npz',**{
        f'coherent_{key}':value for key,value in coherent.items()})
    scores,positions=replay_navigation(tracking,trace,coherent,case,folder)
    segmented=path_segments(tracking,folder,case)
    view=dict(available=tracking['available_time'].copy(),
              observation=tracking['time'].copy(),
              support_count={mode:quality_support_counts(tracking,
                  tracking['support'] if mode=='original' else coherent['support'])
                  for mode in MODES},
              position=positions)
    result=dict(output_sha256={file:digest(folder/file) for file in OUTPUT_FILES},
                scratch_bytes=scratch_bytes,loop_history=loop_counts,
                trigger=trigger_counts(trace['trace_raw_trigger'],
                                       case['attack_start']),
                scores=scores,segments=segmented)
    del observations,tracking,trace,history,groups,coherent
    gc.collect()
    return result,view


def plot_results(views):
    figure,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    colors=dict(original='#536b78',short='#d35d3b')
    for row,scene in enumerate(SCENES):
        for path,style in (('off',':'),('on','-')):
            view=views[scene][path]
            available=view['available'];time=view['observation']
            truth=np.column_stack((2.*time,np.zeros(len(time))))
            for mode in MODES:
                position=view['position'][mode]
                error=np.linalg.norm(position[:,:2]-truth,axis=1)
                label=f'{path} {mode}'
                axes[row,0].plot(available,view['support_count'][mode],style,
                                 color=colors[mode],
                                 linewidth=.7,label=label)
                axes[row,1].plot(available,error,style,color=colors[mode],
                                 linewidth=.7,marker='.',markersize=.7,label=label)
        axes[row,0].set(title=scene,xlabel='Available time (s)',
                        ylabel='Quality-gated direction-supported stars',xlim=(0,20.05))
        axes[row,1].set(title=scene,xlabel='Available time (s)',
                        ylabel='Published horizontal error (m)',xlim=(0,20.05))
        for axis in axes[row]:axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=6);axes[0,1].legend(fontsize=6)
    figure.savefig(OUTPUT/'overview.png',dpi=150)
    plt.close(figure)


def main():
    stages,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip',sources)
    plan=dict(**TRANSITION_PLAN,scenes=SCENES,paths=PATHS,
              beam_template_carrier_aiding=True,beam_coherent_ms=5,
              beam_startup_amplitude_paths=dict(off=False,on=True),
              temporal_beam_trigger_enabled=False,
              loop_history=dict(scope='accepted receive_measured segments, including settle and truncated bad tail; failed acquisition attempts excluded',
                  naming='prn is 1-based; arc is segment identity; epoch_start_samples are FS ticks',
                  nco='carrier phase rad, carrier frequency Hz, code phase chips, code rate Hz at epoch block start; carrier_frequency_hz is updated after block',
                  prompt='complex 1 ms prompt; PLL rad, FLL Hz, DLL discriminator, C/N0 dB-Hz, phase-lock unitless',
                  fields=LOOP_HISTORY_FIELDS),
              source_sha256=sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(plan),indent=2,
        allow_nan=False),encoding='utf-8')
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(dict(
        source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        reference_summary_sha256={name:digest(folder/'summary.json')
            for name,folder in (('stage35',STAGE35),('stage38',STAGE38),
                                ('stage39',STAGE39),('stage40',STAGE40),
                                ('stage41',STAGE41))})),indent=2,
        allow_nan=False),encoding='utf-8')
    onset=round(TRANSITION_PLAN['attack_start_s']*FS)
    assert (onset-1)/FS<TRANSITION_PLAN['attack_start_s']<=onset/FS
    identities={};views={};cases={}
    first_prefix=None;first_geometry=None
    for scene in SCENES:
        folder=OUTPUT/scene;folder.mkdir(exist_ok=True)
        record,identity=generate_new_record(TRANSITION_PLAN,scene,
                                              fake_rate=TRANSITION_PLAN['fake_rate_mps'])
        for reference in REFERENCES[scene]:
            old,old_identity=saved_record(reference)
            verify_power_metadata(record,identity,old,old_identity,onset)
            del old
        input_sha,first_prefix,first_geometry=save_new_input(
            folder,record,identity,first_prefix,first_geometry)
        del record
        gc.collect()
        print(scene,'-4 dB IQ saved',flush=True)
        replay,identity=saved_record(folder)
        identities[scene]=identity
        low,_=saved_record(REFERENCES[scene][1])
        high,_=saved_record(REFERENCES[scene][0])
        interpolation=verify_interpolated_samples(replay['samples'],
            low['samples'],high['samples'],round(.1*FS))
        del low,high
        startup=startup_reference(scene)
        fit,fit_sha=fit_input(replay,startup,folder)
        reports={};views[scene]={};positions={}
        for path in PATHS:
            reports[path],views[scene][path]=run_path(
                replay,identity['case'],scene,path,fit,startup,folder/path)
            positions[path]=views[scene][path]['position']
        np.testing.assert_array_equal(views[scene]['off']['observation'],
                                      views[scene]['on']['observation'])
        np.testing.assert_array_equal(views[scene]['off']['available'],
                                      views[scene]['on']['available'])
        cases[scene]=dict(input_sha256=input_sha,
            reference_input_sha256=dict(zero=(stages['stage35']['cases']['weak'][
                'input_sha256'] if scene=='weak' else stages['stage38']['input_sha256']),
                minus_five=stages['stage39']['cases'][scene]['input_sha256']),
            interpolation_maxabs=interpolation,fit_sha256=fit_sha,paths=reports,
            matched=compare_paths(dict(time=views[scene]['off']['observation'],
                available_time=views[scene]['off']['available']),positions,2.))
        del replay,fit,startup,positions
        gc.collect()
        print(scene,'off/on fit, navigation, and loop history saved',flush=True)
    weak,_=saved_record(OUTPUT/'weak')
    absent,_=saved_record(OUTPUT/'genuine_absent')
    compare_power_pair((weak,identities['weak']),(absent,identities['genuine_absent']),
                       onset)
    old_weak,_=saved_record(REFERENCES['weak'][0])
    old_absent,_=saved_record(REFERENCES['genuine_absent'][0])
    component=compare_iq_components(weak['samples'],absent['samples'],
        old_weak['samples'],old_absent['samples'],round(.1*FS))
    plot_results(views)
    assert source_hashes()==sources
    summary=dict(scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        manifest_sha256=digest(OUTPUT/'manifest.json'),
        figure_sha256=digest(OUTPUT/'overview.png'),
        paired_genuine_component_maxabs=component,cases=cases)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),
        indent=2,allow_nan=False),encoding='utf-8')
    return summary


if __name__=='__main__':
    main()
