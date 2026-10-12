"""Replay saved five-degree IQ with the fixed aided 10 ms beam prompt.

The 5 ms beam, 100 ms navigation support, PLL, and navigation modes are saved
references. Each scene runs only one new 10 ms frontend and two fresh navs.
"""
import gc
import json

import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from core.spoofing.single_source import frontend,navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_aided_coherent_rf import (
    OUTPUT as STAGE28,plot_cases,sources_and_inputs,
)
from experiments.spoofing.single_source_beam_coherence_lengths import OUTPUT as STAGE30
from experiments.spoofing.single_source_code_blend import assert_saved_navigation
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import build_coherent_support
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_long_coherent import OUTPUT as STAGE21,source_snapshot
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_spoof_power import run_navigation
from experiments.spoofing.single_source_spoof_power_minus_five import OUTPUT as STAGE29
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS,OUTPUT as STUDY_OUTPUT,RECEIVER_PLL,digest,
)
from experiments.spoofing.single_source_temporal_trigger import (
    compare_publications,navigation_input,
)


OUTPUT=RESULTS / STUDY_OUTPUT / 'aided_ten_ms_rf'
SCENES=('weak','genuine_absent','shared_control')
MODES=('original','short')


def preflight():
    """Verify each saved source map and all IQ/fit/navigation inputs used."""
    _,current=sources_and_inputs()
    saved={name:json.loads((folder / 'summary.json').read_text(
        encoding='utf-8')) for name,folder in (
            ('stage28',STAGE28),('stage29',STAGE29),('stage30',STAGE30))}
    for name,folder in (('stage28',STAGE28),('stage29',STAGE29),
                        ('stage30',STAGE30)):
        case=saved[name]
        assert case['source_snapshot_sha256']==digest(folder / 'source_snapshot.zip')
        verify_source_snapshot(folder / 'source_snapshot.zip',
                               case['source_sha256'])
        assert case['plan_sha256']==digest(folder / 'plan.json')
        assert case['manifest_sha256']==digest(folder / 'manifest.json')
        assert case['figure_sha256']==digest(folder / 'overview.png')
    assert {key:value for key,value in current.items()
            if key.startswith('core/') and key!='core/single_source.py'}=={
                key:value for key,value in saved['stage30']['source_sha256'].items()
                if key.startswith('core/') and key!='core/single_source.py'}
    for scene in SCENES:
        if scene=='shared_control':
            expected=saved['stage28']['cases'][scene]
            input_folder=STAGE21 / 'no_attack_control' / 'input'
            baseline_folder=STAGE28 / scene
            for filename,sha in expected['output_sha256'].items():
                assert sha==digest(baseline_folder / filename)
        else:
            expected=saved['stage29']['cases'][scene]
            input_folder=STAGE29 / scene / 'input'
            baseline_folder=STAGE29 / scene / 'candidate'
            for filename,sha in expected['output_sha256']['candidate'].items():
                assert sha==digest(baseline_folder / filename)
        for filename,sha in expected['input_sha256'].items():
            assert sha==digest(input_folder / filename)
        fit_case=saved['stage30']['cases'][scene]
        assert fit_case['source_input_sha256']==expected['input_sha256']
        assert fit_case['output_sha256']['fit.npz']==digest(
            STAGE30 / scene / 'fit.npz')
    return saved,current


def scene_paths(scene):
    if scene=='shared_control':
        return (STAGE21 / 'no_attack_control' / 'input',
                STAGE28 / scene)
    return STAGE29 / scene / 'input',STAGE29 / scene / 'candidate'


def replay_saved_five_ms(tracking,coherent,saved):
    """Verify both independently initialized old-mode navs against all fields."""
    supports=dict(original=tracking['support'],short=coherent)
    results={}
    for name,blend,loo in (('original',.1,False),('short',0.,True)):
        navigation=navigate(navigation_input(tracking,supports[name]),
                            diagnostics=True,code_blend=blend,
                            phase_leave_one_out=loo)
        assert_saved_navigation(navigation,saved,prefix=f'{name}_')
        results[name]=navigation
    return results


def ten_ms_frontend(record,folder):
    observations=frontend(record,ANTENNAS,split_support=True,
                          receiver_pll=RECEIVER_PLL,evidence_trace=True,
                          temporal_beam_trigger_enabled=False,
                          beam_template_carrier_aiding=True,beam_coherent_ms=10)
    trace={key:value for key,value in observations.items()
           if key.startswith('trace_')}
    tracking={key:value for key,value in observations.items()
              if not key.startswith('trace_')}
    np.savez_compressed(folder / 'tracking.npz',**tracking)
    np.savez_compressed(folder / 'trace.npz',**trace)
    print(folder.name,'ten-ms frontend saved',flush=True)
    return observations,tracking,trace


def audit_ten_ms_fit(tracking,trace,baseline,fit):
    """All real RF block decisions must match the frozen L=10 fit."""
    np.testing.assert_array_equal(tracking['startup_prns'],baseline['startup_prns'])
    np.testing.assert_array_equal(tracking['calibration'],baseline['calibration'])
    np.testing.assert_array_equal(tracking['calibration_rate'],
                                  baseline['calibration_rate'])
    np.testing.assert_array_equal(tracking['time'],baseline['time'])
    np.testing.assert_array_equal(tracking['available_time'],
                                  baseline['available_time'])
    np.testing.assert_allclose(trace['trace_raw_trigger'][5:],fit['raw'][2,5:],
                               rtol=0,atol=1e-10)
    np.testing.assert_allclose(trace['trace_projectors'][5:],
                               fit['would_projector'][2,5:],rtol=0,atol=1e-9)
    indices=(tracking['time']/.02).astype(int)
    np.testing.assert_allclose(tracking['gain'],
                               fit['would_gain'][2,indices],rtol=0,atol=1e-9)
    np.testing.assert_array_equal(trace['trace_dimensions'][5:],
                                  np.where(fit['raw'][2,5:]>=3.,2,3))


def main():
    saved,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip',sources)
    inputs={scene:(saved['stage28']['cases'][scene] if scene=='shared_control'
                   else saved['stage29']['cases'][scene])['input_sha256']
            for scene in SCENES}
    plan=dict(scope=__doc__,scenes=SCENES,
              beam_template_carrier_aiding=True,beam_coherent_ms=10,
              temporal_beam_trigger_enabled=False,
              navigation_support='same 5 ms groups and 100 ms coherent support',
              navigation_modes=dict(original='support, code_blend .1',
                                    short='coherent support, code_blend 0, short phase LOO'),
              receiver_pll=RECEIVER_PLL,processed_antennas=ANTENNAS,
              input_sha256=inputs,
              reference_summary_sha256={name:digest(folder / 'summary.json')
                  for name,folder in (('stage28',STAGE28),('stage29',STAGE29),
                                      ('stage30',STAGE30))},
              source_sha256=sources,
              source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'))
    (OUTPUT / 'plan.json').write_text(json.dumps(serializable(plan),
        indent=2,allow_nan=False),encoding='utf-8')
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT / 'plan.json'),input_sha256=inputs)),
        indent=2,allow_nan=False),encoding='utf-8')
    results={};views={}
    for scene in SCENES:
        input_folder,baseline_folder=scene_paths(scene)
        with np.load(input_folder / 'observations.npz') as saved_input:
            record=dict(saved_input)
        record['samples']=np.load(input_folder / 'samples.npy',mmap_mode='r')
        metadata=json.loads((input_folder / 'metadata.json').read_text(
            encoding='utf-8'))
        with np.load(baseline_folder / 'tracking.npz') as cached:
            baseline_tracking=dict(cached)
        with np.load(baseline_folder / 'support.npz') as cached:
            baseline_coherent=cached['coherent_support'].copy()
        with np.load(baseline_folder / 'navigation.npz') as cached:
            baseline_navigation=replay_saved_five_ms(
                baseline_tracking,baseline_coherent,cached)
        with np.load(STAGE30 / scene / 'fit.npz') as cached:
            fit=dict(cached)
        folder=OUTPUT / scene
        folder.mkdir(exist_ok=True)
        print(scene,'ten-ms candidate RF start',flush=True)
        observations,tracking,trace=ten_ms_frontend(record,folder)
        audit_ten_ms_fit(tracking,trace,baseline_tracking,fit)
        history=analyze_trace(observations)
        np.savez_compressed(folder / 'history.npz',**history)
        groups=analyze_groups(trace)
        np.savez_compressed(folder / 'groups.npz',**groups)
        coherent=build_coherent_support(tracking,groups)
        np.savez_compressed(folder / 'support.npz',**{
            f'coherent_{key}':value for key,value in coherent.items()})
        candidate_navigation,candidate_scores,_=run_navigation(
            tracking,coherent['support'],metadata['case'],folder)
        comparisons={name:compare_publications(
            tracking,baseline_navigation[name],candidate_navigation[name],
            metadata['case'])[0] for name in MODES}
        results[scene]=dict(input_sha256=inputs[scene],
            baseline_tracking_sha256=digest(baseline_folder / 'tracking.npz'),
            baseline_navigation_sha256=digest(baseline_folder / 'navigation.npz'),
            stage30_fit_sha256=digest(STAGE30 / scene / 'fit.npz'),
            output_sha256={name:digest(folder / name) for name in (
                'tracking.npz','trace.npz','history.npz','groups.npz',
                'support.npz','navigation.npz')},
            trigger_clean=int((trace['trace_raw_trigger'][5:15]>=3.).sum()),
            trigger_post=int((trace['trace_raw_trigger'][15:]>=3.).sum()),
            candidate_scores=candidate_scores,matched_publications=comparisons)
        views[scene]=dict(available=tracking['available_time'].copy(),
            observation=tracking['time'].copy(),
            velocity_east=metadata['case']['velocity_east'],
            baseline=baseline_navigation,candidate=candidate_navigation)
        del record,observations,tracking,trace,history,groups,coherent
        gc.collect()
        print(scene,'ten-ms candidate analysis done',flush=True)
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
