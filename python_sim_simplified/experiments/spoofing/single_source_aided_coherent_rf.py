"""Replay saved zero-dB IQ through the fixed startup-aided 5 ms beam.

The original frontend is verified from frozen cache; each scene runs one new
candidate frontend and two independently initialized navigation paths.
"""
import gc
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from core.spoofing.single_source import frontend,navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_code_blend import assert_saved_navigation
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import build_coherent_support
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_long_coherent import (
    OUTPUT as STAGE21,source_snapshot,
)
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_spoof_power import (
    OUTPUT as STAGE26,run_navigation,
)
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS,OUTPUT as STUDY_OUTPUT,RECEIVER_PLL,digest,
)
from experiments.spoofing.single_source_template_coherence import OUTPUT as STAGE27
from experiments.spoofing.single_source_temporal_trigger import (
    OUTPUT as STAGE25,compare_publications,navigation_input,
)


OUTPUT=RESULTS / STUDY_OUTPUT / 'aided_coherent_rf'
SCENES=('weak','genuine_absent','shared_control')
MODES=('original','short')


def sources_and_inputs():
    """Verify each historic snapshot against its own frozen source map."""
    summaries={name:json.loads((path / 'summary.json').read_text(encoding='utf-8'))
               for name,path in (('stage21',STAGE21),('stage25',STAGE25),
                                 ('stage26',STAGE26),('stage27',STAGE27))}
    for name,path in (('stage21',STAGE21),('stage26',STAGE26),
                      ('stage27',STAGE27)):
        old=summaries[name]
        assert old['source_snapshot_sha256']==digest(path / 'source_snapshot.zip')
        verify_source_snapshot(path / 'source_snapshot.zip',old['source_sha256'])
    old=summaries['stage25']
    assert old['rf_source_snapshot_sha256']==digest(STAGE25 / 'source_snapshot.zip')
    verify_source_snapshot(STAGE25 / 'source_snapshot.zip',old['rf_source_sha256'])
    assert old['finalizer_source_snapshot_sha256']==digest(
        STAGE25 / 'finalizer_source_snapshot.zip')
    verify_source_snapshot(STAGE25 / 'finalizer_source_snapshot.zip',
                           old['finalizer_source_sha256'])
    current=source_hashes()
    assert {key:value for key,value in current.items()
            if key.startswith('core/') and key!='core/single_source.py'}=={
                key:value for key,value in summaries['stage27']['source_sha256'].items()
                if key.startswith('core/') and key!='core/single_source.py'}
    for scene in SCENES:
        if scene=='shared_control':
            expected=summaries['stage21']['cases']['no_attack_control']
            input_folder=STAGE21 / 'no_attack_control' / 'input'
            baseline=STAGE21 / 'no_attack_control'
            cached=STAGE25 / 'shared_control'
            for filename,sha in summaries['stage25']['cases'][scene]['output_sha256'].items():
                assert digest(cached / filename)==sha
            for filename,sha in expected['output_sha256'].items():
                assert digest(baseline / filename)==sha
        else:
            expected=summaries['stage26']['cases'][scene]
            input_folder=STAGE26 / scene / 'input'
            baseline=STAGE26 / scene / 'off'
            cached=baseline
            for filename,sha in expected['output_sha256']['off'].items():
                assert digest(baseline / filename)==sha
        for filename,sha in expected['input_sha256'].items():
            assert digest(input_folder / filename)==sha
        assert summaries['stage27']['cases'][scene]['source_input_sha256']==expected[
            'input_sha256']
        fit=STAGE27 / scene / 'fit.npz'
        assert digest(fit)==summaries['stage27']['cases'][scene]['output_sha256']['fit.npz']
    return summaries,current


def scene_paths(scene):
    if scene=='shared_control':
        folder=STAGE21 / 'no_attack_control'
        navigation=STAGE25 / 'shared_control' / 'navigation.npz'
    else:
        folder=STAGE26 / scene
        navigation=folder / 'off' / 'navigation.npz'
    baseline=folder if scene=='shared_control' else folder / 'off'
    return folder / 'input',baseline,navigation


def replay_cached_navigation(tracking,coherent,saved,scene):
    """Verify two fresh old-mode states against every frozen navigation field."""
    supports=dict(original=tracking['support'],short=coherent)
    navigations={}
    for name,blend,loo in (('original',.1,False),('short',0.,True)):
        navigation=navigate(navigation_input(tracking,supports[name]),
                            diagnostics=True,code_blend=blend,
                            phase_leave_one_out=loo)
        prefix=(f'baseline_{name}_' if scene=='shared_control' else f'{name}_')
        assert_saved_navigation(navigation,saved,prefix=prefix)
        navigations[name]=navigation
    return navigations


def candidate_frontend(record,folder):
    observations=frontend(record,ANTENNAS,split_support=True,
                          receiver_pll=RECEIVER_PLL,evidence_trace=True,
                          temporal_beam_trigger_enabled=False,
                          beam_template_carrier_aiding=True,beam_coherent_ms=5)
    trace={key:value for key,value in observations.items() if key.startswith('trace_')}
    tracking={key:value for key,value in observations.items()
              if not key.startswith('trace_')}
    np.savez_compressed(folder / 'tracking.npz',**tracking)
    np.savez_compressed(folder / 'trace.npz',**trace)
    print(folder.name,'candidate frontend saved',flush=True)
    return observations,tracking,trace


def audit_fit_alignment(scene,tracking,trace,baseline,stage27):
    """Full-block trigger, actual P and gain must match the offline factor fit."""
    with np.load(stage27 / scene / 'fit.npz') as saved:
        fit=dict(saved)
    np.testing.assert_array_equal(tracking['startup_prns'],baseline['startup_prns'])
    np.testing.assert_array_equal(tracking['calibration'],baseline['calibration'])
    np.testing.assert_array_equal(tracking['calibration_rate'],
                                  baseline['calibration_rate'])
    np.testing.assert_allclose(trace['trace_raw_trigger'][5:],fit['raw'][3,5:],
                               rtol=0,atol=1e-10)
    np.testing.assert_allclose(trace['trace_projectors'][5:],
                               fit['would_projector'][3,5:],rtol=0,atol=1e-9)
    indices=(tracking['time']/.02).astype(int)
    np.testing.assert_allclose(tracking['gain'],
                               fit['would_gain'][3,indices],rtol=0,atol=1e-9)
    np.testing.assert_array_equal(trace['trace_dimensions'][5:],
                                  np.where(fit['raw'][3,5:]>=3.,2,3))


def plot_cases(views,output=OUTPUT):
    figure,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    colors=dict(original='#606c76',short='#d05339')
    for row,scene in enumerate(SCENES):
        view=views[scene]
        time=view['available']
        truth=np.column_stack((view['velocity_east']*view['observation'],
                               np.zeros(len(time))))
        for mode in MODES:
            for path,style in (('baseline',':'),('candidate','-')):
                navigation=view[path][mode]
                selected=navigation['selected'].sum(axis=1)
                error=np.linalg.norm(navigation['position'][:,:2]-truth,axis=1)
                label=f'{mode} {path}'
                axes[row,0].plot(time,selected,color=colors[mode],
                                 linestyle=style,linewidth=.8,label=label)
                axes[row,1].plot(time,error,color=colors[mode],
                                 linestyle=style,linewidth=.8,label=label)
                finite=np.isfinite(error)
                axes[row,1].scatter(time[finite],error[finite],
                                    color=colors[mode],s=4)
        axes[row,0].set(title=scene,ylabel='Selected PRNs')
        axes[row,1].set(ylabel='Published horizontal error (m)')
        for axis in axes[row]:
            axis.set(xlabel='Output available time (s)')
            axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    axes[0,1].legend(fontsize=7)
    figure.savefig(output / 'overview.png',dpi=150)
    plt.close(figure)


def main():
    summaries,sources=sources_and_inputs()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip',sources)
    plan=dict(scope=__doc__,scenes=SCENES,
              beam_template_carrier_aiding=True,beam_coherent_ms=5,
              temporal_beam_trigger_enabled=False,
              receiver_pll=RECEIVER_PLL,processed_antennas=ANTENNAS,
              navigation_paths=dict(original='original support, code_blend .1',
                                    short='100ms coherent support, code_blend 0, short phase LOO'),
              source_sha256=sources,
              source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
              historic_summary_sha256={name:digest(path / 'summary.json')
                  for name,path in (('stage21',STAGE21),('stage25',STAGE25),
                                    ('stage26',STAGE26),('stage27',STAGE27))})
    (OUTPUT / 'plan.json').write_text(json.dumps(serializable(plan),
        indent=2,allow_nan=False),encoding='utf-8')
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT / 'plan.json'),
        input_sha256={scene:(summaries['stage21']['cases']['no_attack_control']
            if scene=='shared_control' else summaries['stage26']['cases'][scene])
            ['input_sha256'] for scene in SCENES})),
        indent=2,allow_nan=False),encoding='utf-8')
    results={};views={}
    for scene in SCENES:
        input_folder,baseline_folder,navigation_path=scene_paths(scene)
        with np.load(input_folder / 'observations.npz') as saved:record=dict(saved)
        record['samples']=np.load(input_folder / 'samples.npy',mmap_mode='r')
        metadata=json.loads((input_folder / 'metadata.json').read_text(encoding='utf-8'))
        with np.load(baseline_folder / 'tracking.npz') as saved:baseline_tracking=dict(saved)
        with np.load(baseline_folder / 'support.npz') as saved:
            baseline_coherent=saved['coherent_support'].copy()
        with np.load(navigation_path) as saved:
            baseline_navigation=replay_cached_navigation(baseline_tracking,
                                                          baseline_coherent,saved,
                                                          scene)
        folder=OUTPUT / scene
        folder.mkdir(exist_ok=True)
        print(scene,'candidate RF start',flush=True)
        observations,tracking,trace=candidate_frontend(record,folder)
        audit_fit_alignment(scene,tracking,trace,baseline_tracking,STAGE27)
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
        results[scene]=dict(
            input_sha256=(summaries['stage21']['cases']['no_attack_control']
                if scene=='shared_control' else summaries['stage26']['cases'][scene])
                ['input_sha256'],
            baseline_tracking_sha256=digest(baseline_folder / 'tracking.npz'),
            baseline_navigation_sha256=digest(navigation_path),
            stage27_fit_sha256=digest(STAGE27 / scene / 'fit.npz'),
            output_sha256={name:digest(folder / name) for name in (
                'tracking.npz','trace.npz','history.npz','groups.npz',
                'support.npz','navigation.npz')},
            trigger_clean=int((trace['trace_raw_trigger'][5:15]>=3.).sum()),
            trigger_post=int((trace['trace_raw_trigger'][15:]>=3.).sum()),
            candidate_scores=candidate_scores,matched_publications=comparisons)
        views[scene]=dict(available=tracking['available_time'].copy(),
                          observation=tracking['time'].copy(),
                          velocity_east=metadata['case']['velocity_east'],
                          baseline=baseline_navigation,
                          candidate=candidate_navigation)
        del record,observations,tracking,trace,history,groups,coherent
        gc.collect()
        print(scene,'candidate analysis done',flush=True)
    plot_cases(views)
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
