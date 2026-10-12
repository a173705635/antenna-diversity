"""Paired saved-IQ receiver replay for the three old strong no-attack records.

The only receiver factor is startup magnitude steering, off versus on. These
are reused development inputs, not new RF draws or an authentication test.
"""
from core.provenance import current_path_for_historical_source, ROOT
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
from experiments.spoofing.single_source_code_blend import published_metrics
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import (
    build_coherent_support,quality_support_counts,
)
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_long_coherent import navigation_prefix,source_snapshot
from experiments.spoofing.single_source_nav_bit_epoch_shift import anchor_transport
from experiments.spoofing.single_source_phase_exclusion import verify_source_snapshot
from experiments.spoofing.single_source_phase_state_gate import projection_eligibility
from experiments.spoofing.single_source_startup_amplitude import (
    H0,H0_SEEDS,OUTPUT as STAGE40,load_record,
)
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS,OUTPUT as STUDY_OUTPUT,RECEIVER_PLL,digest,
)
from experiments.spoofing.single_source_temporal_trigger import navigation_input


OUTPUT=RESULTS/STUDY_OUTPUT/'startup_amplitude_rf'
CASES=tuple(f'h0_{seed}' for seed in H0_SEEDS)
PATHS=('off','on')
MODES=('original','short')
OUTPUT_FILES=('tracking.npz','trace.npz','history.npz','groups.npz',
              'support.npz','navigation.npz')


def preflight():
    """Bind every consumed H0 IQ and Stage 40 fit to its frozen source."""
    old=json.loads((STAGE40/'summary.json').read_text(encoding='utf-8'))
    current=source_hashes()
    assert old['source_snapshot_sha256']==digest(STAGE40/'source_snapshot.zip')
    verify_source_snapshot(STAGE40/'source_snapshot.zip',old['source_sha256'])
    for field,file in (('plan_sha256','plan.json'),
                       ('manifest_sha256','manifest.json'),
                       ('figure_sha256','overview.png')):
        assert old[field]==digest(STAGE40/file)
    for name,sha in old['source_sha256'].items():
        if name.startswith('core/') and name not in (
                'core/array_processing.py','core/single_source.py'):
            assert current[current_path_for_historical_source(ROOT, name).relative_to(ROOT).as_posix()]==sha
    for name in CASES:
        case=old['cases'][name]
        folder=H0/'inputs'/f'{name[3:]}_normal_24'
        for file,sha in case['input_sha256'].items():
            assert digest(folder/file)==sha
        for key,file in (('startup_sha256','startup.npz'),
                         ('fit_sha256','fit.npz')):
            assert case[key]==digest(STAGE40/name/file)
        assert case['truth']['result_sha256']==digest(STAGE40/name/'truth.npz')
    return old,current


def audit_alignment(tracking,trace,fit,startup,nominal,mode):
    """Check actual full-receiver beam results against frozen offline fits."""
    np.testing.assert_array_equal(tracking['startup_prns'],startup['prns'])
    np.testing.assert_array_equal(tracking['calibration'],startup['calibration'])
    np.testing.assert_array_equal(tracking['calibration_rate'],
                                  startup['calibration_rate'])
    model=nominal if mode==0 else fit['relative_amplitude'][:,None]*nominal
    np.testing.assert_allclose(trace['trace_nominal'],model,rtol=0,atol=1e-12)
    if mode==1:
        np.testing.assert_array_equal(fit['relative_amplitude'],
                                      startup['relative_amplitude'])
        np.testing.assert_allclose(tracking['startup_relative_amplitude'],
                                   startup['relative_amplitude'],rtol=0,atol=1e-12)
    else:
        assert 'startup_relative_amplitude' not in tracking
    np.testing.assert_allclose(trace['trace_raw_trigger'][5:],fit['raw'][mode,5:],
                               rtol=0,atol=1e-10)
    np.testing.assert_allclose(trace['trace_projectors'][5:],
                               fit['would_projector'][mode,5:],rtol=0,atol=1e-9)
    np.testing.assert_array_equal(trace['trace_dimensions'][5:],
                                  np.where(fit['raw'][mode,5:]>=3.,2,3))
    index=(tracking['time']/.02).astype(int)
    np.testing.assert_allclose(tracking['gain'],fit['would_gain'][mode,index],
                               rtol=0,atol=1e-9)


def score_navigation(tracking,navigation,velocity_east):
    """Use observation-time truth and separate all from .3 s evaluation."""
    time=tracking['time'];available=tracking['available_time']
    truth=np.column_stack((velocity_east*time,np.zeros(len(time))))
    position=navigation['position']
    published=np.isfinite(position).all(axis=1)
    error=np.linalg.norm(position[:,:2]-truth,axis=1)
    evaluation=available>=.3
    first=np.flatnonzero(published)
    anchor=anchor_transport(navigation,time,velocity_east)
    return dict(published_all=int(published.sum()),
        published_evaluation=int((published&evaluation).sum()),
        evaluation_opportunities=int(evaluation.sum()),
        first_published_available_time_s=None if not len(first) else float(available[first[0]]),
        last_published_available_time_s=None if not len(first) else float(available[first[-1]]),
        initialization_time_s=None if navigation['initialization_time'] is None
            else float(navigation['initialization_time']),
        initial_horizontal_error_m=anchor['initial_horizontal_error_m'],
        max_relative_motion_error_m=anchor['max_relative_motion_error_m'],
        all=published_metrics(error,published),
        evaluation=published_metrics(error,published&evaluation))


def check_prefix(tracking,support,eligible,navigation,mode):
    count=len(tracking['time'])-5
    inputs=navigation_input(tracking,support)
    prefix=navigation_prefix(inputs,count)
    options=dict(diagnostics=True,code_blend=.1 if mode=='original' else 0.,
                 phase_leave_one_out=mode=='short',
                 phase_state_continuity=mode=='short')
    if mode=='short':options['phase_state_eligible']=eligible[:count]
    replay=navigate(prefix,**options)
    for key,value in replay.items():
        if key=='diagnostics':
            for diagnostic,array in value.items():
                np.testing.assert_array_equal(array,
                    navigation[key][diagnostic][:count],err_msg=diagnostic)
        elif key=='initialization_time':
            np.testing.assert_array_equal(value,navigation[key])
        else:
            np.testing.assert_array_equal(value,navigation[key][:count],err_msg=key)


def replay_navigation(tracking,trace,coherent,case,folder):
    eligible,latest,recent=projection_eligibility(tracking['available_time'],trace)
    supports=dict(original=tracking['support'],short=coherent['support'])
    arrays=dict(observation_time=tracking['time'],
                available_time=tracking['available_time'],
                phase_state_eligible=eligible,latest_complete_block=latest,
                recent_block_dimensions=recent)
    result={};positions={}
    for mode in MODES:
        support=supports[mode]
        inputs=navigation_input(tracking,support)
        options=dict(diagnostics=True,code_blend=.1 if mode=='original' else 0.,
                     phase_leave_one_out=mode=='short',
                     phase_state_continuity=mode=='short')
        if mode=='short':options['phase_state_eligible']=eligible
        navigation=navigate(inputs,**options)
        check_prefix(tracking,support,eligible,navigation,mode)
        count=quality_support_counts(tracking,support)
        arrays[f'{mode}_support_count']=count
        for key,value in navigation.items():
            if key=='diagnostics':
                for diagnostic,data in value.items():
                    arrays[f'{mode}_diagnostic_{diagnostic}']=data
            else:arrays[f'{mode}_{key}']=np.nan if value is None else value
        result[mode]=dict(**score_navigation(tracking,navigation,case['velocity_east']),
                          quality_support_ge4=int((count>=4).sum()),
                          quality_support_ge6=int((count>=6).sum()))
        positions[mode]=navigation['position'].copy()
    np.savez_compressed(folder/'navigation.npz',**arrays)
    return result,positions


def compare_paths(tracking,positions,velocity_east):
    time=tracking['time'];available=tracking['available_time']
    truth=np.column_stack((velocity_east*time,np.zeros(len(time))))
    result={}
    for mode in MODES:
        off=positions['off'][mode];on=positions['on'][mode]
        off_pub=np.isfinite(off).all(axis=1)
        on_pub=np.isfinite(on).all(axis=1)
        common=(available>=.3)&off_pub&on_pub
        off_error=np.linalg.norm(off[:,:2]-truth,axis=1)
        on_error=np.linalg.norm(on[:,:2]-truth,axis=1)
        result[mode]=dict(common=int(common.sum()),
            new_only=int(((available>=.3)&on_pub&~off_pub).sum()),
            lost_only=int(((available>=.3)&off_pub&~on_pub).sum()),
            off_common=published_metrics(off_error,common),
            on_common=published_metrics(on_error,common))
    return result


def run_path(record,identity,name,path,fit,startup,folder):
    folder.mkdir(exist_ok=True)
    scratch=folder/'beam_stream.dat'
    observations=frontend(record,ANTENNAS,split_support=True,
        receiver_pll=RECEIVER_PLL,evidence_trace=True,
        temporal_beam_trigger_enabled=False,
        beam_template_carrier_aiding=True,beam_coherent_ms=5,
        beam_startup_amplitude=path=='on',stream_memmap_path=scratch)
    scratch_bytes=scratch.stat().st_size
    assert scratch_bytes==len(record['codes'])*record['samples'].shape[1]*16
    trace={key:value for key,value in observations.items() if key.startswith('trace_')}
    tracking={key:value for key,value in observations.items() if not key.startswith('trace_')}
    np.savez_compressed(folder/'tracking.npz',**tracking)
    np.savez_compressed(folder/'trace.npz',**trace)
    scratch.unlink()
    print(name,path,'frontend saved; scratch removed',flush=True)
    audit_alignment(tracking,trace,fit,startup,record['nominal'][ANTENNAS],
                    mode=int(path=='on'))
    history=analyze_trace(observations)
    np.savez_compressed(folder/'history.npz',**history)
    groups=analyze_groups(trace)
    np.savez_compressed(folder/'groups.npz',**groups)
    coherent=build_coherent_support(tracking,groups)
    np.savez_compressed(folder/'support.npz',**{
        f'coherent_{key}':value for key,value in coherent.items()})
    scores,positions=replay_navigation(tracking,trace,coherent,identity['case'],folder)
    result=dict(output_sha256={file:digest(folder/file) for file in OUTPUT_FILES},
                scratch_bytes=scratch_bytes,
                trigger_blocks=int((trace['trace_raw_trigger'][5:]>=3.).sum()),
                scores=scores)
    view=dict(available=tracking['available_time'].copy(),
              observation=tracking['time'].copy(),
              velocity_east=identity['case']['velocity_east'],
              support={mode:quality_support_counts(tracking,
                  tracking['support'] if mode=='original' else coherent['support'])
                  for mode in MODES},position=positions)
    del observations,tracking,trace,history,groups,coherent
    gc.collect()
    return result,view


def plot_results(views):
    figure,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    colors=dict(original='#526270',short='#d05a3c')
    for row,name in enumerate(CASES):
        for path,style in (('off',':'),('on','-')):
            view=views[name][path]
            time=view['available']
            truth=np.column_stack((view['velocity_east']*view['observation'],
                                   np.zeros(len(time))))
            for mode in MODES:
                count=view['support'][mode]
                error=np.linalg.norm(view['position'][mode][:,:2]-truth,axis=1)
                label=f'{mode} {path}'
                axes[row,0].plot(time,count,color=colors[mode],linestyle=style,
                                 linewidth=.7,label=label)
                axes[row,1].plot(time,error,color=colors[mode],linestyle=style,
                                 linewidth=.7,label=label)
                finite=np.isfinite(error)
                axes[row,1].scatter(time[finite],error[finite],
                                    color=colors[mode],s=3)
        axes[row,0].set(title=name,ylabel='Quality-gated direction-supported PRNs')
        axes[row,1].set(ylabel='Published horizontal error (m)')
        for axis in axes[row]:
            axis.set(xlabel='Available time (s)')
            axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=6)
    figure.savefig(OUTPUT/'overview.png',dpi=150)
    plt.close(figure)


def main():
    old,sources=preflight()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip',sources)
    plan=dict(scope=__doc__,cases=CASES,paths=PATHS,
        startup_amplitude='three-channel RMS-one covariance from first 100 phase-calibrated 1ms prompts',
        beam_template_carrier_aiding=True,beam_coherent_ms=5,
        temporal_beam_trigger_enabled=False,receiver_pll=RECEIVER_PLL,
        evaluation_start_s=.3,source_sha256=sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(plan),indent=2),
                                    encoding='utf-8')
    manifest=dict(source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        stage40_summary_sha256=digest(STAGE40/'summary.json'),
        stage40_snapshot_sha256=digest(STAGE40/'source_snapshot.zip'),
        stage40_case_sha256={name:old['cases'][name]['input_sha256'] for name in CASES})
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(manifest),indent=2),
                                        encoding='utf-8')
    results={};views={}
    for name in CASES:
        record,identity=load_record(name)
        (OUTPUT/name).mkdir(exist_ok=True)
        with np.load(STAGE40/name/'fit.npz') as saved:fit=dict(saved)
        with np.load(STAGE40/name/'startup.npz') as saved:startup=dict(saved)
        reports={};views[name]={};positions={}
        for path in PATHS:
            reports[path],views[name][path]=run_path(
                record,identity,name,path,fit,startup,OUTPUT/name/path)
            positions[path]=views[name][path]['position']
        np.testing.assert_array_equal(views[name]['off']['observation'],
                                      views[name]['on']['observation'])
        np.testing.assert_array_equal(views[name]['off']['available'],
                                      views[name]['on']['available'])
        results[name]=dict(input_sha256=old['cases'][name]['input_sha256'],
            stage40_fit_sha256=old['cases'][name]['fit_sha256'],
            stage40_startup_sha256=old['cases'][name]['startup_sha256'],
            paths=reports,matched=compare_paths(
                dict(time=views[name]['off']['observation'],
                     available_time=views[name]['off']['available']),positions,
                     identity['case']['velocity_east']))
        del record,fit,startup,positions
        gc.collect()
        print(name,'off/on receiver and navigation complete',flush=True)
    plot_results(views)
    assert source_hashes()==sources
    summary=dict(scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        manifest_sha256=digest(OUTPUT/'manifest.json'),
        figure_sha256=digest(OUTPUT/'overview.png'),cases=results)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),
        indent=2,allow_nan=False),encoding='utf-8')
    return summary


if __name__=='__main__':
    main()
