"""Replay a causal five-block raw beam trigger on three saved eight-second IQ inputs.

The current 20 ms residual projector always supplies the beam; recent raw
scores only keep that projector enabled. Default cached tracking is verified
by hashes and saved-observation navigation, without another default RF pass.
"""
import gc
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from core.single_source import frontend, navigate
from experiments.paths import RESULTS
from experiments.single_source_close_angle import (
    OUTPUT as STAGE24, preflight_references,
)
from experiments.single_source_code_blend import (
    assert_saved_navigation, published_metrics,
)
from experiments.single_source_coherence_diagnosis import analyze_groups
from experiments.single_source_coherent_navigation import (
    build_coherent_support, quality_support_counts, score_mode,
)
from experiments.single_source_evidence_history import analyze_trace
from experiments.single_source_long_coherent import (
    OUTPUT as STAGE21, navigation_prefix, save_navigation, source_snapshot,
)
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, RECEIVER_PLL, digest,
)


OUTPUT=RESULTS / STUDY_OUTPUT / 'temporal_beam_trigger'
SCENES=('angle5_weak','angle5_genuine_absent','shared_control')
NAVIGATION_FIELDS=('time','available_time','satellites','ranges','lock',
                   'cn0','gain','support','arc','carrier')


def navigation_input(tracking,support):
    """Pass only observation-time arrays into an independent navigation run."""
    return {key:support if key=='support' else tracking[key]
            for key in NAVIGATION_FIELDS}


def compare_publications(observations,baseline,candidate,case):
    """Score both trajectories at their own and their common published epochs."""
    time=observations['time'];available=observations['available_time']
    truth=np.column_stack((case['velocity_east']*time,np.zeros(len(time))))
    baseline_error=np.linalg.norm(baseline['position'][:,:2]-truth,axis=1)
    candidate_error=np.linalg.norm(candidate['position'][:,:2]-truth,axis=1)
    baseline_published=np.isfinite(baseline['position']).all(axis=1)
    candidate_published=np.isfinite(candidate['position']).all(axis=1)
    post=available>=case['attack_start']
    common=post&baseline_published&candidate_published
    result=dict(common_published=int(common.sum()),
                baseline_only_published=int((post&baseline_published&~candidate_published).sum()),
                candidate_only_published=int((post&candidate_published&~baseline_published).sum()),
                baseline_common=published_metrics(baseline_error,common),
                candidate_common=published_metrics(candidate_error,common))
    return result,baseline_error,candidate_error,common


def preflight_inputs():
    stage21,_,_=preflight_references()
    stage24=json.loads((STAGE24 / 'summary.json').read_text(encoding='utf-8'))
    assert stage24['manifest_sha256']==digest(STAGE24 / 'manifest.json')
    assert stage24['plan_sha256']==digest(STAGE24 / 'plan.json')
    assert stage24['source_snapshot_sha256']==digest(STAGE24 / 'source_snapshot.zip')
    verify_source_snapshot(STAGE24 / 'source_snapshot.zip',stage24['source_sha256'])
    for scene in SCENES:
        if scene=='shared_control':
            folder=STAGE21 / 'no_attack_control'
            expected=stage21['cases']['no_attack_control']
            assert stage24['cases'][scene]['navigation_sha256']==digest(
                STAGE24 / scene / 'navigation.npz')
        else:
            folder=STAGE24 / scene
            expected=stage24['cases'][scene]
            for name,value in expected['output_sha256'].items():
                assert value==digest(folder / name)
        for name,value in expected['input_sha256'].items():
            assert value==digest(folder / 'input' / name)
    return stage21,stage24


def assert_prefix(observations,navigation,blend,leave_one_out):
    count=len(observations['time'])-20
    prefix=navigate(navigation_prefix(observations,count),diagnostics=True,
                    code_blend=blend,phase_leave_one_out=leave_one_out)
    for key,value in prefix.items():
        if key=='diagnostics':
            for diagnostic,data in value.items():
                np.testing.assert_array_equal(data,
                    navigation[key][diagnostic][:count],err_msg=diagnostic)
        elif key=='initialization_time':
            np.testing.assert_array_equal(value,navigation[key])
        else:
            np.testing.assert_array_equal(value,navigation[key][:count],err_msg=key)


def current_projector_diagnostics(tracking,trace):
    raw=tracking['beam_trigger_raw']
    mean=tracking['beam_trigger_history_mean']
    effective=tracking['beam_trigger_effective']
    computed=np.flatnonzero(np.isfinite(raw))
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(mean)),computed)
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(effective)),computed)
    for rank,block in enumerate(computed):
        independent_mean=float(np.mean(raw[computed[max(0,rank-4):rank+1]]))
        np.testing.assert_allclose(mean[block],independent_mean,rtol=0.,atol=1e-14)
        np.testing.assert_allclose(effective[block],max(raw[block],independent_mean),
                                   rtol=0.,atol=1e-14)
    on=np.isfinite(effective)&(effective>=3.)
    np.testing.assert_array_equal(on,trace['trace_dimensions']==2)
    np.testing.assert_array_equal(tracking['beam_trigger_history_only'],
                                  on&(raw<3.))
    nominal=trace['trace_nominal']
    projector=trace['trace_projectors']
    gain=np.einsum('mi,bmn,ni->bi',nominal.conj(),projector,nominal).real/len(nominal)
    indices=(tracking['time']/.02).astype(int)
    np.testing.assert_allclose(gain[indices],tracking['gain'],atol=1e-9)
    measured=np.isfinite(raw[indices])
    np.testing.assert_array_equal(tracking['ratio'][measured],raw[indices][measured])
    return dict(block_start_time_s=np.arange(len(raw))*.02,
                raw_trigger=raw,history_mean_trigger=mean,
                effective_trigger=effective,trigger_on=on,
                history_only=tracking['beam_trigger_history_only'],
                projector_dimension=trace['trace_dimensions'],
                current_projector_gain=gain)


def block_summary(block,trace,index):
    """Reference the saved complex projector without putting it into JSON."""
    assert np.linalg.matrix_rank(trace['trace_projectors'][index],tol=1e-8) == \
           block['projector_dimension'][index]
    return dict(raw=float(block['raw_trigger'][index]),
                history_mean=float(block['history_mean_trigger'][index]),
                effective=float(block['effective_trigger'][index]),
                on=bool(block['trigger_on'][index]),
                dimension=int(block['projector_dimension'][index]),
                projector_trace_index=index,
                nominal_gain=block['current_projector_gain'][index])


def load_default(scene,stage21,stage24):
    if scene=='shared_control':
        source=STAGE21 / 'no_attack_control'
        expected=stage21['cases']['no_attack_control']
    else:
        source=STAGE24 / scene
        expected=stage24['cases'][scene]
    with np.load(source / 'tracking.npz') as saved:
        tracking=dict(saved)
    with np.load(source / 'support.npz') as saved:
        coherent=saved['coherent_support'].copy()
    with np.load(STAGE24 / scene / 'navigation.npz') as saved:
        reference=dict(saved)
    metadata=json.loads((source / 'input' / 'metadata.json').read_text(encoding='utf-8'))
    baseline={}
    for name,support,blend,leave_one_out in (
            ('original',tracking['support'],.1,False),
            ('short',coherent,0.,True)):
        inputs=navigation_input(tracking,support)
        baseline[name]=navigate(inputs,diagnostics=True,code_blend=blend,
                                phase_leave_one_out=leave_one_out)
        assert_saved_navigation(baseline[name],reference,prefix=f'{name}_')
    hashes=dict(input_sha256=expected['input_sha256'],
                tracking_sha256=digest(source / 'tracking.npz'),
                support_sha256=digest(source / 'support.npz'),
                navigation_sha256=digest(STAGE24 / scene / 'navigation.npz'))
    return source,tracking,coherent,baseline,metadata,hashes


def run_case(scene,stage21,stage24):
    source,default_tracking,default_coherent,baseline,metadata,hashes=(
        load_default(scene,stage21,stage24))
    folder=OUTPUT / scene
    folder.mkdir(exist_ok=True)
    with np.load(source / 'input' / 'observations.npz') as saved:
        record=dict(saved)
    record['samples']=np.load(source / 'input' / 'samples.npy',mmap_mode='r')
    print(scene,'candidate frontend start',flush=True)
    observations=frontend(record,ANTENNAS,split_support=True,
                          receiver_pll=RECEIVER_PLL,evidence_trace=True,
                          temporal_beam_trigger_enabled=True)
    del record
    gc.collect()
    trace_keys=[key for key in observations if key.startswith('trace_')]
    trace={key:observations[key] for key in trace_keys}
    tracking={key:value for key,value in observations.items() if key not in trace_keys}
    np.savez_compressed(folder / 'tracking.npz',**tracking)
    np.savez_compressed(folder / 'trace.npz',**trace)
    print(scene,'candidate frontend saved',flush=True)

    np.testing.assert_array_equal(tracking['time'],default_tracking['time'])
    np.testing.assert_array_equal(tracking['available_time'],
                                  default_tracking['available_time'])
    for key in ('startup_prns','calibration','calibration_rate'):
        np.testing.assert_array_equal(tracking[key],default_tracking[key])
    # An output is clean only when its availability also precedes the onset.
    preattack=tracking['available_time']<=metadata['case']['attack_start']+1e-9
    preattack_mismatches={key:int((~np.isclose(
        tracking[key][preattack],default_tracking[key][preattack],
        rtol=0.,atol=0.,equal_nan=True)).sum())
        for key in ('ranges','lock','cn0','carrier','gain','support','arc','ratio')}
    block=current_projector_diagnostics(tracking,trace)
    np.savez_compressed(folder / 'blocks.npz',**block)
    history=analyze_trace(observations)
    np.savez_compressed(folder / 'history.npz',**history)
    groups=analyze_groups(trace)
    np.savez_compressed(folder / 'groups.npz',**groups)
    coherent=build_coherent_support(tracking,groups)
    np.savez_compressed(folder / 'support.npz',**{
        f'coherent_{key}':value for key,value in coherent.items()})

    supports=dict(original=tracking['support'],short=coherent['support'])
    default_supports=dict(original=default_tracking['support'],
                          short=default_coherent)
    candidate={}
    scores={}
    arrays=dict(observation_time=tracking['time'],
                available_time=tracking['available_time'])
    for name,blend,leave_one_out in (('original',.1,False),('short',0.,True)):
        inputs=navigation_input(tracking,supports[name])
        candidate[name]=navigate(inputs,diagnostics=True,code_blend=blend,
                                 phase_leave_one_out=leave_one_out)
        assert_prefix(inputs,candidate[name],blend,leave_one_out)
        comparison,old_error,new_error,common=compare_publications(
            tracking,baseline[name],candidate[name],metadata['case'])
        scores[name]=dict(
            baseline=score_mode(baseline[name],default_tracking,metadata['case'],
                                quality_support_counts(default_tracking,default_supports[name]),
                                default_supports[name]),
            candidate=score_mode(candidate[name],tracking,metadata['case'],
                                 quality_support_counts(tracking,supports[name]),
                                 supports[name]),
            matched_publications=comparison)
        arrays[f'{name}_baseline_horizontal_error_m']=old_error
        arrays[f'{name}_candidate_horizontal_error_m']=new_error
        arrays[f'{name}_common_published']=common
        arrays[f'{name}_baseline_support_count']=quality_support_counts(
            default_tracking,default_supports[name])
        arrays[f'{name}_candidate_support_count']=quality_support_counts(
            tracking,supports[name])
    save_navigation(arrays,{f'baseline_{name}':value
                            for name,value in baseline.items()})
    save_navigation(arrays,{f'candidate_{name}':value
                            for name,value in candidate.items()})
    np.savez_compressed(folder / 'navigation.npz',**arrays)
    files=('tracking.npz','trace.npz','blocks.npz','history.npz',
           'groups.npz','support.npz','navigation.npz')
    report=dict(**hashes,output_sha256={name:digest(folder / name) for name in files},
                preattack_epoch_count=int(preattack.sum()),
                preattack_mismatch_count=preattack_mismatches,
                history_only_block_count=int(block['history_only'].sum()),
                newly_enabled_blocks=np.flatnonzero(block['history_only']),
                block388=block_summary(block,trace,388),
                modes=scores)
    plot=dict(available=tracking['available_time'].copy(),
              errors={name:(arrays[f'{name}_baseline_horizontal_error_m'].copy(),
                            arrays[f'{name}_candidate_horizontal_error_m'].copy())
                      for name in supports},
              counts={name:(arrays[f'{name}_baseline_support_count'].copy(),
                            arrays[f'{name}_candidate_support_count'].copy())
                      for name in supports})
    print(scene,'candidate analysis done',flush=True)
    return report,plot


def finalize_saved():
    """Recover the summary from frozen RF outputs after JSON rejected complex P."""
    stage21,stage24=preflight_inputs()
    rf_manifest=json.loads((OUTPUT / 'manifest.json').read_text(encoding='utf-8'))
    assert rf_manifest['source_snapshot_sha256']==digest(OUTPUT / 'source_snapshot.zip')
    verify_source_snapshot(OUTPUT / 'source_snapshot.zip',rf_manifest['source_sha256'])
    sources=source_hashes()
    source_snapshot(OUTPUT / 'finalizer_source_snapshot.zip',sources)
    reports={}
    names=('tracking.npz','trace.npz','blocks.npz','history.npz',
           'groups.npz','support.npz','navigation.npz')
    for scene in SCENES:
        folder=OUTPUT / scene
        source,default_tracking,default_coherent,baseline,metadata,hashes=(
            load_default(scene,stage21,stage24))
        with np.load(folder / 'tracking.npz') as saved:tracking=dict(saved)
        with np.load(folder / 'trace.npz') as saved:trace=dict(saved)
        with np.load(folder / 'blocks.npz') as saved:saved_blocks=dict(saved)
        with np.load(folder / 'history.npz') as saved:saved_history=dict(saved)
        with np.load(folder / 'groups.npz') as saved:saved_groups=dict(saved)
        with np.load(folder / 'support.npz') as saved:saved_support=dict(saved)
        with np.load(folder / 'navigation.npz') as saved:arrays=dict(saved)
        np.testing.assert_array_equal(tracking['time'],default_tracking['time'])
        np.testing.assert_array_equal(tracking['available_time'],
                                      default_tracking['available_time'])
        for key in ('startup_prns','calibration','calibration_rate'):
            np.testing.assert_array_equal(tracking[key],default_tracking[key])
        blocks=current_projector_diagnostics(tracking,trace)
        for key,value in blocks.items():
            np.testing.assert_array_equal(value,saved_blocks[key],err_msg=key)
        computed_history=analyze_trace({**tracking,**trace})
        for key,value in computed_history.items():
            np.testing.assert_array_equal(value,saved_history[key],err_msg=key)
        computed_groups=analyze_groups(trace)
        for key,value in computed_groups.items():
            np.testing.assert_array_equal(value,saved_groups[key],err_msg=key)
        coherent=build_coherent_support(tracking,computed_groups)
        for key,value in coherent.items():
            np.testing.assert_array_equal(value,saved_support[f'coherent_{key}'],
                                          err_msg=key)
        support=dict(original=tracking['support'],short=coherent['support'])
        old_support=dict(original=default_tracking['support'],
                         short=default_coherent)
        modes={}
        for name,blend,leave_one_out in (('original',.1,False),('short',0.,True)):
            old=baseline[name]
            assert_saved_navigation(old,arrays,prefix=f'baseline_{name}_')
            inputs=navigation_input(tracking,support[name])
            candidate=navigate(inputs,diagnostics=True,code_blend=blend,
                               phase_leave_one_out=leave_one_out)
            assert_saved_navigation(candidate,arrays,prefix=f'candidate_{name}_')
            assert_prefix(inputs,candidate,blend,leave_one_out)
            matched,old_error,new_error,common=compare_publications(
                tracking,old,candidate,metadata['case'])
            for key,value in ((f'{name}_baseline_horizontal_error_m',old_error),
                              (f'{name}_candidate_horizontal_error_m',new_error),
                              (f'{name}_common_published',common)):
                np.testing.assert_array_equal(value,arrays[key],err_msg=key)
            old_count=quality_support_counts(default_tracking,old_support[name])
            new_count=quality_support_counts(tracking,support[name])
            np.testing.assert_array_equal(old_count,
                arrays[f'{name}_baseline_support_count'])
            np.testing.assert_array_equal(new_count,
                arrays[f'{name}_candidate_support_count'])
            modes[name]=dict(
                baseline=score_mode(old,default_tracking,metadata['case'],
                                    old_count,old_support[name]),
                candidate=score_mode(candidate,tracking,metadata['case'],
                                     new_count,support[name]),
                matched_publications=matched)
        preattack=tracking['available_time']<=metadata['case']['attack_start']+1e-9
        mismatches={key:int((~np.isclose(
            tracking[key][preattack],default_tracking[key][preattack],
            rtol=0.,atol=0.,equal_nan=True)).sum())
            for key in ('ranges','lock','cn0','carrier','gain','support','arc','ratio')}
        reports[scene]=dict(**hashes,
            output_sha256={name:digest(folder / name) for name in names},
            preattack_epoch_count=int(preattack.sum()),
            preattack_mismatch_count=mismatches,
            history_only_block_count=int(blocks['history_only'].sum()),
            newly_enabled_blocks=np.flatnonzero(blocks['history_only']),
            block388=block_summary(blocks,trace,388),modes=modes)
        print(scene,'saved-output finalization verified',flush=True)
    assert source_hashes()==sources
    summary=dict(scope=rf_manifest['scope'],
        recovery_reason='RF and navigation completed; original summary JSON rejected redundant complex block-388 projector',
        rf_manifest_sha256=digest(OUTPUT / 'manifest.json'),
        rf_source_sha256=rf_manifest['source_sha256'],
        rf_source_snapshot_sha256=rf_manifest['source_snapshot_sha256'],
        finalizer_source_sha256=sources,
        finalizer_source_snapshot_sha256=digest(OUTPUT / 'finalizer_source_snapshot.zip'),
        figure_sha256=digest(OUTPUT / 'overview.png'),cases=reports)
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(summary),
        indent=2,allow_nan=False),encoding='utf-8')
    return summary


def main():
    stage21,stage24=preflight_inputs()
    sources=source_hashes()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip',sources)
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        stage21_summary_sha256=digest(STAGE21 / 'summary.json'),
        stage24_summary_sha256=digest(STAGE24 / 'summary.json'),
        fixed_window_blocks=5,threshold=3.,
        trigger='max(raw_current, mean(current and up to four preceding computed blocks))',
        candidate_phase_leave_one_out=True,candidate_code_blend=0.,
        scenes=SCENES)),indent=2,allow_nan=False),encoding='utf-8')
    reports={}
    plots={}
    for scene in SCENES:
        reports[scene],plots[scene]=run_case(scene,stage21,stage24)
    fig,axes=plt.subplots(len(SCENES),2,figsize=(12,10),
                          constrained_layout=True)
    for row,scene in enumerate(SCENES):
        plot=plots[scene]
        for name,color in (('original','#777777'),('short','#cf5635')):
            old,new=plot['counts'][name]
            axes[row,0].plot(plot['available'],old,color=color,
                             linestyle=':',label=f'{name} cached')
            axes[row,0].plot(plot['available'],new,color=color,
                             label=f'{name} candidate')
            old_error,new_error=plot['errors'][name]
            axes[row,1].plot(plot['available'],old_error,color=color,
                             linestyle=':',label=f'{name} cached')
            old_finite=np.isfinite(old_error)
            axes[row,1].scatter(plot['available'][old_finite],old_error[old_finite],
                                color=color,s=4,zorder=3,alpha=.65)
            axes[row,1].plot(plot['available'],new_error,color=color,
                             linewidth=.8,label=f'{name} candidate')
            finite=np.isfinite(new_error)
            axes[row,1].scatter(plot['available'][finite],new_error[finite],
                                color=color,s=5,zorder=3)
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
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,manifest_sha256=digest(OUTPUT / 'manifest.json'),
        source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        figure_sha256=digest(OUTPUT / 'overview.png'),cases=reports)),
        indent=2,allow_nan=False),encoding='utf-8')
    return reports


if __name__=='__main__':
    main()
