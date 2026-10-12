"""Replay one exact 1 s phase anchor after the short single-star bridge.

The long bridge is a fixed saved-observation candidate. It uses only its own
published trajectory and the existing carrier-fit gates; no RF is regenerated.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from core.spoofing.single_source import navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_carrier_fault import compare_fault
from experiments.spoofing.single_source_code_blend import (
    assert_saved_navigation, published_metrics,
)
from experiments.spoofing.single_source_long_coherent import navigation_prefix, source_snapshot
from experiments.spoofing.single_source_phase_exclusion import (
    CASES22, OUTPUT as STAGE22, fault_window_mask, fit_geometry,
    preflight_inputs, saved_cases, score_pair, verify_source_snapshot,
)
from experiments.spoofing.single_source_suppression_study import OUTPUT as STUDY_OUTPUT, digest


OUTPUT = RESULTS / STUDY_OUTPUT / 'long_phase_anchor'


def preflight_stage22():
    stage19, _, stage21 = preflight_inputs()
    stage22 = json.loads((STAGE22 / 'summary.json').read_text(encoding='utf-8'))
    assert stage22['manifest_sha256'] == digest(STAGE22 / 'manifest.json')
    assert stage22['source_snapshot_sha256'] == digest(STAGE22 / 'source_snapshot.zip')
    verify_source_snapshot(STAGE22 / 'source_snapshot.zip', stage22['source_sha256'])
    for name in CASES22:
        assert stage22['cases'][name]['result_sha256'] == digest(STAGE22 / f'{name}.npz')
    return stage19, stage21, stage22


def replay_pair(observations, saved):
    """The short branch is a complete cached reference, not shared state."""
    short = navigate(observations, diagnostics=True, code_blend=0.,
                     phase_leave_one_out=True)
    assert_saved_navigation(short, saved, prefix='on_')
    long = navigate(observations, diagnostics=True, code_blend=0.,
                    phase_leave_one_out=True, phase_long_anchor=True)
    np.testing.assert_array_equal(short['selected'], long['selected'])
    return short, long


def long_geometry(observations, navigation):
    """Independently recompute geometry only where the long path was used."""
    n = len(observations['time'])
    arrays = dict(full_condition=np.full(n,np.nan),
                  full_horizontal_std_m=np.full(n,np.nan),
                  full_max_residual_m=np.full(n,np.nan),
                  candidate_condition=np.full(n,np.nan),
                  candidate_horizontal_std_m=np.full(n,np.nan),
                  candidate_max_residual_m=np.full(n,np.nan))
    diagnostic = navigation['diagnostics']
    for k in np.flatnonzero(diagnostic['phase_long_path'] == 'long'):
        j = int(diagnostic['phase_long_anchor_index'][k])
        common = (navigation['selected'][k] & navigation['selected'][j]
                  & np.isfinite(observations['carrier'][k])
                  & np.isfinite(observations['carrier'][j])
                  & (observations['arc'][k] == observations['arc'][j]))
        indices = np.flatnonzero(common)
        full = fit_geometry(observations,k,j,indices)
        excluded = int(diagnostic['phase_long_excluded_prn'][k]-1)
        retained = indices[indices != excluded]
        candidate = fit_geometry(observations,k,j,retained)
        for label,fit,subset in (('full',full,indices),
                                 ('candidate',candidate,retained)):
            for field in ('condition','horizontal_std_m','max_residual_m'):
                arrays[f'{label}_{field}'][k]=fit[field]
            recorded=diagnostic[f'phase_long_{label}_residual_m'][k,subset]
            if np.isfinite(recorded).all():
                np.testing.assert_allclose(fit['residual_m'],recorded,atol=1e-8)
    return arrays


def anchor_errors(observations, navigation, metadata):
    """Score candidate-owned anchor absolute error against truth, offline only."""
    diagnostic=navigation['diagnostics']
    indices=diagnostic['phase_long_anchor_index']
    horizontal=np.full(len(indices),np.nan)
    spatial=np.full(len(indices),np.nan)
    for k in np.flatnonzero(diagnostic['phase_long_path']=='long'):
        j=indices[k]
        truth=np.array([metadata['case']['velocity_east']*observations['time'][j],
                        0.,0.])
        delta=diagnostic['phase_long_anchor_position_m'][k]-truth
        horizontal[k]=np.linalg.norm(delta[:2])
        spatial[k]=np.linalg.norm(delta)
    return horizontal,spatial


def finite_summary(values):
    selected=values[np.isfinite(values)]
    return dict(count=int(len(selected)),
                median_m=None if not len(selected) else float(np.median(selected)),
                max_m=None if not len(selected) else float(selected.max()))


def main():
    stage19, stage21, stage22 = preflight_stage22()
    sources=source_hashes()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source_snapshot(OUTPUT / 'source_snapshot.zip',sources)
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        stage22_summary_sha256=digest(STAGE22 / 'summary.json'),
        long_anchor_elapsed_s=1., long_anchor_time_tolerance_s=1e-9,
        code_blend=0., baseline_phase_leave_one_out=True,
        candidate_phase_long_anchor=True,cases=CASES22)),
        indent=2,allow_nan=False),encoding='utf-8')
    reports={}
    plots={}
    for name,observations,_,_,metadata,input_hashes in saved_cases(stage19,stage21):
        with np.load(STAGE22 / f'{name}.npz') as saved:
            reference=dict(saved)
        short,long=replay_pair(observations,reference)
        count=len(observations['time'])-20
        prefix=navigate(navigation_prefix(observations,count),diagnostics=True,
                        code_blend=0.,phase_leave_one_out=True,
                        phase_long_anchor=True)
        for key,value in prefix.items():
            if key=='diagnostics':
                for diagnostic,data in value.items():
                    np.testing.assert_array_equal(data,long[key][diagnostic][:count])
            elif key!='initialization_time':
                np.testing.assert_array_equal(value,long[key][:count])
        modes,errors,common=score_pair(observations,metadata,short,long)
        modes={'short':modes['off'],'long':modes['on']}
        used=long['diagnostics']['phase_long_path']=='long'
        post=observations['available_time']>=metadata['case']['attack_start']
        anchor_horizontal,anchor_3d=anchor_errors(observations,long,metadata)
        modes['long']['long_used_post_epochs']=int((used&post).sum())
        modes['long']['long_used_all_epochs']=int(used.sum())
        modes['long']['long_excluded_prn_counts']={
            str(prn):int((used&post&(long['diagnostics']['phase_long_excluded_prn']==prn)).sum())
            for prn in np.unique(long['diagnostics']['phase_long_excluded_prn'][used&post])}
        modes['long']['anchor_horizontal_error_m']=finite_summary(anchor_horizontal)
        modes['long']['anchor_3d_error_m']=finite_summary(anchor_3d)
        modes['long']['short_to_long_horizontal_jump_m']=finite_summary(
            long['diagnostics']['phase_long_horizontal_jump_m'])
        modes['long']['short_to_long_3d_jump_m']=finite_summary(
            long['diagnostics']['phase_long_jump_m'])
        truth=np.column_stack((metadata['case']['velocity_east']*observations['time'],
                               np.zeros((len(observations['time']),2))))
        three_d_errors={label:np.linalg.norm(nav['position']-truth,axis=1)
                        for label,nav in (('short',short),('long',long))}
        modes['short']['common_3d_error_m']=finite_summary(three_d_errors['short'][common])
        modes['long']['common_3d_error_m']=finite_summary(three_d_errors['long'][common])
        arrays=dict(observation_time=observations['time'],
                    available_time=observations['available_time'],
                    common_published=common,
                    long_anchor_horizontal_error_m=anchor_horizontal,
                    long_anchor_3d_error_m=anchor_3d)
        arrays.update({f'geometry_{key}':value for key,value in
                       long_geometry(observations,long).items()})
        fault_comparison=None
        if name.startswith('fault_'):
            mask=fault_window_mask(observations)
            fault_comparison,_=compare_fault(short,long,observations,metadata,mask)
            arrays['fault_window']=mask
        for label,navigation in (('short',short),('long',long)):
            arrays[f'{label}_horizontal_error_m']=errors['off' if label=='short' else 'on']
            arrays[f'{label}_3d_error_m']=three_d_errors[label]
            arrays[f'{label}_vertical_error_m']=navigation['position'][:,2]
            for key,value in navigation.items():
                if key=='diagnostics':
                    for diagnostic,data in value.items():
                        arrays[f'{label}_diagnostic_{diagnostic}']=data
                else:
                    arrays[f'{label}_{key}']=np.nan if value is None else value
        path=OUTPUT / f'{name}.npz'
        np.savez_compressed(path,**arrays)
        reports[name]=dict(**input_hashes,
                           stage22_reference_sha256=stage22['cases'][name]['result_sha256'],
                           result_sha256=digest(path),modes=modes)
        if fault_comparison is not None:
            reports[name]['fault_window']=fault_comparison
        plots[name]=(observations['available_time'].copy(),
                     {label:errors['off' if label=='short' else 'on'].copy()
                      for label in ('short','long')})
        print(name,'done',flush=True)

    fig,axes=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
    for axis,name in zip(axes.flat,CASES22):
        available,errors=plots[name]
        for label,values in errors.items():
            axis.plot(available,values,label=label)
        axis.axhline(20.,color='gray',linestyle=':',linewidth=1)
        axis.set(title=name,xlabel='Output available time (s)',
                 ylabel='Published horizontal error (m)')
        axis.grid(alpha=.3)
    axes[0,0].legend()
    fig.savefig(OUTPUT / 'overview.png',dpi=150)
    plt.close(fig)
    assert source_hashes()==sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__,source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        figure_sha256=digest(OUTPUT / 'overview.png'),cases=reports)),
        indent=2,allow_nan=False),encoding='utf-8')
    return reports


if __name__=='__main__':
    main()
