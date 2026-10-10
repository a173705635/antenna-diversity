"""Replay saved single-source observations and audit navigation publication gates.

Runs all selected weak three/four-antenna rows and the no-attack control.
The archived RF is never loaded; each replay is checked against its saved
navigation arrays before diagnostics are reported.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from core.single_source import navigate
from experiments.paths import RESULTS, reset_output


SOURCE = RESULTS/'mechanism-four'/'summary.json'
OUTPUT = 'single_source_diagnosis'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def selected_rows(summary):
    return [row for row in summary['selected_rows']
            if row['subset'] in ('three', 'four') and
            ('_weak_' in row['input'] or '_control_' in row['input'])]


def outage_events(reason, published, start):
    """First missed publication and first >=3-epoch uninterrupted outage."""
    missed=np.flatnonzero((np.arange(len(published))>=start)&~published)
    first=None if len(missed)==0 else int(missed[0])
    sustained=None
    for k in missed:
        if k+2<len(published) and not published[k:k+3].any():
            sustained=int(k)
            break
    return dict(first_epoch=first,first_reason=None if first is None else str(reason[first]),
                sustained_epoch=sustained,
                sustained_reason=None if sustained is None else str(reason[sustained]))


def replay(row, output):
    root=Path(__file__).resolve().parents[1]
    folder=root/row['after_root'].replace('\\','/')/'evaluations'/row['after_id']
    tracking=folder/'tracking.npz'; saved_navigation=folder/'navigation.npz'
    with np.load(tracking) as observations:
        navigation=navigate(observations,diagnostics=True)
        observation_time=observations['time'].copy()
        available_time=observations['available_time'].copy()
    with np.load(saved_navigation) as saved:
        for key in saved.files:
            np.testing.assert_array_equal(navigation[key],saved[key],err_msg=f'{row["after_id"]}: {key}')
    assert navigation['initialization_time']==row['after']['initialization_time']
    detail=navigation['diagnostics']
    published=np.isfinite(navigation['position']).all(axis=1)
    candidate=np.isfinite(navigation['candidate_position']).all(axis=1)
    attack_start=row['after']['attack_start_s']
    post=available_time>=attack_start
    assert int(published[post].sum())==row['after']['solved']
    name=Path(row['input'].replace('\\','/')).name+'_'+row['subset']
    np.savez_compressed(output/f'{name}.npz',observation_time=observation_time,
                        available_time=available_time,published=published,candidate=candidate,
                        position=navigation['position'],candidate_position=navigation['candidate_position'],
                        **detail)
    reasons,counts=np.unique(detail['reason'][post],return_counts=True)
    separations=detail['code_separation_m'][post]
    finite_separation=separations[np.isfinite(separations)]
    ages=detail['mean_smoothing_age'][post]
    events=outage_events(detail['reason'],published,int(np.flatnonzero(post)[0]))
    for event in ('first_epoch','sustained_epoch'):
        k=events[event]
        events[event.replace('epoch','observation_time_s')]=None if k is None else float(observation_time[k])
        events[event.replace('epoch','available_time_s')]=None if k is None else float(available_time[k])
    first_published=np.flatnonzero(published)
    return dict(name=name,input=row['input'],subset=row['subset'],after_id=row['after_id'],
                original_source_sha256=row['after']['source_sha256'],
                original_input_sha256=row['after']['input_sha256'],
                tracking_sha256=digest(tracking),navigation_sha256=digest(saved_navigation),
                epochs=int(len(available_time)),post_epochs=int(post.sum()),
                attack_start_s=attack_start,availability_delay_s=float(np.median(available_time-observation_time)),
                initialization_available_time_s=navigation['initialization_time'],
                first_published_available_time_s=None if len(first_published)==0 else float(available_time[first_published[0]]),
                published=int(published[post].sum()),candidates=int(candidate[post].sum()),
                mean_raw=float(detail['raw_count'][post].mean()),
                mean_selected=float(detail['selected_count'][post].mean()),
                mean_unanchored_selected_common_arc=float(detail['unanchored_selected_common_arc_count'][post].mean()),
                mean_anchored_selected_common_arc=float(detail['anchored_selected_common_arc_count'][post].mean()),
                mean_smoothing_age=float(np.nanmean(ages)),
                finite_code_separation_epochs=int(len(finite_separation)),
                code_separation_over_20=int((finite_separation>20).sum()),
                median_code_separation_m=None if len(finite_separation)==0 else float(np.median(finite_separation)),
                phase_ok=int(detail['phase_ok'][post].sum()),
                isolated=int(detail['isolated'][post].sum()),
                recovery_confirmed=int(detail['recovery_confirmed'][post].sum()),
                max_recovery_attempt_count=int(detail['recovery_attempt_count'][post].max()),
                reasons=dict(zip(reasons.tolist(),counts.tolist())),events=events)


def main():
    summary=json.loads(SOURCE.read_text(encoding='utf-8'))
    rows=selected_rows(summary)
    output=reset_output(OUTPUT)
    results=[replay(row,output) for row in rows]
    report=dict(scope=__doc__,source_sha256=source_hashes(),
                selected_summary_sha256=digest(SOURCE),rows=results)
    (output/'summary.json').write_text(json.dumps(serializable(report),indent=2,allow_nan=False),encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    labels=[row['name'] for row in results]
    x=np.arange(len(labels))
    axes[0].bar(x-.2,[r['mean_raw'] for r in results],.4,label='Finite code ranges')
    axes[0].bar(x+.2,[r['mean_selected'] for r in results],.4,label='Selected after quality gates')
    axes[0].set(ylabel='Mean satellites / epoch')
    axes[0].legend()
    axes[1].bar(x-.2,[r['published']/r['post_epochs'] for r in results],.4,label='Published')
    axes[1].bar(x+.2,[r['phase_ok']/r['post_epochs'] for r in results],.4,label='Phase prediction')
    axes[1].set(ylabel='Fraction of post-attack epochs',xticks=x,xticklabels=labels)
    axes[1].tick_params(axis='x',rotation=40)
    axes[1].legend()
    for axis in axes:axis.grid(axis='y',alpha=.3)
    figure.savefig(output/'summary.png',dpi=150)
    plt.close(figure)
    for row in results:print(row['name'],row['published'],'/',row['post_epochs'],row['reasons'],flush=True)


if __name__=='__main__':
    main()
