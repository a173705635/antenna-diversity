"""Compare default and geometry checked 105 ms history.

All 26 mechanism-four selected rows are replayed. New RF is generated only
when RUN_NEW_RF is enabled after the fixed plan has been reviewed.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import platform

import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from core.spoofing.single_source import navigate
from core.spoofing.single_source import frontend
from core.spoofing.single_source_scene import Case, generate
from core.spoofing.recovery_geometry import FS
from experiments.paths import SPOOFING_RESULTS as RESULTS, reset_output
from experiments.spoofing.single_source_diagnosis import SOURCE, digest


OUTPUT = 'single_source_history_study'
CANDIDATE_GAP_S = .105
RUN_NEW_RF = True
NEW_RF_PLAN = [
    dict(seed=2026092401, duration_s=4., scenarios=['weak','no_attack_control','genuine_absent'],
         antennas=[3,4], satellites=24, startup_cn0_db_hz=45.,
         post_0p3s_cn0_db_hz=30., attack_start_s=.3,
         spoof_db=10.,angle_deg=60.,fake_east_m=100.),
    dict(seed=2026092402, duration_s=10., scenarios=['weak'],antennas=[4],
         satellites=24, startup_cn0_db_hz=45., post_0p3s_cn0_db_hz=30.,
         attack_start_s=.3,spoof_db=10.,angle_deg=60.,fake_east_m=100.),
]


def longest_gap(published):
    misses=np.concatenate(([0],(~published).astype(int),[0]))
    starts=np.flatnonzero(np.diff(misses)==1)
    ends=np.flatnonzero(np.diff(misses)==-1)
    return int((ends-starts).max()) if len(starts) else 0


def metrics(navigation,time,attack_start,velocity_east):
    published=np.isfinite(navigation['position']).all(axis=1)
    truth_east=velocity_east*time
    error=np.linalg.norm(navigation['position'][:,:2]-np.column_stack((truth_east,np.zeros(len(time)))),axis=1)
    post=time>=attack_start
    finite=error[post][np.isfinite(error[post])]
    tail=time>=time[-1]-.5
    return dict(opportunities=int(post.sum()),solved=int(published[post].sum()),
                good20=int((finite<=20).sum()),error30=int((finite>=30).sum()),
                error50=int((finite>=50).sum()),
                p95_error_m=None if not len(finite) else float(np.percentile(finite,95)),
                max_error_m=None if not len(finite) else float(finite.max()),
                longest_unavailable_s=longest_gap(published[post])*.01,
                pre_attack_published=int(published[~post].sum()),
                tail_published=int(published[tail].sum()),tail_opportunities=int(tail.sum()),
                initialization_available_time_s=navigation['initialization_time'])


def replay(row,output):
    root=Path(__file__).resolve().parents[2]
    input_folder=root/row['input'].replace('\\','/')
    metadata=json.loads((input_folder/'metadata.json').read_text(encoding='utf-8'))
    folder=root/row['after_root'].replace('\\','/')/'evaluations'/row['after_id']
    tracking=folder/'tracking.npz';saved_navigation=folder/'navigation.npz'
    with np.load(tracking) as observations:
        baseline=navigate(observations)
        candidate=navigate(observations,diagnostics=True,history_gap_s=CANDIDATE_GAP_S)
        time=observations['available_time'].copy()
    with np.load(saved_navigation) as saved:
        for key in saved.files:
            np.testing.assert_array_equal(baseline[key],saved[key],err_msg=f'{row["after_id"]}: {key}')
    assert baseline['initialization_time']==row['after']['initialization_time']
    attack_start=row['after']['attack_start_s']
    velocity=metadata['case']['velocity_east']
    before=metrics(baseline,time,attack_start,velocity)
    after=metrics(candidate,time,attack_start,velocity)
    for key in ('opportunities','solved','good20','error30','error50'):
        assert before[key]==row['after'][key],(row['after_id'],key)
    detail=candidate['diagnostics']
    extended=detail['history_extended']
    gaps=detail['history_extension_gap_s'][extended]
    name=Path(row['input'].replace('\\','/')).name+'_'+row['subset']
    np.savez_compressed(output/f'{name}.npz',available_time=time,
                        baseline_position=baseline['position'],candidate_position=candidate['position'],
                        baseline_selected=baseline['selected'],candidate_selected=candidate['selected'],
                        history_extended=extended,history_extension_gap_s=detail['history_extension_gap_s'])
    return dict(name=name,input=row['input'],subset=row['subset'],after_id=row['after_id'],
                tracking_sha256=digest(tracking),navigation_sha256=digest(saved_navigation),
                input_metadata_sha256=digest(input_folder/'metadata.json'),
                original_source_sha256=row['after']['source_sha256'],
                original_input_sha256=row['after']['input_sha256'],
                baseline=before,candidate=after,
                extended_satellite_epochs=int(extended.sum()),
                extended_epochs=int(extended.any(axis=1).sum()),
                extension_gap_s=None if not len(gaps) else dict(
                    minimum=float(gaps.min()),median=float(np.median(gaps)),maximum=float(gaps.max())))


def generate_new_record(plan,scenario,nav_bit_epoch_s=0.,fake_rate=0.):
    case=Case(duration=plan['duration_s'],stars=plan['satellites'],
              cn0=plan['startup_cn0_db_hz'],attack_start=plan['attack_start_s'],
              phase_deg=10.,amplitude_db=1.,velocity_east=2.,
              spoof_db=plan['spoof_db'],angle=plan['angle_deg'],fake_east=plan['fake_east_m'],
              genuine=scenario!='genuine_absent',
              nav_bit_epoch_s=nav_bit_epoch_s,fake_rate=fake_rate)
    attacked=scenario!='no_attack_control'
    samples,nominal,codes,times,satellites,radii=generate(case,plan['seed'],attacked)
    noise_rng=np.random.default_rng(plan['seed']+10_000_000)
    noise_scale=np.sqrt(FS*(10**(-plan['post_0p3s_cn0_db_hz']/10)-
                            10**(-plan['startup_cn0_db_hz']/10))/2)
    start=round(plan['attack_start_s']*FS)
    block=round(.1*FS)
    for begin in range(start,samples.shape[1],block):
        end=min(begin+block,samples.shape[1])
        shape=samples[:,begin:end].shape
        samples[:,begin:end]+=noise_scale*(noise_rng.normal(size=shape)+1j*noise_rng.normal(size=shape))
    rf_sha256=hashlib.sha256(memoryview(samples).cast('B')).hexdigest()
    record=dict(samples=samples,nominal=nominal,codes=np.array(codes),times=times,
                satellites=satellites,coarse=np.round((radii+75000)/10000)*10000)
    identity=dict(seed=plan['seed'],scenario=scenario,case=asdict(case),attacked=attacked,
                  post_cn0_db_hz=plan['post_0p3s_cn0_db_hz'],noise_seed=plan['seed']+10_000_000,
                  noise_block_s=.1,rf_sha256=rf_sha256)
    return record,identity


def new_rf_runs(output):
    results=[]
    for plan in NEW_RF_PLAN:
        for scenario in plan['scenarios']:
            print('RF start',plan['seed'],scenario,flush=True)
            record,identity=generate_new_record(plan,scenario)
            label=f"{plan['seed']}_{scenario}_{plan['duration_s']:g}s"
            folder=output/label
            folder.mkdir()
            metadata_path=folder/'metadata.json'
            metadata_path.write_text(json.dumps(serializable(identity),indent=2),encoding='utf-8')
            for antennas in plan['antennas']:
                print('  antennas start',antennas,flush=True)
                observations=frontend(record,list(range(antennas)))
                baseline=navigate(observations)
                candidate=navigate(observations,diagnostics=True,history_gap_s=CANDIDATE_GAP_S)
                tracking_path=folder/f'tracking_{antennas}.npz'
                navigation_path=folder/f'navigation_{antennas}.npz'
                np.savez_compressed(tracking_path,**observations)
                np.savez_compressed(navigation_path,
                                    baseline_position=baseline['position'],
                                    baseline_selected=baseline['selected'],
                                    candidate_position=candidate['position'],
                                    candidate_selected=candidate['selected'],
                                    history_extended=candidate['diagnostics']['history_extended'],
                                    history_extension_gap_s=candidate['diagnostics']['history_extension_gap_s'])
                time=observations['available_time']
                before=metrics(baseline,time,plan['attack_start_s'],identity['case']['velocity_east'])
                after=metrics(candidate,time,plan['attack_start_s'],identity['case']['velocity_east'])
                result=dict(label=label,antennas=antennas,rf_sha256=identity['rf_sha256'],
                            metadata_sha256=digest(metadata_path),
                            tracking_sha256=digest(tracking_path),
                            navigation_sha256=digest(navigation_path),
                            baseline=before,candidate=after,
                            extended_satellite_epochs=int(candidate['diagnostics']['history_extended'].sum()))
                results.append(result)
                print('  antennas done',antennas,before['solved'],'->',after['solved'],flush=True)
            del record
            print('RF done',plan['seed'],scenario,flush=True)
    return results


def main():
    summary=json.loads(SOURCE.read_text(encoding='utf-8'))
    output=reset_output(RESULTS/OUTPUT)
    initial_source=source_hashes()
    plan=dict(candidate_gap_s=CANDIDATE_GAP_S,new_rf_plan=NEW_RF_PLAN,
              run_new_rf=RUN_NEW_RF,noise_method='independent complex Gaussian in 100 ms blocks',
              python=platform.python_version(),numpy=np.__version__)
    plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode('utf-8')).hexdigest()
    (output/'manifest.json').write_text(json.dumps(serializable(dict(
        source_sha256=initial_source,plan=plan,plan_sha256=plan_sha256,
        selected_summary_sha256=digest(SOURCE))),indent=2),encoding='utf-8')
    rows=[replay(row,output) for row in summary['selected_rows']]
    new_rf=[]
    if RUN_NEW_RF:
        assert source_hashes()==initial_source
        new_rf=new_rf_runs(output)
    assert source_hashes()==initial_source
    report=dict(scope=__doc__,source_sha256=initial_source,plan_sha256=plan_sha256,
                selected_summary_sha256=digest(SOURCE),candidate_gap_s=CANDIDATE_GAP_S,
                new_rf_plan=NEW_RF_PLAN,new_rf_executed=RUN_NEW_RF,
                new_rf_results=new_rf,rows=rows)
    (output/'summary.json').write_text(json.dumps(serializable(report),indent=2,allow_nan=False),encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    x=np.arange(len(rows));labels=[r['name'] for r in rows]
    axes[0].bar(x,[r['candidate']['solved']-r['baseline']['solved'] for r in rows])
    axes[0].set(ylabel='Change in published epochs')
    axes[1].bar(x,[0 if r['candidate']['max_error_m'] is None else r['candidate']['max_error_m'] for r in rows],label='105 ms')
    axes[1].scatter(x,[0 if r['baseline']['max_error_m'] is None else r['baseline']['max_error_m'] for r in rows],s=15,c='black',label='25 ms')
    axes[1].set(ylabel='Maximum horizontal error (m)',xticks=x,xticklabels=labels)
    axes[1].tick_params(axis='x',rotation=60)
    axes[1].legend()
    for axis in axes:axis.grid(axis='y',alpha=.3)
    fig.savefig(output/'summary.png',dpi=150)
    plt.close(fig)
    if new_rf:
        fig,axis=plt.subplots(figsize=(9,4),constrained_layout=True)
        labels=[f"{r['label']} {r['antennas']}A" for r in new_rf]
        x=np.arange(len(new_rf))
        axis.bar(x-.2,[r['baseline']['solved'] for r in new_rf],.4,label='25 ms')
        axis.bar(x+.2,[r['candidate']['solved'] for r in new_rf],.4,label='105 ms')
        axis.set(ylabel='Published epochs',xticks=x,xticklabels=labels)
        axis.tick_params(axis='x',rotation=40)
        axis.legend();axis.grid(axis='y',alpha=.3)
        fig.savefig(output/'new_rf_summary.png',dpi=150)
        plt.close(fig)
    for row in rows:
        print(row['name'],row['baseline']['solved'],'->',row['candidate']['solved'],
              'max',row['baseline']['max_error_m'],'->',row['candidate']['max_error_m'],flush=True)


if __name__=='__main__':
    main()
