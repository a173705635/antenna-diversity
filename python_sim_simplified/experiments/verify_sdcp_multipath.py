"""Independently recompute saved NCO decisions, then render the experiment evidence."""
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.sdcp_support import verify_sources, plot_case_scores

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.paths import RESULTS

OUTPUT = RESULTS/'sdcp_multipath'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def phase_distance(path, times, fs, wavelength):
    with np.load(path) as h:
        starts = h['epoch_start_samples']/fs
        assert times[0] >= starts[0] and times[-1] < starts[-1]+.001
        ix = np.searchsorted(starts,times,side='right')-1
        phase = h['carrier_nco_phase_rad'][ix]+2*np.pi*h['carrier_nco_frequency_hz'][ix]*(times-starts[ix])
    return -phase*wavelength/(2*np.pi)


def plot_paired_changes(traces, seed, out):
    fig=plt.figure(figsize=(14,18),layout='constrained')
    panels=fig.add_gridspec(3,2,hspace=.12)
    names=('mp_late_strong','mp_both_diff','mp_medium',
           'mp_late_weak','mp_long','mp_other_direction')
    for panel,name in zip(panels,names):
        rows=panel.subgridspec(3,1,hspace=.05)
        upper=fig.add_subplot(rows[0])
        middle=fig.add_subplot(rows[1],sharex=upper,sharey=upper)
        lower=fig.add_subplot(rows[2],sharex=upper)
        d=traces[name,seed]
        upper.plot(d['time_s'],d['paired_A_mm'],color='tab:blue',lw=2,
                   label='A: change from LOS-only reference')
        middle.plot(d['time_s'],d['paired_B_mm'],color='tab:orange',lw=1.2,ls='--',
                   label='B: change from LOS-only reference')
        lower.plot(d['time_s'],d['paired_B_mm']-d['paired_A_mm'],color='tab:green',lw=.9,
                   label='SDCP change = B change - A change')
        upper.set(title=name,ylabel='A change (mm)')
        middle.set_ylabel('B change (mm)')
        upper.tick_params(labelbottom=False)
        middle.tick_params(labelbottom=False)
        lower.set(xlabel='Time (s)',ylabel='SDCP change (mm)')
        for ax in (upper,middle,lower):
            ax.axvline(.6,color='gray',ls=':',lw=.8)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7,loc='upper right')
    fig.suptitle(f'Changes from matched-noise LOS-only reference | seed {seed}\n'
                 'Offline comparison only; not detector inputs. Dotted line: echo onset at 0.6 s.')
    fig.savefig(out/'paired_carrier_changes.png',dpi=160)
    plt.close(fig)


def main():
    out = OUTPUT
    plan, cfg, threshold, completion, rows = [read(out/name) for name in
        ('plan.json','config.json','threshold.json','completion.json','results.json')]
    assert completion['records'] == len(rows) == len(plan['cases'])*len(plan['test_seeds'])
    assert hashlib.sha256((out/'threshold.json').read_bytes()).hexdigest() == completion['threshold_sha256']
    source_mode = verify_sources(out, plan['source_sha256'])
    fs = cfg['signal']['fs_rf_hz']/cfg['dfe']['decimation']
    wavelength = 299792458./cfg['signal']['carrier_hz']
    geom = threshold['geometry_m']
    limit = threshold['threshold_m']
    calibrated = []
    calibration_diagnostics = []
    traces = {}
    for seed in plan['calibration_seeds']:
        with np.load(out/f'cal_{seed}_decision.npz') as d:
            times = d['time_s']
            recomputed = (phase_distance(out/f'cal_{seed}_B.npz',times,fs,wavelength)
                -phase_distance(out/f'cal_{seed}_A.npz',times,fs,wavelength)-geom)
            np.testing.assert_allclose(recomputed,d['residual_m'],atol=1e-10,rtol=0)
            means = np.array([np.mean(recomputed[j-199:j+1]) for j in range(199,len(times),10)])
            np.testing.assert_allclose(means,d['score_m'],atol=1e-10,rtol=0)
            calibrated.append(float(np.max(abs(means))))
            with np.load(out/f'cal_{seed}_A.npz') as a, np.load(out/f'cal_{seed}_B.npz') as b:
                selection=(a['epoch_start_samples']/fs>=.2) & (b['epoch_start_samples']/fs>=.2)
                calibration_diagnostics.append(dict(seed=seed,
                    mean_mm=float(np.mean(recomputed)*1000),std_mm=float(np.std(recomputed)*1000),
                    max_abs_mean_mm=float(np.max(abs(means))*1000),
                    prompt_cross_angle_deg=float(np.angle(np.sum(b['prompt'][selection]*np.conj(a['prompt'][selection])),deg=True)),
                    prompt_epoch_offset_samples=np.unique(b['epoch_start_samples']-a['epoch_start_samples']).tolist()))
    np.testing.assert_allclose(limit,plan['margin']*max(calibrated),atol=1e-10,rtol=0)
    (out/'calibration_diagnostics.json').write_text(json.dumps(calibration_diagnostics,indent=2),encoding='utf-8')
    enriched = []
    case_map = {c['name']:c for c in plan['cases']}
    for row in rows:
        name, seed = row['case'],row['seed']
        c = case_map[name]
        a_file = out/(f'{seed}_{name}_A.npz' if c.get('channels')=='both' else f'{seed}_clean_A.npz')
        with np.load(out/f'{seed}_{name}_decision.npz') as data:
            d = {k:data[k] for k in data.files}
        t = d['time_s']
        a = phase_distance(a_file,t,fs,wavelength)
        b = phase_distance(out/f'{seed}_{name}_B.npz',t,fs,wavelength)
        r = b-a-geom
        np.testing.assert_allclose(r,d['residual_m'],atol=1e-10,rtol=0)
        means = np.array([np.mean(r[j-199:j+1]) for j in range(199,len(t),10)])
        dt = t[np.arange(199,len(t),10)]
        np.testing.assert_array_equal(dt,d['decision_time_s'])
        np.testing.assert_allclose(means,d['score_m'],atol=1e-10,rtol=0)
        alarm = np.array([j>=2 and all(abs(means[j-2:j+1])>limit) for j in range(len(means))])
        np.testing.assert_array_equal(alarm,d['alarm'])
        onset = c.get('onset_s',.4 if name=='drift' else 0.)
        event = dt>=onset
        fired = dt[event & alarm]
        assert bool(len(fired)) == row['detected']
        assert int(np.sum(event & alarm)) == row['event_alarm_count']
        assert int(np.sum(event)) == row['event_decisions']
        assert int(np.sum(~event & alarm)) == row['pre_event_alarm_count']
        if len(fired):
            assert fired[0] == row['first_alarm_s']
            assert fired[0]-onset == row['delay_s']
        for mask,key in ((t>=1.1,'tail_residual'),(t>=.2,'all_residual')):
            x=r[mask]*1000
            for k,v in dict(mean_mm=np.mean(x),std_mm=np.std(x),rms_mm=np.sqrt(np.mean(x*x)),max_abs_mm=np.max(abs(x))).items():
                np.testing.assert_allclose(v,row[key][k],atol=1e-7,rtol=0)
        np.testing.assert_allclose(np.mean(means[dt>=1.1])*1000,row['tail_score_mean_mm'],atol=1e-7,rtol=0)
        quality = np.ones(len(t),bool)
        for file in (a_file,out/f'{seed}_{name}_B.npz'):
            with np.load(file) as h:
                ix = np.searchsorted(h['epoch_start_samples']/fs,t,side='right')-1
                quality &= (h['phase_lock_metric'][ix]>=.8) & (abs(h['pll_error_rad'][ix])<np.pi/2)
        np.testing.assert_array_equal(quality,d['quality'])
        assert np.mean(quality) == row['quality_fraction']
        clean_a = phase_distance(out/f'{seed}_clean_A.npz',t,fs,wavelength)
        clean_b = phase_distance(out/f'{seed}_los_B.npz',t,fs,wavelength)
        pa,pb = (a-clean_a)*1000,(b-clean_b)*1000
        enriched.append(dict(row,paired_A_tail_mm=float(np.mean(pa[t>=1.1])),
            paired_B_tail_mm=float(np.mean(pb[t>=1.1]))))
        d['paired_A_mm']=pa
        d['paired_B_mm']=pb
        traces[name,seed] = d
    (out/'verified_results.json').write_text(json.dumps(enriched,indent=2,allow_nan=False),encoding='utf-8')
    change_dir=out/'window_change'
    change_threshold=read(change_dir/'threshold.json')
    change_rows=read(change_dir/'results.json')
    change_max=[]
    for seed in plan['calibration_seeds']:
        with np.load(out/f'cal_{seed}_decision.npz') as d:
            averaged=np.convolve(d['residual_m'],np.ones(200)/200,'valid')
            change_max.append(np.max(abs((averaged[200:]-averaged[:-200])[::10])))
    change_limit=change_threshold['threshold_m']
    np.testing.assert_allclose(change_limit,1.25*max(change_max),atol=1e-12,rtol=0)
    for row in change_rows:
        name,seed=row['case'],row['seed']
        d=traces[name,seed]
        averaged=np.convolve(d['residual_m'],np.ones(200)/200,'valid')
        s=(averaged[200:]-averaged[:-200])[::10]
        dt=d['time_s'][399::10]
        alarm=np.zeros(len(s),bool)
        alarm[2:]=np.convolve((abs(s)>change_limit).astype(int),np.ones(3,int),'valid')==3
        with np.load(change_dir/f'{seed}_{name}.npz') as saved:
            np.testing.assert_array_equal(dt,saved['time_s'])
            np.testing.assert_allclose(s,saved['score_m'],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(alarm,saved['alarm'])
        c=case_map[name]
        onset=c.get('onset_s',.4 if name=='drift' else 0.)
        event=dt>=onset
        fired=dt[event & alarm]
        assert bool(len(fired))==row['detected']
        assert int(np.sum(event & alarm))==row['event_alarm_count']
        assert int(np.sum(event))==row['event_decisions']
        assert int(np.sum(~event & alarm))==row['pre_event_alarm_count']
        if len(fired):
            assert fired[0]==row['first_alarm_s'] and fired[0]-onset==row['delay_s']
        np.testing.assert_allclose(np.max(abs(s))*1000,row['max_abs_score_mm'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(np.mean(s[dt>=1.1])*1000,row['tail_score_mean_mm'],atol=1e-8,rtol=0)
    plot_case_scores(plan,
        {key:(d['decision_time_s'],d['score_m'],d['alarm']) for key,d in traces.items()},
        limit,'200 ms mean (mm)','Absolute SDCP residual',out/'all_cases.png')
    selected=('los','mp_late_strong','mp_long','mp_both_diff','hardware','drift')
    fig,axs=plt.subplots(3,2,figsize=(13,11))
    for ax,name in zip(axs.flat,selected):
        d=traces[name,plan['test_seeds'][0]]
        ax.plot(d['time_s'],d['residual_m']*1000,alpha=.35,lw=.7,label='1 ms residual')
        ax.plot(d['decision_time_s'],d['score_m']*1000,label='200 ms causal mean')
        ax.scatter(d['decision_time_s'][d['alarm']],d['score_m'][d['alarm']]*1000,s=8,color='red',label='Confirmed alarm')
        ax.axhline(limit*1000,color='black',ls='--',lw=.7)
        ax.axhline(-limit*1000,color='black',ls='--',lw=.7)
        ax.set(title=name,xlabel='Time (s)',ylabel='Residual (mm)')
        ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'representative_traces.png',dpi=160)
    plt.close(fig)
    plot_paired_changes(traces,plan['test_seeds'][0],out)
    checks = dict(source_verification=source_mode, records_verified=len(rows),calibrations_verified=len(calibrated),
        source_files_verified=len(plan['source_sha256']),decision_arrays_verified=len(rows)+len(calibrated),
        window_change_records_verified=len(change_rows),
        approach='Separate reconstruction of NCO phase, explicit window means and confirmation loops; quality and metrics checked.',
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    lines=['| 场景 | 报警记录数 | 事件后报警判定数/判定数 | 首次报警延迟 ms | 尾段平均 SDCP mm | A/B 配对载波变化 mm |',
        '|---|---:|---:|---|---|---|']
    for c in plan['cases']:
        rs=[r for r in enriched if r['case']==c['name']]
        delays=' / '.join('—' if r['delay_s'] is None else f"{r['delay_s']*1000:.0f}" for r in rs)
        tail=' / '.join(f"{r['tail_score_mean_mm']:.3f}" for r in rs)
        pair=' / '.join(f"{r['paired_A_tail_mm']:.2f},{r['paired_B_tail_mm']:.2f}" for r in rs)
        lines.append(f"| {c['name']} | {sum(r['detected'] for r in rs)}/3 | {sum(r['event_alarm_count'] for r in rs)}/{sum(r['event_decisions'] for r in rs)} | {delays} | {tail} | {pair} |")
    (out/'result_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(threshold_mm=limit*1000,verification=checks,elapsed_s=completion['seconds']),indent=2))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
