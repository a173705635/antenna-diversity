"""双通道无定位多径实验：一次运行完成仿真、独立复算和绘图。

阅读顺序：实验参数 → 场景与检测函数 → run_simulation → verify_and_plot → main。
复算直接读取保存的 NCO 数据，独立计算窗口均值及连续报警，不调用生成阶段
的 scores/confirmed，避免同一个计算错误在生成和核验中重复而未被发现。
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.antenna import build_antenna_model, direction_vector
from core.config import Direction, MultipathPath
from core.gnss import generate_scenario
from core.multipath.scene import _multipath_scene
from core.presets import default_config
from core.multipath.sdcp_observation import raw_sdcp, receive as receive_observation
from core.provenance import source_hashes, freeze_source
from experiments.multipath.sdcp_support import save_json, statistics, verify_sources, plot_case_scores, case_plot_title
from experiments.paths import MULTIPATH_RESULTS as RESULTS, reset_output

# 实验参数：时间以秒计，载波距离以米计；表格和图中转换为毫米。
OUTPUT = RESULTS/'sdcp_multipath'
FREEZE_SOURCE = False


CALIBRATION_SEEDS = (11900, 11902, 11904)
TEST_SEEDS = (12000, 12002, 12004)
DURATION_MS = 1500  # 唯一时长设置；例如改成 700。
OBSERVATION_START_MS = 200
END_MARGIN_MS = 10
TAIL_DURATION_MS = 400
WINDOW = 200
STRIDE = 10
CONFIRM = 3
MARGIN = 1.25


def analysis_timing(duration_ms):
    """共同观测网格留出末端 10 ms；尾段取记录最后 400 ms。"""
    times = np.arange(OBSERVATION_START_MS, duration_ms-END_MARGIN_MS+1)*.001
    tail_start_s = (duration_ms-TAIL_DURATION_MS)*.001
    return times, tail_start_s


def cases():
    result = [dict(name=n, kind=n) for n in ('los', 'hardware', 'drift')]
    specs = [
        ('mp_t0ms_p-3dB_d0p3chip_phi90_az110_el15', -3, .3, 90, 0, 110, 15),
        ('mp_t600ms_p-3dB_d0p3chip_phi90_az110_el15', -3, .3, 90, .6, 110, 15),
        ('mp_t0ms_p-20dB_d0p3chip_phi90_az110_el15', -20, .3, 90, 0, 110, 15),
        ('mp_t600ms_p-20dB_d0p3chip_phi90_az110_el15', -20, .3, 90, .6, 110, 15),
        ('mp_t600ms_p-6dB_d0p05chip_phi90_az110_el15', -6, .05, 90, .6, 110, 15),
        ('mp_t600ms_p-6dB_d1p2chip_phi90_az110_el15', -6, 1.2, 90, .6, 110, 15),
        ('mp_t600ms_p-6dB_d0p3chip_phi0_az110_el15', -6, .3, 0, .6, 110, 15),
        ('mp_t600ms_p-6dB_d0p3chip_phi180_az110_el15', -6, .3, 180, .6, 110, 15),
        ('mp_t600ms_p-6dB_d0p3chip_phi90_az210_el35', -6, .3, 90, .6, 210, 35),
        ('mp_t600ms_p-6dB_d0p3chip_phi90_az110_el15', -6, .3, 90, .6, 110, 15),
    ]
    for name, db, delay, phase, onset, az, el in specs:
        result.append(dict(name=name, kind='multipath', power_db=db,
            delay_chips=delay, reflection_deg=phase, onset_s=onset,
            channels='both', azimuth_deg=az, elevation_deg=el))
    return result


def scores(residual):
    """绝对残差的因果均值；保留固定偏差，不做前后窗口相减。"""
    ends = np.arange(WINDOW-1, len(residual), STRIDE)
    cumulative = np.r_[0., np.cumsum(residual)]
    means = (cumulative[WINDOW:]-cumulative[:-WINDOW])/WINDOW
    return ends, means[::STRIDE]


def confirmed(score, threshold):
    """每 STRIDE 个采样判决一次，连续 CONFIRM 次越限才确认报警。"""
    exceed = np.abs(score) > threshold
    alarm = np.zeros(len(score), dtype=bool)
    count = np.r_[0, np.cumsum(exceed)]
    alarm[CONFIRM-1:] = (count[CONFIRM:]-count[:-CONFIRM]) == CONFIRM
    return alarm


def receive(signal, cfg, seed, out, name, hardware=False, *, channel_id):
    h, metadata = receive_observation(signal, cfg, seed, hardware, channel_id=channel_id)
    np.savez_compressed(out/(name+'.npz'), **{k: v for k, v in h.items() if isinstance(v, np.ndarray)})
    metadata['acquisition'] = {k: v for k, v in metadata['acquisition'].items()
                               if not isinstance(v, np.ndarray)}
    save_json(out/(name+'.json'), metadata)
    print(name, h['status'], len(h['epoch_start_samples']), flush=True)
    return h


def residual(first, second, cfg, geometry, times):
    fs = cfg.signal.fs_rf_hz/cfg.dfe.decimation
    for h in (first, second):
        assert times[0] >= h['epoch_start_samples'][0]/fs
        assert times[-1] < h['epoch_start_samples'][-1]/fs + .001
    return raw_sdcp(first, second, fs, times, cfg.signal.wavelength_m)-geometry


def run_simulation():
    """生成标定和测试数据，冻结门限，保存双路 NCO 与初次判决。"""
    cfg = default_config()
    if not cfg.enable_multipath_research:
        print('Multipath research is disabled in core/config.py; experiment skipped.', flush=True)
        return None
    # 1. 配置与来源记录。重跑只清理本实验的固定输出目录。
    minimum_ms = OBSERVATION_START_MS + WINDOW-1 + STRIDE*(CONFIRM-1) + END_MARGIN_MS
    if DURATION_MS < minimum_ms:
        raise ValueError(f'Multipath detection needs at least {minimum_ms} ms for a complete window and confirmation; got {DURATION_MS} ms.')
    out = reset_output(OUTPUT)
    (out/'.gitattributes').write_text('** -text\n', encoding='utf-8')
    started = time.perf_counter()
    cfg.phone.active_antennas = ('TL', 'TR')
    cfg.signal.duration_ms = cfg.tracking.max_epochs = DURATION_MS
    times, tail_start_s = analysis_timing(cfg.signal.duration_ms)
    cfg.signal.multipath_paths = []
    cfg.interferers = []
    cfg.tracking.carrier_discriminator = 'costas'
    cfg.phone.channel_delay_ns = {k:0. for k in cfg.phone.channel_delay_ns}
    cfg.phone.estimated_channel_delay_ns = dict(cfg.phone.channel_delay_ns)
    save_json(out/'config.json', asdict(cfg))
    hashes = source_hashes()
    if FREEZE_SOURCE:
        freeze_source(out)
    save_json(out/'plan.json', dict(cases=cases(), calibration_seeds=CALIBRATION_SEEDS,
        test_seeds=TEST_SEEDS, window_samples=WINDOW, stride_samples=STRIDE,
        confirmation=CONFIRM, margin=MARGIN, observation_interval_s=.001,
        ignored_start_s=times[0], late_onset_ramp_s=.01,
        duration_s=cfg.signal.duration_ms/1000, tail_start_s=tail_start_s,
        phase_policy='Raw NCO, no wrap, no centering, no integer adjustment.',
        threshold='1.25 times largest absolute calibration window mean; frozen before tests.',
        assumptions='Exact known LOS geometry; zero calibrated hardware offset; controlled initial carrier branch.',
        quality_policy='Diagnostic only; no sample filtering.',
        data_symbols=[1,1,-1,1,-1,-1,1,-1], data_symbol_s=.020,
        data_note='Synthetic 50 bps, not valid navigation words; delayed together with echo waveform.',
        source_sha256=hashes, python=sys.version, numpy=np.__version__))
    # 2. 生成直达波及测试数据位，回波随后从同一波形延迟生成。
    scene, truth = generate_scenario(cfg)
    waveform = truth['desired_waveform']
    del truth
    fs0 = cfg.signal.fs_rf_hz
    bits = np.array([1,1,-1,1,-1,-1,1,-1])
    for start in range(0, waveform.size, 1000000):
        stop = min(start+1000000, waveform.size)
        b = bits[(np.arange(start,stop)//round(.020*fs0)) % len(bits)]
        waveform[start:stop] *= b
        scene[:,start:stop] *= b
    baseline = np.array(cfg.phone.phase_centers_m['TR'])-np.array(cfg.phone.phase_centers_m['TL'])
    geometry = float(baseline @ direction_vector(cfg.signal.true_direction))
    # 3. 独立无多径标定。门限在测试之前确定，测试过程中不调整。
    calibration = []
    for seed in CALIBRATION_SEEDS:
        a = receive(scene[0], cfg, seed, out, f'cal_{seed}_A', channel_id='TL')
        b = receive(scene[1], cfg, seed+1, out, f'cal_{seed}_B', channel_id='TR')
        r = residual(a,b,cfg,geometry,times)
        ends, score = scores(r)
        calibration.append(float(np.max(abs(score))))
        np.savez_compressed(out/f'cal_{seed}_decision.npz', time_s=times,
            residual_m=r, decision_time_s=times[ends], score_m=score)
    threshold = MARGIN*max(calibration)
    save_json(out/'threshold.json', dict(threshold_m=threshold,
        calibration_max_abs_mean_m=calibration, margin=MARGIN, geometry_m=geometry,
        frozen_before_test=True))
    threshold_hash = hashlib.sha256((out/'threshold.json').read_bytes()).hexdigest()
    print('FROZEN THRESHOLD mm:', threshold*1000, flush=True)
    # 4. 逐场景、逐种子通过 RF → DFE → 独立跟踪，保存观测与判决。
    clean_a = {seed:receive(scene[0],cfg,seed,out,f'{seed}_clean_A',channel_id='TL') for seed in TEST_SEEDS}
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    time_s = np.arange(waveform.size)/fs0
    rows = []
    for case in cases():
        name = case['name']
        is_multipath = case['kind'] == 'multipath'
        signal_b = scene[1].copy()
        if is_multipath:
            path = MultipathPath(Direction(case['azimuth_deg'],case['elevation_deg']),
                relative_power_db=case['power_db'], excess_delay_chips=case['delay_chips'],
                reflection_phase_rad=np.deg2rad(case['reflection_deg']))
            echo = _multipath_scene(path,cfg,model,waveform,time_s,np.zeros(2)).truth['component']
            # 所有多径场景都向两个通道加入回波，不提供单路回波选项。
            signal_a = scene[0].copy()
            for start in range(0, waveform.size, 1000000):
                stop = min(start+1000000, waveform.size)
                envelope = (np.ones(stop-start) if case['onset_s'] == 0 else
                    .5-.5*np.cos(np.pi*np.clip((time_s[start:stop]-case['onset_s'])/.01,0,1)))
                signal_b[start:stop] += echo[1,start:stop]*envelope
                signal_a[start:stop] += echo[0,start:stop]*envelope
            del echo
        elif name == 'hardware':
            signal_b *= 10**(-1/20)*np.exp(1j*np.deg2rad(5.))
        elif name == 'drift':
            for start in range(0, waveform.size, 1000000):
                stop = min(start+1000000, waveform.size)
                signal_b[start:stop] *= np.exp(1j*np.deg2rad(20*np.maximum(time_s[start:stop]-.4,0)))
        for seed in TEST_SEEDS:
            a = receive(signal_a,cfg,seed,out,f'{seed}_{name}_A',channel_id='TL') if is_multipath else clean_a[seed]
            b = receive(signal_b,cfg,seed+1,out,f'{seed}_{name}_B',name=='hardware',channel_id='TR')
            r = residual(a,b,cfg,geometry,times)
            ends, score = scores(r)
            dt = times[ends]
            alarm = confirmed(score,threshold)
            onset = case.get('onset_s',.4 if name=='drift' else 0.)
            event = dt >= onset
            first_alarm = dt[alarm & event]
            quality = np.ones(len(times),bool)
            fs = fs0/cfg.dfe.decimation
            for h in (a,b):
                ix = np.searchsorted(h['epoch_start_samples']/fs,times,side='right')-1
                quality &= (h['phase_lock_metric'][ix]>=.8) & (abs(h['pll_error_rad'][ix])<np.pi/2)
            np.savez_compressed(out/f'{seed}_{name}_decision.npz', time_s=times,
                residual_m=r, decision_time_s=dt, score_m=score, alarm=alarm, quality=quality)
            row = dict(case=name, seed=seed,
                assessment_status='evaluated' if np.any(event) else 'not_evaluated_event_outside_record',
                detected=bool(len(first_alarm)) if np.any(event) else None,
                full_post_event_window=bool(np.any(dt >= onset+(WINDOW-1)*.001)),
                first_alarm_s=float(first_alarm[0]) if len(first_alarm) else None,
                delay_s=float(first_alarm[0]-onset) if len(first_alarm) else None,
                event_alarm_count=int(np.sum(alarm & event)), event_decisions=int(np.sum(event)),
                pre_event_alarm_count=int(np.sum(alarm & ~event)),
                tail_score_mean_mm=float(np.mean(score[dt>=tail_start_s])*1000),
                tail_residual=statistics(r[times>=tail_start_s]), all_residual=statistics(r),
                quality_fraction=float(np.mean(quality)))
            rows.append(row)
            save_json(out/'results.json', rows)
            print('RESULT',name,seed,row['detected'],row['delay_s'],flush=True)
        del signal_b
        if is_multipath:
            del signal_a
    # 5. 生成阶段完整性检查；completion.json 仅表示仿真阶段结束。
    assert threshold_hash == hashlib.sha256((out/'threshold.json').read_bytes()).hexdigest()
    assert hashes == source_hashes()
    assert 'core.positioning' not in sys.modules and 'core.gps_simulation' not in sys.modules
    save_json(out/'completion.json',dict(records=len(rows),seconds=time.perf_counter()-started,
        threshold_sha256=threshold_hash,source_unchanged=True,no_positioning_import=True))
    return out


# 独立复算与图表：读取落盘数据，而不是复用仿真中的临时计算结果。
def read_json(path):
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
    names=('mp_t600ms_p-3dB_d0p3chip_phi90_az110_el15','mp_t600ms_p-6dB_d0p3chip_phi90_az110_el15','mp_t600ms_p-6dB_d0p3chip_phi0_az110_el15',
           'mp_t600ms_p-20dB_d0p3chip_phi90_az110_el15','mp_t600ms_p-6dB_d1p2chip_phi90_az110_el15','mp_t600ms_p-6dB_d0p3chip_phi90_az210_el35')
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
        upper.set(title=case_plot_title(name),ylabel='A change (mm)')
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


def verify_and_plot(out):
    """独立重建观测、核对判决及统计，成功后生成核验记录和图表。"""
    # 1. 核对实验来源、记录数量和冻结门限。
    plan, cfg, threshold, completion, rows = [read_json(out/name) for name in
        ('plan.json','config.json','threshold.json','completion.json','results.json')]
    assert completion['records'] == len(rows) == len(plan['cases'])*len(plan['test_seeds'])
    assert hashlib.sha256((out/'threshold.json').read_bytes()).hexdigest() == completion['threshold_sha256']
    source_mode = verify_sources(out, plan['source_sha256'])
    tail_start_s = plan['tail_start_s']
    window = plan['window_samples']
    stride = plan['stride_samples']
    confirmation = plan['confirmation']
    fs = cfg['signal']['fs_rf_hz']/cfg['dfe']['decimation']
    wavelength = 299792458./cfg['signal']['carrier_hz']
    geom = threshold['geometry_m']
    limit = threshold['threshold_m']
    # 2. 从标定 NCO 重新计算残差和绝对均值门限。
    calibrated = []
    calibration_diagnostics = []
    traces = {}
    for seed in plan['calibration_seeds']:
        with np.load(out/f'cal_{seed}_decision.npz') as d:
            times = d['time_s']
            recomputed = (phase_distance(out/f'cal_{seed}_B.npz',times,fs,wavelength)
                -phase_distance(out/f'cal_{seed}_A.npz',times,fs,wavelength)-geom)
            np.testing.assert_allclose(recomputed,d['residual_m'],atol=1e-10,rtol=0)
            means = np.array([np.mean(recomputed[j-window+1:j+1]) for j in range(window-1,len(times),stride)])
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
    # 3. 从每条测试 NCO 独立复算观测、报警、质量诊断及配对变化。
    enriched = []
    case_map = {c['name']:c for c in plan['cases']}
    for row in rows:
        name, seed = row['case'],row['seed']
        c = case_map[name]
        a_file = out/(f'{seed}_{name}_A.npz' if c['kind']=='multipath' else f'{seed}_clean_A.npz')
        with np.load(out/f'{seed}_{name}_decision.npz') as data:
            d = {k:data[k] for k in data.files}
        t = d['time_s']
        a = phase_distance(a_file,t,fs,wavelength)
        b = phase_distance(out/f'{seed}_{name}_B.npz',t,fs,wavelength)
        r = b-a-geom
        np.testing.assert_allclose(r,d['residual_m'],atol=1e-10,rtol=0)
        means = np.array([np.mean(r[j-window+1:j+1]) for j in range(window-1,len(t),stride)])
        dt = t[np.arange(window-1,len(t),stride)]
        np.testing.assert_array_equal(dt,d['decision_time_s'])
        np.testing.assert_allclose(means,d['score_m'],atol=1e-10,rtol=0)
        alarm = np.array([j>=confirmation-1 and all(abs(means[j-confirmation+1:j+1])>limit) for j in range(len(means))])
        np.testing.assert_array_equal(alarm,d['alarm'])
        onset = c.get('onset_s',.4 if name=='drift' else 0.)
        event = dt>=onset
        fired = dt[event & alarm]
        assert (bool(len(fired)) if np.any(event) else None) == row['detected']
        assert row['assessment_status'] == ('evaluated' if np.any(event) else 'not_evaluated_event_outside_record')
        assert row['full_post_event_window'] == bool(np.any(dt >= onset+(window-1)*plan['observation_interval_s']))
        assert int(np.sum(event & alarm)) == row['event_alarm_count']
        assert int(np.sum(event)) == row['event_decisions']
        assert int(np.sum(~event & alarm)) == row['pre_event_alarm_count']
        if len(fired):
            assert fired[0] == row['first_alarm_s']
            assert fired[0]-onset == row['delay_s']
        for mask,key in ((t>=tail_start_s,'tail_residual'),(t>=plan['ignored_start_s'],'all_residual')):
            x=r[mask]*1000
            for k,v in dict(mean_mm=np.mean(x),std_mm=np.std(x),rms_mm=np.sqrt(np.mean(x*x)),max_abs_mm=np.max(abs(x))).items():
                np.testing.assert_allclose(v,row[key][k],atol=1e-7,rtol=0)
        np.testing.assert_allclose(np.mean(means[dt>=tail_start_s])*1000,row['tail_score_mean_mm'],atol=1e-7,rtol=0)
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
        enriched.append(dict(row,paired_A_tail_mm=float(np.mean(pa[t>=tail_start_s])),
            paired_B_tail_mm=float(np.mean(pb[t>=tail_start_s]))))
        d['paired_A_mm']=pa
        d['paired_B_mm']=pb
        traces[name,seed] = d
    (out/'verified_results.json').write_text(json.dumps(enriched,indent=2,allow_nan=False),encoding='utf-8')
    # 4. 所有记录核对完成后，输出绝对残差图和分轴载波变化图。
    plot_case_scores(plan,
        {key:(d['decision_time_s'],d['score_m'],d['alarm']) for key,d in traces.items()},
        limit,'200 ms mean (mm)','Absolute SDCP residual',out/'all_cases.png')
    selected=('los','mp_t600ms_p-3dB_d0p3chip_phi90_az110_el15','mp_t600ms_p-6dB_d1p2chip_phi90_az110_el15','mp_t600ms_p-6dB_d0p3chip_phi90_az110_el15','hardware','drift')
    fig,axs=plt.subplots(3,2,figsize=(13,11))
    for ax,name in zip(axs.flat,selected):
        d=traces[name,plan['test_seeds'][0]]
        ax.plot(d['time_s'],d['residual_m']*1000,alpha=.35,lw=.7,label='1 ms residual')
        ax.plot(d['decision_time_s'],d['score_m']*1000,label='200 ms causal mean')
        ax.scatter(d['decision_time_s'][d['alarm']],d['score_m'][d['alarm']]*1000,s=8,color='red',label='Confirmed alarm')
        ax.axhline(limit*1000,color='black',ls='--',lw=.7)
        ax.axhline(-limit*1000,color='black',ls='--',lw=.7)
        ax.set(title=case_plot_title(name),xlabel='Time (s)',ylabel='Residual (mm)')
        ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'representative_traces.png',dpi=160)
    plt.close(fig)
    plot_paired_changes(traces,plan['test_seeds'][0],out)
    # 5. 保存核验凭据与便于阅读的统计表。
    checks = dict(source_verification=source_mode, records_verified=len(rows),calibrations_verified=len(calibrated),
        source_files_verified=len(plan['source_sha256']),decision_arrays_verified=len(rows)+len(calibrated),
        approach='Separate reconstruction of NCO phase, explicit window means and confirmation loops; quality and metrics checked.',
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    lines=['| 场景 | 报警/已评估记录数（0/0 表示未评估） | 事件后报警判定数/判定数 | 首次报警延迟 ms | 尾段平均 SDCP mm | A/B 配对载波变化 mm |',
        '|---|---:|---:|---|---|---|']
    for c in plan['cases']:
        rs=[r for r in enriched if r['case']==c['name']]
        delays=' / '.join('—' if r['delay_s'] is None else f"{r['delay_s']*1000:.0f}" for r in rs)
        tail=' / '.join(f"{r['tail_score_mean_mm']:.3f}" for r in rs)
        pair=' / '.join(f"{r['paired_A_tail_mm']:.2f},{r['paired_B_tail_mm']:.2f}" for r in rs)
        lines.append(f"| {c['name']} | {sum(r['detected'] is True for r in rs)}/{sum(r['detected'] is not None for r in rs)} | {sum(r['event_alarm_count'] for r in rs)}/{sum(r['event_decisions'] for r in rs)} | {delays} | {tail} | {pair} |")
    (out/'result_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(threshold_mm=limit*1000,verification=checks,elapsed_s=completion['seconds']),indent=2))
    print('\n'.join(lines))


def main():
    """唯一运行入口：生成 → 独立复算 → 图表和统计输出。"""
    print('[1/2] Simulating calibration and multipath cases...', flush=True)
    output = run_simulation()
    if output is None:
        return
    print('[2/2] Independently verifying observations and rendering plots...', flush=True)
    verify_and_plot(output)
    print(f'Experiment and verification complete: {output}', flush=True)


if __name__ == '__main__':
    main()
