"""无定位 SDCP 基线：一次运行完成双路仿真、绘图和独立核验。

run_simulation() 生成信号、跟踪和相位响应结果；verify_results() 从保存的
原始 NCO 独立重建观测及统计；main() 按顺序执行。基线不加入回波。
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.antenna import direction_vector
from core.gnss import generate_scenario
from core.presets import default_config
from core.multipath.sdcp_observation import raw_sdcp, receive as receive_observation
from core.provenance import source_hashes, freeze_source
from experiments.multipath.sdcp_support import save_json, statistics, verify_sources
from experiments.paths import MULTIPATH_RESULTS as RESULTS, reset_output

OUTPUT = RESULTS/'sdcp_baseline'
FREEZE_SOURCE = False
DURATION_MS = 1500
TAIL_DURATION_MS = 400


CASES = ('zero', 'fixed_plus10', 'fixed_minus10', 'ramp', 'sine',
         'step', 'gap', 'hardware')
SEEDS = (11600, 11602, 11604)


def analysis_timing(duration_ms):
    """观测止于记录结束前 10 ms；事件时间保持物理定义，不随记录缩放。"""
    times = np.arange(10, duration_ms-10+1)*.001
    intervals = dict(all=[.01, duration_ms/1000], settled=[.2, duration_ms/1000],
                     before_event=[.2, .55], event=[.6, .8],
                     tail=[(duration_ms-TAIL_DURATION_MS)/1000, duration_ms/1000])
    return times, intervals


def region_statistics(values):
    """中断或截短导致没有有效观测时，明确没有统计值，而不是零误差。"""
    if len(values) == 0:
        return dict(epochs=0, status='not_evaluated', reason='no_valid_samples_in_region',
                    mean_mm=None, std_mm=None, rms_mm=None, max_abs_mm=None)
    return dict(statistics(values), status='evaluated')


def phase_degrees(case, t):
    if case == 'fixed_plus10':
        return np.full_like(t, 10.)
    if case == 'fixed_minus10':
        return np.full_like(t, -10.)
    if case == 'ramp':
        return 20.*np.maximum(t-.4, 0.)
    if case == 'sine':
        return 10.*np.sin(4*np.pi*(t-.4))*(t >= .4)
    if case == 'step':
        return 30.*(t >= .6)
    if case == 'hardware':
        return np.full_like(t, 8.)
    return np.zeros_like(t)


def receive(signal, cfg, seed, out, name, hardware=False, *, channel_id):
    h, metadata = receive_observation(signal, cfg, seed, hardware, channel_id=channel_id)
    np.savez_compressed(out/(name+'.npz'), **{k: v for k, v in h.items() if isinstance(v, np.ndarray)})
    save_json(out/(name+'.json'), metadata)
    print(name, h['status'], len(h['epoch_start_samples']), flush=True)
    return h


def run_simulation():
    """生成双路跟踪记录、统计和图表，保存核验所需的数据哈希。"""
    cfg = default_config()
    if not cfg.enable_multipath_research:
        print('Multipath research is disabled in core/config.py; baseline skipped.', flush=True)
        return None
    # 1. 配置无多径接收场景并记录实验来源。
    if DURATION_MS < 30:
        raise ValueError("Baseline needs at least 30 ms for its 10 ms observation start and end margin.")
    times, intervals = analysis_timing(DURATION_MS)
    out = reset_output(OUTPUT)
    (out/'.gitattributes').write_text('** -text\n', encoding='utf-8')
    started = time.perf_counter()
    cfg.phone.active_antennas = ('TL', 'TR')
    cfg.signal.duration_ms = cfg.tracking.max_epochs = DURATION_MS
    cfg.interferers = []
    cfg.signal.multipath_paths = []
    cfg.tracking.carrier_discriminator = 'costas'
    cfg.phone.channel_delay_ns = {k: 0. for k in cfg.phone.channel_delay_ns}
    cfg.phone.estimated_channel_delay_ns = dict(cfg.phone.channel_delay_ns)
    save_json(out/'config.json', asdict(cfg))
    hashes = source_hashes()
    if FREEZE_SOURCE:
        freeze_source(out)
    bits = np.array([1, 1, -1, 1, -1, -1, 1, -1])
    save_json(out/'manifest.json', {'scope': __doc__, 'cases': CASES, 'seeds': SEEDS,
        'duration_s': cfg.signal.duration_ms/1000, 'statistic_intervals_s': intervals, 'data_symbols': bits, 'symbol_duration_s': .020,
        'data_note': 'Synthetic repeated 50 bps symbols, not valid LNAV.',
        'gap_s': [.6, .75], 'step_deg': 30., 'step_start_s': .6,
        'ramp_deg_per_s': 20., 'dynamic_start_s': .4, 'sine_amplitude_deg': 10., 'sine_hz': 2.,
        'hardware_B': {'antenna_db': -1., 'antenna_deg': 5., 'rf_db': 1., 'rf_deg': 3.},
        'phase_policy': 'Raw accumulated NCO, no wrap, no centering, no integer correction.',
        'quality_policy': 'Report only, never used to select statistics.',
        'source_sha256': hashes, 'python_version': sys.version, 'numpy_version': np.__version__})
    # 2. 生成两天线共同时间轴上的信号及合成数据位。
    print(f'Generating {cfg.signal.duration_ms/1000:g} s dual-antenna source', flush=True)
    scene, truth = generate_scenario(cfg)
    del truth
    fs0 = cfg.signal.fs_rf_hz
    for first in range(0, scene.shape[1], 1000000):
        last = min(first+1000000, scene.shape[1])
        t = np.arange(first, last)/fs0
        scene[:, first:last] *= bits[(np.arange(first, last)//round(.020*fs0)) % len(bits)]
    fs = fs0/cfg.dfe.decimation
    wave = cfg.signal.wavelength_m
    baseline = np.array(cfg.phone.phase_centers_m['TR'])-np.array(cfg.phone.phase_centers_m['TL'])
    geometry = float(baseline @ direction_vector(cfg.signal.true_direction))
    masks = {name: (times >= start) & (times < stop)
             for name, (start, stop) in intervals.items()}
    rows, arrays = [], {'time_s': times}
    # 3. 逐种子运行正常 A，逐场景运行 B，并对齐相位与计算响应误差。
    for seed in SEEDS:
        first_h = receive(scene[0], cfg, seed, out, f'{seed}_A', channel_id='TL')
        for case in CASES:
            signal = scene[1].copy()
            for first in range(0, signal.size, 1000000):
                last = min(first+1000000, signal.size)
                t = np.arange(first, last)/fs0
                phase = phase_degrees(case, t)
                if case == 'hardware':
                    signal[first:last] *= 10**(-1/20)*np.exp(1j*np.deg2rad(5.))
                else:
                    signal[first:last] *= np.exp(1j*np.deg2rad(phase))
                if case == 'gap':
                    signal[first:last] *= ~((t >= .6) & (t < .75))
            h = receive(signal, cfg, seed+1, out, f'{seed}_{case}_B', case == 'hardware', channel_id='TR')
            del signal
            d = raw_sdcp(first_h, h, fs, times, wave)
            injected = -wave*phase_degrees(case, times)/360
            expected = geometry+injected
            if case == 'zero':
                control = d.copy()
            quality = np.ones(len(times), dtype=bool)
            for channel in (first_h, h):
                ix = np.searchsorted(channel['epoch_start_samples']/fs, times, side='right')-1
                quality &= (channel['phase_lock_metric'][ix] >= .8) & (abs(channel['pll_error_rad'][ix]) < np.pi/2)
            present = ~((times >= .6) & (times < .75)) if case == 'gap' else np.ones(len(times), bool)
            regions = {region: {'raw_sdcp': region_statistics(d[mask]),
                       'truth_error': region_statistics((d-expected)[mask & present]),
                       'paired_error': region_statistics((d-control-injected)[mask & present]),
                       'quality_pass': int(quality[mask].sum()), 'quality_total': int(mask.sum())}
                       for region, mask in masks.items()}
            row = {'seed': seed, 'case': case, 'regions': regions}
            if case == 'gap':
                row['recovery_status'] = ('observed' if np.any(times >= .75)
                                          else 'not_evaluated_record_ends_before_recovery')
            fit_mask = times >= .55
            recovered_deg = -(d-control)*360/wave
            if case in ('ramp', 'sine'):
                minimum = 2 if case == 'ramp' else 3
                row['fit_status'] = ('evaluated' if fit_mask.sum() >= minimum
                                     else 'not_evaluated_insufficient_fit_samples')
            if case == 'ramp' and row['fit_status'] == 'evaluated':
                slope, intercept = np.polyfit(times[fit_mask]-.4, recovered_deg[fit_mask], 1)
                row['ramp_fit'] = {'slope_deg_s': float(slope), 'intercept_deg': float(intercept)}
            if case == 'sine' and row['fit_status'] == 'evaluated':
                angle = 4*np.pi*(times[fit_mask]-.4)
                design = np.column_stack([np.sin(angle), np.cos(angle), np.ones(len(angle))])
                sine, cosine, dc = np.linalg.lstsq(design, recovered_deg[fit_mask], rcond=None)[0]
                row['sine_fit'] = {'amplitude_deg': float(np.hypot(sine, cosine)),
                    'lag_ms': float(-np.arctan2(cosine, sine)/(4*np.pi)*1000), 'dc_deg': float(dc)}
            rows.append(row)
            label = f'{seed}_{case}'
            arrays[label+'_d_m'] = d
            arrays[label+'_expected_m'] = expected
            arrays[label+'_quality'] = quality
            save_json(out/'summary.json', {'geometry_m': geometry, 'wavelength_m': wave, 'rows': rows})
            np.savez_compressed(out/'aligned.npz', **arrays)
    # 4. 绘制原始几何残差与离线配对响应。
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), constrained_layout=True)
    for case, ax in zip(CASES, axes.flat):
        for seed in SEEDS:
            ax.plot(times, (arrays[f'{seed}_{case}_d_m']-geometry)*1000, lw=.65, label=str(seed))
        ax.plot(times, -wave*phase_degrees(case, times)/360*1000, 'k--', lw=1.4, label='injected reference')
        if case == 'gap':
            ax.axvspan(.6, .75, color='gray', alpha=.2)
        ax.set(title=case, xlabel='Time (s)', ylabel='Raw SDCP - geometry (mm)')
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(out/'sdcp_all.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for seed in SEEDS:
        for ax, case in zip(axes, ('step', 'gap', 'sine')):
            paired = arrays[f'{seed}_{case}_d_m']-arrays[f'{seed}_zero_d_m']
            ax.plot(times, paired*1000, lw=.8, label=str(seed))
    for ax, case in zip(axes, ('step', 'gap', 'sine')):
        ax.plot(times, -wave*phase_degrees(case, times)/360*1000, 'k--', label='injected reference')
        ax.set(title=case+' paired response (offline control comparison)', xlabel='Time (s)', ylabel='Change (mm)')
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(out/'paired_response.png', dpi=160)
    plt.close(fig)
    # 5. completion 只表示仿真阶段完成；核验由下一阶段执行。
    assert hashes == source_hashes()
    assert 'core.positioning' not in sys.modules and 'core.gps_simulation' not in sys.modules
    save_json(out/'completion.json', {'elapsed_s': time.perf_counter()-started,
        'rows': len(rows), 'rf_tracking_runs': len(SEEDS)*(len(CASES)+1),
        'source_unchanged': True, 'positioning_imported': False})
    save_json(out/'sha256.json', {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in out.rglob('*') if p.is_file()})
    return out


def verify_results(root):
    """独立核对文件哈希、NCO 重建、参考注入、统计和动态拟合。"""
    # 1. 数据完整性与源码来源；不改写原始清单以迁就核验。
    hashes = json.loads((root/'sha256.json').read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    manifest = json.loads((root/'manifest.json').read_text())
    source_mode = verify_sources(root, manifest['source_sha256'])
    summary = json.loads((root/'summary.json').read_text())
    cfg = json.loads((root/'config.json').read_text())
    fs = cfg['signal']['fs_rf_hz']/cfg['dfe']['decimation']
    wave = 299792458./cfg['signal']['carrier_hz']
    direction = cfg['signal']['true_direction']
    az, el = np.deg2rad([direction['azimuth_deg'], direction['elevation_deg']])
    vector = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    centers = cfg['phone']['phase_centers_m']
    geometry = (np.array(centers['TR'])-np.array(centers['TL'])) @ vector
    np.testing.assert_allclose(geometry, summary['geometry_m'], atol=1e-14)
    checked_metrics = 0
    fitted_parameters = 0
    # 2. 逐记录重算；不调用生成阶段的观测与统计函数。
    with np.load(root/'aligned.npz') as arrays:
        times = arrays['time_s']
        masks = {name: (times >= start) & (times < stop)
                 for name, (start, stop) in manifest['statistic_intervals_s'].items()}
        for row in summary['rows']:
            seed, case = row['seed'], row['case']
            phase = []
            quality = np.ones(len(times), bool)
            for name in (f'{seed}_A', f'{seed}_{case}_B'):
                metadata = json.loads((root/(name+'.json')).read_text())
                assert metadata['acquisition']['success']
                assert metadata['tracking_status'] == 'tracking'
                with np.load(root/(name+'.npz')) as h:
                    starts = h['epoch_start_samples']/fs
                    ix = np.searchsorted(starts, times, side='right')-1
                    assert np.all(ix >= 0)
                    assert np.all(times < starts[ix]+.001)
                    phase.append(np.array([h['carrier_nco_phase_rad'][i] +
                        2*np.pi*h['carrier_nco_frequency_hz'][i]*(t-starts[i])
                        for t, i in zip(times, ix)]))
                    quality &= (h['phase_lock_metric'][ix] >= .8) & (abs(h['pll_error_rad'][ix]) < np.pi/2)
            reconstructed = (phase[0]-phase[1])*wave/(2*np.pi)
            label = f'{seed}_{case}'
            np.testing.assert_allclose(reconstructed, arrays[label+'_d_m'], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(quality, arrays[label+'_quality'])
            injection = np.zeros(len(times))
            if case in ('fixed_plus10', 'fixed_minus10', 'hardware'):
                injection[:] = {'fixed_plus10': 10., 'fixed_minus10': -10., 'hardware': 8.}[case]
            if case == 'ramp':
                injection[times >= .4] = 20*(times[times >= .4]-.4)
            if case == 'sine':
                injection[times >= .4] = 10*np.sin(2*np.pi*2*(times[times >= .4]-.4))
            if case == 'step':
                injection[times >= .6] = 30
            expected = geometry-injection*wave/360
            np.testing.assert_allclose(expected, arrays[label+'_expected_m'], atol=1e-14, rtol=0)
            fit = times >= .55
            recovered_deg = -(reconstructed-arrays[f'{seed}_zero_d_m'])*360/wave
            if case in ('ramp', 'sine'):
                minimum = 2 if case == 'ramp' else 3
                assert row['fit_status'] == ('evaluated' if fit.sum() >= minimum
                                             else 'not_evaluated_insufficient_fit_samples')
            if case == 'gap':
                assert row['recovery_status'] == ('observed' if np.any(times >= .75)
                                                  else 'not_evaluated_record_ends_before_recovery')
            if case == 'ramp' and row['fit_status'] == 'evaluated':
                design = np.column_stack([times[fit]-.4, np.ones(fit.sum())])
                slope, intercept = np.linalg.solve(design.T @ design, design.T @ recovered_deg[fit])
                np.testing.assert_allclose([slope, intercept],
                    [row['ramp_fit']['slope_deg_s'], row['ramp_fit']['intercept_deg']], atol=1e-8, rtol=0)
                fitted_parameters += 2
            if case == 'sine' and row['fit_status'] == 'evaluated':
                angle = 4*np.pi*(times[fit]-.4)
                design = np.column_stack([np.sin(angle), np.cos(angle), np.ones(fit.sum())])
                sine, cosine, dc = np.linalg.solve(design.T @ design, design.T @ recovered_deg[fit])
                np.testing.assert_allclose([np.hypot(sine, cosine), -np.arctan2(cosine, sine)/(4*np.pi)*1000, dc],
                    [row['sine_fit']['amplitude_deg'], row['sine_fit']['lag_ms'], row['sine_fit']['dc_deg']], atol=1e-8, rtol=0)
                fitted_parameters += 3
            present = ~((times >= .6) & (times < .75)) if case == 'gap' else np.ones(len(times), bool)
            for region, mask in masks.items():
                values = {'raw_sdcp': reconstructed[mask],
                    'truth_error': (reconstructed-expected)[mask & present],
                    'paired_error': (reconstructed-arrays[f'{seed}_zero_d_m']+injection*wave/360)[mask & present]}
                for label, x in values.items():
                    assert np.isfinite(x).all()
                    recorded = row['regions'][region][label]
                    assert len(x) == recorded['epochs']
                    if len(x) == 0:
                        assert recorded['status'] == 'not_evaluated'
                        assert recorded['reason'] == 'no_valid_samples_in_region'
                        assert all(recorded[k] is None for k in ('mean_mm', 'std_mm', 'rms_mm', 'max_abs_mm'))
                        continue
                    assert recorded['status'] == 'evaluated'
                    for metric, value in {'mean_mm': np.mean(x)*1000, 'std_mm': np.std(x)*1000,
                        'rms_mm': np.linalg.norm(x)/np.sqrt(len(x))*1000,
                        'max_abs_mm': np.max(abs(x))*1000}.items():
                        np.testing.assert_allclose(value, recorded[metric], atol=1e-8, rtol=0)
                        checked_metrics += 1
                assert row['regions'][region]['quality_pass'] == int(quality[mask].sum())
    checks = {'source_verification': source_mode, 'hashes_verified': len(hashes), 'records_reconstructed': len(summary['rows']),
                      'metrics_verified': checked_metrics, 'fitted_parameters_verified': fitted_parameters,
                      'common_time_domains_verified': True}
    # 3. 只有全部检查通过，才写核验完成记录。
    save_json(root/'verification.json', checks)
    print(json.dumps(checks))


def main():
    """唯一入口：仿真与绘图完成后，自动执行独立核验。"""
    print('[1/2] Simulating baseline cases and plotting responses...', flush=True)
    output = run_simulation()
    if output is None:
        return
    print('[2/2] Independently verifying saved NCO observations...', flush=True)
    verify_results(output)
    print(f'Baseline experiment and verification complete: {output}', flush=True)


if __name__ == '__main__':
    main()
