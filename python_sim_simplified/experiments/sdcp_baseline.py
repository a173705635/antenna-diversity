"""Dual RF/DFE/tracking SDCP experiment, with no positioning dependency."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.antenna import direction_vector
from core.gnss import generate_scenario
from core.presets import default_config
from core.sdcp_observation import raw_sdcp, receive as receive_observation
from core.provenance import source_hashes, freeze_source
from experiments.sdcp_support import save_json, statistics


CASES = ('zero', 'fixed_plus10', 'fixed_minus10', 'ramp', 'sine',
         'step', 'gap', 'hardware')
SEEDS = (11600, 11602, 11604)


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


def receive(signal, cfg, seed, out, name, hardware=False):
    h, metadata = receive_observation(signal, cfg, seed, hardware)
    np.savez_compressed(out/(name+'.npz'), **{k: v for k, v in h.items() if isinstance(v, np.ndarray)})
    save_json(out/(name+'.json'), metadata)
    print(name, h['status'], len(h['epoch_start_samples']), flush=True)
    return h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze-source', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'.gitattributes').write_text('** -text\n', encoding='utf-8')
    started = time.perf_counter()
    cfg = default_config()
    cfg.phone.active_antennas = ('TL', 'TR')
    cfg.signal.duration_ms = cfg.tracking.max_epochs = 1500
    cfg.interferers = []
    cfg.signal.multipath_paths = []
    cfg.tracking.carrier_discriminator = 'costas'
    cfg.phone.channel_delay_ns = {k: 0. for k in cfg.phone.channel_delay_ns}
    cfg.phone.estimated_channel_delay_ns = dict(cfg.phone.channel_delay_ns)
    save_json(out/'config.json', asdict(cfg))
    hashes = source_hashes()
    if args.freeze_source:
        freeze_source(out)
    bits = np.array([1, 1, -1, 1, -1, -1, 1, -1])
    save_json(out/'manifest.json', {'scope': __doc__, 'cases': CASES, 'seeds': SEEDS,
        'duration_s': 1.5, 'data_symbols': bits, 'symbol_duration_s': .020,
        'data_note': 'Synthetic repeated 50 bps symbols, not valid LNAV.',
        'gap_s': [.6, .75], 'step_deg': 30., 'step_start_s': .6,
        'ramp_deg_per_s': 20., 'dynamic_start_s': .4, 'sine_amplitude_deg': 10., 'sine_hz': 2.,
        'hardware_B': {'antenna_db': -1., 'antenna_deg': 5., 'rf_db': 1., 'rf_deg': 3.},
        'phase_policy': 'Raw accumulated NCO, no wrap, no centering, no integer correction.',
        'quality_policy': 'Report only, never used to select statistics.',
        'source_sha256': hashes, 'python_version': sys.version, 'numpy_version': np.__version__})
    print('Generating 1.5 s dual-antenna source', flush=True)
    scene, truth = generate_scenario(cfg)
    del truth
    fs0 = cfg.signal.fs_rf_hz
    for first in range(0, scene.shape[1], 1000000):
        last = min(first+1000000, scene.shape[1])
        t = np.arange(first, last)/fs0
        scene[:, first:last] *= bits[(np.arange(first, last)//round(.020*fs0)) % len(bits)]
    fs = fs0/cfg.dfe.decimation
    times = np.arange(10, 1491)*.001
    wave = cfg.signal.wavelength_m
    baseline = np.array(cfg.phone.phase_centers_m['TR'])-np.array(cfg.phone.phase_centers_m['TL'])
    geometry = float(baseline @ direction_vector(cfg.signal.true_direction))
    masks = {'all': times >= .01, 'settled': times >= .2,
             'before_event': (times >= .2) & (times < .55),
             'event': (times >= .6) & (times < .8), 'tail': times >= 1.1}
    rows, arrays = [], {'time_s': times}
    for seed in SEEDS:
        first_h = receive(scene[0], cfg, seed, out, f'{seed}_A')
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
            h = receive(signal, cfg, seed+1, out, f'{seed}_{case}_B', case == 'hardware')
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
            regions = {region: {'raw_sdcp': statistics(d[mask]),
                       'truth_error': statistics((d-expected)[mask & present]),
                       'paired_error': statistics((d-control-injected)[mask & present]),
                       'quality_pass': int(quality[mask].sum()), 'quality_total': int(mask.sum())}
                       for region, mask in masks.items()}
            row = {'seed': seed, 'case': case, 'regions': regions}
            fit_mask = times >= .55
            recovered_deg = -(d-control)*360/wave
            if case == 'ramp':
                slope, intercept = np.polyfit(times[fit_mask]-.4, recovered_deg[fit_mask], 1)
                row['ramp_fit'] = {'slope_deg_s': float(slope), 'intercept_deg': float(intercept)}
            if case == 'sine':
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
    assert hashes == source_hashes()
    assert 'core.positioning' not in sys.modules and 'core.gps_simulation' not in sys.modules
    save_json(out/'completion.json', {'elapsed_s': time.perf_counter()-started,
        'rows': len(rows), 'rf_tracking_runs': len(SEEDS)*(len(CASES)+1),
        'source_unchanged': True, 'positioning_imported': False})
    save_json(out/'sha256.json', {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in out.rglob('*') if p.is_file()})
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
