"""Paired 2–4 antenna spatial-detection experiment; observation model, not RF."""
import argparse
import hashlib
import json
import platform
from os import fspath
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from presets import mate70_phase_centers
from spatial_spoofing import spatial_scores, calibrate_threshold

W = 299792458 / 1575.42e6
LAYOUTS = {'2_short': [0, 1], '2_diagonal': [0, 3], '3': [0, 1, 2], '4': [0, 1, 2, 3]}
NAMES = ['TL', 'TR', 'BL', 'BR']
METHODS = ('global', 'cluster3')
CASES = [('los', False), ('all_spoof', True), ('partial_2', True), ('partial_3', True),
         ('partial_4', True), ('partial_6', True), ('two_sources', True),
         ('mixture_-6', True), ('mixture_0', True), ('mixture_3', True),
         ('mixture_6', True), ('mixture_12', True), ('single_mp', False),
         ('clustered_los', False), ('short_aperture_los', False),
         ('short_aperture_spoof', True), ('low_signal_los', False),
         ('low_signal_spoof', True), ('prn_bias_spoof', True),
         ('bad_fourth_spoof', True), ('common_reflection', False),
         ('spatial_matched', True)]


def model_observations(seed, n, case, *, noise_m=.003 / np.sqrt(2), positions_m=None):
    """Independent 100 ms events, 10 correlated epochs, four physical antennas.

    Spatial carrier phasors are mixed BEFORE angle extraction. This models
    perfectly overlapping correlator components, not DLL/PLL takeover. Each
    antenna has independent AR(1) complex noise (rho=.7); shared baselines thus
    inherit their real covariance. Truth is used only here, never by detector.
    """
    if case not in dict(CASES):
        raise ValueError('unknown model case')
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 2 * np.pi, (n, 8))
    up = rng.uniform(np.sin(np.deg2rad(15)), 1, (n, 8))
    direction = np.stack((np.sqrt(1-up**2)*np.cos(az), np.sqrt(1-up**2)*np.sin(az), up), -1)
    if case == 'clustered_los':
        direction[:, :3] = direction[:, :1] + rng.normal(0, .01, (n, 3, 3))
        direction[:, :3] /= np.linalg.norm(direction[:, :3], axis=-1, keepdims=True)
    rotation = Rotation.random(n, random_state=rng).as_matrix()
    direction = np.einsum('nij,nsj->nsi', rotation, direction)
    geometry = mate70_phase_centers() if positions_m is None else positions_m
    positions = np.array([geometry[key] for key in NAMES], dtype=float)
    positions -= positions[0]
    if case.startswith('short_aperture'):
        positions *= .1
    genuine = -np.einsum('mc,nsc->nms', positions, direction)
    source = rng.normal(size=(n, 2, 3)); source /= np.linalg.norm(source, axis=-1, keepdims=True)
    fake = -np.einsum('mc,nkc->nmk', positions, source)
    phase = genuine.copy()
    if case in ('all_spoof', 'short_aperture_spoof', 'low_signal_spoof',
                'prn_bias_spoof', 'bad_fourth_spoof'):
        phase[:] = fake[:, :, :1]
    elif case.startswith('partial_'):
        phase[:, :, :int(case.split('_')[1])] = fake[:, :, :1]
    elif case == 'two_sources':
        phase[:, :, :4] = fake[:, :, :1]; phase[:, :, 4:] = fake[:, :, 1:]
    if case == 'prn_bias_spoof':
        phase += rng.normal(0, .02, phase.shape)
    z = np.exp(-2j * np.pi * phase / W)
    if case.startswith('mixture_') or case == 'common_reflection':
        db = float(case.split('_')[1]) if case.startswith('mixture_') else 10.
        # Genuine and fake are aligned at TL. Other antenna offsets differ.
        z += 10**(db/20) * np.exp(-2j * np.pi * fake[:, :, :1] / W)
    if case == 'single_mp':
        z[:, :, :1] += .6 * np.exp(-2j*np.pi*fake[:, :, :1]/W + .7j)
    if case.startswith('low_signal'):
        noise_m = .02
    sigma = np.full((1, 1, 4, 1), noise_m * 2*np.pi/W)
    if case == 'bad_fourth_spoof':
        sigma[:, :, 3] = .04 * 2*np.pi/W
    noise = rng.normal(size=(n, 10, 4, 8)) + 1j*rng.normal(size=(n, 10, 4, 8))
    for k in range(1, 10):
        noise[:, k] = .7 * noise[:, k-1] + np.sqrt(1-.7**2)*noise[:, k]
    x = -np.angle(z[:, None] + noise*sigma)*W/(2*np.pi)
    # Unknown common receiver phase/drift and satellite phase origins cancel.
    x += rng.normal(0, .2, (n, 10, 4, 1)) + rng.normal(0, 2., (n, 10, 1, 8))
    x += rng.integers(-10, 11, (n, 1, 4, 8)) * W/2
    return x


def wilson(hits, total):
    z = 1.959963984540054
    p = hits/total; den = 1+z*z/total
    center = (p+z*z/(2*total))/den
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [max(0., center-half), min(1., center+half)]


def paired_gain(a, b):
    """Paired-event mean delta and approximate normal 95% interval (not Wilson)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError('aligned independent-event decisions required')
    d = b.astype(float)-a
    delta = float(d.mean()); half = float(1.959963984540054*d.std(ddof=1)/np.sqrt(len(d)))
    return {'delta': delta, 'normal95': [max(-1., delta-half), min(1., delta+half)],
            'gained': int(np.sum(~a & b)), 'lost': int(np.sum(a & ~b))}


def score_events(seed, n, case, *, positions_m=None):
    arrays = {f'{layout}_{method}': [] for layout in LAYOUTS for method in METHODS}
    for chunk, start in enumerate(range(0, n, 250)):
        x = model_observations(seed+chunk, min(250, n-start), case, positions_m=positions_m)
        for layout, indices in LAYOUTS.items():
            scores = spatial_scores(x[:, :, indices], W)
            for method, values in scores.items():
                arrays[f'{layout}_{method}'].append(values)
    return {key: np.concatenate(value) for key, value in arrays.items()}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=fspath), encoding='utf-8')


def plot_results(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = ['los', 'all_spoof', 'partial_2', 'partial_3', 'partial_4', 'two_sources',
                'mixture_0', 'mixture_3', 'mixture_6', 'low_signal_spoof', 'prn_bias_spoof']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    for ax, method in zip(axes, METHODS):
        for j, layout in enumerate(LAYOUTS):
            chosen = [next(r for r in rows if r['case'] == case and r['layout'] == layout and r['method'] == method)
                      for case in selected]
            p = np.array([r['fraction'] for r in chosen]); ci = np.array([r['wilson95'] for r in chosen])
            ax.errorbar(np.arange(len(selected)) + (j-1.5)*.12, p*100,
                        yerr=np.maximum(0, np.stack((p-ci[:, 0], ci[:, 1]-p)))*100,
                        marker='o', ms=4, lw=1, label=layout)
        ax.set_xticks(range(len(selected)), selected, rotation=65, ha='right')
        ax.set_ylim(-3, 103); ax.set_ylabel('Alert fraction (%)'); ax.set_title(method)
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Independent observation-model events; each configuration calibrated to 1% nominal H0')
    fig.savefig(out/'detection_comparison.png', dpi=160); plt.close(fig)
    stress = ['single_mp', 'clustered_los', 'short_aperture_los', 'low_signal_los', 'common_reflection', 'spatial_matched']
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for j, layout in enumerate(LAYOUTS):
        vals = [next(r['fraction'] for r in rows if r['case'] == case and r['layout'] == layout and r['method'] == 'cluster3') for case in stress]
        ax.bar(np.arange(len(stress)) + (j-1.5)*.19, np.array(vals)*100, .19, label=layout)
    ax.set_xticks(range(len(stress)), stress, rotation=20, ha='right'); ax.set_ylim(0, 105)
    ax.set_ylabel('Alert fraction (%)'); ax.legend(); ax.grid(axis='y', alpha=.2)
    ax.set_title('Boundary cases at frozen nominal thresholds (last column is an attack)')
    fig.savefig(out/'boundaries.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-events', type=int, default=5000)
    parser.add_argument('--events', type=int, default=5000)
    parser.add_argument('--width-m', type=float, default=.0759)
    parser.add_argument('--height-m', type=float, default=.1609)
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.width_m, args.height_m)):
        parser.error('array width and height must be finite and positive')
    x, y = args.width_m / 2, args.height_m / 2
    geometry = dict(zip(NAMES, [(-x, y, 0.), (x, y, 0.), (-x, -y, 0.), (x, -y, 0.)]))
    if min(args.events, args.calibration_events) < 100 or max(args.events, args.calibration_events) > 100000:
        parser.error('event counts must be between 100 and 100000')
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    calibration = score_events(200000, args.calibration_events, 'los', positions_m=geometry)
    thresholds = {key: calibrate_threshold(value) for key, value in calibration.items()}
    np.savez_compressed(out/'calibration.npz', **calibration)
    save_json(out/'thresholds.json', {'target_alpha': .01, 'thresholds': thresholds,
              'calibration_seed_base': 200000, 'events': args.calibration_events,
              'layouts': LAYOUTS, 'antenna_names': NAMES, 'positions_m': geometry, 'prns': 8, 'epochs': 10,
              'wavelength_m': W, 'scope': 'local research; observation-model H0; not GPSPATRON thresholds'})
    rows, gains = [], []
    for i, (case, attack) in enumerate(CASES):
        # Matched attack and LOS deliberately reuse the exact random observations.
        seed = 300000 if case == 'spatial_matched' else 300000 + i*10000
        scores = score_events(seed, args.events, case, positions_m=geometry)
        np.savez_compressed(out/f'{case}.npz', **scores)
        decisions = {key: values < thresholds[key] for key, values in scores.items()}
        for layout in LAYOUTS:
            for method in METHODS:
                key = f'{layout}_{method}'; hits = int(decisions[key].sum())
                rows.append({'case': case, 'is_attack': attack, 'layout': layout, 'method': method,
                    'events': args.events, 'alerts': hits, 'fraction': hits/args.events,
                    'wilson95': wilson(hits, args.events), 'threshold': thresholds[key],
                    'score_quantiles_01_50_99': np.quantile(scores[key], [.01, .5, .99]).tolist(),
                    'seed_base': seed})
        for method in METHODS:
            for a, b in [('2_short', '3'), ('2_diagonal', '3'), ('3', '4'), ('2_short', '4')]:
                gains.append({'case': case, 'comparison': f'{a}->{b}', 'method': method,
                              **paired_gain(decisions[f'{a}_{method}'], decisions[f'{b}_{method}'])})
        print(f'{case}: ' + ', '.join(f'{layout} cluster={decisions[layout+"_cluster3"].mean():.3f}' for layout in LAYOUTS), flush=True)
        save_json(out/'summary.json', {'rows': rows, 'paired_gains': gains})
    manifest = {'scope': __doc__, 'python': platform.python_version(), 'numpy': np.__version__,
        'calibration_events': args.calibration_events, 'evaluation_events_per_case': args.events,
        'positions_m': geometry, 'layouts': LAYOUTS, 'cases': CASES,
        'chunk_events': 250, 'calibration_seed_base': 200000, 'evaluation_seed_base': 300000,
        'seed_stride_per_case': 10000, 'step_s': .01, 'epochs_per_event': 10,
        'noise_m_per_antenna_small_angle': .003/np.sqrt(2), 'temporal_ar1': .7,
        'known_limitations': ['synthetic correlator phasors, not tracking loops', 'fixed 8 PRNs',
            'uniform random sky and orientation is a chosen population', 'no missing data in nominal model',
            'independent antenna noises and common oscillator reference', 'no certified output authenticity'],
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
             [Path(__file__), Path(__file__).with_name('spatial_spoofing.py'), Path(__file__).with_name('presets.py')]}}
    save_json(out/'manifest.json', manifest)
    plot_results(out, rows)
    save_json(out/'sha256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})


if __name__ == '__main__':
    main()
