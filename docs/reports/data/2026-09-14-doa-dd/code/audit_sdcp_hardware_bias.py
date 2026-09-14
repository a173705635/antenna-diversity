"""Fixed antenna/RF errors through dual RF/DFE/loops; no modulo in evaluated SDCP."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audit_dual_tracking_phase import run_case
from positioning import carrier_range
from presets import default_config
from run_gps_spp import serializable


CASES = {
    'zero': (0., 0., 0., 0.),
    'antenna_amplitude': (-1., 0., 0., 0.),
    'antenna_phase': (0., 5., 0., 0.),
    'rf_amplitude': (0., 0., 1., 0.),
    'rf_phase': (0., 0., 0., 3.),
    'combined': (-1., 5., 1., 3.),
}


def stats(values):
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    if not x.size:
        return {'epochs': 0}
    return {'epochs': int(x.size), 'mean_mm': float(x.mean()*1000),
            'std_mm': float(x.std()*1000),
            'rms_mm': float(np.sqrt(np.mean(x*x))*1000),
            'max_abs_mm': float(np.max(abs(x))*1000)}


def evaluate(case, times):
    """Reconstruct raw ranges directly, ignoring the legacy wrapped output."""
    ranges, qualities = [], []
    for ch in case['channels']:
        h = dict(ch, success=True)
        ranges.append(carrier_range(h, times, case['fs_hz'], .001, case['wavelength_m']))
        starts = np.array(ch['epoch_start_samples'])/case['fs_hz']
        ix = np.searchsorted(starts, times, side='right')-1
        safe = np.clip(ix, 0, len(starts)-1)
        q = (np.array(ch['phase_lock_metric']) >= .8) & (abs(np.array(ch['pll_error_rad'])) < np.pi/2)
        qualities.append((ix >= 0) & (times < starts[safe]+.001) & q[safe])
    d = ranges[1]-ranges[0]
    # This generator's geometric NCO phase uses -k*b.u; range has opposite sign.
    g = -case['ideal_geometry_path_m']
    return d, d-g, qualities[0] & qualities[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (out/name).write_text(json.dumps(serializable(obj), indent=2, allow_nan=False), encoding='utf-8')
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(__file__).parent.glob('*.py')}
    save('manifest.json', {'scope': __doc__, 'duration_ms': 1000,
        'seed_pairs': [[9900, 9901], [9902, 9903]], 'cases_TR': CASES,
        'parameter_order': ['antenna_db', 'antenna_deg', 'rf_db', 'rf_deg'],
        'parameters_are': 'Synthetic sensitivity settings, NOT measured hardware specifications.',
        'reference': 'No reference centering, branch rounding, phase unwrap or modulo in evaluated ranges.',
        'source_sha256': source})
    times = np.arange(200, 999)*.001
    rows, traces = [], {'time_s': times}
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for seed, ax in zip((9900, 9902), axes):
        control = None
        for label, (ant_db, ant_deg, rf_db, rf_deg) in CASES.items():
            cfg = default_config()
            cfg.phone.active_antennas = ('TL', 'TR')
            cfg.signal.duration_ms = cfg.tracking.max_epochs = 1000
            cfg.random_seed = seed
            cfg.interferers = []
            cfg.signal.multipath_paths = []
            cfg.phone.antenna_model.true_state.amplitude_error_db = {'TR': ant_db}
            cfg.phone.antenna_model.true_state.phase_error_deg = {'TR': ant_deg}
            cfg.rf.amplitude_error_db = (0., rf_db)
            cfg.rf.phase_error_deg = (0., rf_deg)
            config_record = asdict(cfg)
            config_record['plot']['output_dir'] = str(cfg.plot.output_dir)
            save(f'{seed}_{label}_config.json', config_record)
            print(f'RUN {seed} {label}', flush=True)
            case = run_case(duration_ms=1000, seed_base=seed, receiver_config=cfg)
            save(f'{seed}_{label}.json', case)
            row = {'seed': seed, 'case': label, 'status': case['status']}
            if case['status'] != 'evaluated_not_lock_certified':
                rows.append(row)
                continue
            d, error, quality = evaluate(case, times)
            if label == 'zero':
                control = error.copy()
            paired = error-control if control is not None else np.full(error.shape, np.nan)
            row.update({'geometry_range_m': -case['ideal_geometry_path_m'],
                'expected_injected_range_mm': -(ant_deg+rf_deg)/360*case['wavelength_m']*1000,
                'raw_sdcp': stats(d), 'geometry_residual': stats(error),
                'paired_change': stats(paired), 'quality_pass': int(quality.sum()),
                'finite_epochs': int(np.isfinite(error).sum()),
                'max_adjacent_step_mm': float(np.nanmax(abs(np.diff(error)))*1000),
                'saturation_fraction': [ch['saturation_fraction'] for ch in case['channels']]})
            rows.append(row)
            traces[f'{seed}_{label}_residual_m'] = error
            traces[f'{seed}_{label}_quality'] = quality
            ax.plot(times, error*1000, lw=.7, label=label)
            print(json.dumps(row), flush=True)
        ax.set(title=f'Seed pair {seed}/{seed+1}; no modulo or centering',
               xlabel='Time (s)', ylabel='D - geometry (mm)')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, ncol=3)
    save('summary.json', {'statistics': 'Population std (ddof=0), all finite points 200..998 ms; quality reported separately.',
                          'rows': rows})
    np.savez_compressed(out/'traces.npz', **traces)
    fig.savefig(out/'residuals.png', dpi=150)
    plt.close(fig)
    assert source == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path(__file__).parent.glob('*.py')}, 'source changed during audit'
    save('sha256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})


if __name__ == '__main__':
    main()
