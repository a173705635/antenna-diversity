"""One-second, multiple-seed LOS-only dual-loop baseline with fixed windows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audit_dual_tracking_phase import run_case, statistics


def summarize_case(case):
    if case['status'] != 'evaluated_not_lock_certified':
        return {'status': case['status']}
    time = np.asarray(case['time_s'])
    phase = np.asarray(case['phase_difference_rad'])
    if (not len(time) or not np.all(np.isfinite(time)) or
            not np.all(np.isfinite(phase))):
        return {'status': 'nonfinite_tracking_output'}
    for channel in case['channels']:
        for key in ('carrier_nco_phase_rad', 'carrier_nco_frequency_hz', 'pll_error_rad'):
            if not np.all(np.isfinite(channel[key])):
                return {'status': 'nonfinite_tracking_output'}
    expected = np.radians(case['ideal_geometry_phase_deg'])
    def window(start, stop):
        selected = (time >= start) & (time < stop)
        if not np.any(selected):
            return None
        return statistics(phase[selected], case['wavelength_m'], expected)
    end = float(time[-1]+.001)
    result = {'status': 'evaluated_not_lock_certified',
              'time_start_s': float(time[0]), 'time_end_s': float(time[-1]),
              'all_epochs': window(0., end),
              'post_200ms': window(.2, end), 'post_500ms': window(.5, end),
              'windows_100ms': [dict(start_ms=i*100, statistics=window(i*.1, (i+1)*.1))
                                for i in range(int(np.ceil(end/.1)))],
              'phase_step_gt_90deg_count': int(np.sum(abs(np.diff(phase)) > np.pi/2)),
              'post_200ms_phase_step_gt_90deg_count': int(np.sum(
                  (abs(np.diff(phase)) > np.pi/2) & (time[:-1] >= .2))),
              'channel_diagnostics': []}
    for channel in case['channels']:
        t = np.array(channel['epoch_start_samples'])/case['fs_hz']
        mask = t >= .2
        pll = np.array(channel['pll_error_rad'])
        frequency = np.array(channel['carrier_nco_frequency_hz'])
        result['channel_diagnostics'].append({
            'channel': channel['channel'],
            'post_200ms_epochs': int(mask.sum()),
            'post_200ms_pll_error_rms_deg': float(np.degrees(np.sqrt(np.mean(pll[mask]**2)))) if mask.any() else None,
            'post_200ms_pll_error_gt_90deg_count': int(np.sum(abs(pll[mask]) > np.pi/2)),
            'all_pll_error_gt_90deg_count': int(np.sum(abs(pll) > np.pi/2)),
            'post_200ms_frequency_mean_hz': float(np.mean(frequency[mask])) if mask.any() else None,
            'post_200ms_frequency_std_hz': float(np.std(frequency[mask])) if mask.any() else None})
    return result


def aggregate(records):
    valid = [r for r in records if r['summary']['status'] == 'evaluated_not_lock_certified']
    result = {'records': len(records), 'evaluated': len(valid),
              'failures': [{'seed_base': r['seed_base'], 'status': r['summary']['status']}
                           for r in records if r not in valid]}
    for label in ('all_epochs', 'post_200ms', 'post_500ms'):
        values = [r['summary'][label] for r in valid if r['summary'][label] is not None]
        result[label] = {'records_with_window': len(values)}
        for metric in ('wrapped_error_mean_m', 'wrapped_error_std_m',
                       'wrapped_error_rms_m', 'wrapped_error_max_abs_m'):
            x = [v[metric] for v in values]
            result[label][metric] = ({'min': min(x), 'median': float(np.median(x)),
                                     'max': max(x)} if x else None)
    result['phase_step_gt_90deg_count'] = sum(r['summary']['phase_step_gt_90deg_count'] for r in valid)
    result['post_200ms_pll_error_gt_90deg_count'] = sum(
        c['post_200ms_pll_error_gt_90deg_count'] for r in valid
        for c in r['summary']['channel_diagnostics'])
    return result


def plot(records, destination):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for record in records:
        if record['summary']['status'] != 'evaluated_not_lock_certified':
            continue
        case = record['case']
        time = np.array(case['time_s'])
        error = np.angle(np.exp(1j*(np.array(case['phase_difference_rad'])-
                                   np.radians(case['ideal_geometry_phase_deg']))))
        error *= case['wavelength_m']/(2*np.pi)*1000
        axes[0].plot(time, error, alpha=.6, linewidth=.8, label=str(record['seed_base']))
        for ax in axes[1:]:
            selected = time >= .2
            ax.plot(time[selected], error[selected], alpha=.6, linewidth=.8)
    axes[0].set(ylabel='Error vs geometry (mm)', title='Independent TL/TR tracking; 10 RF seed pairs; no multipath')
    axes[0].legend(ncol=5, fontsize=8, title='TL seed (TR seed = TL + 1)')
    axes[1].set(xlabel='Time (s)', ylabel='Error after 0.2 s (mm)')
    for ax in axes:
        ax.axhline(0, color='black', linewidth=.7)
        ax.grid(alpha=.25)
    fig.savefig(destination/'stability.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results'/'dual_tracking_stability_1s')
    args = parser.parse_args()
    # Refuse implicit reuse of old results; each run has explicit source provenance.
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in Path(__file__).resolve().parent.glob('*.py')}
    manifest = {'duration_ms': 1000, 'seed_pairs': [[1000+2*i, 1001+2*i] for i in range(10)],
                'post_windows_s': [.2, .5], 'source_sha256': sources,
                'scope': 'default 100 ms experiment extended to 1 s; identical ideal clock/LO; no interference/multipath or channel delay; default RF/DFE quantization and loop gains; independent seed pairs',
                'interpretation': 'NCO phase aligned in physical time; pointwise phase remains modulo integer cycles; fixed windows are not lock certification; 90 degree flags are diagnostics, not a complete cycle-slip detector; no IID assumption across epochs'}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    records = []
    for seed, _ in manifest['seed_pairs']:
        print(f'RUN seed pair {seed}/{seed+1}', flush=True)
        case = run_case(True, duration_ms=1000, seed_base=seed)
        record = {'seed_base': seed, 'case': case, 'summary': summarize_case(case)}
        (args.output/f'seed_{seed}.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
        records.append(record)
        print(json.dumps({'seed': seed, 'summary': record['summary']['post_200ms'] if 'post_200ms' in record['summary'] else record['summary']}), flush=True)
    summary = aggregate(records)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    plot(records, args.output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
