"""Validate the single-frequency SDCP/weighting front end, not PVT accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audit_dual_tracking_phase import run_case
from config import Direction, MultipathPath
from presets import default_config
from sdcp import geometry_offset_m, modified_sdcp, observation_weights


def assumed_geometry():
    cfg = default_config()
    positions = cfg.phone.assumed_phase_centers_m
    baseline = np.array(positions['TR'])-positions['TL']
    direction = cfg.signal.assumed_direction
    return geometry_offset_m(baseline, direction.azimuth_deg, direction.elevation_deg)


def analyze(case, threshold):
    if case['status'] != 'evaluated_not_lock_certified':
        return {'status': case['status']}
    time = np.array(case['time_s'])
    values = modified_sdcp(time, case['phase_difference_rad'], case['wavelength_m'],
                           assumed_geometry())
    weights = observation_weights(values['residual_m'], threshold)
    mask = time >= .6
    residual = values['residual_m'][mask]
    if residual.size == 0:
        return {'status': 'insufficient_evaluation_epochs'}
    return {'status': 'evaluated_not_lock_certified',
            'reference_offset_m': values['reference_offset_m'],
            'residual_m': values['residual_m'].tolist(),
            'soft_pair_weights': weights['soft_pair'].tolist(),
            'hard_pair_weights': weights['hard_pair'].tolist(),
            'evaluation': {
                'epochs': int(mask.sum()),
                'residual_mean_m': float(residual.mean()),
                'residual_std_m': float(residual.std()),
                'residual_rms_m': float(np.sqrt(np.mean(residual**2))),
                'max_abs_residual_m': float(np.max(abs(residual))),
                'hard_zero_weight_fraction': float(np.mean(weights['hard_pair'][mask,0] == 0)),
                'soft_weight_mean': float(np.mean(weights['soft_pair'][mask,0])),
            }}


def scenarios():
    common = dict(direction=Direction(110.,15.), excess_delay_chips=.25,
                  reflection_phase_rad=np.pi/2)
    return [
        ('los_only', [], None),
        ('different_direction_onset', [MultipathPath(relative_power_db=-6., **common)], 500),
        ('weak_echo_onset', [MultipathPath(relative_power_db=-18., **common)], 500),
        ('common_mode_boundary', [MultipathPath(Direction(30.,60.), relative_power_db=-6.,
             excess_delay_chips=0., reflection_phase_rad=np.pi/2)], 500),
        ('static_contaminated_reference', [MultipathPath(relative_power_db=-6., **common)], None),
    ]


def phase_bias_vs_los(case, control):
    """Post-hoc individual-channel comparison; never enters SDCP weights."""
    if case['status'] != 'evaluated_not_lock_certified' or control['status'] != 'evaluated_not_lock_certified':
        return None
    biases = {}
    for channel, baseline in zip(case['channels'], control['channels']):
        def mean_phase(data, fs):
            t = np.array(data['epoch_start_samples'])/fs
            phase = np.array(data['carrier_nco_phase_rad'])-2*np.pi*1500*t
            return np.angle(np.mean(np.exp(1j*phase[t >= .6])))
        delta = mean_phase(channel, case['fs_hz'])-mean_phase(baseline, control['fs_hz'])
        biases[channel['channel']] = float(np.angle(np.exp(1j*delta))*case['wavelength_m']/(2*np.pi))
    return biases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results'/'trimble_sdcp')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    calibration_root = Path(__file__).resolve().parents[1]/'docs/reports/data/2026-09-11-dual-tracking-stability'
    scores, calibration_hashes, calibration_seeds = [], {}, []
    for path in sorted(calibration_root.glob('seed_*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        case = record['case']
        residual = modified_sdcp(case['time_s'], case['phase_difference_rad'],
                                 case['wavelength_m'], assumed_geometry())['residual_m']
        scores.extend(abs(residual[np.array(case['time_s']) >= .6]).tolist())
        calibration_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        calibration_seeds.extend(c['rf_noise_seed'] for c in case['channels'])
    if len(calibration_hashes) != 10:
        raise ValueError('all ten archived H0 records required')
    threshold = float(np.quantile(scores, .99, method='higher'))
    frozen = {'threshold_m': threshold, 'calibration_epochs': len(scores),
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).resolve().parent.glob('*.py')},
              'calibration_sha256': calibration_hashes, 'calibration_rf_seeds': calibration_seeds,
              'calibration_rule': '99th percentile of absolute residual in t>=0.6s; per-record reference median on [0.2,0.4)s; correlated epochs, not record-level CFAR',
              'soft_rule': 'expit((threshold-abs(residual))/(threshold/4)); same weight for both antennas',
              'hard_rule': '1 when abs(residual)<=threshold else 0',
              'evaluation_rf_pairs': [[2000+2*i,2001+2*i] for i in range(5)],
              'scope': 'single L1 satellite, TL/TR 75.9 mm ideal baseline, assumed exact geometry, common ideal clock, no interference; process 400 observation weighting only; no RTK/PVT',
              'reference_warning': 'reference presumed clean for onset tests; static contamination deliberately violates this assumption'}
    threshold_path = args.output/'threshold.json'
    threshold_path.write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    threshold = json.loads(threshold_path.read_text())['threshold_m']
    frozen_bytes = threshold_path.read_bytes()
    print(f'FROZEN threshold {threshold*1000:.6f} mm before evaluation', flush=True)
    records = []
    controls = {}
    for kind, paths, onset in scenarios():
        for seed, _ in frozen['evaluation_rf_pairs']:
            print(f'RUN {kind} {seed}/{seed+1}', flush=True)
            case = run_case(True, duration_ms=1000, seed_base=seed,
                            multipath_paths=paths, multipath_start_ms=onset)
            result = analyze(case, threshold)
            if kind == 'los_only':
                controls[seed] = case
            row = {'kind': kind, 'seed_base': seed,
                   'scenario': {'onset_ms': onset, 'paths': [dict(azimuth_deg=p.direction.azimuth_deg,
                       elevation_deg=p.direction.elevation_deg,power_db=p.relative_power_db,
                       delay_chips=p.excess_delay_chips,reflection_phase_rad=p.reflection_phase_rad) for p in paths]},
                   'case': case, 'sdcp': result,
                   'posthoc_bias_vs_paired_los_m': phase_bias_vs_los(case, controls[seed])}
            (args.output/f'{kind}_{seed}.json').write_text(json.dumps(row, indent=2, allow_nan=False), encoding='utf-8')
            records.append(row)
            print(json.dumps(result.get('evaluation',result)), flush=True)
    assert threshold_path.read_bytes() == frozen_bytes
    summary = []
    for kind, _, _ in scenarios():
        selected = [r for r in records if r['kind'] == kind]
        valid = [r for r in selected if r['sdcp']['status'] == 'evaluated_not_lock_certified']
        summary.append({'kind': kind, 'records': len(selected), 'evaluated': len(valid),
                        'results': [{'seed_base': r['seed_base'],
                                     'evaluation': r['sdcp'].get('evaluation'),
                                     'status': r['sdcp']['status'],
                                     'posthoc_bias_vs_paired_los_m': r['posthoc_bias_vs_paired_los_m']} for r in selected]})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    fig, axes = plt.subplots(5,2,figsize=(12,12),constrained_layout=True)
    for rowindex,(kind,_,_) in enumerate(scenarios()):
        for record in records:
            if record['kind'] != kind or record['sdcp']['status'] != 'evaluated_not_lock_certified':
                continue
            t = np.array(record['case']['time_s'])
            selected = t >= .2
            axes[rowindex,0].plot(t[selected], np.array(record['sdcp']['residual_m'])[selected]*1000,alpha=.6,linewidth=.8)
            axes[rowindex,1].plot(t[selected], np.array(record['sdcp']['soft_pair_weights'])[selected,0],alpha=.6,linewidth=.8)
        axes[rowindex,0].set(ylabel='Residual (mm)',title=kind)
        for sign in [-1,1]: axes[rowindex,0].axhline(sign*threshold*1000,color='black',linestyle='--',linewidth=.7)
        axes[rowindex,1].set(ylabel='Pair weight',ylim=(-.05,1.05))
        for ax in axes[rowindex]:
            ax.axvline(.5,color='grey',linestyle=':'); ax.grid(alpha=.2)
    axes[-1,0].set_xlabel('Time (s)'); axes[-1,1].set_xlabel('Time (s)')
    fig.savefig(args.output/'sdcp_weights.png',dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
