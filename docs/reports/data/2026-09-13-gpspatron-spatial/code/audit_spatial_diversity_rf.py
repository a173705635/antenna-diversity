"""Synthetic four-RF-chain validation of 2/3/4-antenna evidence and output gating.

No transmit hardware. Static startup/replacement/mixture cases, not a dynamic
coherent takeover demonstration. Receiver and PVT use original timing aid.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from audit_spatial_diversity import W, LAYOUTS, NAMES, METHODS, save_json
from presets import mate70_phase_centers
from run_gps_spp import satellite_setup, receiver_reference, build_scene, run_record, serializable
from spatial_spoofing import observations_from_records, detect_windows, output_gate

RF_CASES = ('los', 'all_spoof', 'partial_3', 'partial_4', 'mixture_0', 'mixture_6')


def channel_parameters(satellites, channel):
    """Generator-only genuine and false ranges. Receiver gets neither list."""
    rx, rotation = receiver_reference()
    positions = np.array([mate70_phase_centers()[name] for name in NAMES])
    offset = positions[channel] - positions[0]
    fake_rx = rx + rotation.T @ np.array([100., 0., 0.])
    az, el = np.deg2rad([100., 10.])
    source = np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
    genuine, forged = copy.deepcopy(satellites), copy.deepcopy(satellites)
    for s, g, f in zip(satellites, genuine, forged):
        direction = rotation @ (s['position_m']-rx)
        direction /= np.linalg.norm(direction)
        g['range_at_zero_m'] -= float(offset @ direction)
        f['range_at_zero_m'] = float(np.linalg.norm(s['position_m']-fake_rx)+75000.-offset @ source)
    return genuine, forged


def channel_scene(satellites, channel, case, duration_ms):
    if case not in RF_CASES:
        raise ValueError('unknown RF case')
    genuine, forged = channel_parameters(satellites, channel)
    if case == 'los':
        chosen = genuine
    elif case == 'all_spoof':
        chosen = forged
    elif case.startswith('partial_'):
        count = int(case.split('_')[1]); chosen = forged[:count] + genuine[count:]
    else:
        chosen = genuine
    scene, cfg = build_scene(chosen, duration_ms)
    if case.startswith('mixture_'):
        extra, _ = build_scene(forged, duration_ms)
        scene += 10**(float(case.split('_')[1])/20) * extra
        del extra
    return scene, cfg


def policy_metrics(errors_m, states, limit_m=30.):
    error, state = np.asarray(errors_m, float), np.asarray(states)
    available = np.isfinite(error)
    danger = available & (error > limit_m)
    safe = available & ~danger
    retained = available & (state == 0)
    return {'epochs': len(error), 'available_epochs': int(available.sum()),
        'danger_limit_m': limit_m, 'dangerous_available_epochs': int(danger.sum()),
        'dangerous_retained_epochs': int(np.sum(danger & retained)),
        'dangerous_blocked_by_alert': int(np.sum(danger & (state == 1))),
        'dangerous_blocked_by_unknown': int(np.sum(danger & (state == -1))),
        'safe_available_epochs': int(safe.sum()), 'safe_blocked_epochs': int(np.sum(safe & ~retained)),
        'safe_blocked_by_alert': int(np.sum(safe & (state == 1))),
        'safe_blocked_by_unknown': int(np.sum(safe & (state == -1))),
        'retained_epochs': int(retained.sum()),
        'retained_rms_m': float(np.sqrt(np.mean(error[retained]**2))) if retained.any() else None,
        'unprotected_rms_m': float(np.sqrt(np.mean(error[available]**2))) if available.any() else None}


def evaluate_records(records, thresholds):
    t, carrier, valid = observations_from_records(records, W)
    errors = np.linalg.norm(np.asarray(records[0]['enu_error_m'], float), axis=1)
    results = []
    for layout, indices in LAYOUTS.items():
        for method in METHODS:
            windows = detect_windows(t, carrier[:, indices], valid[:, indices],
                                     thresholds[f'{layout}_{method}'], W, method=method)
            states = np.array([r['state'] for r in windows])
            gate = output_gate(t, windows)
            results.append({'layout': layout, 'method': method,
                'windows': windows, 'window_count': len(windows),
                'alerts': int(np.sum(states == 1)), 'unknown': int(np.sum(states == -1)),
                'no_evidence': int(np.sum(states == 0)),
                'observation_valid_fraction': float(valid[:, indices].mean()),
                'causal_gate_states': gate.tolist(), 'policy': policy_metrics(errors, gate)})
    return results, (t, carrier, valid, errors)


def plot_rf(out, rows, traces, thresholds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = list(dict.fromkeys(r['case'] for r in rows))
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, max(4, len(cases)*2.5)),
                             squeeze=False, layout='constrained')
    for i, case in enumerate(cases):
        first = next(key for key in traces if key[0] == case)
        t, _, _, error = traces[first]
        axes[i, 0].plot(t, error, color='black'); axes[i, 0].axhline(30, color='red', ls=':')
        axes[i, 0].set_ylabel('3D error (m)'); axes[i, 0].set_title(f'{case}: unprotected TL, seed {first[1]}')
        for layout in ('2_short', '3', '4'):
            row = next(r for r in rows if r['case'] == case and r['seed'] == first[1]
                       and r['layout'] == layout and r['method'] == 'cluster3')
            windows = row['windows']; threshold = thresholds[layout+'_cluster3']
            scores = [r['score']/threshold if r['score'] is not None and threshold > 0 else np.nan for r in windows]
            axes[i, 1].plot([r['decision_s'] for r in windows], scores, 'o-', label=layout)
        axes[i, 1].axhline(1, color='red', ls=':'); axes[i, 1].set_ylabel('score / frozen threshold')
        axes[i, 1].set_title('cluster3: below 1 = evidence; gaps = unknown'); axes[i, 1].legend()
        for ax in axes[i]: ax.grid(alpha=.2); ax.set_xlabel('Receiver time (s)')
    fig.suptitle('Synthetic RF / DFE / acquisition / tracking; static cases; first seed shown')
    fig.savefig(out/'rf_results.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True, help='frozen model thresholds.json')
    parser.add_argument('--duration-ms', type=int, default=600)
    parser.add_argument('--seeds', type=int, nargs='+', default=[610000, 620000])
    parser.add_argument('--cases', nargs='+', choices=RF_CASES, default=list(RF_CASES))
    args = parser.parse_args()
    if args.duration_ms < 400 or args.duration_ms % 100:
        parser.error('duration must be a multiple of 100 ms, at least 400 ms')
    payload = json.loads(args.model.read_text(encoding='utf-8'))
    if payload['layouts'] != LAYOUTS or payload['prns'] != 8 or not np.isclose(payload['wavelength_m'], W):
        parser.error('threshold calibration contract does not match RF configuration')
    thresholds = payload['thresholds']
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    sats = satellite_setup(); rows = []; traces = {}; started = time.monotonic()
    save_json(out/'manifest.json', {'scope': __doc__, 'duration_ms': args.duration_ms,
        'seeds': args.seeds, 'cases': args.cases, 'antenna_names': NAMES, 'layouts': LAYOUTS,
        'coordinates_interpretation': 'phone x=east, y=north, z=up; TL is local reference',
        'physical_positions_m': mate70_phase_centers(), 'false_position_offset_enu_m': [100., 0., 0.],
        'source_az_el_deg': [100., 10.], 'method': 'local GPSPATRON-inspired spatial phase detection',
        'thresholds': thresholds, 'threshold_file_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'noise_seed_policy': 'seed + channel, paired across all cases',
        'truth_boundary': 'reference errors evaluated only AFTER receiver; gate decisions use carrier/validity only',
        'rf_sampling_hz': 33000000, 'dfe_sampling_hz': 6600000,
        'scope_limits': ['static cases, no gradual power capture or carry-off', '8 GPS L1 PRNs, no navigation bits',
            'synthetic ephemerides and external coarse timing aid', 'ideal isotropic elements, no coupling',
            'common sampling/frequency reference; independent RF noise', 'model-calibrated thresholds not refitted on RF',
            'gating is not recovered navigation or a clock holdover implementation'],
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})
    for case in args.cases:
        for seed in args.seeds:
            records = []
            for ch, name in enumerate(NAMES):
                scene, cfg = channel_scene(sats, ch, case, args.duration_ms)
                save_json(out/f'{case}_{seed}_{name}_config.json', serializable(asdict(cfg)))
                record = run_record(seed+ch, sats, scene, cfg)
                del scene
                save_json(out/f'{case}_{seed}_{name}.json', serializable(record))
                records.append(record)
                print(f'DONE {case}/{seed}/{name}, elapsed={time.monotonic()-started:.1f}s', flush=True)
            evaluation, trace = evaluate_records(records, thresholds)
            traces[(case, seed)] = trace
            np.savez_compressed(out/f'{case}_{seed}_observations.npz', time_s=trace[0], carrier_m=trace[1],
                                valid=trace[2], tl_error_m=trace[3])
            rows.extend({'case': case, 'seed': seed, **row} for row in evaluation)
            save_json(out/'summary.json', {'rows': rows, 'elapsed_s': time.monotonic()-started,
                'denominator': 'windows within RF records are correlated; no independent-event CI claimed'})
    plot_rf(out, rows, traces, thresholds)
    save_json(out/'sha256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})


if __name__ == '__main__':
    main()
