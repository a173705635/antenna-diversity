"""Single-antenna assisted GPS L1/L5 Q5 RF-to-position comparison.

Two frequency streams share physical trajectories and a clock, not RF noise.
No ionosphere, inter-frequency bias, CNAV decoding or integer ambiguity fixing.
"""

from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.serialization import serializable
from core.provenance import source_hashes, freeze_source
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np

from core.config import SPEED_OF_LIGHT_MPS
from core.positioning import PositioningFailure
from core.positioning import carrier_range, solve_position, solve_code_carrier_float
from core.gps_simulation import build_scene, satellite_setup, run_record, receiver_reference
from experiments.paths import PROJECT_ROOT_DIR, RESULTS_DIR, reset_output

OUTPUT_DIR = RESULTS_DIR/'gps_l1_l5'
WEIGHTS_PATH = PROJECT_ROOT_DIR/'configs/dual_frequency_weights.json'
DURATION_MS = 1000
SEEDS = [4000, 4001, 4002]
FREEZE_SOURCE_FLAG = False


def position_stats(positions):
    error = np.asarray(positions)-receiver_reference()[0]
    valid_flag = np.isfinite(error).all(axis=1)
    result = {'epochs': len(error), 'solved_epochs': int(valid_flag.sum())}
    if valid_flag.any():
        e = error[valid_flag]
        result.update(position_rms_m=float(np.sqrt(np.mean(np.sum(e*e, axis=1)))),
                      demeaned_rms_m=float(np.sqrt(np.mean(np.sum((e-e.mean(axis=0))**2, axis=1)))))
    return result


def compare(records, satellites, weights=None):
    times = np.asarray(records['L1CA']['time_s'])
    if not np.array_equal(times, records['L5Q']['time_s']):
        raise ValueError('frequency streams must share observation times')
    states = np.array([[s['position_m']+s['velocity_mps']*t for s in satellites] for t in times])
    phases = {}
    codes = {}
    for kind, frequency in [('L1CA', 1575.42e6), ('L5Q', 1176.45e6)]:
        record = records[kind]
        codes[kind] = np.asarray(record['pseudoranges_m'], dtype=float)
        phases[kind] = np.full_like(codes[kind], np.nan)
        for column, history in enumerate(record['satellite_tracking']):
            if (history['acquired_flag'] and history['tracking_status'] == 'tracking'):
                phases[kind][:, column] = carrier_range(dict(history, success_flag=True), times,
                    record['fs_dfe_hz'], .001, SPEED_OF_LIGHT_MPS/frequency)
        phases[kind][~np.isfinite(codes[kind])] = np.nan
    output = {}
    modes = [('L1', ['L1CA']), ('L5', ['L5Q']), ('L1_L5', ['L1CA', 'L5Q'])]
    if weights is not None:
        modes += [('L5_weighted', ['L5Q']), ('dual_weighted', ['L1CA', 'L5Q']),
                  ('L5_code_dual_phase', ['L1CA', 'L5Q'])]
    for name, kinds in modes:
        p = np.concatenate([codes[k] for k in kinds], axis=1)
        l = np.concatenate([phases[k] for k in kinds], axis=1)
        geometry = np.concatenate([states for k in kinds], axis=1)
        cs, ls = np.full(p.shape[1], 30.), np.full(p.shape[1], .01)
        if name in ('L5_weighted', 'dual_weighted', 'L5_code_dual_phase'):
            cs = np.concatenate([np.full(len(satellites), weights[k]['code_sigma_m']) for k in kinds])
            ls = np.concatenate([np.full(len(satellites), weights[k]['carrier_sigma_m']) for k in kinds])
        if name == 'L5_code_dual_phase':
            cs[:len(satellites)] = np.inf
        positions = np.full((len(times), 3), np.nan)
        code_failures = []
        for i in range(len(times)):
            valid_flag = np.isfinite(p[i])
            try:
                positions[i] = solve_position(geometry[i, valid_flag], p[i, valid_flag], sigma_m=cs[valid_flag])['position_m']
            except PositioningFailure as exc:
                code_failures.append({'index': i, 'reason': str(exc)})
        result = {'code_failures': code_failures, 'code': position_stats(positions), 'code_position_ecef_m': positions,
                  'observation_labels': [f'{k}:PRN{s["prn"]}' for k in kinds for s in satellites]}
        # Offline float ambiguities are independent per satellite AND frequency.
        # Do not bridge missing epochs or silently supply truth for rejected data.
        if np.isfinite(p).all() and np.isfinite(l).all():
            try:
                solution = solve_code_carrier_float(geometry, p, l, code_sigma_m=cs, carrier_sigma_m=ls)
                result['float'] = position_stats(solution['position_m'])
                result['float_solution'] = solution
                result['carrier_postfit_max_m'] = float(abs(solution['carrier_residual_m']).max())
                result['phase_residual_screen_passed_flag'] = result['carrier_postfit_max_m'] <= .05
            except PositioningFailure as exc:
                result['float_failure'] = str(exc)
        else:
            result['float_failure'] = 'incomplete continuous common arc'
        output[name] = result
    return output


def main():
    if DURATION_MS < 210:
        raise ValueError('GPS positioning is not evaluated: need at least 210 ms to include the first 200 ms observation and end margin.')
    output_dir = reset_output(OUTPUT_DIR)
    if FREEZE_SOURCE_FLAG:
        freeze_source(output_dir)
    weights = json.loads(WEIGHTS_PATH.read_text(encoding='utf-8'))
    def save(name, data):
        (output_dir/name).write_text(json.dumps(serializable(data), indent=2, allow_nan=False), encoding='utf-8')
    satellites = satellite_setup()
    manifest = {'scope': __doc__, 'satellites': satellites, 'duration_ms': DURATION_MS,
                'seeds': SEEDS, 'frequency_noise_seeds': 'L1=seed; L5=seed+100000',
                'source_sha256': source_hashes(),
                'time_aid': 'per-satellite range rounded to 10 km; selects primary 1 ms ambiguity',
                'power': 'same configured power per tracked component; Q5 only, no I5 component',
                'float': 'offline independent real bias per satellite/frequency; no integer fixing',
                'combination': 'equal-weight baseline plus frozen band weights and L5-code/dual-phase mode; common clock',
                'recommended_mode': 'L5_code_dual_phase',
                'weight_calibration': 'seeds 4000..4002, observation scatter only; this receiver configuration',
                'weights': weights,
                'weights_sha256': hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest(),
                'phase_screen': '50 mm maximum postfit residual, not certified cycle-slip protection',
                'noise_model': 'Fixed independent 130 K element-input noise passes once through antenna A=C diag(sqrt(eta)); source direction gain/phase acts on each wave only; passive element and feedline loss noise, then RF-added noise; one RF input filter per band',
                'frequency_configs': {}}
    records = {seed: {} for seed in SEEDS}
    for kind in ['L1CA', 'L5Q']:
        print(f'Generating {kind} scene...', flush=True)
        scene, cfg = build_scene(satellites, DURATION_MS, signal_kind=kind)
        manifest['frequency_configs'][kind] = {'signal': asdict(cfg.signal),
                                             'antenna_noise_temperature_k': cfg.phone.antenna_noise_temperature_k,
                                             'antenna_model': asdict(cfg.phone.antenna_model),
                                             'feedline': asdict(cfg.feedline), 'rf': asdict(cfg.rf),
                                             'dfe': asdict(cfg.dfe), 'tracking': asdict(cfg.tracking)}
        save('manifest.json', manifest)
        for seed in SEEDS:
            record = run_record(seed+(100000 if kind == 'L5Q' else 0), satellites, scene, cfg)
            records[seed][kind] = record
            save(f'{kind}_{seed}.json', record)
        del scene
    summaries = []
    for seed in SEEDS:
        result = compare(records[seed], satellites, weights)
        save(f'comparison_{seed}.json', result)
        summary = {'seed': seed, 'modes': {name: {key: value for key, value in mode.items()
                    if key in ('code', 'float', 'float_failure', 'carrier_postfit_max_m', 'phase_residual_screen_passed_flag')}
                   for name, mode in result.items()}}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    save('summary.json', summaries)
    save('sha256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in output_dir.glob('*.json') if p.name != 'sha256.json'})


if __name__ == '__main__':
    main()
