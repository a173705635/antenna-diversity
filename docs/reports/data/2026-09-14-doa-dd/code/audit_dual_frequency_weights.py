"""Freeze observation-only band weights, then evaluate unchanged receiver records."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from positioning import carrier_range
from run_gps_l1_l5 import compare
from run_gps_spp import serializable


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def calibrate(root):
    manifest = read(root/'manifest.json')
    weights = {}
    for kind, mode, frequency in [('L1CA', 'L1', 1575.42e6), ('L5Q', 'L5', 1176.45e6)]:
        scatter, phase_scatter = [], []
        for seed in manifest['seeds']:
            record = read(root/f'{kind}_{seed}.json')
            phase = np.column_stack([carrier_range(dict(h, success=True), record['time_s'],
                record['fs_dfe_hz'], .001, 299792458/frequency) for h in record['satellite_tracking']])
            # Mean removal discards unknown phase origins, not true range errors.
            scatter.extend(np.std(np.asarray(record['pseudoranges_m'])-phase, axis=0, ddof=1))
            residual = np.asarray(read(root/f'comparison_{seed}.json')[mode]['float_solution']['carrier_residual_m'])
            # Approximate degrees-of-freedom correction for eight observations/four geometric states.
            phase_scatter.extend(np.std(residual, axis=0, ddof=1)/np.sqrt(1-4/residual.shape[1]))
        weights[kind] = {'code_sigma_m': float(np.median(scatter)),
                         'carrier_sigma_m': max(.0005, float(np.median(phase_scatter)))}
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output/name).write_text(json.dumps(serializable(value), indent=2, allow_nan=False), encoding='utf-8')
    weights = read(args.weights) if args.weights else calibrate(args.input)
    save('weights.json', weights)
    manifest = read(args.input/'manifest.json')
    satellites = manifest['satellites']
    for s in satellites:
        s['position_m'], s['velocity_mps'] = np.array(s['position_m']), np.array(s['velocity_mps'])
    save('manifest.json', {'input': str(args.input), 'seeds': manifest['seeds'],
         'calibration': 'frozen external weights' if args.weights else 'observation-only calibration on these records',
         'weight_file': str(args.weights), 'weights': weights,
         'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.input.glob('*.json')},
         'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})
    summaries = []
    for seed in manifest['seeds']:
        records = {kind: read(args.input/f'{kind}_{seed}.json') for kind in ('L1CA', 'L5Q')}
        result = compare(records, satellites, weights)
        save(f'comparison_{seed}.json', result)
        summary = {'seed': seed, 'modes': {name: {k: v for k,v in mode.items() if k in
            ('code', 'float', 'float_failure', 'carrier_postfit_max_m', 'phase_residual_screen_passed')}
            for name, mode in result.items()}}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    save('summary.json', summaries)
    save('sha256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in args.output.glob('*.json') if p.name != 'sha256.json'})


if __name__ == '__main__':
    main()
