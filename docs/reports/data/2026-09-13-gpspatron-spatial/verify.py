"""Recompute archived results and verify immutable data/source hashes."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'python_sim_simplified'))
from audit_spatial_diversity import calibrate_threshold, paired_gain, wilson
from audit_spatial_diversity_rf import evaluate_records
from run_gps_spp import serializable


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


checks = []


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    checks.append(message)


thresholds = read(HERE/'model/thresholds.json')['thresholds']
calibration = np.load(HERE/'model/calibration.npz')
for key, threshold in thresholds.items():
    check(calibrate_threshold(calibration[key]) == threshold, f'calibration/{key}')

summary = read(HERE/'model/summary.json')
check(len(summary['rows']) == 22*8, 'complete model matrix 22 cases x 8 configurations')
scores = {}
for row in summary['rows']:
    case = row['case']
    if case not in scores:
        scores[case] = dict(np.load(HERE/'model'/f'{case}.npz'))
    key = row['layout']+'_'+row['method']
    values = scores[case][key]
    hits = int(np.sum(values < thresholds[key]))
    check(len(values) == row['events'] == 5000 and hits == row['alerts'], f'model count/{case}/{key}')
    check(row['fraction'] == hits/len(values), f'model rate/{case}/{key}')
    check(np.allclose(wilson(hits, len(values)), row['wilson95'], rtol=0, atol=1e-14), f'model Wilson/{case}/{key}')
    check(np.allclose(np.quantile(values, [.01, .5, .99]), row['score_quantiles_01_50_99']), f'model score quantiles/{case}/{key}')
for row in summary['paired_gains']:
    a, b = row['comparison'].split('->'); method = row['method']; values = scores[row['case']]
    gain = paired_gain(values[a+'_'+method] < thresholds[a+'_'+method], values[b+'_'+method] < thresholds[b+'_'+method])
    check(all(gain[key] == row[key] for key in gain), f'paired/{row["case"]}/{row["comparison"]}/{method}')
for key in thresholds:
    check(np.array_equal(scores['los'][key], scores['spatial_matched'][key]), f'indistinguishable-input/{key}')

rf_records = 0
for part in ('rf_a', 'rf_b'):
    directory = HERE/part
    manifest = read(directory/'manifest.json')
    check(manifest['thresholds'] == thresholds, f'{part}/threshold frozen')
    check(manifest['threshold_file_sha256'] == hashlib.sha256((HERE/'model/thresholds.json').read_bytes()).hexdigest(), f'{part}/threshold file hash')
    actual = read(directory/'summary.json')['rows']
    check(len(actual) == 3*2*8, f'{part}/complete RF matrix')
    for case in manifest['cases']:
        for seed in manifest['seeds']:
            records = [read(directory/f'{case}_{seed}_{name}.json') for name in manifest['antenna_names']]
            rf_records += len(records)
            expected, trace = evaluate_records(records, thresholds)
            recorded = np.load(directory/f'{case}_{seed}_observations.npz')
            for name, array in zip(('time_s', 'carrier_m', 'valid', 'tl_error_m'), trace):
                check(np.array_equal(array, recorded[name], equal_nan=True), f'{part}/{case}/{seed}/{name}')
            for row in expected:
                match = next(r for r in actual if r['case'] == case and r['seed'] == seed and
                             r['layout'] == row['layout'] and r['method'] == row['method'])
                check(all(serializable(value) == match[key] for key, value in row.items()),
                      f'{part}/{case}/{seed}/{row["layout"]}/{row["method"]}/windows+causal-policy')
                p = row['policy']
                check(p['dangerous_retained_epochs']+p['dangerous_blocked_by_alert']+p['dangerous_blocked_by_unknown'] == p['dangerous_available_epochs'], 'dangerous denominator conservation')
                check(p['safe_blocked_by_alert']+p['safe_blocked_by_unknown'] == p['safe_blocked_epochs'], 'safe denominator conservation')
            for record in records:
                check(record['fs_dfe_hz'] == 6600000, 'RF actual DFE sample rate')

check(rf_records == 48, '48 RF channel records reconstructed')
for part in ('model', 'rf_a', 'rf_b'):
    directory = HERE/part
    manifest = read(directory/'manifest.json')
    source = directory/'code' if part == 'model' else HERE/'code'
    for name, digest in manifest['source_sha256'].items():
        check(hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, f'{part}/source/{name}')
    for name, digest in read(directory/'sha256.json').items():
        check(hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest, f'{part}/data/{name}')

index = HERE/'sha256.json'
if index.exists():
    for name, digest in read(index).items():
        check(hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest, f'archive/{name}')

result = {'passed': True, 'checks': len(checks), 'model_events': 110000,
          'rf_channel_records': rf_records, 'rf_prn_tracking_histories': rf_records*8,
          'detail': checks}
(HERE/'validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'detail'}))
