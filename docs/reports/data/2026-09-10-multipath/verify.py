"""Verify archived bytes and recompute detection summaries without simulation."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'python_sim_simplified'))
from audit_detection_calibration import evaluate
from detection_calibration import calibrate_threshold


def read(path):
    return json.loads((HERE / path).read_text(encoding='utf-8'))


def main():
    manifest = read('sha256.json')
    for name, digest in manifest.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    frozen = read('detection_calibration/threshold.json')
    threshold = frozen['threshold_db']
    assert calibrate_threshold([r['score_db'] for r in frozen['calibration']], .01) == threshold
    evaluation = read('detection_calibration/evaluation.json')
    used = {r['seed'] for r in frozen['calibration']}
    for kind, rows in evaluation['datasets'].items():
        seeds = {r['seed'] for r in rows}
        assert len(seeds) == len(rows) and not used.intersection(seeds)
        used.update(seeds)
        assert evaluate(rows, threshold) == evaluation['summary'][kind], kind
    boundary = read('detection_boundaries/summary.json')
    assert boundary['threshold_sha256'] == manifest['detection_calibration/threshold.json']
    assert boundary['threshold_db'] == threshold
    total = 0
    for case in boundary['cases']:
        assert case['threshold_sha256'] == boundary['threshold_sha256']
        assert not used.intersection(r['seed'] for r in case['rows'])
        assert evaluate(case['rows'], threshold) == case['summary']
        total += len(case['rows'])
    assert total == 2200 and len(boundary['cases']) == 22
    print(f'PASS: {len(manifest)} file hashes; frozen threshold; independent evaluation; 22 cells / {total} records')


if __name__ == '__main__':
    main()
