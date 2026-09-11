"""Validate frozen-weight provenance, independent seeds and archive integrity."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
count = 0
for folder in ('calibration', 'holdout_rf', 'evaluation'):
    for name, digest in read(root/folder/'sha256.json').items():
        assert Path(name).name == name
        assert hashlib.sha256((root/folder/name).read_bytes()).hexdigest() == digest
        count += 1
cal = read(root/'calibration/manifest.json')
evaluation = read(root/'evaluation/manifest.json')
assert not set(cal['seeds']) & set(evaluation['seeds'])
assert read(root/'calibration/weights.json') == read(root/'evaluation/weights.json')
assert evaluation['calibration'] == 'frozen external weights'
for name, digest in evaluation['input_sha256'].items():
    assert hashlib.sha256((root/'holdout_rf'/name).read_bytes()).hexdigest() == digest
for summary in read(root/'evaluation/summary.json'):
    modes = summary['modes']
    for mode in modes.values():
        assert mode['float']['solved_epochs'] == 80
        assert mode['phase_residual_screen_passed']
    reference = modes['L5_weighted']['float']
    optimized = modes['L5_code_dual_phase']['float']
    assert optimized['demeaned_rms_m'] < reference['demeaned_rms_m']
    assert abs(optimized['position_rms_m']-reference['position_rms_m']) < .001
print(f'Verified {count} hashes, disjoint calibration/evaluation seeds, frozen weights and 3 held-out improvements.')
