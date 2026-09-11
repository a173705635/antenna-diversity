"""Verify archive bytes, RF/DFE settings, tracking coverage and solution counts."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
def read(name):
    return json.loads((root/name).read_text(encoding='utf-8'))

hashes = read('sha256.json')
for name, digest in hashes.items():
    assert Path(name).name == name
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
manifest = read('manifest.json')
assert manifest['duration_ms'] == 1000
for kind, decimation in [('L1CA', 5), ('L5Q', 1)]:
    cfg = manifest['frequency_configs'][kind]
    assert cfg['signal']['fs_rf_hz'] == 33e6
    assert cfg['dfe']['decimation'] == decimation
    for seed in manifest['seeds']:
        record = read(f'{kind}_{seed}.json')
        assert record['fs_dfe_hz'] == 33e6/decimation
        assert np.asarray(record['pseudoranges_m'], dtype=float).shape == (80, 8)
        assert np.isfinite(np.asarray(record['pseudoranges_m'], dtype=float)).all()
        assert record['summary']['solved_epochs'] == 80
        for history in record['satellite_tracking']:
            assert history['acquired'] and history['tracking_status'] == 'tracking'
            if kind == 'L5Q':
                assert history['nh_sync']['metric'] >= 1.5
                assert 0 <= history['nh_sync']['offset'] < 20
solutions = 0
for seed in manifest['seeds']:
    comparison = read(f'comparison_{seed}.json')
    for mode, result in comparison.items():
        assert result['code']['solved_epochs'] == result['float']['solved_epochs'] == 80
        assert result['phase_residual_screen_passed']
        solution = result['float_solution']
        assert solution['integer_fixed'] is False
        assert len(solution['ambiguity_m']) == (16 if mode == 'L1_L5' else 8)
        assert np.isfinite(np.asarray(solution['position_m'])).all()
        assert np.max(np.abs(solution['carrier_residual_m'])) <= .05
        solutions += 80
print(f'Verified {len(hashes)} file hashes, 48 tracks, {solutions} code and {solutions} float positions.')
