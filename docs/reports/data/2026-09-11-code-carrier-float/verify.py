"""Independent archive, residual and position-statistics verification."""
import hashlib
import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent
hashes = json.loads((root/'sha256.json').read_text(encoding='utf-8'))
for name, expected in hashes.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
meta = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for name, expected in meta['input_sha256'].items():
    assert hashlib.sha256((root.parent/'2026-09-11-gps-spp'/name).read_bytes()).hexdigest() == expected
source = meta['source_manifest']
reference = np.array(source['receiver_truth_ecef_m'])
count = 0
for seed in source['seeds']:
    record = json.loads((root/f'seed_{seed}.json').read_text(encoding='utf-8'))
    positions = np.array(record['position_ecef_m'], dtype=float)
    clocks = np.array(record['clock_bias_m'], dtype=float)
    valid = np.isfinite(positions).all(axis=1)
    error = positions[valid]-reference
    stats = record['summary']['joint_float']
    assert valid.sum() == stats['epochs']
    assert np.isclose(np.sqrt(np.mean(np.sum(error**2, axis=1))), stats['absolute_3d_rms_m'])
    assert np.isclose(np.sqrt(np.mean(np.sum((error-error.mean(axis=0))**2, axis=1))), stats['demeaned_3d_rms_m'])
    times = np.array(record['time_s'])
    states = np.array([[np.array(s['position_m'])+t*np.array(s['velocity_mps']) for s in source['satellites']] for t in times])
    code = np.array(record['pseudoranges_m'], dtype=float)
    phase = np.array(record['carrier_range_m'], dtype=float)
    for arc in record['arcs']:
        if arc['status'] != 'accepted':
            continue
        assert not arc['integer_fixed'] and arc['rank'] == arc['unknowns']
        indices = np.array(arc['indices'])
        delta = positions[indices, None, :]-states[indices]
        distance = np.linalg.norm(delta, axis=2)
        rp = code[indices]-distance-clocks[indices, None]
        rl = phase[indices]+np.array(arc['phase_origin_shift_m'])-distance-clocks[indices, None]-np.array(arc['centered_ambiguity_m'])
        np.testing.assert_allclose(rp, np.array(record['code_residual_m'])[indices], atol=1e-6)
        np.testing.assert_allclose(rl, np.array(record['carrier_residual_m'])[indices], atol=1e-6)
        geometry = np.concatenate((delta/distance[:, :, None], np.ones((*distance.shape, 1))), axis=2)
        position_gradient = np.einsum('tij,ti->tj', geometry, rp/meta['code_sigma_m']**2+rl/meta['phase_sigma_m']**2)
        bias_gradient = rl.sum(axis=0)/meta['phase_sigma_m']**2
        assert np.max(np.abs(position_gradient)) < .01
        assert np.max(np.abs(bias_gradient)) < .01
    count += valid.sum()
print(f'Verified {len(hashes)} archive files, input hashes and {count} joint position solutions.')
