"""Verify archived bytes and independently recompute SPP result consistency."""
import hashlib
import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent
hashes = json.loads((root/'sha256.json').read_text(encoding='utf-8'))
for name, digest in hashes.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
reference = np.array(manifest['receiver_truth_ecef_m'])
total = 0
for seed in manifest['seeds']:
    data = json.loads((root/f'seed_{seed}.json').read_text(encoding='utf-8'))
    position = np.array(data['position_ecef_m'], dtype=float)
    clock = np.array(data['clock_bias_m'], dtype=float)
    ranges = np.array(data['pseudoranges_m'], dtype=float)
    times = np.array(data['time_s'])
    valid = np.isfinite(position).all(axis=1)
    errors = np.linalg.norm(position[valid]-reference, axis=1)
    assert np.isclose(np.sqrt(np.mean(errors**2)), data['summary']['position_rms_m'])
    assert np.isclose(np.quantile(errors, .95), data['summary']['position_p95_m'])
    assert np.isclose(np.max(errors), data['summary']['position_max_m'])
    assert int(valid.sum()) == data['summary']['solved_epochs']
    for epoch in np.flatnonzero(valid):
        states = np.array([np.array(s['position_m'])+times[epoch]*np.array(s['velocity_mps'])
                           for s in manifest['satellites']])
        used = np.isfinite(ranges[epoch])
        difference = position[epoch]-states[used]
        distance = np.linalg.norm(difference, axis=1)
        residual = ranges[epoch, used]-distance-clock[epoch]
        np.testing.assert_allclose(residual, np.array(data['residual_m'])[epoch, used], atol=1e-6)
        design = np.column_stack((difference/distance[:, None], np.ones(used.sum())))
        assert np.linalg.norm(design.T @ residual) < 1e-4
    total += int(valid.sum())
print(f'Verified {len(hashes)} archived files and {total} position solutions.')
