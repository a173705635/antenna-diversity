"""Independently verify archived per-PRN errors and position metrics."""
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
summaries = json.loads((root/'summary.json').read_text(encoding='utf-8'))
count = 0
for summary in summaries:
    record = json.loads((root/f"{summary['name']}.json").read_text(encoding='utf-8'))
    times = np.array(record['time_s'])
    states = np.array([[np.array(s['position_m'])+t*np.array(s['velocity_mps']) for s in source['satellites']] for t in times])
    error = np.array(record['pseudoranges_m'], dtype=float)-np.linalg.norm(states-reference, axis=2)-source['clock_bias_truth_m']
    np.testing.assert_allclose(error, record['pseudorange_error_m'], atol=1e-7)
    for key, values in [('mean_m', np.nanmean(error, axis=0)), ('std_m', np.nanstd(error, axis=0)),
                        ('rms_m', np.sqrt(np.nanmean(error**2, axis=0)))]:
        np.testing.assert_allclose(values, record['per_prn'][key], atol=1e-7)
        np.testing.assert_allclose(values, summary['per_prn'][key], atol=1e-7)
    for name, position_key in [('spp', 'spp_positions_m'), ('joint_float', 'joint_positions_m')]:
        if name in record:
            position = np.array(record[position_key], dtype=float)
            delta = position[np.isfinite(position).all(axis=1)]-reference
            assert np.isclose(np.sqrt(np.mean(np.sum(delta**2, axis=1))), record[name]['absolute_3d_rms_m'])
            assert np.isclose(np.sqrt(np.mean(np.sum((delta-delta.mean(axis=0))**2, axis=1))), record[name]['demeaned_3d_rms_m'])
    count += np.isfinite(error).sum()
assert meta['baseline_reproduction_max_difference_m'] == 0.
print(f'Verified {len(hashes)} files, {len(summaries)} controls and {count} per-satellite observations.')
