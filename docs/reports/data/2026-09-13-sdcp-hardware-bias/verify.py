"""Independently reconstruct saved NCO residuals and check all reported metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
hashes = json.loads((root/'sha256.json').read_text())
for name, expected in hashes.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
summary = json.loads((root/'summary.json').read_text())
times = np.arange(200, 999)*.001
controls = {}
checked = 0
for row in summary['rows']:
    label = f"{row['seed']}_{row['case']}"
    case = json.loads((root/(label+'.json')).read_text())
    assert row['status'] == case['status']
    if row['status'] != 'evaluated_not_lock_certified':
        continue
    ranges, quality = [], np.ones(times.size, dtype=bool)
    for ch in case['channels']:
        starts = np.asarray(ch['epoch_start_samples'])/case['fs_hz']
        values, flags = [], []
        for t in times:
            ix = np.searchsorted(starts, t, side='right')-1
            if ix < 0 or not t < starts[ix]+.001:
                values.append(np.nan)
                flags.append(False)
                continue
            phase = ch['carrier_nco_phase_rad'][ix]+2*np.pi*ch['carrier_nco_frequency_hz'][ix]*(t-starts[ix])
            values.append(-phase*case['wavelength_m']/(2*np.pi))
            flags.append(ch['phase_lock_metric'][ix] >= .8 and abs(ch['pll_error_rad'][ix]) < np.pi/2)
        ranges.append(np.asarray(values))
        quality &= flags
    d = ranges[1]-ranges[0]
    residual = d+case['ideal_geometry_path_m']
    if row['case'] == 'zero':
        controls[row['seed']] = residual
    changes = residual-controls[row['seed']]
    for key, x in [('raw_sdcp', d), ('geometry_residual', residual), ('paired_change', changes)]:
        x = x[np.isfinite(x)]*1000
        result = row[key]
        assert result['epochs'] == len(x)
        for name, value in [('mean_mm', x.mean()), ('std_mm', x.std()),
                            ('rms_mm', np.sqrt(np.mean(x*x))), ('max_abs_mm', max(abs(x)))]:
            np.testing.assert_allclose(value, result[name], atol=1e-9, rtol=1e-9)
    assert int(quality.sum()) == row['quality_pass']
    with np.load(root/'traces.npz') as traces:
        np.testing.assert_allclose(traces[label+'_residual_m'], residual, atol=1e-12)
        np.testing.assert_array_equal(traces[label+'_quality'], quality)
    checked += 1
result = {'files_hashed': len(hashes), 'records_reconstructed': checked,
          'reported_rows': len(summary['rows']), 'metrics_per_record': 15,
          'method': 'Raw accumulated NCO, no wrap, no branch rounding, no mean subtraction.'}
(root/'validation_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
