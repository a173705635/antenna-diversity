"""Replay archived RF/DFE tracking: calibrated modulo SDCP versus reference SDCP.

No new RF realizations. Known zero hardware bias is a simulation assumption.
Geometry uses each epoch's TL SPP; no reference interval or true position.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_sdcp_positioning import BASELINE, WAVELENGTH
from positioning import carrier_range, solve_position
from sdcp import calibrated_sdcp, modified_sdcp

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'docs/reports/data/2026-09-11-sdcp-positioning'
OUTPUT = ROOT/'docs/reports/data/2026-09-11-calibrated-sdcp'


def read(name):
    return json.loads((SOURCE/name).read_text(encoding='utf-8'))


def residuals(pair, satellites):
    times = np.asarray(pair[0]['time_s'])
    assert np.array_equal(times, pair[1]['time_s'])
    phase = [np.column_stack([carrier_range(dict(h, success=True), times,
             r['fs_dfe_hz'], .001, WAVELENGTH) for h in r['satellite_tracking']]) for r in pair]
    states = np.array([[np.asarray(s['position_m'])+np.asarray(s['velocity_mps'])*t
                       for s in satellites] for t in times])
    approximate = np.array([solve_position(s, p)['position_m'] for s, p in
                            zip(states, pair[0]['pseudoranges_m'])])
    geometry = -(np.linalg.norm(states-approximate[:, None, :]-BASELINE, axis=2)
                 -np.linalg.norm(states-approximate[:, None, :], axis=2))
    sdcp = -(phase[1]-phase[0])
    reference = np.column_stack([modified_sdcp(times, sdcp[:, i]*2*np.pi/WAVELENGTH,
                 WAVELENGTH, geometry[:, i])['residual_m'] for i in range(sdcp.shape[1])])
    return times, sdcp, geometry, reference


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (OUTPUT/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    hashes = read('sha256.json')
    for name, digest in hashes.items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == digest
    satellites = read('manifest.json')['satellites']
    scores = {'reference': [], 'calibrated': []}
    for seed in (6000, 6002, 6004):
        t, s, g, ref = residuals([read(f'cal_{seed}_{ch}.json') for ch in (0, 1)], satellites)
        for name, r in [('reference', ref), ('calibrated', calibrated_sdcp(s, g, WAVELENGTH, 0))]:
            scores[name].extend(abs(r[t >= .6]).ravel())
    thresholds = {k: float(np.quantile(v, .99, method='higher')) for k, v in scores.items()}
    rows = []
    for seed in (6100, 6102):
        for kind in ('los', 'strong', 'weak', 'common', 'static'):
            pair = [read(f'{"common" if kind == "common" else "los"}_{seed}_0.json'),
                    read(f'{kind}_{seed}_1.json')]
            t, s, g, ref = residuals(pair, satellites)
            r = calibrated_sdcp(s, g, WAVELENGTH, 0)
            row = {'seed': seed, 'kind': kind, 'methods': {}}
            for name, value in [('reference', ref), ('calibrated', r)]:
                masks = {'early': (t >= .2) & (t < .4), 'late': t >= .6}
                row['methods'][name] = {window: {
                    'prn1_reject_fraction': float(np.mean(abs(value[mask, 0]) > thresholds[name])),
                    'other_reject_fraction': float(np.mean(abs(value[mask, 1:]) > thresholds[name])),
                    'prn1_mean_m': float(value[mask, 0].mean())} for window, mask in masks.items()}
            if kind == 'los':
                row['uncalibrated_bias_sensitivity'] = {str(b): float(np.mean(
                    abs(calibrated_sdcp(s[t >= .6]+b, g[t >= .6], WAVELENGTH, 0)) > thresholds['calibrated']))
                    for b in (0, .001, .003, .005, .01, WAVELENGTH/2)}
            save(f'{kind}_{seed}.json', {'time_s': t.tolist(), 'sdcp_m': s.tolist(),
                 'geometry_m': g.tolist(), 'reference_m': ref.tolist(), 'calibrated_m': r.tolist()})
            rows.append(row)
    save('summary.json', {'thresholds_m': thresholds, 'rows': rows})
    save('manifest.json', {'scope': __doc__, 'source_hashes': hashes,
         'code_hashes': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                        for name in ('sdcp.py', 'audit_calibrated_sdcp.py')},
         'calibration_seeds': [6000,6002,6004], 'evaluation_seeds': [6100,6102],
         'threshold': '99th percentile higher, independent LOS, t>=.6 pooled PRNs',
         'channel_bias_m': 0, 'weighting_or_pvt_changes': False})
    save('sha256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.glob('*.json')})
    print(json.dumps({'thresholds_m': thresholds, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
