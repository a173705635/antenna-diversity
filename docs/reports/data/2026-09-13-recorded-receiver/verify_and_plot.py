"""Recompute archived recorded-GPS navigation/PVT and evaluate metadata offsets."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3] / 'python_sim_simplified'))
from recorded_receiver import position_channels
from gps_observables import solve_gps_epoch


def main():
    stored = json.loads((ROOT / 'run/position.json').read_text())
    manifest = json.loads((ROOT / 'run/manifest.json').read_text())
    channels = {int(p.stem[3:]): dict(np.load(p), success=True)
                for p in (ROOT / 'run').glob('prn*.npz')}
    times = np.array([e['sample_time_s'] for e in stored['epochs']])
    recomputed = position_channels(channels, manifest['dfe']['fs_out_hz'], times)
    assert recomputed['status'] == stored['status'] == 'evaluated'
    for prn, nav in stored['navigation'].items():
        assert recomputed['navigation'][int(prn)] == nav
    checked = 0
    for a, b in zip(stored['epochs'], recomputed['epochs']):
        assert a['status'] == b['status']
        if a['status'] != 'position':
            continue
        assert a['prns'] == b['prns']
        for key in ('position_m', 'clock_bias_m', 'residual_m', 'raw_pseudorange_m',
                    'corrected_pseudorange_m', 'satellite_positions_m',
                    'satellite_clock_correction_s', 'transmit_tow_s', 'carrier_range_m'):
            np.testing.assert_allclose(a[key], b[key], atol=1e-8, rtol=1e-12)
            checked += 1
    solved = [e for e in stored['epochs'] if e['status'] == 'position']
    first = solved[0]
    ephs = [stored['navigation'][str(p)]['ephemeris'] for p in first['prns']]
    shifted = solve_gps_epoch(ephs, first['transmit_tow_s'], first['receiver_tow_s'] + .02)
    clock_origin_position_difference = float(np.linalg.norm(np.array(first['position_m']) - shifted['position_m']))
    assert clock_origin_position_difference < .1
    # Coordinates enter only here, after navigation and PVT have been recomputed.
    xml = ET.parse(ROOT / 'source/HackRF_Bands-L1.sdrx')
    meta = xml.find('.//{*}position').attrib
    lat, lon = np.deg2rad([float(meta['lat']), float(meta['lon'])])
    height = float(meta['height'])
    a, e2 = 6378137., 6.6943799901413165e-3
    n = a / np.sqrt(1-e2*np.sin(lat)**2)
    reference = np.array([(n+height)*np.cos(lat)*np.cos(lon),
                          (n+height)*np.cos(lat)*np.sin(lon),
                          (n*(1-e2)+height)*np.sin(lat)])
    rotation = np.array([[-np.sin(lon), np.cos(lon), 0.],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    position = np.array([e['position_m'] for e in solved])
    enu = (position-reference) @ rotation.T
    norm = np.linalg.norm(enu, axis=1)
    horizontal = np.linalg.norm(enu[:, :2], axis=1)
    nav_summary = []
    for prn, nav in stored['navigation'].items():
        h = channels[int(prn)]
        d = nav['diagnostics']
        nav_summary.append({'prn': int(prn), 'status': nav['status'],
            'tracking_epochs': len(h['prompt']), 'valid_words': d['valid_words'],
            'tested_words': d['tested_words'], 'complete_subframes': len(nav['subframes']),
            'time_anchors': len(nav['time_anchors']), 'iode': nav['ephemeris']['iode'],
            'used_position_epochs': sum(int(prn) in e['prns'] for e in solved),
            'first_tlm_segment_time_s': nav['time_anchors'][0]['sample_time_s'],
            'first_tlm_tow_s': nav['time_anchors'][0]['tow_s']})
    summary = {'epochs_attempted': len(stored['epochs']), 'positions': len(solved),
        'status_counts': dict(Counter(e['status'] for e in stored['epochs'])),
        'satellite_count_distribution': dict(Counter(len(e['prns']) for e in solved)),
        'navigation': nav_summary, 'metadata_llh': meta,
        'reference_assumption': 'WGS84 ellipsoidal height; metadata has no survey uncertainty or explicit height-datum statement',
        'reference_ecef_m': reference.tolist(), 'mean_enu_m': enu.mean(axis=0).tolist(),
        'std_enu_m': enu.std(axis=0, ddof=0).tolist(),
        'rms_enu_m': np.sqrt(np.mean(enu**2, axis=0)).tolist(),
        'horizontal_rms_m': float(np.sqrt(np.mean(horizontal**2))),
        'three_d_rms_m': float(np.sqrt(np.mean(norm**2))),
        'three_d_median_m': float(np.median(norm)), 'three_d_p95_m': float(np.percentile(norm, 95)),
        'three_d_max_m': float(norm.max()),
        'three_d_demeaned_rms_m': float(np.sqrt(np.mean(np.sum((enu-enu.mean(axis=0))**2, axis=1)))),
        'residual_rms_m': float(np.sqrt(np.mean([r*r for e in solved for r in e['residual_m']]))),
        'clock_bias_first_last_m': [solved[0]['clock_bias_m'], solved[-1]['clock_bias_m']],
        'geometry_condition_min_max': [min(e['geometry_condition'] for e in solved), max(e['geometry_condition'] for e in solved)],
        'numeric_array_or_scalar_checks': checked,
        'clock_origin_shift_s': .02, 'clock_origin_position_change_m': clock_origin_position_difference,
        'evaluation_time_from_file_s': [times[0]+manifest['arguments']['start'], times[-1]+manifest['arguments']['start']]}
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    file_t = np.array([e['sample_time_s'] for e in solved])+manifest['arguments']['start']
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax[0, 0].plot(enu[:, 0], enu[:, 1], '.', ms=3)
    ax[0, 0].plot(0, 0, '+', color='black', ms=12, label='Metadata reference')
    ax[0, 0].set(xlabel='East offset (m)', ylabel='North offset (m)', title='Horizontal code-SPP positions')
    ax[0, 0].axis('equal'); ax[0, 0].legend()
    for i, name in enumerate(('East', 'North', 'Up')):
        ax[0, 1].plot(file_t, enu[:, i], lw=1, label=name)
    ax[0, 1].set(xlabel='Time from IQ file start (s)', ylabel='Offset from metadata (m)', title='Offset includes any metadata datum error')
    ax[0, 1].legend()
    for p in stored['navigation']:
        v = [(e['sample_time_s']+manifest['arguments']['start'], e['residual_m'][e['prns'].index(int(p))])
             for e in solved if int(p) in e['prns']]
        ax[1, 0].plot(*np.array(v).T, lw=.8, label=f'G{int(p):02}')
    ax[1, 0].set(xlabel='Time from IQ file start (s)', ylabel='Post-fit residual (m)', title='Residuals are not pseudorange truth errors')
    ax[1, 0].legend(ncol=5, fontsize=8)
    ax[1, 1].step(file_t, [len(e['prns']) for e in solved], where='mid')
    ax[1, 1].set(xlabel='Time from IQ file start (s)', ylabel='Satellites used', yticks=range(4, 11), ylim=(7.5, 10.5), title='Current-epoch tracking quality gate')
    for a in ax.flat: a.grid(alpha=.25)
    fig.suptitle('Real HackRF GPS L1: 380 offline code positions; no atmospheric correction')
    fig.savefig(ROOT / 'position_validation.png', dpi=150); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
