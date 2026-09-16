"""Independently reconstruct the no-positioning audit from saved raw NCOs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    root = parser.parse_args().directory
    hashes = json.loads((root/'sha256.json').read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    summary = json.loads((root/'summary.json').read_text())
    cfg = json.loads((root/'config.json').read_text())
    fs = cfg['signal']['fs_rf_hz']/cfg['dfe']['decimation']
    wave = 299792458./cfg['signal']['carrier_hz']
    direction = cfg['signal']['true_direction']
    az, el = np.deg2rad([direction['azimuth_deg'], direction['elevation_deg']])
    vector = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    centers = cfg['phone']['phase_centers_m']
    geometry = (np.array(centers['TR'])-np.array(centers['TL'])) @ vector
    np.testing.assert_allclose(geometry, summary['geometry_m'], atol=1e-14)
    checked_metrics = 0
    fitted_parameters = 0
    with np.load(root/'aligned.npz') as arrays:
        times = arrays['time_s']
        masks = {'all': times >= .01, 'settled': times >= .2,
                 'before_event': (times >= .2) & (times < .55),
                 'event': (times >= .6) & (times < .8), 'tail': times >= 1.1}
        for row in summary['rows']:
            seed, case = row['seed'], row['case']
            phase = []
            quality = np.ones(len(times), bool)
            for name in (f'{seed}_A', f'{seed}_{case}_B'):
                metadata = json.loads((root/(name+'.json')).read_text())
                assert metadata['acquisition']['success']
                assert metadata['tracking_status'] == 'tracking'
                with np.load(root/(name+'.npz')) as h:
                    starts = h['epoch_start_samples']/fs
                    ix = np.searchsorted(starts, times, side='right')-1
                    assert np.all(ix >= 0)
                    assert np.all(times < starts[ix]+.001)
                    phase.append(np.array([h['carrier_nco_phase_rad'][i] +
                        2*np.pi*h['carrier_nco_frequency_hz'][i]*(t-starts[i])
                        for t, i in zip(times, ix)]))
                    quality &= (h['phase_lock_metric'][ix] >= .8) & (abs(h['pll_error_rad'][ix]) < np.pi/2)
            reconstructed = (phase[0]-phase[1])*wave/(2*np.pi)
            label = f'{seed}_{case}'
            np.testing.assert_allclose(reconstructed, arrays[label+'_d_m'], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(quality, arrays[label+'_quality'])
            injection = np.zeros(len(times))
            if case in ('fixed_plus10', 'fixed_minus10', 'hardware'):
                injection[:] = {'fixed_plus10': 10., 'fixed_minus10': -10., 'hardware': 8.}[case]
            if case == 'ramp':
                injection[times >= .4] = 20*(times[times >= .4]-.4)
            if case == 'sine':
                injection[times >= .4] = 10*np.sin(2*np.pi*2*(times[times >= .4]-.4))
            if case == 'step':
                injection[times >= .6] = 30
            expected = geometry-injection*wave/360
            np.testing.assert_allclose(expected, arrays[label+'_expected_m'], atol=1e-14, rtol=0)
            fit = times >= .55
            recovered_deg = -(reconstructed-arrays[f'{seed}_zero_d_m'])*360/wave
            if case == 'ramp':
                design = np.column_stack([times[fit]-.4, np.ones(fit.sum())])
                slope, intercept = np.linalg.solve(design.T @ design, design.T @ recovered_deg[fit])
                np.testing.assert_allclose([slope, intercept],
                    [row['ramp_fit']['slope_deg_s'], row['ramp_fit']['intercept_deg']], atol=1e-8, rtol=0)
                fitted_parameters += 2
            if case == 'sine':
                angle = 4*np.pi*(times[fit]-.4)
                design = np.column_stack([np.sin(angle), np.cos(angle), np.ones(fit.sum())])
                sine, cosine, dc = np.linalg.solve(design.T @ design, design.T @ recovered_deg[fit])
                np.testing.assert_allclose([np.hypot(sine, cosine), -np.arctan2(cosine, sine)/(4*np.pi)*1000, dc],
                    [row['sine_fit']['amplitude_deg'], row['sine_fit']['lag_ms'], row['sine_fit']['dc_deg']], atol=1e-8, rtol=0)
                fitted_parameters += 3
            present = ~((times >= .6) & (times < .75)) if case == 'gap' else np.ones(len(times), bool)
            for region, mask in masks.items():
                values = {'raw_sdcp': reconstructed[mask],
                    'truth_error': (reconstructed-expected)[mask & present],
                    'paired_error': (reconstructed-arrays[f'{seed}_zero_d_m']+injection*wave/360)[mask & present]}
                for label, x in values.items():
                    assert np.isfinite(x).all()
                    recorded = row['regions'][region][label]
                    assert len(x) == recorded['epochs']
                    for metric, value in {'mean_mm': np.mean(x)*1000, 'std_mm': np.std(x)*1000,
                        'rms_mm': np.linalg.norm(x)/np.sqrt(len(x))*1000,
                        'max_abs_mm': np.max(abs(x))*1000}.items():
                        np.testing.assert_allclose(value, recorded[metric], atol=1e-8, rtol=0)
                        checked_metrics += 1
                assert row['regions'][region]['quality_pass'] == int(quality[mask].sum())
    print(json.dumps({'hashes_verified': len(hashes), 'records_reconstructed': len(summary['rows']),
                      'metrics_verified': checked_metrics, 'fitted_parameters_verified': fitted_parameters,
                      'common_time_domains_verified': True}))


if __name__ == '__main__':
    main()
