"""Run externally generated dual-antenna IQ through the existing receiver chains."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import tomllib

import numpy as np

from acquisition import acquire
from audit_sdcp_hardware_bias import CASES, evaluate, stats
from dfe import process_dfe
from gnss import ca_code, _desired_waveform
from presets import default_config
from rf import process_rf
from run_gps_spp import serializable
from tracking import track


def receive(x, cfg, seed, name):
    adc, ri = process_rf(x[None, :], cfg.rf, cfg.reference_impedance_ohm, seed)
    baseband, di = process_dfe(adc, cfg.dfe, cfg.signal.fs_rf_hz, cfg.signal.if_hz)
    del adc
    fs = di['fs_out_hz']
    acq = acquire(baseband[0], fs, cfg.signal.prn, cfg.signal.code_rate_hz, cfg.acquisition)
    if not acq['success']:
        return {'status': 'acquisition_failed', 'channel': name, 'acquisition': acq}
    h = track(baseband[0], fs, cfg.signal.prn, acq, cfg.tracking,
              cfg.signal.carrier_hz, cfg.signal.code_rate_hz)
    return {'status': h['status'], 'success': h['success'], 'channel': name,
        'rf_noise_seed': seed, 'fs_hz': fs, 'acquisition': acq,
        'rf_info': ri, 'saturation_fraction': float(ri['saturation_fraction'][0]),
        **{key: h[key] for key in ('epoch_start_samples', 'carrier_nco_phase_rad',
            'carrier_nco_frequency_hz', 'pll_error_rad', 'phase_lock_metric')}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iq', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output/name).write_text(json.dumps(serializable(value), indent=2, allow_nan=False), encoding='utf-8')
    meta = tomllib.loads((args.iq/'source_metadata.toml').read_text())
    checks = tomllib.loads((args.iq/'source_checks.toml').read_text())
    assert checks['all_pass'], 'External source acceptance did not pass'
    for name, expected in meta['files_sha256'].items():
        with (args.iq/name).open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == expected
        assert (args.iq/name).stat().st_size == meta['samples_per_channel']*16
    for name in ('source_metadata.toml', 'source_checks.toml', 'prn1.i8'):
        shutil.copyfile(args.iq/name, args.output/name)
    cfg = default_config()
    cfg.signal.duration_ms = cfg.tracking.max_epochs = 1000
    cfg.interferers = []
    cfg.signal.multipath_paths = []
    assert meta['sample_rate_hz'] == cfg.signal.fs_rf_hz and meta['if_hz'] == cfg.signal.if_hz
    assert meta['carrier_hz'] == cfg.signal.carrier_hz and not meta['source_noise']
    source_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in Path(__file__).parent.glob('*.py')}
    config_record = asdict(cfg); config_record['plot']['output_dir'] = str(cfg.plot.output_dir)
    save('manifest.json', {'source': meta, 'receiver_config': config_record,
        'scope': 'No multipath/interference; external native noiseless PRN/carrier/array primitives.',
        'code_polarity_mapping': 'Native PRN1 is exactly the negative of local PRN1; apply -1 to BOTH input IQ files, fixed code convention, no fitted phase.',
        'seed_pairs': [[9900, 9901], [9902, 9903]], 'source_sha256': source_hashes,
        'generator_adapter_sha256': hashlib.sha256(Path(__file__).with_name('external_gnss_generate.jl').read_bytes()).hexdigest()})
    a = np.memmap(args.iq/'TL.c128', dtype='<c16', mode='r')
    b = np.memmap(args.iq/'TR.c128', dtype='<c16', mode='r')
    prn = np.fromfile(args.iq/'prn1.i8', dtype='i1')
    assert np.array_equal(-prn, ca_code(1)), 'PRN chips differ beyond documented bipolar convention'
    # Compare sparse RF samples before receiver noise; no adjustment from observed phase.
    indices = np.unique(np.linspace(0, len(a)-1, 200000, dtype=np.int64))
    wave, _, _ = _desired_waveform(cfg, indices, indices/cfg.signal.fs_rf_hz)
    amplitude = np.sqrt(10**((cfg.signal.power_dbm-30)/10)*cfg.reference_impedance_ohm)
    positions = cfg.phone.phase_centers_m
    direct = {}
    for name, signal in [('TL', a), ('TR', b)]:
        steering = np.exp(-2j*np.pi/cfg.signal.wavelength_m*np.dot(positions[name], meta['look_vector']))
        reference = wave/amplitude*steering
        diff = abs(-signal[indices]-reference)
        direct[name] = {'samples': len(indices), 'max_complex_error': float(diff.max()),
            'rms_complex_error': float(np.sqrt(np.mean(diff**2))),
            'sample_count_error_gt_1e_5': int(np.sum(diff>1e-5))}
    save('independent_waveform_comparison.json', {'prn_chips_equal_after_fixed_negation': True, 'channels': direct,
        'note': 'Native bipolar convention negated for both channels; no fitted phase/scale/code shift.'})
    times = np.arange(200,999)*.001
    rows = []; traces = {'time_s': times}
    for seed in (9900,9902):
        print(f'RUN {seed} TL shared', flush=True)
        first = receive(np.asarray(a)*(-amplitude), cfg, seed, 'TL')
        save(f'{seed}_TL.json', first)
        control = None
        for label,(ant_db,ant_deg,rf_db,rf_deg) in CASES.items():
            print(f'RUN {seed} {label} TR', flush=True)
            cfg.rf = replace(cfg.rf, amplitude_error_db=rf_db, phase_error_deg=rf_deg)
            factor = -amplitude*10**(ant_db/20)*np.exp(1j*np.deg2rad(ant_deg))
            second = receive(np.asarray(b)*factor, cfg, seed+1, 'TR')
            case = {'status': 'evaluated_not_lock_certified' if first.get('success') and second.get('success') else 'receiver_failed',
                    'fs_hz': first.get('fs_hz'), 'wavelength_m': cfg.signal.wavelength_m,
                    'ideal_geometry_path_m': -meta['geometry_range_m'], 'channels': [first,second],
                    'parameters_TR': {'antenna_db':ant_db,'antenna_deg':ant_deg,'rf_db':rf_db,'rf_deg':rf_deg}}
            save(f'{seed}_{label}.json',case)
            row = {'seed':seed,'case':label,'status':case['status']}
            if case['status'] == 'evaluated_not_lock_certified':
                d,error,q = evaluate(case,times)
                if label == 'zero': control = error.copy()
                row.update({'raw_sdcp':stats(d), 'geometry_residual':stats(error),
                    'paired_change':stats(error-control) if control is not None else {'epochs':0},
                    'quality_pass':int(q.sum()),'finite_epochs':int(np.isfinite(error).sum()),
                    'expected_injected_range_mm':-(ant_deg+rf_deg)/360*cfg.signal.wavelength_m*1000,
                    'max_adjacent_step_mm':float(np.nanmax(abs(np.diff(error)))*1000)})
                traces[f'{seed}_{label}_residual_m']=error
                traces[f'{seed}_{label}_quality']=q
            rows.append(row); print(json.dumps(row),flush=True)
        cfg.rf = replace(cfg.rf, amplitude_error_db=0., phase_error_deg=0.)
    save('summary.json',{'rows':rows,'statistics':'All finite points, 200..998 ms, 1 ms; ddof=0; no modulo or mean subtraction.'})
    np.savez_compressed(args.output/'traces.npz',**traces)
    assert source_hashes == {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                            for f in Path(__file__).parent.glob('*.py')}
    save('sha256.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.output.iterdir()})


if __name__ == '__main__': main()
