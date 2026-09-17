"""Frozen single-PRN multipath experiment through two independent receiver chains."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.antenna import build_antenna_model, direction_vector
from core.config import Direction, MultipathPath
from core.gnss import generate_scenario, _multipath_scene
from core.presets import default_config
from core.sdcp_observation import raw_sdcp, receive as receive_observation
from core.provenance import source_hashes, freeze_source
from experiments.sdcp_support import save_json, statistics
from experiments.paths import RESULTS, reset_output

OUTPUT = RESULTS/'sdcp_multipath'
FREEZE_SOURCE = False


CALIBRATION_SEEDS = (11900, 11902, 11904)
TEST_SEEDS = (12000, 12002, 12004)
TIMES = np.arange(200, 1491)*.001
WINDOW = 200
STRIDE = 10
CONFIRM = 3
MARGIN = 1.25


def cases():
    result = [dict(name=n, kind=n) for n in ('los', 'hardware', 'drift')]
    specs = [
        ('mp_start_strong', -3, .3, 90, 0, 'B', 110, 15),
        ('mp_late_strong', -3, .3, 90, .6, 'B', 110, 15),
        ('mp_start_weak', -20, .3, 90, 0, 'B', 110, 15),
        ('mp_late_weak', -20, .3, 90, .6, 'B', 110, 15),
        ('mp_short', -6, .05, 90, .6, 'B', 110, 15),
        ('mp_medium', -6, .3, 90, .6, 'B', 110, 15),
        ('mp_long', -6, 1.2, 90, .6, 'B', 110, 15),
        ('mp_phase0', -6, .3, 0, .6, 'B', 110, 15),
        ('mp_phase180', -6, .3, 180, .6, 'B', 110, 15),
        ('mp_other_direction', -6, .3, 90, .6, 'B', 210, 35),
        ('mp_both_diff', -6, .3, 90, .6, 'both', 110, 15),
    ]
    for name, db, delay, phase, onset, channels, az, el in specs:
        result.append(dict(name=name, kind='multipath', power_db=db,
            delay_chips=delay, reflection_deg=phase, onset_s=onset,
            channels=channels, azimuth_deg=az, elevation_deg=el))
    return result


def scores(residual):
    ends = np.arange(WINDOW-1, len(residual), STRIDE)
    return ends, np.convolve(residual, np.ones(WINDOW)/WINDOW, mode='valid')[::STRIDE]


def confirmed(score, threshold):
    exceed = np.abs(score) > threshold
    alarm = np.zeros(len(score), dtype=bool)
    alarm[CONFIRM-1:] = np.convolve(exceed.astype(int), np.ones(CONFIRM, int), 'valid') == CONFIRM
    return alarm


def receive(signal, cfg, seed, out, name, hardware=False):
    h, metadata = receive_observation(signal, cfg, seed, hardware)
    np.savez_compressed(out/(name+'.npz'), **{k: v for k, v in h.items() if isinstance(v, np.ndarray)})
    metadata['acquisition'] = {k: v for k, v in metadata['acquisition'].items()
                               if not isinstance(v, np.ndarray)}
    save_json(out/(name+'.json'), metadata)
    print(name, h['status'], len(h['epoch_start_samples']), flush=True)
    return h


def residual(first, second, cfg, geometry):
    fs = cfg.signal.fs_rf_hz/cfg.dfe.decimation
    for h in (first, second):
        assert TIMES[0] >= h['epoch_start_samples'][0]/fs
        assert TIMES[-1] < h['epoch_start_samples'][-1]/fs + .001
    return raw_sdcp(first, second, fs, TIMES, cfg.signal.wavelength_m)-geometry


def main():
    out = reset_output(OUTPUT.name)
    (out/'.gitattributes').write_text('** -text\n', encoding='utf-8')
    started = time.perf_counter()
    cfg = default_config()
    cfg.phone.active_antennas = ('TL', 'TR')
    cfg.signal.duration_ms = cfg.tracking.max_epochs = 1500
    cfg.signal.multipath_paths = []
    cfg.interferers = []
    cfg.tracking.carrier_discriminator = 'costas'
    cfg.phone.channel_delay_ns = {k:0. for k in cfg.phone.channel_delay_ns}
    cfg.phone.estimated_channel_delay_ns = dict(cfg.phone.channel_delay_ns)
    save_json(out/'config.json', asdict(cfg))
    hashes = source_hashes()
    if FREEZE_SOURCE:
        freeze_source(out)
    save_json(out/'plan.json', dict(cases=cases(), calibration_seeds=CALIBRATION_SEEDS,
        test_seeds=TEST_SEEDS, window_samples=WINDOW, stride_samples=STRIDE,
        confirmation=CONFIRM, margin=MARGIN, observation_interval_s=.001,
        ignored_start_s=.2, late_onset_ramp_s=.01, duration_s=1.5,
        phase_policy='Raw NCO, no wrap, no centering, no integer adjustment.',
        threshold='1.25 times largest absolute calibration window mean; frozen before tests.',
        assumptions='Exact known LOS geometry; zero calibrated hardware offset; controlled initial carrier branch.',
        quality_policy='Diagnostic only; no sample filtering.',
        data_symbols=[1,1,-1,1,-1,-1,1,-1], data_symbol_s=.020,
        data_note='Synthetic 50 bps, not valid navigation words; delayed together with echo waveform.',
        source_sha256=hashes, python=sys.version, numpy=np.__version__))
    scene, truth = generate_scenario(cfg)
    waveform = truth['desired_waveform']
    del truth
    fs0 = cfg.signal.fs_rf_hz
    bits = np.array([1,1,-1,1,-1,-1,1,-1])
    for start in range(0, waveform.size, 1000000):
        stop = min(start+1000000, waveform.size)
        b = bits[(np.arange(start,stop)//round(.020*fs0)) % len(bits)]
        waveform[start:stop] *= b
        scene[:,start:stop] *= b
    baseline = np.array(cfg.phone.phase_centers_m['TR'])-np.array(cfg.phone.phase_centers_m['TL'])
    geometry = float(baseline @ direction_vector(cfg.signal.true_direction))
    calibration = []
    for seed in CALIBRATION_SEEDS:
        a = receive(scene[0], cfg, seed, out, f'cal_{seed}_A')
        b = receive(scene[1], cfg, seed+1, out, f'cal_{seed}_B')
        r = residual(a,b,cfg,geometry)
        ends, score = scores(r)
        calibration.append(float(np.max(abs(score))))
        np.savez_compressed(out/f'cal_{seed}_decision.npz', time_s=TIMES,
            residual_m=r, decision_time_s=TIMES[ends], score_m=score)
    threshold = MARGIN*max(calibration)
    save_json(out/'threshold.json', dict(threshold_m=threshold,
        calibration_max_abs_mean_m=calibration, margin=MARGIN, geometry_m=geometry,
        frozen_before_test=True))
    threshold_hash = hashlib.sha256((out/'threshold.json').read_bytes()).hexdigest()
    print('FROZEN THRESHOLD mm:', threshold*1000, flush=True)
    clean_a = {seed:receive(scene[0],cfg,seed,out,f'{seed}_clean_A') for seed in TEST_SEEDS}
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    time_s = np.arange(waveform.size)/fs0
    rows = []
    for case in cases():
        name = case['name']
        both = case.get('channels') == 'both'
        signal_b = scene[1].copy()
        if case['kind'] == 'multipath':
            path = MultipathPath(Direction(case['azimuth_deg'],case['elevation_deg']),
                relative_power_db=case['power_db'], excess_delay_chips=case['delay_chips'],
                reflection_phase_rad=np.deg2rad(case['reflection_deg']))
            echo = _multipath_scene(path,cfg,model,waveform,time_s,np.zeros(2)).truth['component']
            if both:
                signal_a = scene[0].copy()
            for start in range(0, waveform.size, 1000000):
                stop = min(start+1000000, waveform.size)
                envelope = (np.ones(stop-start) if case['onset_s'] == 0 else
                    .5-.5*np.cos(np.pi*np.clip((time_s[start:stop]-case['onset_s'])/.01,0,1)))
                signal_b[start:stop] += echo[1,start:stop]*envelope
                if both:
                    signal_a[start:stop] += echo[0,start:stop]*envelope
            del echo
        elif name == 'hardware':
            signal_b *= 10**(-1/20)*np.exp(1j*np.deg2rad(5.))
        elif name == 'drift':
            for start in range(0, waveform.size, 1000000):
                stop = min(start+1000000, waveform.size)
                signal_b[start:stop] *= np.exp(1j*np.deg2rad(20*np.maximum(time_s[start:stop]-.4,0)))
        for seed in TEST_SEEDS:
            a = receive(signal_a,cfg,seed,out,f'{seed}_{name}_A') if both else clean_a[seed]
            b = receive(signal_b,cfg,seed+1,out,f'{seed}_{name}_B',name=='hardware')
            r = residual(a,b,cfg,geometry)
            ends, score = scores(r)
            dt = TIMES[ends]
            alarm = confirmed(score,threshold)
            onset = case.get('onset_s',.4 if name=='drift' else 0.)
            event = dt >= onset
            first_alarm = dt[alarm & event]
            quality = np.ones(len(TIMES),bool)
            fs = fs0/cfg.dfe.decimation
            for h in (a,b):
                ix = np.searchsorted(h['epoch_start_samples']/fs,TIMES,side='right')-1
                quality &= (h['phase_lock_metric'][ix]>=.8) & (abs(h['pll_error_rad'][ix])<np.pi/2)
            np.savez_compressed(out/f'{seed}_{name}_decision.npz', time_s=TIMES,
                residual_m=r, decision_time_s=dt, score_m=score, alarm=alarm, quality=quality)
            row = dict(case=name, seed=seed, detected=bool(len(first_alarm)),
                first_alarm_s=float(first_alarm[0]) if len(first_alarm) else None,
                delay_s=float(first_alarm[0]-onset) if len(first_alarm) else None,
                event_alarm_count=int(np.sum(alarm & event)), event_decisions=int(np.sum(event)),
                pre_event_alarm_count=int(np.sum(alarm & ~event)),
                tail_score_mean_mm=float(np.mean(score[dt>=1.1])*1000),
                tail_residual=statistics(r[TIMES>=1.1]), all_residual=statistics(r),
                quality_fraction=float(np.mean(quality)))
            rows.append(row)
            save_json(out/'results.json', rows)
            print('RESULT',name,seed,row['detected'],row['delay_s'],flush=True)
        del signal_b
        if both:
            del signal_a
    assert threshold_hash == hashlib.sha256((out/'threshold.json').read_bytes()).hexdigest()
    assert hashes == source_hashes()
    assert 'core.positioning' not in sys.modules and 'core.gps_simulation' not in sys.modules
    save_json(out/'completion.json',dict(records=len(rows),seconds=time.perf_counter()-started,
        threshold_sha256=threshold_hash,source_unchanged=True,no_positioning_import=True))


if __name__ == '__main__':
    main()
