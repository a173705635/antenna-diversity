"""Single-frequency dual-antenna SDCP -> weighted code/float PVT experiment.

Known 75.9-mm east baseline, shared ideal clock, independent RF/DFE/loops.
No external interference, no integer fixing. All weights use tracked data only.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from config import MultipathPath, Direction, SPEED_OF_LIGHT_MPS
from positioning import carrier_range, solve_position, solve_code_carrier_float
from run_gps_spp import build_scene, satellite_setup, run_record, receiver_reference, serializable
from sdcp import modified_sdcp, observation_weights

WAVELENGTH = SPEED_OF_LIGHT_MPS/1575.42e6
BASELINE = receiver_reference()[1].T @ np.array([.0759, 0., 0.])


def channel_satellites(satellites, channel):
    result = copy.deepcopy(satellites)
    receiver = receiver_reference()[0]
    for s in result:
        # Source geometry only. PVT models baseline separately via effective states.
        s['range_at_zero_m'] += (np.linalg.norm(s['position_m']-receiver-channel*BASELINE)
                                -np.linalg.norm(s['position_m']-receiver))
    return result


def receiver(seed, channel, kind, satellites, duration):
    paths, onset = {}, None
    if kind != 'los' and (channel == 1 or kind == 'common'):
        power = -18. if kind == 'weak' else -6.
        # A localized PRN1 echo; equal relative echo on both antennas for common case.
        paths = {1: [MultipathPath(Direction(110.,15.), relative_power_db=power,
                     excess_delay_chips=.25, reflection_phase_rad=np.pi/2)]}
        onset = None if kind == 'static' else 500
    source_satellites = channel_satellites(satellites, channel)
    scene, cfg = build_scene(source_satellites, duration, multipath_by_prn=paths,
                             multipath_start_ms=onset)
    return run_record(seed+channel, source_satellites, scene, cfg)


def observables(pair, satellites):
    times = np.asarray(pair[0]['time_s'])
    if not np.array_equal(times, pair[1]['time_s']):
        raise ValueError('channel time axes differ')
    p = [np.asarray(r['pseudoranges_m'], dtype=float) for r in pair]
    phases = []
    for record in pair:
        if any(h.get('tracking_status') != 'tracking' for h in record['satellite_tracking']):
            raise ValueError('incomplete tracked pair')
        phases.append(np.column_stack([carrier_range(dict(h, success=True), times,
            record['fs_dfe_hz'], .001, WAVELENGTH) for h in record['satellite_tracking']]))
    if not all(np.isfinite(x).all() for x in p+phases):
        raise ValueError('incomplete continuous observation arc')
    states = np.array([[s['position_m']+s['velocity_mps']*t for s in satellites] for t in times])
    # Estimate a fixed approximate position from reference-interval TL SPP, not source position.
    ref = (times >= .2) & (times < .4)
    approximate = np.median([solve_position(s, obs)['position_m'] for s,obs in zip(states[ref], p[0][ref])], axis=0)
    range_difference = (np.linalg.norm(states-approximate-BASELINE, axis=2)
                        -np.linalg.norm(states-approximate, axis=2))
    residuals = []
    for column in range(len(satellites)):
        # NCO convention is negative range: phi_TR-phi_TL = -(L_TR-L_TL)*2pi/lambda.
        residuals.append(modified_sdcp(times, -(phases[1][:,column]-phases[0][:,column])*2*np.pi/WAVELENGTH,
                         WAVELENGTH, -range_difference[:,column])['residual_m'])
    return times, np.concatenate(p,axis=1), np.concatenate(phases,axis=1), np.concatenate([states,states-BASELINE],axis=1), np.column_stack(residuals)


def statistics(positions, times):
    positions = np.asarray(positions)
    selected = times >= .6
    valid = selected & np.isfinite(positions).all(axis=1)
    result = {'evaluation_epochs': int(selected.sum()), 'solved_epochs': int(valid.sum())}
    if valid.any():
        error = positions[valid]-receiver_reference()[0]
        result.update(rms_m=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
            demeaned_rms_m=float(np.sqrt(np.mean(np.sum((error-error.mean(axis=0))**2,axis=1)))),
            max_m=float(np.max(np.linalg.norm(error,axis=1))))
    return result


def evaluate(pair, satellites, threshold):
    times, p, phase, geometry, residual = observables(pair, satellites)
    weights = {name: np.ones_like(residual) for name in ('plain','soft','hard')}
    for column in range(residual.shape[1]):
        w = observation_weights(residual[:,column], threshold)
        weights['soft'][:,column] = w['soft_pair'][:,0]
        weights['hard'][:,column] = w['hard_pair'][:,0]
    result = {'time_s': times, 'sdcp_residual_m': residual, 'threshold_m': threshold, 'modes': {}}
    for name, pair_weight in weights.items():
        weight = np.concatenate([pair_weight,pair_weight],axis=1)
        scale = np.full_like(weight, np.inf)
        np.divide(1., np.sqrt(weight), out=scale, where=weight>0)
        code_positions = np.full((len(times),3),np.nan)
        for k in range(len(times)):
            try:
                code_positions[k] = solve_position(geometry[k],p[k],sigma_m=30.*scale[k])['position_m']
            except ValueError:
                pass
        mode = {'pair_weights': pair_weight, 'code_positions_m': code_positions,
                'code': statistics(code_positions,times),
                'min_retained_prns': int(np.min(np.sum(pair_weight[times>=.6]>1e-3,axis=1))),
                'prn1_mean_weight': float(pair_weight[times>=.6,0].mean()),
                'other_prns_mean_weight': float(pair_weight[times>=.6,1:].mean())}
        try:
            solved = solve_code_carrier_float(geometry,p,phase,code_sigma_m=30.*scale,carrier_sigma_m=.01*scale)
            mode['float_positions_m'] = solved['position_m']
            mode['float'] = statistics(solved['position_m'],times)
            mode['carrier_residual_m'] = solved['carrier_residual_m']
            used = weight > 1e-3
            mode['used_phase_residual_max_m'] = float(abs(solved['carrier_residual_m'][used]).max())
            mode['phase_screen_passed'] = mode['used_phase_residual_max_m'] <= .05
        except ValueError as exc:
            mode['float_failure'] = str(exc)
        result['modes'][name] = mode
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results/sdcp_positioning')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (args.output/name).write_text(json.dumps(serializable(value),indent=2,allow_nan=False),encoding='utf-8')
    satellites = satellite_setup()
    manifest = {'scope': __doc__, 'satellites': satellites, 'baseline_ecef_m': BASELINE,
        'duration_ms':1000, 'calibration_seeds':[6000,6002,6004], 'evaluation_seeds':[6100,6102],
        'reference_s':[.2,.4], 'evaluation_start_s':.6, 'onset_s':.5,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'cases':'los; strong TR-only PRN1 -6dB; weak TR-only -18dB; common both -6dB; static TR-only -6dB from start; all echoes .25 L1 chips, reflection phase pi/2',
        'geometry':'Known east baseline; direction compensation from TL reference-interval SPP, no true receiver position passed to detector/solver',
        'timing':'existing coarse pseudorange aid; shared ideal clock; no integer ambiguity fixing',
        'threshold_rule':'independent current-chain H0 99th percentile abs SDCP at >=.6s, pooled PRNs; not CFAR certification',
        'weight_application':'same per-PRN weight on BOTH antennas, both code and phase; sigma/sqrt(weight)',
        'source_model':'separate point-antenna RF streams with exact initial geometric range; fixed near-static baseline; no external interference'}
    save('manifest.json',manifest)
    scores=[]
    for seed in manifest['calibration_seeds']:
        pair=[]
        for channel in (0,1):
            print(f'CAL {seed} channel {channel}',flush=True)
            r=receiver(seed,channel,'los',satellites,1000)
            save(f'cal_{seed}_{channel}.json',r)
            pair.append(r)
        times,_,_,_,residual=observables(pair,satellites)
        scores.extend(abs(residual[times>=.6]).ravel())
    threshold=float(np.quantile(scores,.99,method='higher'))
    save('threshold.json',{'threshold_m':threshold,'epochs_times_prns':len(scores),'calibration_seeds':manifest['calibration_seeds']})
    print(f'FROZEN {threshold*1000:.6f} mm',flush=True)
    summaries=[]
    for seed in manifest['evaluation_seeds']:
        clean=[]
        for channel in (0,1):
            r=receiver(seed,channel,'los',satellites,1000)
            save(f'los_{seed}_{channel}.json',r)
            clean.append(r)
        control=None
        for kind in ('los','strong','weak','common','static'):
            pair=clean if kind=='los' else [clean[0],None]
            if kind!='los':
                for channel in ((0,1) if kind=='common' else (1,)):
                    print(f'EVAL {kind} {seed} channel {channel}',flush=True)
                    pair[channel]=receiver(seed,channel,kind,satellites,1000)
                    save(f'{kind}_{seed}_{channel}.json',pair[channel])
            result=evaluate(pair,satellites,threshold)
            if kind=='los':
                control=result
            mask=np.asarray(result['time_s'])>=.6
            for name,mode in result['modes'].items():
                for field in ('code','float'):
                    key=f'{field}_positions_m'
                    if key in mode and key in control['modes'][name]:
                        delta=np.asarray(mode[key])[mask]-np.asarray(control['modes'][name][key])[mask]
                        valid=np.isfinite(delta).all(axis=1)
                        mode[f'{field}_paired_los_change_rms_m']=float(np.sqrt(np.mean(np.sum(delta[valid]**2,axis=1)))) if valid.any() else None
            save(f'analysis_{kind}_{seed}.json',result)
            summary={'kind':kind,'seed':seed,'modes':{n:{k:v for k,v in m.items() if k not in
                ('pair_weights','code_positions_m','float_positions_m','carrier_residual_m')} for n,m in result['modes'].items()}}
            summaries.append(summary)
            print(json.dumps(summary),flush=True)
    save('summary.json',summaries)
    save('sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.json') if p.name!='sha256.json'})


if __name__=='__main__':
    main()
