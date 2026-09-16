"""Paired sampled-IQ signal recovery research; see RECOVERY_FIFTY_PLAN.md."""
import argparse
from dataclasses import dataclass, asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from presets import mate70_phase_centers
from run_gps_spp import serializable
from signal_recovery import (N, codes, sky, yaw_matrix, steering, synthesize,
                             array_weights, observe, position_solution, refine_source)


@dataclass
class Case:
    method: str = 'project'
    satellites: int = 8
    antennas: int = 4
    duration: int = 60
    step: int = 1
    spoof_db: float = 10.
    spoof_fraction: float = 1.
    genuine: bool = True
    angle: float = 60.
    cn0: float = 45.
    phase_deg: float = 0.
    amplitude_db: float = 0.
    geometry_mm: float = 0.
    yaw_error_deg: float = 0.
    phase_drift_deg_s: float = 0.
    multipath: float = 0.
    clip: float = 0.
    loading: float = .01
    min_gain: float = .05
    ratio_limit: float = 3.
    motion: bool = True
    clustered: bool = False
    drift_mps: float = 5.
    attack_end: float = 1e9
    power_ramp: float = 0.
    coherent_s: float = .02
    admission_residual: float = 1e9
    refinement_iterations: int = 2


def save(path, data):
    path.write_text(json.dumps(serializable(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def summarize(t, errors, valid, used, residual, genuine):
    post = t >= 10
    e = errors[post]; v = valid[post]
    good = v & (e <= 20)
    longest = run = 0
    for bit in good:
        run = run + 1 if bit else 0
        longest = max(longest, run)
    return dict(opportunities=int(post.sum()), solved=int(v.sum()),
                good20=int(good.sum()), misleading30=int((v & (e>=30)).sum()),
                misleading50=int((v & (e>=50)).sum()), availability=float(v.mean()),
                good20_fraction=float(good.mean()),
                p95_m=float(np.quantile(e[v], .95)) if v.any() else None,
                maximum_m=float(e[v].max()) if v.any() else None,
                minimum_satellites=int(used[post].min()),
                longest_good_sample_span_s=float(max(longest-1, 0)*(t[1]-t[0])),
                first_good_s=float(t[post][good][0]-10) if good.any() else None,
                no_authentic_solutions=int(v.sum()) if not genuine else 0,
                residual_p95_m=float(np.quantile(residual[post][v], .95)) if v.any() else None)


def run_case(case, seed, attacked):
    spec = codes(case.satellites)
    initial_sky = sky(seed, case.satellites, case.clustered)
    nominal = np.array(list(mate70_phase_centers().values()))[:case.antennas]
    rng = np.random.default_rng(seed+100000)
    actual = nominal + case.geometry_mm*.001*rng.normal(size=(4,3))[:case.antennas]
    phase_error = rng.normal(size=4)[:case.antennas]*case.phase_deg
    amplitude_error = rng.normal(size=4)[:case.antennas]*case.amplitude_db
    phase_rate = rng.normal(size=4)[:case.antennas]*case.phase_drift_deg_s
    t = np.arange(0, case.duration+1, case.step, dtype=float)
    methods=('single','gated',case.method) if case.method=='refined' else ('single',case.method)
    outputs = {method:dict(position=[],used=[],quality=[],residual=[],condition=[],ranges=[],valid=[],
                           delay=None,reacquisitions=0) for method in methods}
    truth_position=[]; satellite_history=[]; eigen=[]; clip=[]; suppression=[]; retained=[]; iq_example=None
    for epoch, second in enumerate(t):
        directions = initial_sky @ yaw_matrix(np.radians(.004*second)).T
        pos = np.array([50*np.sin(.05*second), 50*(1-np.cos(.05*second)), 0.]) if case.motion else np.zeros(3)
        yaw = .05*second if case.motion else 0.
        truth_position.append(pos)
        attack = attacked and second >= 10 and second < case.attack_end
        displacement = case.drift_mps*max(second-10, 0)
        fake = pos + np.array([displacement, 0., 0.])
        tangent = np.cross(directions[0], [0.,0.,1.]); tangent /= np.linalg.norm(tangent)
        angle = np.radians(case.angle)
        source = directions[0]*np.cos(angle)+tangent*np.sin(angle)
        gain = 10**(amplitude_error/20)*np.exp(1j*np.radians(phase_error+second*phase_rate))
        # Always draw four channels, preserving paired channel-subset noise.
        noise = (rng.normal(size=(4,N))+1j*rng.normal(size=(4,N)))[:case.antennas]/np.sqrt(2)
        x, satellites, truth = synthesize(spec, directions, pos, fake, source, actual,
            yaw, gain, noise, case.cn0, case.coherent_s,
            case.spoof_db+case.power_ramp*max(second-10,0),
            round(case.satellites*case.spoof_fraction) if attack else 0,
            genuine=case.genuine or not attack, multipath=case.multipath, clip=case.clip)
        desired = steering(directions, nominal, yaw+np.radians(case.yaw_error_deg))
        satellite_history.append(satellites)
        # Coarse satellite range aid is independent of current true/fake receiver location.
        coarse = np.round((21e6+np.arange(case.satellites)*17000+75000)/10000)*10000
        for method, result in outputs.items():
            weights, eligible, info = array_weights(x, desired, 'gated' if method=='refined' else method,
                                                   case.loading, case.ratio_limit, case.min_gain)
            if method=='refined' and info['eigen_ratio']>=case.ratio_limit:
                weights,eligible=refine_source(x,spec,desired,coarse,weights,case.refinement_iterations,case.min_gain)
            ranges, delay, quality, reacquired = observe(x, spec, weights, coarse, result['delay'])
            result['delay'] = delay
            result['reacquisitions'] += int(reacquired.sum()) if epoch else 0
            valid = eligible & (quality>=20.)
            position, condition, residual = position_solution(satellites, ranges, valid)
            if residual > case.admission_residual:
                position[:] = np.nan
            result['position'].append(position); result['used'].append(int(valid.sum()))
            result['quality'].append(quality); result['condition'].append(condition); result['residual'].append(residual)
            result['ranges'].append(ranges); result['valid'].append(valid)
            if method == case.method:
                eigen.append(info['eigen_ratio'])
                # Components used only for evaluation AFTER receiver decisions.
                norm = np.linalg.norm(weights,axis=0)
                suppression.append(abs(weights.conj().T @ truth['source_vector'])/norm)
                retained.append(abs(np.sum(weights.conj()*truth['actual'],axis=0))/norm)
        clip.append(truth['clip_fraction'])
        if second == 10:
            iq_example=x.copy()
    truth_position=np.array(truth_position); rows={}; traces={}
    for method, result in outputs.items():
        position=np.array(result['position']); error=np.linalg.norm((position-truth_position)[:,:2],axis=1)
        valid=np.isfinite(position).all(axis=1)
        rows[method]=summarize(t,error,valid,np.array(result['used']),np.array(result['residual']),case.genuine or not attacked)
        rows[method]['reacquisitions']=result['reacquisitions']
        traces.update({method+'_position':position,method+'_error':error,
                       method+'_used':np.array(result['used']),method+'_quality':np.array(result['quality']),
                       method+'_condition':np.array(result['condition']),method+'_residual':np.array(result['residual'])})
        traces[method+'_ranges']=np.array(result['ranges'])
        traces[method+'_valid']=np.array(result['valid'])
    traces.update(time_s=t,satellites=np.array(satellite_history),truth_position=truth_position,eigen_ratio=np.array(eigen),
                  source_response=np.array(suppression),genuine_response=np.array(retained),clip_fraction=np.array(clip))
    return rows,traces,iq_example


def first_rounds():
    methods=('beam','mvdr','blind','project','gated')
    return [(i+1,f'{method}_power{power}',Case(method=method,spoof_db=power))
            for i,(power,method) in enumerate((p,m) for p in (10.,0.) for m in methods)]


def geometry_rounds():
    base=Case(method='gated')
    changes=[('two_antennas',dict(antennas=2)),('three_antennas',dict(antennas=3)),
             ('12_satellites',dict(satellites=12)),('16_satellites',dict(satellites=16)),
             ('24_satellites',dict(satellites=24)),('half_spoofed',dict(spoof_fraction=.5)),
             ('quarter_spoofed',dict(spoof_fraction=.25)),('no_authentic',dict(genuine=False)),
             ('source_one_degree',dict(angle=1.)),('clustered_sky',dict(clustered=True))]
    return [(11+i,label,replace(base,**change)) for i,(label,change) in enumerate(changes)]


def error_rounds():
    base=Case(method='gated')
    changes=[('phase_10deg',dict(phase_deg=10.)),('amplitude_2db',dict(amplitude_db=2.)),
             ('geometry_3mm',dict(geometry_mm=3.)),('attitude_10deg',dict(yaw_error_deg=10.)),
             ('phase_drift',dict(phase_drift_deg_s=.3)),('reflection',dict(multipath=.5)),
             ('adc_clip',dict(clip=5.)),
             ('refine_once',dict(method='refined',refinement_iterations=1)),
             ('refine_twice',dict(method='refined',refinement_iterations=2)),
             ('refine_four',dict(method='refined',refinement_iterations=4))]
    return [(21+i,label,replace(base,**change)) for i,(label,change) in enumerate(changes)]


def stress_rounds():
    base=Case(method='refined',refinement_iterations=2)
    changes=[('equal_power',dict(spoof_db=0.)),('weak_spoofer',dict(spoof_db=-10.)),
             ('cn0_35',dict(cn0=35.)),('cn0_30',dict(cn0=30.)),
             ('one_ms_coherent',dict(coherent_s=.001)),('same_direction',dict(angle=0.)),
             ('quarter_spoofed_refined',dict(spoof_fraction=.25)),('slow_drag',dict(drift_mps=.1)),
             ('withdrawal_30s',dict(attack_end=30.)),
             ('combined_errors',dict(phase_deg=10.,amplitude_db=2.,geometry_mm=3.,
                  yaw_error_deg=10.,phase_drift_deg_s=.3,multipath=.3,clip=5.))]
    return [(31+i,label,replace(base,**change)) for i,(label,change) in enumerate(changes)]


def holdout_rounds():
    base=Case(method='refined',refinement_iterations=2,duration=600,step=2)
    changes=[('long_8_satellites',dict()),('long_12_satellites',dict(satellites=12)),
             ('long_16_satellites',dict(satellites=16)),('long_24_satellites',dict(satellites=24)),
             ('static_30_minutes',dict(motion=False,duration=1800,step=5)),
             ('long_combined_errors',dict(phase_deg=10.,amplitude_db=2.,geometry_mm=3.,
                  yaw_error_deg=10.,phase_drift_deg_s=.3,multipath=.3,clip=5.)),
             ('long_no_authentic',dict(genuine=False)),('long_same_direction',dict(angle=0.)),
             ('long_cn0_35',dict(cn0=35.)),('long_withdrawal',dict(attack_end=300.))]
    return [(41+i,label,replace(base,**change)) for i,(label,change) in enumerate(changes)]


def aggregate(rows, method, attacked):
    chosen=[r[method] for r in rows if r['attacked']==attacked]
    n=sum(r['opportunities'] for r in chosen)
    return dict(opportunities=n,solved=sum(r['solved'] for r in chosen),
                good20=sum(r['good20'] for r in chosen),
                misleading30=sum(r['misleading30'] for r in chosen),
                misleading50=sum(r['misleading50'] for r in chosen),
                no_authentic_solutions=sum(r['no_authentic_solutions'] for r in chosen),
                good20_fraction=sum(r['good20'] for r in chosen)/n,
                worst_seed_p95_m=max((r['p95_m'] for r in chosen if r['p95_m'] is not None),default=None))


def execute(out, rounds, seeds):
    for number,label,case in rounds:
        directory=out/f'round{number:02d}';directory.mkdir(parents=True,exist_ok=False)
        save(directory/'frozen.json',dict(round=number,label=label,case=asdict(case),seeds=seeds,
            source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ('signal_recovery.py','audit_recovery_fifty.py')}))
        rows=[]; start=time.monotonic()
        for seed in seeds:
            for attacked in (False,True):
                result,traces,iq=run_case(case,seed,attacked)
                rows.append(dict(seed=seed,attacked=attacked,**result))
                np.savez_compressed(directory/f'{seed}_{int(attacked)}.npz',**traces)
                if seed==seeds[0]:
                    np.savez_compressed(directory/f'iq_{int(attacked)}.npz',samples=iq)
        summary=dict(round=number,label=label,method=case.method,case=asdict(case),
                     candidate_normal=aggregate(rows,case.method,False),candidate_attack=aggregate(rows,case.method,True),
                     baseline_normal=aggregate(rows,'single',False),baseline_attack=aggregate(rows,'single',True),
                     elapsed_s=time.monotonic()-start)
        if case.method=='refined':
            summary['previous_normal']=aggregate(rows,'gated',False)
            summary['previous_attack']=aggregate(rows,'gated',True)
        save(directory/'rows.json',rows);save(directory/'summary.json',summary)
        print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',type=int,default=1)
    parser.add_argument('--rounds',type=int,nargs='+')
    parser.add_argument('--seeds',type=int,nargs='+',default=[810001,810002,810003])
    args=parser.parse_args()
    rounds={1:first_rounds,2:geometry_rounds,3:error_rounds,4:stress_rounds,5:holdout_rounds}[args.phase]()
    if args.rounds:rounds=[r for r in rounds if r[0] in args.rounds]
    execute(args.output,rounds,args.seeds)


if __name__=='__main__':main()
