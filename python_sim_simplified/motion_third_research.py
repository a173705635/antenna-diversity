"""Sequential paired experiments. Truth exists only in simulator and scorer."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).parent
OUT = ROOT/'results'/'research-third-hundred'
CASES = ['common', 'diverse', 'offset', 'wrap', 'yaw', 'static', 'uniform',
         'bias', 'scale', 'delay', 'timestamp', 'drift', 'prn_drift',
         'outlier', 'phase_center', 'phase_center_uncertain', 'slip',
         'moving', 'moving_diverse', 'rapid', 'two_antennas', 'three_antennas',
         'body_lever', 'spoof_gyro', 'gaussian']


def simulate(case, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 12, 121); dt = t[1]-t[0]
    rate = np.column_stack((.21*np.cos(.7*t), .28*np.sin(.5*t), .31+.15*np.cos(.9*t)))
    if case in ('static', 'uniform'): rate[:] = 0
    if case == 'yaw': rate[:, :2] = 0
    if case == 'rapid': rate *= 3
    rot = np.repeat(np.eye(3)[None], len(t), axis=0)
    for k in range(1, len(t)):
        rot[k] = rot[k-1] @ Rotation.from_rotvec((rate[k-1]+rate[k])*dt/2).as_matrix()
    p = np.array([[-.03795,.08045,0],[.03795,.08045,0],[-.03795,-.08045,0],[.03795,-.08045,0]])
    if case == 'two_antennas': p = p[:2]
    if case == 'three_antennas': p = p[:3]
    baseline = p[1:]-p[0]
    perturb = rng.normal(0, .0015, baseline.shape) if case.startswith('phase_center') else np.zeros_like(baseline)
    true_baseline = baseline+perturb
    measured_baseline = true_baseline+rng.normal(0,.00005,baseline.shape)
    u0 = np.array([.45,-.35,.821583836]); u0 /= np.linalg.norm(u0)
    directions = np.repeat(u0[None,None,:], len(t), axis=0)
    directions = np.repeat(directions, 7, axis=1)
    diverse = case in ('diverse','moving_diverse')
    if diverse:
        az = np.linspace(0, 2*np.pi,7,endpoint=False)
        dirs = np.column_stack((.85*np.cos(az),.85*np.sin(az),np.full(7,np.sqrt(1-.85**2))))
        directions[:] = dirs
    if case in ('moving','moving_diverse'):
        for k in range(len(t)):
            directions[k] = Rotation.from_rotvec(np.array([0,.035,0])*t[k]).apply(directions[k])
    wave = 2*np.pi/(299792458/1575.42e6)
    x = np.einsum('tij,bj->tbi',rot,true_baseline)*wave
    phase = np.einsum('tbi,tpi->tbp',x,directions)
    if case in ('offset','wrap'): phase += rng.uniform(-2.5,2.5,(1,len(baseline),7))
    if case == 'drift': phase += (.45*np.sin(.24*t)+.09*t)[:,None,None]*np.arange(1,len(baseline)+1)[None,:,None]
    if case == 'prn_drift': phase += t[:,None,None]*rng.normal(0,.04,(1,len(baseline),7))
    if case == 'outlier': phase[:,:,0] += .8*np.sin(.6*t)[:,None]
    if case == 'slip': phase[61:,0,2] += 1.7
    noise = rng.normal(0,.025,phase.shape) if case == 'gaussian' else rng.uniform(-.025,.025,phase.shape)
    phase += noise
    phase = np.angle(np.exp(1j*phase))
    bias = np.array([.009,-.008,.012]) if case in ('bias','spoof_gyro') else np.zeros(3)
    scale = np.array([1.04,.96,1.06]) if case == 'scale' else np.ones(3)
    delay = .18 if case == 'delay' else 0.
    gyro_t = t+delay
    gyro = rate*scale+bias+rng.uniform(-.0008,.0008,rate.shape)
    if case == 'spoof_gyro': gyro += .10*np.sin(t)[:,None]
    # Independent simulated instrument calibration; not online truth or real phone data.
    bias_readings = bias+rng.uniform(-.001,.001,(400,3))
    scale_readings = scale+rng.uniform(-.001,.001,(100,3))
    o = dict(time=t,gyro_time=gyro_t,gyro=gyro,phase=phase,baselines=baseline,
             wave_number=wave,bias_readings=bias_readings,scale_readings=scale_readings,
             clock_offset_readings=delay+rng.uniform(-.001,.001,100),
             timestamp_bound=.06 if case == 'timestamp' else .001,
             measured_baselines=measured_baseline,phase_center_bound=.003 if case == 'phase_center_uncertain' else .00015,
             gyro_error_bound=.002,phase_noise_bound=.025,gyro_certified=case!='spoof_gyro',
             bounded_phase_noise=case!='gaussian',body_lever=np.array([.03,.02,.01]) if case=='body_lever' else np.zeros(3))
    truth = dict(common=not diverse, direction=directions[len(t)//2].mean(axis=0),
                 rotation=rot, origin=np.zeros(3))
    return o,truth


def load(path):
    spec=importlib.util.spec_from_file_location('motion_snapshot',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def evaluate(path, seed):
    module = load(path);rows=[]
    for case in CASES:
        o, truth = simulate(case,seed);result=module.estimate(o)
        direction=np.array(result['direction']);ref=truth['direction'];ref=ref/(np.linalg.norm(ref)+1e-30)
        error=float(np.degrees(np.arccos(np.clip(direction@ref,-1,1))))
        rows.append(dict(case=case,**result,direction_error_deg=error,
                         false_alarm=bool(result['status']=='common' and not truth['common']),
                         missed_common=bool(result['status']=='diverse' and truth['common']),
                         common_truth=truth['common']))
    return rows


def record(number, mechanism, target, rationale):
    folder=OUT/f'iteration{number}';folder.mkdir(parents=True,exist_ok=False)
    prev=OUT/f'iteration{number-1}'/'after.py' if number>276 else ROOT/'motion_third_baseline.py'
    (folder/'before.py').write_bytes(prev.read_bytes());(folder/'after.py').write_bytes((ROOT/'motion_third.py').read_bytes())
    proposal=dict(iteration=number,mechanism=mechanism,target=target,rationale=rationale,seed=2760,
                  acceptance='Same 25 observations before/after; report both improvement and regressions.')
    (folder/'proposal.json').write_text(json.dumps(proposal,indent=2),encoding='utf8')
    before=evaluate(folder/'before.py',2760);after=evaluate(folder/'after.py',2760)
    results=dict(before=before,after=after,seed=2760,source_hashes={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('before.py','after.py')})
    (folder/'results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    b=next(x for x in before if x['case']==target);a=next(x for x in after if x['case']==target)
    summary=dict(iteration=number,mechanism=mechanism,target=target,before=b,after=a,
        all_before=dict(false_alarm=sum(x['false_alarm'] for x in before),missed_common=sum(x['missed_common'] for x in before)),
        all_after=dict(false_alarm=sum(x['false_alarm'] for x in after),missed_common=sum(x['missed_common'] for x in after)),
        interpretation=rationale+' Results are simulated phase-domain evidence, not navigation or authentication.')
    (folder/'conclusion.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('number',type=int);p.add_argument('mechanism');p.add_argument('target');p.add_argument('rationale')
    a=p.parse_args();record(a.number,a.mechanism,a.target,a.rationale)
