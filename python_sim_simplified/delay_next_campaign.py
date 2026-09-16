"""One research proposal per invocation; immutable paired observations."""
import argparse,hashlib,importlib.util,inspect,json,shutil,time
from pathlib import Path
import numpy as np
from audit_lag_support import generate
from lag_calibration import shape
from signal_recovery import steering
from presets import mate70_phase_centers

ROOT=Path(__file__).parent/'results/research-next-hundred'
CASES=[(32051,k) for k in ('normal','attack','overlap','two','absent','bandwidth','skew','noise')]+[(31001,'normal'),(32051,'three'),(32051,'multipath'),(32051,'weak'),(32051,'gain'),(32051,'noise_mismatch')]

def load(path):
    spec=importlib.util.spec_from_file_location('delay_candidate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def observations(seed,kind):
    z,a,lag,g,t,f=generate(seed,'normal' if kind in ('weak','noise_mismatch') else 'attack' if kind in ('three','multipath') else kind)
    if kind=='three':z,a,g=z[:3],a[:3],g[:3]
    if kind=='multipath':
        z+=.35*g[:,None,None]*a[:,:,None]*np.exp(1j*np.arange(len(t)))[None,:,None]*shape(lag[None,None,:],t[None,:,None]+.35,.9)
    if kind=='weak':
        rng=np.random.default_rng(seed);rng.normal(0,.4,len(z));rng.normal(0,.1,len(t))
        alpha=np.exp(1j*rng.uniform(-np.pi,np.pi,len(t)))
        z-=.96*g[:,None,None]*a[:,:,None]*alpha[None,:,None]*shape(lag[None,None,:],t[None,:,None],.9)
    if kind=='noise_mismatch':
        rng=np.random.default_rng(seed);rng.normal(0,.4,len(z));rng.normal(0,.1,len(t))
        alpha=np.exp(1j*rng.uniform(-np.pi,np.pi,len(t)))
        clean=g[:,None,None]*a[:,:,None]*alpha[None,:,None]*shape(lag[None,None,:],t[None,:,None],.9)
        z=clean+(z-clean)*np.linspace(.7,1.3,len(lag))
    if kind=='gain':
        rng=np.random.default_rng(seed);rng.normal(0,.4,len(z));rng.normal(0,.1,len(t))
        alpha=np.exp(1j*rng.uniform(-np.pi,np.pi,len(t)))
        beta=np.sqrt(10)*np.exp(1j*rng.uniform(-np.pi,np.pi,len(t)))
        pos=np.array(list(mate70_phase_centers().values()))
        v=steering(np.array([[.8,.6,0.]]),pos,0.)[:,0]
        clean=g[:,None,None]*(a[:,:,None]*alpha[None,:,None]*shape(lag[None,None,:],t[None,:,None],.9)+v[:,None,None]*beta[None,:,None]*shape(lag[None,None,:],f[None,:,None],.9))
        amplitude=np.array([1.,.7,1.2,.85]);z+=(amplitude[:,None,None]-1)*clean;g*=amplitude
    shifts=np.linspace(0,.06,len(z)) if kind=='skew' else np.zeros(len(z))
    return z,a,lag,g,t,shifts

def evaluate(method,seed,kind):
    z,a,lag,g,t,shifts=observations(seed,kind)
    bandwidth=.7 if kind=='bandwidth' else .9
    covariance=.99*shape(lag[:,None],lag[None,:],bandwidth)+.01*np.eye(len(lag))
    rng=np.random.default_rng(seed+902000)
    noise=.02*(rng.normal(size=(512,len(lag)))+1j*rng.normal(size=(512,len(lag))))@np.linalg.cholesky(covariance).T
    if kind=='noise_mismatch':noise*=np.linspace(.7,1.3,len(lag))
    gain,info=method(z,a,lag,noise) if 'noise' in inspect.signature(method).parameters else method(z,a,lag)
    real=kind not in ('noise','absent')
    code=(np.array(info['selected_delay_chips'])-t)*299792458/1.023e6
    admitted=np.array(info['supported_prns'],dtype=bool)
    return dict(seed=seed,kind=kind,input_sha256=hashlib.sha256(z.tobytes()+a.tobytes()+lag.tobytes()).hexdigest(),noise_calibration_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
        phase_rmse_deg=float(np.sqrt(np.mean(np.degrees(np.angle(gain/g))**2))) if real else None,
        group_delay_rmse_chips=float(np.sqrt(np.mean((np.array(info['channel_delay_chips'])-shifts)**2))),
        code_error_m=code.tolist() if real else None,
        admitted_code_rmse_m=float(np.sqrt(np.mean(code[admitted]**2))) if real and admitted.any() else None,**info)

def summary(rows):
    return {r['kind']+str(r['seed']):dict(usable=r['usable'],phase=r['phase_rmse_deg'],group=r['group_delay_rmse_chips'],code=r['admitted_code_rmse_m'],supported=sum(r['supported_prns'])) for r in rows}

def run(number,parent,hypothesis):
    folder=ROOT/f'iteration{number:03d}';folder.mkdir(parents=True)
    for name in ('delay_next.py','delay_next_campaign.py','lag_calibration.py','lag_source_support.py','audit_lag_calibration.py','audit_lag_support.py'):
        shutil.copy2(Path(__file__).parent/name,folder/name)
    previous=ROOT/f'iteration{parent:03d}'/'delay_next.py' if parent>=101 else Path(__file__).parent/'lag_source_support.py'
    old=load(previous).estimate;new=load(folder/'delay_next.py').estimate
    write(folder/'proposal.json',dict(iteration=number,parent=parent,hypothesis=hypothesis,scope='correlation-domain, known sky, diagonal gain, Mate70 75.9x160.9mm; no navigation/authentication claim',cases=CASES))
    rows=[]
    for seed,kind in CASES:
        before=evaluate(old,seed,kind);after=evaluate(new,seed,kind)
        rows.append(dict(before=before,after=after));print(number,kind,summary([after]),flush=True)
    write(folder/'results.json',dict(rows=rows,after=summary([r['after'] for r in rows])))
    print('DONE',number,flush=True)

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def finish(number,decision,next_step):
    folder=ROOT/f'iteration{number:03d}'
    result=json.loads((folder/'results.json').read_text(encoding='utf-8'))
    write(folder/'conclusion.json',dict(decision=decision,evidence=result['after'],next_step=next_step))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('number',type=int);p.add_argument('parent',type=int);p.add_argument('hypothesis');args=p.parse_args()
    run(args.number,args.parent,args.hypothesis)
