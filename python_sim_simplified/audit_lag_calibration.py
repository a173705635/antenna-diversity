"""Sequential space-delay proposals, same development observations each revision."""
import json,shutil,sys
from pathlib import Path
import numpy as np
from lag_calibration import estimate,shape
from signal_recovery import sky,steering
from presets import mate70_phase_centers
from audit_recovery_fifty import save


def generate(seed,kind):
    rng=np.random.default_rng(seed);m=2 if kind=='two' else 4
    pos=np.array(list(mate70_phase_centers().values()))[:m];a=steering(sky(seed,8),pos,0.)
    v=steering(np.array([[.8,.6,0.]]),pos,0.)[:,0]
    g=np.exp(1j*rng.normal(0,.4,m));g/=g[0]
    lag=np.linspace(-2,2,65);true_delay=rng.normal(0,.1,8)
    fake_delay=true_delay+(.1 if kind=='overlap' else .7)
    alpha=np.exp(1j*rng.uniform(-np.pi,np.pi,8));beta=np.sqrt(10)*np.exp(1j*rng.uniform(-np.pi,np.pi,8))
    if kind=='normal':beta*=0
    if kind=='absent':alpha*=0
    bandwidth=.7 if kind=='bandwidth' else .9
    shifts=np.linspace(0,.06,m) if kind=='skew' else np.zeros(m)
    z=g[:,None,None]*(a[:,:,None]*alpha[None,:,None]*shape(lag[None,None,:],true_delay[None,:,None]+shifts[:,None,None],bandwidth)
        +v[:,None,None]*beta[None,:,None]*shape(lag[None,None,:],fake_delay[None,:,None]+shifts[:,None,None],bandwidth))
    covariance=.99*shape(lag[:,None],lag[None,:],bandwidth)+.01*np.eye(len(lag))
    noise=(rng.normal(size=z.shape)+1j*rng.normal(size=z.shape))*.02
    z+=noise@np.linalg.cholesky(covariance).T
    return z,a,lag,g,true_delay,fake_delay


def run(number):
    root=Path(f'results/research-hundred/iteration{number:03d}');rows=[]
    for name in ('lag_calibration.py','audit_lag_calibration.py'):shutil.copy2(name,root/name)
    for kind in ('normal','attack','overlap','two','absent','bandwidth','skew'):
        for seed in (31001,31002,31003,31004):
            z,a,lag,g,actual,fake=generate(seed,kind)
            estimated,info=estimate(z,a,lag)
            error=np.sqrt(np.mean(np.degrees(np.angle(estimated/g))**2))
            if number==15:
                from research_hundred import load
                wrong=load(Path('results/research-hundred/iteration014/lag_calibration.py')).estimate(z,a,lag)[0]
                info['gaussian_template_error_deg']=float(np.sqrt(np.mean(np.degrees(np.angle(wrong/g))**2)))
            if number==16:
                from research_hundred import load
                wrong=load(Path('results/research-hundred/iteration015/lag_calibration.py')).estimate(z,a,lag)[0]
                info['unwhitened_error_deg']=float(np.sqrt(np.mean(np.degrees(np.angle(wrong/g))**2)))
            if number==17:
                from research_hundred import load
                wrong=load(Path('results/research-hundred/iteration016/lag_calibration.py')).estimate(z,a,lag)[0]
                info['fixed_bandwidth_error_deg']=float(np.sqrt(np.mean(np.degrees(np.angle(wrong/g))**2)))
            if number==20:
                from research_hundred import load
                wrong=load(Path('results/research-hundred/iteration018/lag_calibration.py')).estimate(z,a,lag)[0]
                info['unaligned_error_deg']=float(np.sqrt(np.mean(np.degrees(np.angle(wrong/g))**2)))
            rows.append(dict(kind=kind,seed=seed,phase_rmse_deg=float(error),**info))
    summary={kind:float(np.mean([r['phase_rmse_deg'] for r in rows if r['kind']==kind]))
        for kind in ('normal','attack','overlap','two','absent','bandwidth','skew')}
    save(root/'results.json',dict(domain='synthetic bandlimited correlation shape and correlated noise; no RF claim',rows=rows,summary=summary))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':run(int(sys.argv[1]))
