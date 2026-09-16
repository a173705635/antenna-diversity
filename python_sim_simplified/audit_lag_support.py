"""Evaluate selected code delay separately from gain phase; no PVT inference."""
import sys,shutil,json
import numpy as np
from research_hundred import ROOT
from audit_lag_calibration import generate as original_generate
from lag_calibration import shape
from lag_source_support import estimate
from audit_recovery_fifty import save


KINDS=('normal','attack','overlap','two','absent','bandwidth','skew','noise')


def generate(seed,kind):
    z,a,lag,g,true,fake=original_generate(seed,kind)
    if kind=='noise':
        rng=np.random.default_rng(seed)
        covariance=.99*shape(lag[:,None],lag[None,:])+.01*np.eye(len(lag))
        z=.02*(rng.normal(size=z.shape)+1j*rng.normal(size=z.shape))@np.linalg.cholesky(covariance).T
    return z,a,lag,g,true,fake


def evaluate(seed,kind):
    z,a,lag,g,true,fake=generate(seed,kind);gain,info=estimate(z,a,lag)
    error=(np.array(info['selected_delay_chips'])-true)*299792458/1.023e6
    accepted=np.array(info['supported_prns']);real=kind not in ('noise','absent')
    return dict(seed=seed,kind=kind,phase_rmse_deg=float(np.sqrt(np.mean(np.degrees(np.angle(gain/g))**2))) if real else None,
        code_error_m=error.tolist() if real else None,
        admitted_code_rmse_m=float(np.sqrt(np.mean(error[accepted]**2))) if real and accepted.any() else None,
        original_code_rmse_m=float(np.sqrt(np.mean(error**2))) if real else None,**info)


def run(number):
    folder=ROOT/f'iteration{number:03d}';rows=[]
    for name in ('lag_calibration.py','lag_source_support.py','audit_lag_support.py','audit_lag_calibration.py'):
        shutil.copy2(name,folder/name)
    for kind in KINDS:
        rows.extend(evaluate(seed,kind) for seed in (31001,31002,31003,31004))
        print(kind,[(r['usable'],sum(r['supported_prns'])) for r in rows if r['kind']==kind],flush=True)
    save(folder/'results.json',dict(domain='synthetic correlation domain, not PVT',rows=rows))


if __name__=='__main__':run(int(sys.argv[1]))
