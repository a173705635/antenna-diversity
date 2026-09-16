"""Synthetic correlation-domain calibration, not RF or a navigation receiver.

Estimator functions receive measured outputs and nominal/commanded inputs only.
scene() is simulation/evaluation truth, never a receiver calibration input.
"""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/research-hundred'


def noise(rng, shape):
    return (rng.normal(size=shape) + 1j*rng.normal(size=shape))/np.sqrt(2)


def scene(seed=41060, m=4):
    rng = np.random.default_rng(seed)
    # Exact mate70_phase_centers coordinates, order TL/TR/BL/BR; copied to keep snapshots standalone.
    pos = np.array([[-.0759/2,.1609/2],[.0759/2,.1609/2],[-.0759/2,-.1609/2],[.0759/2,-.1609/2]])[:m]
    az=rng.uniform(0,2*np.pi,40)
    sin_el=rng.uniform(np.sin(np.radians(15)),.97,40)
    uv=np.vstack((np.sqrt(1-sin_el**2)*np.cos(az),np.sqrt(1-sin_el**2)*np.sin(az)))
    steering = np.exp(2j*np.pi/.1903 * pos @ uv)
    h = np.diag(np.exp(1j*np.linspace(-.35, .4, m))) + .12*noise(rng, (m,m))
    amplitudes = np.exp(.4*noise(rng, (24,)))
    v = h @ np.exp(2j*np.pi/.1903 * pos @ np.array([.2, -.6]))
    v /= np.linalg.norm(v)
    return dict(rng=rng, h=h, a=steering[:,:24], test_a=steering[:,24:], uv=uv,
                clean=(h@steering[:,:24])*amplitudes, test_y=h@steering[:,24:],
                amplitudes=amplitudes, v=v, seed=seed)


def constraints(a, y, basis):
    rows=[]
    for x,z in zip(a.T,y.T):
        p=np.eye(len(z))-np.outer(z,z.conj())/np.vdot(z,z).real
        rows.append(p @ np.column_stack([b@x for b in basis]))
    return np.vstack(rows)


def homogeneous_fit(a, y, diagonal=False):
    m=len(a)
    basis=[np.diag(np.eye(m)[j]) for j in range(m)] if diagonal else np.eye(m*m).reshape(m*m,m,m)
    q=constraints(a,y,basis)
    _,s,vh=np.linalg.svd(q,full_matrices=False)
    return np.einsum('i,ijk->jk', vh[-1].conj(), basis),s


def projective_error(pred, actual):
    alpha=np.sum(pred.conj()*actual,axis=0)/np.sum(abs(pred)**2,axis=0)
    return float(np.linalg.norm(pred*alpha-actual)/np.linalg.norm(actual))


def relative_error(pred, actual):
    return float(np.linalg.norm(pred-actual)/np.linalg.norm(actual))


def pilot_fit(command, measured):
    return np.linalg.lstsq(command.T,measured.T,rcond=None)[0].T


def frequency_scene(seed=41060, sigma=.01):
    d=scene(seed);f=np.linspace(-1,1,13)
    tap=.16*noise(d['rng'],(4,4))
    h=d['h'][None,:,:]+np.exp(-2j*np.pi*f*.18)[:,None,None]*tap
    measured=h+sigma*noise(d['rng'],h.shape)
    return f,h,measured


def polynomial_response(f, measured, query, order):
    design=np.vander(f,order+1,increasing=True)
    coefficients=np.linalg.lstsq(design,measured.reshape(len(f),-1),rcond=None)[0]
    return (np.vander(query,order+1,increasing=True)@coefficients).reshape(len(query),4,4)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def conclude(n, decision, reason, next_step):
    write_json(OUT/f'iteration{n:03d}'/'conclusion.json',dict(iteration=n,decision=decision,reason=reason,next_step=next_step,
       scope='Synthetic correlation domain; no RF, PVT, bench or sustained navigation validation.'))


def run(n):
    mod=importlib.import_module(f'matrix_hundred_{n:03d}')
    folder=OUT/f'iteration{n:03d}';folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'proposal.json',dict(iteration=n,**mod.PROPOSAL,development_seed=41060,
       geometry='Mate70 TL/TR/BL/BR, 0.0759 x 0.1609 m; uniform azimuth and uniform sin(elevation), 15 deg to asin(0.97).',
       correction='Same numbered research mechanism rerun after correcting initial geometry; initial records preserved in matrix_initial_geometry.',
       evaluation_contract='Same measured arrays for compared estimators; truth evaluation only; explicit synthetic hardware assumptions.',
       external_data='None. No measured hardware prior.'))
    source=folder/'source';source.mkdir(exist_ok=True)
    for p in (Path(__file__),Path(mod.__file__)):
        shutil.copy2(p,source/p.name)
    results=mod.experiment()
    results['scope']='correlation-domain simulation only; positioning_error_m not measured'
    results['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
    write_json(folder/'results.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    run(int(sys.argv[1]))
