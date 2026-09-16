"""Sequential research ledger. Old RF records are development data in this study."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from audit_recovery_two_hundred import score
from audit_recovery_fifty import save

ROOT=Path('results/research-twenty')
DEVELOPMENT=(201,211,338,350,363,367)


def begin(number,hypothesis,parent):
    folder=ROOT/f'iteration{number:02d}';folder.mkdir(parents=True,exist_ok=False)
    save(folder/'proposal.json',dict(number=number,hypothesis=hypothesis,parent=parent,
                                    status='proposed',data_role='development'))
    return folder


def conclude(number,decision,evidence,next_step):
    save(ROOT/f'iteration{number:02d}'/'conclusion.json',
         dict(decision=decision,evidence=evidence,next_step=next_step))


def navigation(number):
    from navigation_candidate import candidate
    folder=ROOT/f'iteration{number:02d}';rows=[];inputs={}
    for name in ('navigation_candidate.py','research_twenty.py'):
        shutil.copy2(name,folder/name)
    for n in DEVELOPMENT:
        for path in sorted(Path(f'results/recovery-four-hundred/round{n:03d}').glob('*.npz')):
            inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as archive:z={k:archive[k] for k in archive.files}
            p=candidate(z)
            from carrier_admission import Rule,evaluate
            baseline=evaluate(z,'calibrated',Rule(window=5,mature=1,divergence=np.inf,spatial=1.))['position']
            rows.append(dict(record=str(path),attack=path.stem.endswith('_1'),
                             baseline=score(baseline,z['truth_position']),
                             proposal=score(p,z['truth_position'])))
            np.savez_compressed(folder/(path.stem+'.npz'),position=p,baseline=baseline)
    totals={}
    for flag in (False,True):
        totals['attack' if flag else 'normal']={m:{k:sum(r[m][k] for r in rows if r['attack']==flag)
            for k in ('opportunities','solved','good20','error30','error50')} for m in ('baseline','proposal')}
    save(folder/'results.json',dict(inputs=inputs,rows=rows,totals=totals))
    print(json.dumps(totals),flush=True)


def prompt_data(seed,kind):
    from signal_recovery import sky,steering
    from presets import mate70_phase_centers
    rng=np.random.default_rng(seed);m=2 if kind=='two' else 3 if kind=='three' else 4
    positions=np.array(list(mate70_phase_centers().values()))[:m]
    a=steering(sky(seed,8),positions,0.);v=steering(np.array([[.8,.6,0.]]),positions,0.)[:,0]
    g=np.exp(1j*rng.normal(0,np.radians(20),m));g/=g[0]
    if kind=='amplitude':g*=10**(rng.normal(0,3,m)/20);g/=g[0]
    alpha=(rng.normal(size=(8,80))+1j*rng.normal(size=(8,80)))/np.sqrt(2)
    beta=(rng.normal(size=(8,80))+1j*rng.normal(size=(8,80)))*np.sqrt(5)
    if kind=='normal':beta*=0
    if kind=='weak':beta/=np.sqrt(10)
    if kind=='coherent':beta=np.sqrt(10)*alpha
    if kind=='absent':alpha*=0
    if kind=='unequal':beta[0]*=10
    noise=(rng.normal(size=(m,8,80))+1j*rng.normal(size=(m,8,80)))*.2/np.sqrt(2)
    z=g[:,None,None]*(a[:,:,None]*alpha+v[:,None,None]*beta)+noise
    if kind=='multipath':z[:,0]+=np.array([1,1j,-1,-1j])[:,None]*alpha[0]*2
    return z,a,g


PROMPT_KINDS=('normal','attack','weak','coherent','absent','unequal','amplitude','two','three','multipath')


def calibration(number):
    from calibration_candidate import calibrate,cross_calibration
    folder=ROOT/f'iteration{number:02d}';rows=[]
    for name in ('calibration_candidate.py','research_twenty.py'):
        shutil.copy2(name,folder/name)
    for kind in PROMPT_KINDS:
        for seed in (21001,21002,21003,21004):
            z,a,truth=prompt_data(seed,kind)
            if number==19:
                # Explicit simulated external calibration measurement, not exact truth input.
                rng=np.random.default_rng(seed+991)
                prior=truth*np.exp(1j*rng.normal(0,np.radians(5),len(truth)));prior/=prior[0]
                g,info=calibrate(z,a,prior)
                info['external_prior_rms_deg']=float(np.sqrt(np.mean(np.degrees(np.angle(prior/truth))**2)))
            else:g,info=calibrate(z,a)
            base=cross_calibration(z,a)
            def phase_error(estimate):
                return float(np.sqrt(np.mean(np.degrees(np.angle((estimate/estimate[0])/truth))**2)))
            rows.append(dict(kind=kind,seed=seed,baseline_deg=phase_error(base),proposal_deg=phase_error(g),**info))
    summary={kind:dict(baseline_deg=float(np.mean([r['baseline_deg'] for r in rows if r['kind']==kind])),
                     proposal_deg=float(np.mean([r['proposal_deg'] for r in rows if r['kind']==kind])),
                     usable=sum(r['usable'] for r in rows if r['kind']==kind)) for kind in PROMPT_KINDS}
    save(folder/'results.json',dict(domain='synthetic complex prompts, not RF PVT',rows=rows,summary=summary))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('number',type=int)
    n=parser.parse_args().number
    (navigation if n<=10 else calibration)(n)
