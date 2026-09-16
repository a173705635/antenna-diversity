"""One-revision runner; scenarios score a revision but do not count as revisions."""
import json,sys,hashlib,importlib.util,shutil
from pathlib import Path
import numpy as np
from presets import mate70_phase_centers
from signal_recovery import sky,steering
ROOT=Path('results/research-third-hundred')

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')

def observation(seed,kind):
    rng=np.random.default_rng(seed);m,s,n=4,8,240
    a=steering(sky(48020,s),np.array(list(mate70_phase_centers().values())),0.)
    def white():return (rng.normal(size=(m,s,n))+1j*rng.normal(size=(m,s,n)))/np.sqrt(2)
    noise=white();background=white();mixing=np.eye(m,dtype=complex)
    if kind in ('spatial','mixed','gain'):mixing=np.linalg.cholesky(.75*np.ones((m,m))+.25*np.eye(m)).astype(complex)
    if kind=='gain':mixing=np.diag([1.,.25,2.,.5])@mixing
    noise=np.einsum('ij,jsk->isk',mixing,noise);background=np.einsum('ij,jsk->isk',mixing,background)
    if kind in ('temporal','mixed'):
        for t in range(1,n):
            noise[:,:,t]=.8*noise[:,:,t-1]+.6*noise[:,:,t]
            background[:,:,t]=.8*background[:,:,t-1]+.6*background[:,:,t]
    g=np.exp(1j*np.array([0.,.18,-.25,.3]))
    if kind=='gain':g*=np.array([1.,.25,2.,.5])
    bits=np.repeat(rng.choice([-1.,1.],size=(s,n//20)),20,axis=1)
    tone=np.exp(2j*np.pi*np.arange(n)[None,:]*np.linspace(-.004,.004,s)[:,None])
    amp={'weak':.35,'strong':1.,'gain':.6,'mixed':.6,'transient':1.}.get(kind,0.)
    signal=amp*g[:,None,None]*a[:,:,None]*bits[None,:,:]*tone[None,:,:]
    if kind=='transient':signal[:,:,120:]=0
    if kind in ('common','mixed'):
        v=np.exp(1j*np.array([0.,1.2,2.1,-.7]));signal+=2*v[:,None,None]*bits[None,:,:]*tone[None,:,:]
    if kind=='two_common':
        v=np.exp(1j*np.array([0.,1.2,2.1,-.7]));w=np.exp(1j*np.array([0.,-.7,.1,2.]))
        signal=2*v[:,None,None]*bits[None,:,:]*tone[None,:,:]+1.5*w[:,None,None]*np.roll(bits,7,axis=1)[None,:,:]*tone[None,:,:]
    if kind=='noise_drift':background[:,:,80:]*=2
    if kind=='impulsive':background[:,:,::31]*=12;noise[:,:,::29]*=12
    if kind=='three_sources':signal=.8*g[:,None,None]*a[:,:,None]*bits[None,:,:]*tone[None,:,:];signal[:,3:]=0
    if kind=='moving_common':
        v=np.exp(1j*np.array([0.,1.2,2.1,-.7]));motion=np.exp(1j*np.arange(m)[:,None]*np.linspace(0,1.5,n)[None,:])
        signal=2*v[:,None,None]*motion[:,None,:]*bits[None,:,:]*tone[None,:,:]
    if kind=='near_common':
        v=a[:,0];signal=.35*g[:,None,None]*a[:,:,None]*bits[None,:,:]*tone[None,:,:]+2*v[:,None,None]*bits[None,:,:]*tone[None,:,:]
    if kind=='ar2':
        for t in range(2,n):
            noise[:,:,t]+=.75*noise[:,:,t-2];background[:,:,t]+=.75*background[:,:,t-2]
    return background+signal,a,noise

KINDS=['iid','spatial','temporal','common','weak','strong','gain','mixed','transient','two_common','noise_drift','impulsive','three_sources','moving_common','near_common','ar2']

def load(path):
    spec=importlib.util.spec_from_file_location('candidate_'+hashlib.sha256(str(path).encode()).hexdigest()[:8],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def run(revision,parent,mechanism,rationale):
    folder=ROOT/f'iteration{revision:03d}';folder.mkdir(parents=True,exist_ok=True)
    old=Path('cold_next.py') if parent==150 else ROOT/f'iteration{parent:03d}'/'after.py'
    shutil.copy2(old,folder/'before.py');shutil.copy2('source_third.py',folder/'after.py')
    save(folder/'proposal.json',dict(iteration=revision,parent=parent,mechanism=mechanism,rationale=rationale,layer='receiver prompt model, independent off-code noise; no RF/PVT',seed=82600))
    before=load(old);after=load(folder/'after.py');rows=[]
    for i,kind in enumerate(KINDS):
        z,a,noise=observation(82600+i,kind);digest=hashlib.sha256(z.tobytes()+a.tobytes()+noise.tobytes()).hexdigest()
        old_info=before.complex_calibrate(z,a)[1] if parent==150 else before.detect(z,a,noise)
        new_info=after.detect(z,a,noise)
        rows.append(dict(case=kind,input_sha256=digest,before=old_info,after=new_info))
    save(folder/'results.json',dict(rows=rows))
    summary={side:{r['case']:r[side]['usable'] for r in rows} for side in ('before','after')}
    save(folder/'conclusion.json',dict(status='executed_pending_interpretation',**summary))
    print(json.dumps(summary));print('min-p',[(r['case'],round(min(r['after']['p']),6)) for r in rows])

if __name__=='__main__':run(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4])



