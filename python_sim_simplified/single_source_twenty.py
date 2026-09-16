"""Sequential single-source full-receiver loops; immutable IQ and model caches."""
import ast,hashlib,json,shutil,sys,time
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from continuous_improvement import Case,generate
from research_hundred import load
from signal_recovery import FS

ROOT=Path('results/single-source-twenty')
SUBSETS={'two':[0,1],'three':[0,1,2],'four':[0,1,2,3]}


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def create_input(seed,kind,duration=.6):
    folder=ROOT/'inputs'/f'{seed}_{kind}_{duration}'
    folder.mkdir(parents=True)
    changes=dict(cn0=45.,phase_deg=10.,amplitude_db=1.,velocity_east=2.,attack_start=.3,duration=duration)
    if kind=='weak':changes['cn0']=35.
    if kind in ('near','cluster'):changes['angle']=0.
    if kind=='absent':changes['genuine']=False
    if kind=='release':changes['attack_end']=.45
    if kind=='drag':changes.update(fake_east=0.,fake_rate=50.)
    if kind=='return':changes.update(genuine=False,attack_end=.8)
    case=Case(**changes); attacked=kind!='normal'
    generator=generate
    if kind=='cluster':
        from types import FunctionType
        from signal_recovery import sky
        def clustered_sky(seed,stars):
            directions=sky(seed,stars)
            tangent=np.cross(directions[0],[0.,0.,1.]);tangent/=np.linalg.norm(tangent)
            for i,angle in ((1,.5),(2,-.5)):
                directions[i]=directions[0]*np.cos(np.radians(angle))+tangent*np.sin(np.radians(angle))
            return directions
        generator=FunctionType(generate.__code__,dict(generate.__globals__,sky=clustered_sky))
    x,a,codes,times,satellites,radii=generator(case,seed,attacked)
    if kind=='drift':
        x*=np.exp(1j*1.25*np.arange(4)[:,None]*np.arange(x.shape[1])[None,:]/FS)
    np.save(folder/'samples.npy',x)
    np.savez(folder/'observations.npz',nominal=a,codes=np.array(codes),times=times,satellites=satellites,
             coarse=np.round((radii+75000)/10000)*10000)
    save(folder/'metadata.json',dict(seed=seed,kind=kind,case=asdict(case),attacked=attacked,
         samples_sha256=digest(folder/'samples.npy'),observations_sha256=digest(folder/'observations.npz'),
         drift_rad_s_per_channel_index=1.25 if kind=='drift' else 0.,physical_spoof_emitters=1 if attacked else 0))
    return folder


def input_record(folder):
    with np.load(folder/'observations.npz') as z:record={k:z[k] for k in z.files}
    record['samples']=np.load(folder/'samples.npy',mmap_mode='r')
    return record


def evaluate(model_path,folder,subset):
    source=digest(model_path);meta=read(folder/'metadata.json')
    tree=ast.parse(model_path.read_text(encoding='utf-8'))
    tree.body=[node for node in tree.body if not isinstance(node,ast.FunctionDef) or node.name!='navigate']
    frontend_source=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    front_identity=hashlib.sha256((frontend_source+meta['samples_sha256']+meta['observations_sha256']+subset).encode()).hexdigest()
    front=ROOT/'frontends'/front_identity
    identity=hashlib.sha256((source+meta['samples_sha256']+meta['observations_sha256']+subset).encode()).hexdigest()
    cache=ROOT/'evaluations'/identity
    # This is an explicit immutable research cache, not an error fallback.
    if cache.exists():
        if not front.exists():
            front.mkdir(parents=True);shutil.copy2(cache/'tracking.npz',front/'tracking.npz')
        return read(cache/'score.json'),identity
    cache.mkdir(parents=True)
    model=load(model_path);t=time.perf_counter()
    reused=front.exists()
    if reused:
        with np.load(front/'tracking.npz') as data:z={k:data[k] for k in data.files}
    else:
        z=model.frontend(input_record(folder),SUBSETS[subset])
        front.mkdir(parents=True);np.savez_compressed(front/'tracking.npz',**z)
    result=model.navigate(z)
    np.savez_compressed(cache/'tracking.npz',**z)
    np.savez_compressed(cache/'navigation.npz',**{k:v for k,v in result.items() if k!='initialization_time'})
    attack_start=meta['case']['attack_start']
    eligible=z['available_time']>=attack_start
    truth=np.zeros_like(result['position']);truth[:,0]=meta['case']['velocity_east']*z['available_time']
    init=result['initialization_time'];init_before=init is not None and init<attack_start
    error=np.linalg.norm((result['position']-truth)[:,:2],axis=1)[eligible]
    good=np.isfinite(error)
    runs=np.diff(np.flatnonzero(np.r_[True,good,True]))-1
    score=dict(opportunities=len(error),solved=int(good.sum()),good20=int((error<=20).sum()),
        error30=int((error>=30).sum()),error50=int((error>=50).sum()),
        errors_m=[float(e) if np.isfinite(e) else None for e in error],
        initialization_time=init,attack_start_s=attack_start,initialized_before_attack=init_before,
        initialization_gap_s=None if init is None else attack_start-init,
        longest_unavailable_s=float(runs.max()*.01),elapsed_s=time.perf_counter()-t,
        source_sha256=source,input_sha256=meta['samples_sha256'],subset=subset,
        frontend_reused=reused,frontend_ast_sha256=frontend_source)
    save(cache/'score.json',score);return score,identity


def propose(number,parent,problem,hypothesis,kinds=('normal','drift')):
    folder=ROOT/f'round{number:03d}';folder.mkdir(parents=True)
    save(folder/'proposal.json',dict(number=number,parent=parent,problem=problem,hypothesis=hypothesis,
         proposed_utc=datetime.now(timezone.utc).isoformat(),scope='normal startup, single emitter',
         development_seed=113001,duration=2.,kinds=list(kinds),subsets=['three','four']))


def run(number):
    folder=ROOT/f'round{number:03d}';proposal=read(folder/'proposal.json')
    shutil.copy2('single_source_receiver.py',folder/'receiver.py')
    parent=ROOT/'baseline.py' if proposal['parent']==0 else ROOT/f"round{proposal['parent']:03d}/receiver.py"
    jobs=[]
    for kind in proposal['kinds']:
        inp=ROOT/'inputs'/f"{proposal['development_seed']}_{kind}_{proposal['duration']}"
        for subset in proposal['subsets']:
            jobs.append((parent,folder/'receiver.py',inp,kind,subset))
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(evaluate_pair,jobs):
            rows.append(row);print(json.dumps({k:row[k] for k in ('kind','subset')}|{
                v:{k:row[v][k] for k in ('solved','good20','error30','error50','initialized_before_attack')} for v in ('before','after')}),flush=True)
    totals={v:{key:sum(r[v][key] for r in rows) for key in ('opportunities','solved','good20','error30','error50')} for v in ('before','after')}
    save(folder/'results.json',dict(rows=rows,totals=totals));print(json.dumps(totals),flush=True)


def evaluate_pair(job):
    parent,current,inp,kind,subset=job
    before,bid=evaluate(parent,inp,subset);after,aid=evaluate(current,inp,subset)
    return dict(kind=kind,subset=subset,before=before,after=after,before_id=bid,after_id=aid)


def finish(number,decision,evidence,next_problem):
    save(ROOT/f'round{number:03d}'/'conclusion.json',dict(decision=decision,evidence=evidence,next_problem=next_problem,
          completed_utc=datetime.now(timezone.utc).isoformat()))
    completed=sorted(int(p.parent.name[5:]) for p in ROOT.glob('round*/conclusion.json'))
    save(ROOT/'progress.json',dict(requested_rounds=20,completed_rounds=completed,
         next_problem=next_problem,scope='single emitter; no cold attack',commercial_goal_met=False))


if __name__=='__main__':run(int(sys.argv[1]))
