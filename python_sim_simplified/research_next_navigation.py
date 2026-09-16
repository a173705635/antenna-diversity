"""Sequential actual navigation mechanisms, fixed observed input pairs."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from research_hundred import load
from navigation_hundred import candidate as navigate
from audit_recovery_two_hundred import score
from audit_recovery_fifty import save

ROOT=Path('results/research-next-hundred')


def start(n,hypothesis,parent):
    folder=ROOT/f'iteration{n:03d}';folder.mkdir()
    save(folder/'proposal.json',dict(number=n,hypothesis=hypothesis,parent=parent,
         scope='navigation postprocessing of already observed paired RF traces'))


def finish(n,decision,evidence,next_step):
    save(ROOT/f'iteration{n:03d}'/'conclusion.json',dict(decision=decision,evidence=evidence,next_step=next_step))
    save(ROOT/'status.json',dict(requested_iterations=100,completed_iterations=len(list(ROOT.glob('iteration*/conclusion.json'))),phase='development'))


def inputs():
    for seed in range(43001,43009):
        pair={};hashes={}
        for method in ('capture','joint'):
            path=Path(f'results/research-hundred/rf_validation/{seed}_{method}.npz')
            with np.load(path) as z:pair[method]={k:z[k] for k in z.files}
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        truth=pair['joint'].pop('truth_position');pair['capture'].pop('truth_position')
        yield seed,pair,truth,hashes
    from navigation_next_cases import generate,KINDS
    for i,kind in enumerate(KINDS):
        seed=81001+i;pair,truth=generate(seed,kind)
        yield f'{seed}_{kind}',pair,truth,{'generator_sha256':hashlib.sha256(Path('navigation_next_cases.py').read_bytes()).hexdigest()}


def run(n):
    folder=ROOT/f'iteration{n:03d}'
    for name in ('navigation_next.py','research_next_navigation.py','navigation_next_cases.py'):shutil.copy2(name,folder/name)
    current=load(Path('navigation_next.py')).candidate
    parent=load(ROOT/f"iteration{json.loads((folder/'proposal.json').read_text())['parent']:03d}"/'navigation_next.py').candidate
    rows=[]
    for seed,pair,truth,hashes in inputs():
        baseline=navigate(pair['joint'])
        if not pair['joint']['joint_usable']:baseline[:]=np.nan
        result=current(pair);previous=parent(pair)
        np.savez_compressed(folder/f'{seed}.npz',baseline=baseline,parent=previous,proposal=result)
        rows.append(dict(seed=seed,domain='RF-derived' if isinstance(seed,int) else 'observation',input_hashes=hashes,baseline=score(baseline,truth),parent=score(previous,truth),proposal=score(result,truth)))
    totals={name:{key:sum(r[name][key] for r in rows) for key in ('opportunities','solved','good20','error30','error50')} for name in ('baseline','parent','proposal')}
    save(folder/'results.json',dict(rows=rows,totals=totals));print(json.dumps(totals),flush=True)


if __name__=='__main__':run(int(sys.argv[1]))
