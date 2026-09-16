"""Actual source-change navigation experiments; independent scoring."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from research_hundred import load
from audit_recovery_fifty import save
from audit_recovery_two_hundred import score

ROOT=Path('results/research-third-hundred')

def start(n,hypothesis,parent):
    folder=ROOT/f'iteration{n}';folder.mkdir()
    save(folder/'proposal.json',dict(number=n,hypothesis=hypothesis,parent=parent))

def finish(n,decision,evidence,next_step):
    save(ROOT/f'iteration{n}'/'conclusion.json',dict(decision=decision,evidence=evidence,next_step=next_step))

def inputs():
    for seed in range(86101,86108):
        pair={}
        for route in ('joint','capture'):
            p=Path(f'results/research-next-hundred/navigation_corrected_validation/{seed}_{route}.npz')
            with np.load(p) as z:pair[route]={k:z[k] for k in z.files}
        truth=pair['joint'].pop('truth_position');pair['capture'].pop('truth_position')
        yield str(seed),'RF-derived',pair,truth
    from navigation_third_cases import generate,KINDS
    for i,kind in enumerate(KINDS):
        seed=90101+i;pair,truth=generate(seed,kind)
        yield f'{seed}_{kind}','observation',pair,truth

def run(n):
    folder=ROOT/f'iteration{n}';proposal=json.loads((folder/'proposal.json').read_text())
    for name in ('navigation_third.py','navigation_third_cases.py','research_third_navigation.py'):shutil.copy2(name,folder/name)
    current=load(Path('navigation_third.py')).candidate
    parent=load(ROOT/'baseline_navigation.py' if proposal['parent']==200 else ROOT/f"iteration{proposal['parent']}/navigation_third.py").candidate
    rows=[]
    for seed,domain,pair,truth in inputs():
        before=parent(pair);after=current(pair)
        np.savez_compressed(folder/f'{seed}.npz',before=before,after=after)
        rows.append(dict(seed=seed,domain=domain,before=score(before,truth),after=score(after,truth)))
    totals={domain:{version:{k:sum(r[version][k] for r in rows if r['domain']==domain)
        for k in ('opportunities','solved','good20','error30','error50')} for version in ('before','after')}
        for domain in ('RF-derived','observation')}
    save(folder/'results.json',dict(rows=rows,totals=totals));print(json.dumps(totals),flush=True)

if __name__=='__main__':run(int(sys.argv[1]))
