"""Actual research revisions; replays and random seeds are not iteration counts."""
import hashlib,importlib.util,json,shutil
from pathlib import Path
import numpy as np
from audit_recovery_fifty import save
from audit_recovery_two_hundred import score

ROOT=Path('results/research-hundred')
NUMBERS=(201,350,365,367,381,382)


def start(number,hypothesis,parent):
    folder=ROOT/f'iteration{number:03d}';folder.mkdir()
    save(folder/'proposal.json',dict(number=number,hypothesis=hypothesis,parent=parent))
    save(ROOT/'status.json',dict(requested_iterations=100,
         completed_iterations=len(list(ROOT.glob('iteration*/conclusion.json'))),current_iteration=number))


def finish(number,decision,evidence,next_step):
    save(ROOT/f'iteration{number:03d}'/'conclusion.json',dict(decision=decision,evidence=evidence,next_step=next_step))
    save(ROOT/'status.json',dict(requested_iterations=100,
         completed_iterations=len(list(ROOT.glob('iteration*/conclusion.json'))),current_iteration=None))


def load(path):
    spec=importlib.util.spec_from_file_location('research_proposal',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def navigation(number):
    folder=ROOT/f'iteration{number:03d}';shutil.copy2('navigation_hundred.py',folder/'navigation_hundred.py')
    method=load(Path('navigation_hundred.py')).candidate
    baseline=load(ROOT/'baseline_navigation.py').candidate
    rows=[]
    for n in NUMBERS:
        for path in sorted(Path(f'results/recovery-four-hundred/round{n:03d}').glob('*.npz')):
            with np.load(path) as data:z={k:data[k] for k in data.files}
            old=baseline(z);new=method(z)
            np.savez_compressed(folder/(path.stem+'.npz'),position=new,baseline=old)
            rows.append(dict(number=n,attack=path.stem.endswith('_1'),path=str(path),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             baseline=score(old,z['truth_position']),proposal=score(new,z['truth_position'])))
    totals={str(flag):{m:{k:sum(r[m][k] for r in rows if r['attack']==flag)
            for k in ('opportunities','solved','good20','error30','error50')}
            for m in ('baseline','proposal')} for flag in (False,True)}
    save(folder/'results.json',dict(rows=rows,totals=totals));print(json.dumps(totals),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('number',type=int)
    navigation(parser.parse_args().number)
