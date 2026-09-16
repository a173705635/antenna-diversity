"""Four mechanism loops; reuse only identical frozen RF frontend identities."""
import ast,hashlib,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import single_source_twenty as campaign
from single_source_twenty import read,save,digest,load

ROOT=Path('results/mechanism-four')
OLD=Path('results/weak-ten')
PRIOR=Path('results/spatial-ten')


def frontend_hash(model):
    tree=ast.parse(model.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='navigate']
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def evaluate(model,folder,subset):
    meta=read(folder/'metadata.json');tail=meta['samples_sha256']+meta['observations_sha256']+subset
    identity=hashlib.sha256((digest(model)+tail).encode()).hexdigest()
    front_id=hashlib.sha256((frontend_hash(model)+tail).encode()).hexdigest()
    for base in (OLD,PRIOR):
        cached=base/'evaluations'/identity/'score.json'
        if cached.is_file():return read(cached),identity,str(base)
    front=ROOT/'frontends'/front_id
    if not front.exists():
        for base in (OLD,PRIOR):
            cached=base/'frontends'/front_id/'tracking.npz'
            if cached.is_file():
                front.mkdir(parents=True);shutil.copy2(cached,front/'tracking.npz')
                save(front/'reuse.json',dict(source=str(cached),sha256=digest(cached)));break
    campaign.ROOT=ROOT;score,identity=campaign.evaluate(model,folder,subset)
    return score,identity,str(ROOT)


def run(number,model,folder,subset):
    row=dict(input=str(folder),subset=subset,model=str(model))
    for label,path in [('before',ROOT/'baseline.py'),('after',model)]:
        score,identity,base=evaluate(path,folder,subset)
        row[label]=score;row[label+'_id']=identity;row[label+'_root']=base
    save(ROOT/f'round{number:02d}'/f'{folder.name}_{subset}_{model.stem}.json',row)
    print(number,folder.name,subset,model.stem,{label:{k:row[label][k] for k in ['solved','good20','error30','error50']} for label in ['before','after']},flush=True)


def record(number,problem,hypothesis):
    p=ROOT/f'round{number:02d}';p.mkdir(parents=True,exist_ok=True)
    save(p/'proposal.json',dict(number=number,problem=problem,hypothesis=hypothesis,utc=datetime.now(timezone.utc).isoformat()))


def finish(number,decision,evidence):
    save(ROOT/f'round{number:02d}/conclusion.json',dict(decision=decision,evidence=evidence))


if __name__=='__main__':run(int(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4])
