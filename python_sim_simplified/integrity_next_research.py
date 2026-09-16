"""One invocation closes one actual source-change experiment; replay is read-only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from integrity_next_model import make_case, score, NAMES

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/research-next-hundred'
SOURCE=ROOT/'integrity_next.py'

def execute(source, cases):
    ns={}; exec(compile(source,'<frozen-integrity>','exec'),ns)
    return {name: score(ns['receive'](o),truth) for name,(o,truth) in cases.items()}

def run(number, old, new, hypothesis, target, conclusion):
    folder=OUT/f'iteration{number}'; folder.mkdir()
    before=SOURCE.read_text(encoding='utf-8')
    assert old in before
    proposal=dict(iteration=number, hypothesis=hypothesis, target=target, seed=1510,
                  model='600 s 3-D synthetic independent bench and observation inputs; not measured phone data',
                  requirements='all epochs normal and <=20m; normal >=30m and >=50m are failures; protected epochs fail continuity')
    (folder/'proposal.json').write_text(json.dumps(proposal,indent=2,ensure_ascii=False),encoding='utf-8')
    after=before.replace(old,new,1)
    SOURCE.write_text(after,encoding='utf-8')
    cases={name:make_case(name,1510) for name in NAMES}
    bm=execute(before,cases); am=execute(after,cases)
    (folder/'before.py').write_text(before,encoding='utf-8'); (folder/'after.py').write_text(after,encoding='utf-8')
    record=dict(**proposal,before=bm,after=am,conclusion=conclusion,status='closed',
                before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest())
    (folder/'record.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(iteration=number,target=target,before=bm[target],after=am[target]),ensure_ascii=False))

def replay(number):
    folder=OUT/f'iteration{number}'
    record=json.loads((folder/'record.json').read_text(encoding='utf-8'))
    cases={name:make_case(name,record['seed']) for name in NAMES}
    for phase in ['before','after']:
        assert execute((folder/f'{phase}.py').read_text(encoding='utf-8'),cases)==record[phase]
    print(f'{number}: replay identical')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--replay',type=int,required=True)
    replay(parser.parse_args().replay)
