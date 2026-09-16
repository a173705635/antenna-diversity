"""Freeze, reproduce snapshots, and evaluate new seeds; none count as research rounds."""
from datetime import datetime, timezone
import subprocess
from matrix_hundred_core import *


def validate():
    sources=[ROOT/'matrix_hundred_core.py']+[ROOT/f'matrix_hundred_{n:03d}.py' for n in range(41,61)]
    frozen=dict(utc=datetime.now(timezone.utc).isoformat(),development_seed=41060,holdout_seeds=[61061,61062],
        scope='Frozen synthetic model validation; no novel RF data or measured hardware.',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    write_json(OUT/'matrix_freeze.json',frozen)
    replay=[]
    for n in range(41,61):
        folder=OUT/f'iteration{n:03d}';saved=json.loads((folder/'results.json').read_text(encoding='utf-8'))
        for name,digest in saved['source_sha256'].items():
            assert hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest()==digest
        command=f'import json; from matrix_hundred_{n:03d} import experiment; print(json.dumps(experiment()))'
        completed=subprocess.run([sys.executable,'-c',command],cwd=folder/'source',capture_output=True,text=True,check=True)
        actual=json.loads(completed.stdout)
        expected={k:v for k,v in saved.items() if k not in ('scope','source_sha256')}
        assert actual==expected
        replay.append(dict(iteration=n,exact_json_match=True,source_hash_verified=True))
    write_json(OUT/'matrix_reproduction.json',dict(rounds=replay,all_exact=True))
    holdout=[]
    for seed in frozen['holdout_seeds']:
        for n in range(41,61):
            mod=importlib.import_module(f'matrix_hundred_{n:03d}')
            holdout.append(dict(iteration=n,seed=seed,results=mod.experiment(seed)))
    write_json(OUT/'matrix_holdout.json',dict(frozen_before_evaluation=True,experiments=holdout))
    assert frozen['source_sha256']=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    print(json.dumps(dict(reproduced=len(replay),new_seed_evaluations=len(holdout),source_unchanged=True)))


if __name__=='__main__':
    validate()
