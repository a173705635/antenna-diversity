"""Conditional RF replay with illustrative bounds/threshold; not detection-rate validation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from gf_quality_gate import classify

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/reports/data/2026-09-12-gf-startup-rf'
OUTPUT=ROOT/'docs/reports/data/2026-09-12-gf-gate'


def counts(state):
    return {name:int(np.count_nonzero(state==value)) for name,value in
            [('unknown',-1),('no_exceedance_evidence',0),('suspected_anomaly',1)]}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    hashes=json.loads((SOURCE/'sha256.json').read_text())
    for name,digest in hashes.items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest: raise ValueError('hash mismatch')
    d=json.loads((SOURCE/'summary.json').read_text())
    rows=[]
    for scene,key in [('los','los_gf_m'),('startup','static_gf_m')]:
        r=np.asarray(d[key],float)
        for branch in [False,True]:
            for bound in [None,.003,.008]:
                out=classify(r,.010,nuisance_bound_m=np.nan if bound is None else bound,
                             valid=np.asarray(d['valid'],bool),branch_verified=branch)
                rows.append({'scene':scene,'external_branch_assumed_verified':branch,
                    'illustrative_nuisance_bound_m':bound,'prn1':counts(out['state'][:,0]),
                    'other_prns':counts(out['state'][:,1:])})
    result={'scope':__doc__,'threshold_m':.010,
        'warning':'10 mm threshold and 3/8 mm bounds are illustrative, not independently calibrated. '
                  'True branch flags are conditional experiments, not receiver certification. No detector probabilities.',
        'rows':rows,'inputs':hashes,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [Path(__file__),Path(__file__).with_name('gf_quality_gate.py')]}}
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
