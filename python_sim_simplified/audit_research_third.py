"""Readonly historical replay and archive integrity; not counted research rounds."""
import hashlib,json
from pathlib import Path
import numpy as np
from research_hundred import load
from audit_recovery_fifty import save
ROOT=Path('results/research-third-hundred')

def run():
    archives=[]
    for name in ('research-twenty','recovery-four-hundred','research-hundred','research-next-hundred'):
        folder=Path('results')/name;hashes=json.loads((folder/'manifest.json').read_text())['sha256']
        for path,digest in hashes.items():assert hashlib.sha256((folder/path).read_bytes()).hexdigest()==digest,(name,path)
        archives.append(dict(archive=name,verified=len(hashes)))
    save(ROOT/'archive_verification.json',archives)
    observed={}
    for route in ('joint','capture'):
        with np.load(f'results/research-next-hundred/navigation_corrected_validation/86102_{route}.npz') as z:observed[route]={k:z[k] for k in z.files if k!='truth_position'}
    rows=[]
    for n in range(201,226):
        folder=ROOT/f'iteration{n}';model=load(folder/'navigation_third.py').candidate
        with np.load(folder/'86102.npz') as z:expected=z['after']
        np.testing.assert_allclose(model(observed),expected,atol=1e-7,equal_nan=True)
        generator=load(folder/'navigation_third_cases.py').generate
        pair,_=generator(90101,'normal')
        with np.load(folder/'90101_normal.npz') as z:expected=z['after']
        np.testing.assert_allclose(model(pair),expected,atol=1e-7,equal_nan=True)
        rows.append(dict(iteration=n,replayed=['86102_RF','90101_normal'],equal=True))
    save(ROOT/'navigation_replay.json',dict(cases=50,rows=rows))
    print(json.dumps(dict(archives=archives,navigation_replays=50)))

if __name__=='__main__':run()
