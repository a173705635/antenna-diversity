"""Archive and selected historical replay checks; never research iterations."""
import hashlib,json
from pathlib import Path
import numpy as np
from research_hundred import load
from research_next_navigation import ROOT
from audit_recovery_fifty import save

def run():
    archives=[]
    for name in ('research-twenty','recovery-four-hundred','research-hundred'):
        folder=Path('results')/name
        manifest=json.loads((folder/'manifest.json').read_text())
        hashes=manifest['sha256'];bad=[]
        for path,digest in hashes.items():
            if hashlib.sha256((folder/path).read_bytes()).hexdigest()!=digest:bad.append(path)
        assert not bad,(name,bad)
        archives.append(dict(archive=name,verified=len(hashes),mismatches=bad))
    save(ROOT/'archive_verification.json',archives)
    pair={}
    for route in ('capture','joint'):
        with np.load(f'results/research-hundred/rf_validation/43002_{route}.npz') as z:pair[route]={k:z[k] for k in z.files if k!='truth_position'}
    rows=[]
    for n in range(176,201):
        folder=ROOT/f'iteration{n:03d}';model=load(folder/'navigation_next.py').candidate
        with np.load(folder/'43002.npz') as z:expected=z['proposal']
        np.testing.assert_allclose(model(pair),expected,atol=1e-8,equal_nan=True)
        rows.append(dict(iteration=n,case='43002_RF',match=True))
        if n>176:
            generator=load(folder/'navigation_next_cases.py').generate
            observed,_=generator(81001,'normal')
            with np.load(folder/'81001_normal.npz') as z:expected=z['proposal']
            np.testing.assert_allclose(model(observed),expected,atol=1e-8,equal_nan=True)
            rows.append(dict(iteration=n,case='81001_normal_observation',match=True))
    save(ROOT/'navigation_replay.json',dict(replayed=len(rows),rows=rows))
    print(json.dumps(dict(archives=archives,replayed=len(rows))))

if __name__=='__main__':run()
