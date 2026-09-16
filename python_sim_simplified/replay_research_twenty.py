"""Read-only reproduction of every saved proposal; never overwrites round outputs."""
import importlib.util,json
from pathlib import Path
import numpy as np
from research_twenty import prompt_data
from audit_recovery_fifty import save

ROOT=Path('results/research-twenty')


def load(path):
    spec=importlib.util.spec_from_file_location('saved_proposal',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


navigation_records=0;prompt_records=0;rf_records=0
for number in range(1,20):
    folder=ROOT/f'iteration{number:02d}'
    result=json.loads((folder/'results.json').read_text())
    if number<=10:
        method=load(folder/'navigation_candidate.py').candidate
        for row in result['rows']:
            path=Path(row['record'])
            with np.load(path) as data:z={k:data[k] for k in data.files}
            with np.load(folder/(path.stem+'.npz')) as data:
                np.testing.assert_allclose(method(z),data['position'],atol=1e-8,equal_nan=True)
            navigation_records+=1
    else:
        method=load(folder/'calibration_candidate.py').calibrate
        for row in result['rows']:
            z,a,truth=prompt_data(row['seed'],row['kind'])
            if number==19:
                rng=np.random.default_rng(row['seed']+991)
                prior=truth*np.exp(1j*rng.normal(0,np.radians(5),len(truth)));prior/=prior[0]
                g,info=method(z,a,prior)
            else:g,info=method(z,a)
            error=float(np.sqrt(np.mean(np.degrees(np.angle((g/g[0])/truth))**2)))
            np.testing.assert_allclose(error,row['proposal_deg'],atol=1e-8)
            assert info['usable']==row['usable'];prompt_records+=1
for folder in sorted((ROOT/'iteration20').glob('*_*')):
    if not folder.is_dir():continue
    record=json.loads((folder/'result.json').read_text())
    source=ROOT/('iteration20' if record['seed']<23001 else 'final_source')/'navigation_candidate.py'
    method=load(source).candidate
    with np.load(folder/'positions.npz') as data:expected={k:data[k] for k in data.files}
    for name,key in (('baseline','navigation'),('joint','joint')):
        with np.load(folder/(name+'.npz')) as data:z={k:data[k] for k in data.files}
        p=method(z)
        if name=='joint' and not record['calibration']['usable']:p[:]=np.nan
        np.testing.assert_allclose(p,expected[key],atol=1e-8,equal_nan=True)
    rf_records+=1
save(ROOT/'reproduction.json',dict(iterations=20,navigation_proposal_replays=navigation_records,
                                 prompt_proposal_replays=prompt_records,rf_observation_replays=rf_records))
print(navigation_records,prompt_records,rf_records,flush=True)
