"""Check frozen threshold, weights, independent seeds and position summaries."""
import hashlib
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
def read(name):
    return json.loads((root/name).read_text(encoding='utf-8'))
hashes=read('sha256.json')
for name,digest in hashes.items():
    assert Path(name).name==name
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
manifest=read('manifest.json')
cal={s+c for s in manifest['calibration_seeds'] for c in (0,1)}
evaluation={s+c for s in manifest['evaluation_seeds'] for c in (0,1)}
assert not cal & evaluation
threshold=read('threshold.json')['threshold_m']
count=0
for path in root.glob('analysis_*.json'):
    record=read(path.name)
    assert record['threshold_m']==threshold
    residual=np.asarray(record['sdcp_residual_m'])
    hard=(abs(residual)<=threshold).astype(float)
    soft=1/(1+np.exp(np.clip((abs(residual)-threshold)/(threshold/4),-700,700)))
    np.testing.assert_array_equal(record['modes']['hard']['pair_weights'],hard)
    np.testing.assert_allclose(record['modes']['soft']['pair_weights'],soft,atol=1e-14)
    times=np.asarray(record['time_s'])
    mask=times>=.6
    seed=path.stem.rsplit('_',1)[1]
    control=read(f'analysis_los_{seed}.json')
    for name,mode in record['modes'].items():
        for field in ('code','float'):
            key=f'{field}_positions_m'
            if key not in mode:
                continue
            p=np.asarray(mode[key],dtype=float)[mask]
            p=p[np.isfinite(p).all(axis=1)]
            assert len(p)==mode[field]['solved_epochs']
            if len(p):
                jitter=np.sqrt(np.mean(np.sum((p-p.mean(axis=0))**2,axis=1)))
                np.testing.assert_allclose(jitter,mode[field]['demeaned_rms_m'],atol=1e-8)
            if key in control['modes'][name]:
                delta=np.asarray(mode[key],dtype=float)[mask]-np.asarray(control['modes'][name][key],dtype=float)[mask]
                delta=delta[np.isfinite(delta).all(axis=1)]
                if len(delta):
                    rms=np.sqrt(np.mean(np.sum(delta**2,axis=1)))
                    np.testing.assert_allclose(rms,mode[f'{field}_paired_los_change_rms_m'],atol=1e-8)
    count+=1
assert count==10
print(f'Verified {len(hashes)} hashes and {count} paired scenarios, SDCP weights and position scatter.')
