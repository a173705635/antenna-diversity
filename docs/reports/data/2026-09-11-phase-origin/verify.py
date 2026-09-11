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
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
rows=read('summary.json')
assert sum(r['status']=='zero_dfe_output' for r in rows)==3
evaluated=[r for r in rows if r['status']=='evaluated']
assert len(evaluated)==36
for path in root.glob('*_*_*.json'):
    r=read(path.name)
    if 'cycles_after_geometry' not in r:
        continue
    c=np.asarray(r['cycles_after_geometry'])
    f=c-np.rint(c)
    np.testing.assert_allclose(f,r['fractional_cycles'],atol=1e-13)
    tail=np.asarray(r['time_s'])>=.3
    maximum=np.max(abs(f[tail]))*(299792458/1575.42e6)
    np.testing.assert_allclose(maximum,r['tail_fractional_max_abs_m'],atol=1e-13)
for r in evaluated:
    if r['layer'] in ('ideal','linear_dfe'):
        assert r['tail_known_bias_corrected_max_abs_m'] < 1e-5
        if r['label']=='known_channel_bias':
            assert abs(r['tail_fractional_mean_cycles']-.4/(2*np.pi)) < 1e-5
print(f'Verified {len(hashes)} hashes, 36 evaluated variants and 3 unavailable nominal noiseless cases.')
