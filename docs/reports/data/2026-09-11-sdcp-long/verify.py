"""Verify completed long-window archive and recompute reported detection rates."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
hashes=json.loads((root/'sha256.json').read_text())
for name,digest in hashes.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
rows=json.loads((root/'summary.json').read_text())
assert len(rows)==30
for row in rows:
    d=json.loads((root/f"w{row['window_ms']//10}_{row['kind']}_{row['seed']}.json").read_text())
    t=np.array(d['time_s'])
    r=np.array(d['residual_m'])
    a=abs(r)>row['threshold_mm']/1000
    assert np.array_equal(a,d['alarm'])
    assert np.isclose(a[t>=1.5,0].mean(),row['prn1_fraction'])
    assert np.sum(t>=1.5)==150
print(len(hashes),'hashes and 30 evaluation rows verified')
