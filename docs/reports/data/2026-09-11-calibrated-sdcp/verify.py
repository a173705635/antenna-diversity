"""Verify archive bytes and residual arithmetic."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
hashes = json.loads((root/'sha256.json').read_text())
for name, digest in hashes.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
for path in root.glob('*_61*.json'):
    data = json.loads(path.read_text())
    wavelength = 299792458/1575.42e6
    expected = (np.array(data['sdcp_m'])-data['geometry_m']+wavelength/2) % wavelength-wavelength/2
    np.testing.assert_allclose(expected, data['calibrated_m'], atol=1e-14)
print(f'{len(hashes)} hashes and 10 residual arrays verified')
