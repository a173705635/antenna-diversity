"""Compare GPS output with the package just before research-code removal."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

pytestmark = pytest.mark.local_evidence

ROOT = Path(__file__).resolve().parents[1]
PRE_CLEANUP = ROOT/'archive/pre_interference_only_2026-10-14_156b994'/'python_sim_simplified'

PROBE = r'''
import dataclasses, hashlib, importlib, json, pathlib, sys
import numpy as np
prefix, mode, destination = sys.argv[1:]
def encode(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=value.shape,
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, np.generic): return encode(value.item())
    if isinstance(value, complex): return [value.real, value.imag]
    if dataclasses.is_dataclass(value): return encode(dataclasses.asdict(value))
    if isinstance(value, dict): return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [encode(v) for v in value]
    if isinstance(value, pathlib.Path): return str(value)
    return value
gps = importlib.import_module(prefix + 'gps_simulation')
satellites = gps.satellite_setup()
scene, cfg = gps.build_scene(satellites, 300, signal_kind=mode)
result = gps.run_record(3000, satellites, scene, cfg)
pathlib.Path(destination).write_text(json.dumps(encode(result), sort_keys=True), encoding='utf-8')
'''


@pytest.mark.parametrize('mode', ['L1CA', 'L5Q'])
def test_gps_outputs_match_pre_cleanup_snapshot(tmp_path, mode):
    for label, cwd, prefix in [('old', PRE_CLEANUP, 'core.'), ('new', ROOT, 'core.')]:
        process = subprocess.run([sys.executable, '-B', '-c', PROBE, prefix, mode,
                                  str(tmp_path/f'{label}.json')], cwd=cwd,
                                 capture_output=True, text=True)
        assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads((tmp_path/'old.json').read_text()) == json.loads((tmp_path/'new.json').read_text())
