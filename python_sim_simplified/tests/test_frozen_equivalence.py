"""Separate interpreters prevent old flat modules from contaminating the package."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

pytestmark = pytest.mark.local_evidence

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/'archive/pre_refactor_2026-09-16'

PROBE = r'''
import dataclasses, hashlib, importlib, json, pathlib, sys
import numpy as np
prefix, mode, destination = sys.argv[1:]
def module(name): return importlib.import_module(prefix + name)
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
if mode == 'pipeline':
    cfg = module('presets').default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    result = module('pipeline').run_end_to_end(cfg)
    del result['config']  # output path changes intentionally with packaging
else:
    gps = module('gps_simulation' if prefix else 'run_gps_spp')
    satellites = gps.satellite_setup()
    scene, cfg = gps.build_scene(satellites, 300, signal_kind=mode)
    result = gps.run_record(3000, satellites, scene, cfg)
pathlib.Path(destination).write_text(json.dumps(encode(result), sort_keys=True), encoding='utf-8')
'''


@pytest.mark.parametrize('mode', ['pipeline', 'L1CA', 'L5Q'])
def test_frozen_and_refactored_numeric_outputs_are_identical(tmp_path, mode):
    for label, cwd, prefix in [('old', FROZEN, ''), ('new', ROOT, 'core.')]:
        process = subprocess.run([sys.executable, '-c', PROBE, prefix, mode,
                                  str(tmp_path/f'{label}.json')], cwd=cwd,
                                 capture_output=True, text=True)
        assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads((tmp_path/'old.json').read_text()) == json.loads((tmp_path/'new.json').read_text())
