"""Current experiment CLIs work without the old top-level script directory."""
import importlib
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['full_chain', 'gps_spp', 'gps_l1_l5', 'recorded_iq', 'single_source'])
def test_direct_entry_help_from_outside_project(name, tmp_path):
    module = importlib.import_module(f'experiments.{name}')
    assert module.__doc__
    process = subprocess.run([sys.executable, str(ROOT/'experiments'/f'{name}.py'), '--help'],
                             cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert '--output' in process.stdout
    assert '--freeze-source' in process.stdout
