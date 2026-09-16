"""Current experiments run from fixed in-file paths without CLI parameters."""
import importlib
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', [
    'full_chain', 'gps_spp', 'gps_l1_l5', 'recorded_iq', 'single_source',
    'sdcp_baseline', 'sdcp_multipath', 'sdcp_window_change',
    'verify_sdcp_baseline', 'verify_sdcp_multipath',
])
def test_entry_has_fixed_paths_and_no_argument_parser(name):
    module = importlib.import_module(f'experiments.{name}')
    assert module.__doc__
    source = (ROOT/'experiments'/f'{name}.py').read_text(encoding='utf-8')
    assert 'ArgumentParser' not in source
    assert module.OUTPUT.parent == ROOT/'results'


def test_fixed_output_is_refreshed_for_reruns(tmp_path, monkeypatch):
    from experiments import paths
    monkeypatch.setattr(paths, 'RESULTS', tmp_path)
    output = paths.reset_output('demo')
    (output/'stale.txt').write_text('old', encoding='utf-8')
    assert paths.reset_output('demo') == output
    assert not (output/'stale.txt').exists()
