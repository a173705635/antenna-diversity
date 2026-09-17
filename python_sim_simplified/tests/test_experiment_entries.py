"""Current experiments run from fixed in-file paths without CLI parameters."""
import importlib
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', [
    'full_chain', 'gps_spp', 'gps_l1_l5', 'recorded_iq', 'single_source',
    'sdcp_baseline', 'sdcp_multipath',
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


def test_baseline_entry_runs_verification_after_simulation(monkeypatch, tmp_path):
    from experiments import sdcp_baseline

    stages = []

    def simulate():
        stages.append('simulation')
        return tmp_path

    def verify(output):
        assert output == tmp_path
        assert stages == ['simulation']
        stages.append('verification')

    monkeypatch.setattr(sdcp_baseline, 'run_simulation', simulate)
    monkeypatch.setattr(sdcp_baseline, 'verify_results', verify)
    sdcp_baseline.main()
    assert stages == ['simulation', 'verification']
