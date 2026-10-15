"""Current experiments run from fixed in-file paths without CLI parameters."""
import importlib
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', [
    'interference.full_chain', 'gps_spp', 'gps_l1_l5', 'recorded_iq',
])
def test_entry_has_fixed_paths_and_no_argument_parser(name):
    module = importlib.import_module(f'experiments.{name}')
    assert module.__doc__
    source = (ROOT/'experiments'/Path(*name.split('.'))).with_suffix('.py').read_text(encoding='utf-8')
    assert 'ArgumentParser' not in source
    direction = name.split('.')[0]
    expected_parent = ROOT/'results'
    if direction == 'interference':
        expected_parent /= direction
    assert module.OUTPUT.parent == expected_parent


def test_fixed_output_is_refreshed_for_reruns(tmp_path, monkeypatch):
    from experiments import paths
    output = paths.reset_output(tmp_path/'interference'/'demo')
    (output/'stale.txt').write_text('old', encoding='utf-8')
    sibling = tmp_path/'interference'/'other_experiment'
    sibling.mkdir()
    (sibling/'keep.txt').write_text('keep', encoding='utf-8')
    assert paths.reset_output(output) == output
    assert not (output/'stale.txt').exists()
    assert (sibling/'keep.txt').read_text(encoding='utf-8') == 'keep'


@pytest.mark.parametrize('name', ['gps_spp', 'gps_l1_l5', 'recorded_iq'])
def test_shared_entry_resets_its_declared_result_path(name, monkeypatch):
    module = importlib.import_module(f'experiments.{name}')

    class OutputCaptured(Exception):
        pass

    def capture(output):
        assert output == module.OUTPUT
        raise OutputCaptured

    monkeypatch.setattr(module, 'reset_output', capture)
    with pytest.raises(OutputCaptured):
        module.main()
