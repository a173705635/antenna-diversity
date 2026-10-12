"""The maintained package must run without the frozen flat-module tree."""
import ast
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_core_is_independent_of_experiments_and_archive():
    core = importlib.import_module('core.pipeline')
    assert callable(core.run_end_to_end)
    external = sys.stdlib_module_names | {'numpy', 'scipy', 'matplotlib'}
    for path in (ROOT / 'core').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert node.module.split('.')[0] in external, path.name
            if isinstance(node, ast.Import):
                for item in node.names:
                    assert item.name.split('.')[0] in external, path.name


def test_provenance_covers_all_maintained_code_and_explicit_freeze(tmp_path):
    from core.provenance import source_hashes, freeze_source
    expected = source_hashes()
    assert 'configs/motion_summary.json' not in expected
    assert 'docs/spoofing/motion_summary.json' not in expected
    assert 'core/tracking.py' in expected
    assert 'core/spoofing/single_source.py' in expected
    assert 'core/interference/interference.py' in expected
    assert 'core/multipath/multipath.py' in expected
    assert 'core/multipath/sdcp_observation.py' in expected
    assert 'core/multipath/plotting.py' in expected
    assert 'core/spoofing/array_acquisition.py' in expected
    assert 'core/spoofing/array_processing.py' in expected
    assert 'experiments/multipath/sdcp_baseline.py' in expected
    assert 'experiments/spoofing/single_source.py' in expected
    assert 'tools/spoofing/resume_nominal_yaw.py' in expected
    sources = {p.relative_to(ROOT).as_posix()
               for name in ('core', 'experiments', 'tools')
               for p in (ROOT / name).rglob('*.py')}
    assert sources <= expected.keys()
    assert not (tmp_path / 'code').exists()
    freeze_source(tmp_path)
    import hashlib
    for name, digest in expected.items():
        assert hashlib.sha256((tmp_path / 'code' / name).read_bytes()).hexdigest() == digest


def test_historical_source_keys_resolve_to_current_files():
    from core.provenance import current_path_for_historical_source
    examples = {
        'core/single_source.py': 'core/spoofing/single_source.py',
        'core/interference.py': 'core/interference/interference.py',
        'core/multipath.py': 'core/multipath/multipath.py',
        'core/sdcp.py': 'core/multipath/sdcp.py',
        'core/sdcp_arc.py': 'core/multipath/sdcp_arc.py',
        'core/sdcp_observation.py': 'core/multipath/sdcp_observation.py',
        'core/sdcp_self_check.py': 'core/multipath/sdcp_self_check.py',
        'core/array_acquisition.py': 'core/spoofing/array_acquisition.py',
        'core/array_processing.py': 'core/spoofing/array_processing.py',
        'experiments/single_source_nominal_yaw.py':
            'experiments/spoofing/single_source_nominal_yaw.py',
        'experiments/sdcp_baseline.py': 'experiments/multipath/sdcp_baseline.py',
        'tools/resume_projection_long.py': 'tools/spoofing/resume_projection_long.py',
        'core/tracking.py': 'core/tracking.py',
        'configs/motion_summary.json': 'docs/spoofing/motion_summary.json',
    }
    for old_name, current_name in examples.items():
        assert current_path_for_historical_source(ROOT, old_name) == ROOT/current_name
        assert (ROOT/current_name).is_file()
        assert current_path_for_historical_source(ROOT, current_name) == ROOT/current_name


@pytest.mark.parametrize(('direction', 'name'), [
    ('multipath', 'sdcp'),
    ('multipath', 'sdcp_arc'),
    ('multipath', 'sdcp_observation'),
    ('multipath', 'sdcp_self_check'),
    ('spoofing', 'array_acquisition'),
    ('spoofing', 'array_processing'),
])
def test_dedicated_modules_live_only_in_their_direction(direction, name):
    module = importlib.import_module(f'core.{direction}.{name}')
    assert Path(module.__file__).resolve() == ROOT/'core'/direction/f'{name}.py'
    assert not (ROOT/'core'/f'{name}.py').exists()


def test_multipath_plots_are_owned_by_multipath():
    from core.multipath.plotting import plot_multipath_results
    from core import plotting
    assert callable(plot_multipath_results)
    assert not hasattr(plotting, 'plot_multipath_results')


@pytest.mark.parametrize(('entry', 'check', 'stages', 'cases'), [
    ('single_source_early_truncation', 'verify_inputs', (41, 42), 'CASES'),
    ('single_source_low_power_amplitude_rf', 'preflight', (39, 40, 43), 'SCENES'),
])
def test_historical_preflight_checks_relocated_report_bytes(
        tmp_path, monkeypatch, entry, check, stages, cases):
    import hashlib
    import json
    import zipfile

    module = importlib.import_module(f'experiments.spoofing.{entry}')
    root = tmp_path/'current'
    report = root/'docs/spoofing/motion_summary.json'
    report.parent.mkdir(parents=True)
    original = b'{"historical_result": 1}'
    report.write_bytes(original)
    old_name = 'configs/motion_summary.json'
    sources = {old_name: hashlib.sha256(original).hexdigest()}
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr(module, 'source_hashes', lambda: {})
    monkeypatch.setattr(module, cases, ())

    for stage in stages:
        folder = tmp_path/f'stage{stage}'
        folder.mkdir()
        snapshot = folder/'source_snapshot.zip'
        with zipfile.ZipFile(snapshot, 'w') as archive:
            archive.writestr(old_name, original)
        summary = dict(source_sha256=sources,
                       source_snapshot_sha256=module.digest(snapshot), cases={})
        for stem in ('plan', 'manifest', 'figure'):
            filename = 'overview.png' if stem == 'figure' else f'{stem}.json'
            path = folder/filename
            path.write_bytes(b'unchanged evidence')
            summary[f'{stem}_sha256'] = module.digest(path)
        (folder/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        monkeypatch.setattr(module, f'STAGE{stage}', folder)

    getattr(module, check)()
    report.write_bytes(b'{"historical_result": 2}')
    with pytest.raises(AssertionError, match=old_name.replace('.', r'\.')):
        getattr(module, check)()
