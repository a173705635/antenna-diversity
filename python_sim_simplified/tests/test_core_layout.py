"""The maintained package must run without the frozen flat-module tree."""
import ast
import importlib
import sys
from pathlib import Path

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
    assert 'core/interference/interference.py' in expected
    assert not any('/spoofing/' in name or '/multipath/' in name for name in expected)
    sources = {p.relative_to(ROOT).as_posix()
               for name in ('core', 'experiments', 'tools')
               for p in (ROOT / name).rglob('*.py')}
    assert sources <= expected.keys()
    assert not (tmp_path / 'code').exists()
    freeze_source(tmp_path)
    import hashlib
    for name, digest in expected.items():
        assert hashlib.sha256((tmp_path / 'code' / name).read_bytes()).hexdigest() == digest


def test_dedicated_research_modules_are_removed():
    for direction in ('spoofing', 'multipath'):
        assert not (ROOT/'core'/direction).exists()
        assert not (ROOT/'experiments'/direction).exists()
    assert not (ROOT/'tools'/'spoofing').exists()
