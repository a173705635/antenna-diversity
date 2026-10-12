"""Record the maintained code identity; copy sources only on explicit freeze."""
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def current_path_for_historical_source(root, name):
    """Locate today's file for a key retained in an older source manifest."""
    path = Path(name)
    if path.parts[0] == 'core':
        if path.name in {'interference.py', 'multipath.py'}:
            path = Path('core') / path.stem / path.name
        elif path.name in {'single_source.py', 'single_source_scene.py',
                           'spatial_spoofing.py', 'mechanism_phase.py',
                           'mechanism_smoothing.py', 'recovery_geometry.py'}:
            path = Path('core/spoofing') / path.name
    elif path.parts[0] == 'experiments':
        if path.stem.startswith('single_source') or path.stem == 'source_evidence_model':
            path = Path('experiments/spoofing') / path.name
        elif path.stem.startswith('sdcp_'):
            path = Path('experiments/multipath') / path.name
        elif path.stem == 'full_chain':
            path = Path('experiments/interference') / path.name
    elif path.parts[0] == 'tools' and path.stem != 'frozen_version':
        path = Path('tools/spoofing') / path.name
    return Path(root) / path


def source_hashes():
    paths = sorted([*ROOT.glob('core/**/*.py'), *ROOT.glob('experiments/**/*.py'),
                    *ROOT.glob('tools/**/*.py'), *ROOT.glob('configs/*.json'),
                    ROOT / 'requirements.txt'])
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def freeze_source(output):
    for name in source_hashes():
        target = Path(output) / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
