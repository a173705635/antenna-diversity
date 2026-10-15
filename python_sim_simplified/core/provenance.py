"""Record the maintained code identity; copy sources only on explicit freeze."""
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


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
