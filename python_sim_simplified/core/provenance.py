"""Record the maintained code identity; copy sources only on explicit freeze."""
import hashlib
from pathlib import Path
import shutil

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]


def source_hashes():
    source_paths = sorted([*PROJECT_ROOT_DIR.glob('core/**/*.py'), *PROJECT_ROOT_DIR.glob('experiments/**/*.py'),
                           *PROJECT_ROOT_DIR.glob('tools/**/*.py'), *PROJECT_ROOT_DIR.glob('configs/*.json'),
                           PROJECT_ROOT_DIR / 'requirements.txt'])
    return {source_path.relative_to(PROJECT_ROOT_DIR).as_posix(): hashlib.sha256(source_path.read_bytes()).hexdigest()
            for source_path in source_paths}


def freeze_source(output_dir):
    for name in source_hashes():
        target_path = Path(output_dir) / 'code' / name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT_DIR / name, target_path)
