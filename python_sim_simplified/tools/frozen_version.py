"""Verify the pre-refactor snapshot or explicitly restore its original layout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
FROZEN_DIR = PROJECT_ROOT_DIR/'archive/pre_refactor_2026-09-16'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', type=Path, help='New directory in which to restore the original flat tree')
    args = parser.parse_args()
    manifest = json.loads((FROZEN_DIR/'FREEZE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((FROZEN_DIR/name).read_bytes()).hexdigest() == expected, name
    print(f"Verified {len(manifest['files'])} original files; Git base {manifest['git_head']}")
    if args.restore is not None:
        args.restore.mkdir(parents=True, exist_ok=False)
        for name in manifest['files']:
            target_path = args.restore/name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(FROZEN_DIR/name, target_path)
        print(f'Restored original files to {args.restore.resolve()}')
        print('Historical experiments also need their original results/, validation/ and external data paths.')


if __name__ == '__main__':
    main()
