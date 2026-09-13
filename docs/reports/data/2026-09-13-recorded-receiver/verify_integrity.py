"""Verify frozen evidence bytes and the receiver source used by the real run."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def main():
    expected = json.loads((ROOT/'checksums.json').read_text())
    actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in ROOT.rglob('*') if p.is_file() and p.name != 'checksums.json'
              and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    assert actual == expected, 'Evidence files changed, missing, or unlisted'
    manifest = json.loads((ROOT/'run/manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest() == digest
    print(f"Verified {len(actual)} evidence files and {len(manifest['source_sha256'])} receiver source hashes")


if __name__ == '__main__': main()
