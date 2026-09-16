"""Shared SDCP experiment serialization and source verification."""
import hashlib
import json

import numpy as np

from core.provenance import source_hashes


def statistics(x):
    return {'epochs': len(x), 'mean_mm': float(np.mean(x)*1000),
            'std_mm': float(np.std(x)*1000),
            'rms_mm': float(np.sqrt(np.mean(x*x))*1000),
            'max_abs_mm': float(np.max(abs(x))*1000)}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)), encoding='utf-8')


def verify_sources(root, expected):
    """Verify either a preserved snapshot or the current maintained sources."""
    if (root/'code').is_dir():
        actual = {name: hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()
                  for name in expected}
        mode = 'snapshot'
    else:
        actual = source_hashes()
        mode = 'current_source'
    assert actual == expected, f'{mode} source hashes differ from the recorded run'
    return mode
