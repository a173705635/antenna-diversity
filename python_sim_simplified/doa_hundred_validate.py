"""Frozen snapshot replay plus held-out noise draws; does not add research rounds."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import doa_hundred as d


def compare(actual, expected):
    if isinstance(actual, dict):
        return all(compare(value, expected[key]) for key, value in actual.items())
    if isinstance(actual, (list, tuple, np.ndarray)):
        return bool(np.allclose(actual, expected, rtol=1e-9, atol=1e-10))
    if isinstance(actual, (float, np.floating)):
        return bool(np.isclose(actual, expected, rtol=1e-9, atol=1e-10))
    return actual == expected


def main():
    replay = []
    for number in range(61, 81):
        folder = d.ROOT/f'iteration{number:03d}'
        source = folder/'doa_hundred.py'
        spec = importlib.util.spec_from_file_location(f'doa_snapshot_{number}', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = getattr(module, f'round{number:03d}')()
        expected = json.loads((folder/'results.json').read_text())
        entry = dict(iteration=number, replay_matches=compare(result, expected),
                     source_hash_matches=hashlib.sha256(source.read_bytes()).hexdigest()==expected['source_sha256'],
                     conclusion_present=(folder/'conclusion.json').exists())
        replay.append(entry)
    frozen_hash = hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest()
    holdout = []
    for seed in [9041, 9042, 9043]:
        d.rng = lambda seed=seed: np.random.default_rng(seed)
        results = {str(n): getattr(d, f'round{n:03d}')() for n in [67, 70, 73, 74, 75, 77, 78]}
        holdout.append(dict(seed=seed, results=results))
    summary = dict(frozen_source_sha256=frozen_hash, development_seed=61080,
                   heldout_seeds=[9041, 9042, 9043], replay=replay, holdout=holdout,
                   all_replays_match=all(all(row[k] for k in ['replay_matches', 'source_hash_matches', 'conclusion_present']) for row in replay),
                   interpretation='Held-out noise draws only: geometry and model families fixed. Not unseen hardware, propagation, or adversarial validation.')
    (d.ROOT/'doa_validation.json').write_text(json.dumps(summary, indent=2, default=d.serial), encoding='utf8')
    print(json.dumps(summary, default=d.serial))


if __name__ == '__main__':
    main()
