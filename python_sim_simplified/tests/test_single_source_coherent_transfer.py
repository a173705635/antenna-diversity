"""Source preflight for the fixed four-second coherent transfer replay."""
import json

import pytest

from experiments.single_source_coherent_transfer import preflight_input
from experiments.single_source_suppression_study import digest


def test_transfer_preflight_checks_saved_iq_and_observation_hashes(tmp_path):
    (tmp_path / 'samples.npy').write_bytes(b'samples')
    (tmp_path / 'observations.npz').write_bytes(b'observations')
    metadata = dict(case={'duration': 4.0},
                    samples_sha256=digest(tmp_path / 'samples.npy'),
                    observations_sha256=digest(tmp_path / 'observations.npz'))
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    loaded, hashes = preflight_input(tmp_path)
    assert loaded == metadata
    assert hashes['metadata.json'] == digest(tmp_path / 'metadata.json')
    (tmp_path / 'observations.npz').write_bytes(b'changed')
    with pytest.raises(AssertionError):
        preflight_input(tmp_path)
