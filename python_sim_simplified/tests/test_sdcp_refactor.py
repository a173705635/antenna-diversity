"""Regression checks for shared SDCP observations and explicit source provenance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from core.sdcp_observation import raw_sdcp
from experiments import sdcp_baseline, sdcp_multipath, sdcp_support


def test_raw_observation_preserves_unwrapped_cycles_at_common_times():
    fs, wave = 10000., .19
    times = np.array([.0011, .0021, .0031])
    histories = []
    for starts, frequency, offset in [([2, 12, 22, 32], 1000., 0.),
                                       ([5, 15, 25, 35], 1002., 7*np.pi)]:
        starts = np.array(starts)
        histories.append(dict(epoch_start_samples=starts,
            carrier_nco_phase_rad=2*np.pi*frequency*starts/fs+offset,
            carrier_nco_frequency_hz=np.full(4, frequency)))
    np.testing.assert_allclose(raw_sdcp(*histories, fs, times, wave),
                               -wave*(3.5+2*times), atol=1e-14)


@pytest.mark.parametrize('experiment,keeps_search_array',
                         [(sdcp_baseline, True), (sdcp_multipath, False)])
def test_receive_outputs_preserve_experiment_metadata(monkeypatch, tmp_path,
                                                     experiment, keeps_search_array):
    history = dict(epoch_start_samples=np.array([0, 10]), status='tracking', success=True)
    metadata = dict(noise_seed=17, acquisition=dict(success=True, grid=np.array([2, 3])),
                    tracking_status='tracking', tracking_success=True, rf={}, dfe={})
    monkeypatch.setattr(experiment, 'receive_observation', lambda *args: (history, metadata))
    result = experiment.receive(np.zeros(20), None, 17, tmp_path, 'A')
    assert result is history
    saved = json.loads((tmp_path/'A.json').read_text())
    assert ('grid' in saved['acquisition']) == keeps_search_array
    with np.load(tmp_path/'A.npz') as arrays:
        assert arrays.files == ['epoch_start_samples']
        np.testing.assert_array_equal(arrays['epoch_start_samples'], [0, 10])


@pytest.mark.parametrize('filename', ['tracking.py', 'core/tracking.py'])
def test_verifier_accepts_old_and_new_snapshot_layouts(tmp_path, monkeypatch, filename):
    snapshot = tmp_path/'code'/filename
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b'preserved original source\n')
    expected = {filename: hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    monkeypatch.setattr(sdcp_support, 'source_hashes', lambda: {'current.py': 'different'})
    assert sdcp_support.verify_sources(tmp_path, expected) == 'snapshot'
    snapshot.write_bytes(b'changed\n')
    with pytest.raises(AssertionError, match='snapshot source hashes differ'):
        sdcp_support.verify_sources(tmp_path, expected)


def test_verifier_requires_matching_current_source_for_runs_without_snapshot(tmp_path, monkeypatch):
    expected = {'core/tracking.py': 'recorded hash'}
    monkeypatch.setattr(sdcp_support, 'source_hashes', lambda: expected.copy())
    assert sdcp_support.verify_sources(tmp_path, expected) == 'current_source'
    monkeypatch.setattr(sdcp_support, 'source_hashes', lambda: {'core/tracking.py': 'changed hash'})
    with pytest.raises(AssertionError, match='current_source source hashes differ'):
        sdcp_support.verify_sources(tmp_path, expected)
    assert not (tmp_path/'code').exists()
