"""Observed signal-loss case: stale quality must not admit a false code fix."""
import json
from pathlib import Path
import numpy as np
from research_hundred import load


def test_signal_loss_cannot_publish_the_observed_89m_false_fix():
    row=json.loads(Path('results/weak-ten/round06/114101_absent_30_24_2_three.json').read_text())
    path=Path(row['after_root'])/'evaluations'/row['after_id']/'tracking.npz'
    with np.load(path) as data:z={key:data[key] for key in data.files}
    result=load(Path('single_source_receiver_weak.py')).navigate(z)
    assert np.isfinite(result['position'][z['available_time']<.3]).all(1).any()
    assert np.isnan(result['position'][z['available_time']>=.34]).all()
