"""Weak-signal changes must retain the observed-arc state handoff rejection."""
import numpy as np
from pathlib import Path
from research_hundred import load
from single_source_twenty import ROOT


def test_code_handoff_cannot_seed_a_carrier_bridge():
    with np.load(ROOT/'state_handoff_probe.npz') as data:
        z={k:data[k] for k in data.files if not k.startswith('probe_')}
    # This synthetic handoff probe assumes confirmed directions; encode the
    # same assumption in the new energy-ratio units, independently of RF tests.
    z['support']=np.where(z['support']>=.85,4.,0.)
    result=load(Path('single_source_receiver_weak.py')).navigate(z)
    assert np.isfinite(result['position'][34]).all()
    assert np.isnan(result['position'][35:39]).all()
    assert np.isnan(result['candidate_position'][35:39]).all()
