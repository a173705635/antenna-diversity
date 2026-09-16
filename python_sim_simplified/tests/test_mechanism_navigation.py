"""End-to-end archived observations: preserve availability without false anchors."""
import json
from pathlib import Path
import numpy as np
from research_hundred import load


def test_attack_transition_does_not_seed_the_observed_34m_code_position():
    row=json.loads(Path('results/weak-ten/round10/116921_weak_30_24_4_four.json').read_text())
    with np.load(Path(row['after_root'])/'evaluations'/row['after_id']/'tracking.npz') as d:z=dict(d)
    out=load(Path('results/mechanism-four/round02/combined.py')).navigate(z)['position']
    post=z['available_time']>=.3;truth=np.c_[2*z['available_time'],np.zeros(len(out))]
    assert np.isfinite(out[post]).all()
    assert np.max(np.linalg.norm(out[post,:2]-truth[post],axis=1))<20


def test_bad_code_handoff_is_not_needed_when_earlier_carrier_anchor_is_observed():
    with np.load('results/single-source-twenty/state_handoff_probe.npz') as d:
        z={k:d[k] for k in d.files if not k.startswith('probe_')}
    # Existing synthetic probe assumes known directions; express that same
    # assumption in the new F-ratio units. RF negative controls remain separate.
    z['support']=np.where(z['support']>=.85,4.,0.)
    out=load(Path('results/mechanism-four/round02/combined.py')).navigate(z)['position']
    truth=np.c_[2*z['available_time'],np.zeros(len(out))]
    assert np.max(np.linalg.norm(out[34:39,:2]-truth[34:39],axis=1))<2
