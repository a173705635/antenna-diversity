"""State admission regression found after freezing round 20; no RF success claim."""
import numpy as np
from single_source_receiver_fixed import navigate
from single_source_twenty import ROOT


def test_unchecked_code_candidate_cannot_seed_carrier_bridge_after_prn_handoff():
    with np.load(ROOT/'state_handoff_probe.npz') as stored:
        z={k:stored[k] for k in stored.files if not k.startswith('probe_')}
    result=navigate(z)
    assert np.isfinite(result['position'][34]).all()
    assert np.isnan(result['position'][35:39]).all()
    assert np.isnan(result['candidate_position'][35:39]).all()
