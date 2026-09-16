from pathlib import Path
import json
import numpy as np
from mechanism_phase import phase_prediction


def fixture():
    r=json.loads(Path('results/weak-ten/round10/116921_weak_30_24_4_four.json').read_text())
    folder=Path(r['after_root'])/'evaluations'/r['after_id']
    with np.load(folder/'tracking.npz') as d:z=dict(d)
    with np.load(folder/'navigation.npz') as d:positions=d['position'].copy()
    valid=np.isfinite(z['ranges'])&(z['cn0']>=25)&(z['lock']>=.5)&(z['gain']>=.05)&(z['support']>=3)
    return z,valid,positions


def test_disjoint_adjacent_subsets_can_use_an_earlier_published_anchor():
    z,valid,positions=fixture()
    assert not (valid[16]&valid[17]).any()
    predicted=phase_prediction(z,valid,positions,17)
    assert predicted is not None
    assert np.linalg.norm(predicted[:2]-[2*z['available_time'][17],0])<1


def test_new_tracking_arc_cannot_bridge_an_old_position():
    z,valid,positions=fixture();z['arc'][17]+=1
    assert phase_prediction(z,valid,positions,17) is None


def test_unpublished_positions_cannot_seed_a_bridge():
    z,valid,positions=fixture();positions[:17]=np.nan
    assert phase_prediction(z,valid,positions,17) is None
