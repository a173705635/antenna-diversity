import numpy as np
from audit_antenna_selection import evaluate
from audit_sdcp_positioning import BASELINE
from run_gps_spp import satellite_setup, receiver_reference


def test_selection_models_antenna_displacement_in_position_solution():
    sats = satellite_setup()
    position = receiver_reference()[0]
    pair = []
    for ch in (0,1):
        ranges = [[np.linalg.norm(s['position_m']-position-ch*BASELINE)+75000 for s in sats]]
        pair.append({'time_s':[0.], 'fs_dfe_hz':1e6,'pseudoranges_m':ranges,
            'satellite_tracking':[{'prn':s['prn'],'epoch_start_samples':[0],
                                   'cn0_db_hz':[40+ch]} for s in sats]})
    result = evaluate(pair,'cn0_db_hz')
    assert result['quality_selection']['B_fraction']==1.
    for v in result.values():
        assert v['solved']==1
        assert v['rms_m'] < 1e-4
