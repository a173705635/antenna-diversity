from pathlib import Path

import numpy as np

from audit_sdcp_arc import consume_tracking
from positioning import carrier_range


DATA=Path(__file__).resolve().parents[2]/'docs/reports/data/2026-09-13-sdcp-tracking-gap'
W=299792458/1575.42e6


def test_clean_rf_nco_reconstruction_matches_existing_carrier_observables():
    a=dict(np.load(DATA/'A.npz'));b=dict(np.load(DATA/'B_clean.npz'))
    times=b['epoch_start_samples']/6.6e6
    rows=consume_tracking(a,b,['original']*len(times),session_id='clean')
    expected=(carrier_range(dict(success=True,**b),times,6.6e6,.001,W)
              -carrier_range(dict(success=True,**a),times,6.6e6,.001,W))
    np.testing.assert_allclose([r['raw_sd_m'] for r in rows],expected,atol=1e-12)
    assert all(r['residual_m'] is not None for r in rows if .65<=r['time_s']<.79)


def test_rf_reacquisition_join_preserves_prefix_and_blocks_old_reference():
    a=dict(np.load(DATA/'A.npz'));old=dict(np.load(DATA/'B_gap.npz'))
    new=dict(np.load(DATA/'B_gap_reacquired.npz'))
    before=old['epoch_start_samples']<new['epoch_start_samples'][0]
    b={key:np.concatenate([old[key][before],new[key]]) for key in old}
    n=int(before.sum());ids=['old']*n+['new']*len(new['epoch_start_samples'])
    rows=consume_tracking(a,b,ids,session_id='joined')
    prefix=consume_tracking(a,{key:value[:n] for key,value in b.items()},ids[:n],session_id='joined')
    assert prefix==rows[:n]
    assert any(r['residual_m'] is not None for r in rows if .2<=r['time_s']<.29)
    tail=[r for r in rows if .65<=r['time_s']<.79]
    assert len(tail)==140 and all(r['quality_valid'] for r in tail)
    assert all(r['residual_m'] is None and r['state']=='await_reference' for r in tail)
    assert rows[n]['arc_id'] != rows[n-1]['arc_id']


def test_tracking_identity_change_alone_revokes_reference_on_clean_rf():
    a=dict(np.load(DATA/'A.npz'));b=dict(np.load(DATA/'B_clean.npz'))
    times=b['epoch_start_samples']/6.6e6
    ids=['old' if t<.46 else 'new' for t in times]
    rows=consume_tracking(a,b,ids,session_id='identity')
    assert all(r['residual_m'] is None for r in rows if r['time_s']>=.46)


def test_missing_a_epoch_cannot_be_filled_by_extrapolating_expired_phase():
    # At 0.201 s the preceding A epoch has ended, while B still has data.
    def history(starts):
        n=len(starts)
        return dict(epoch_start_samples=np.array(starts),
                    phase_lock_metric=np.ones(n), pll_error_rad=np.zeros(n),
                    carrier_nco_phase_rad=np.zeros(n), carrier_nco_frequency_hz=np.zeros(n))
    rows=consume_tracking(history([1320000,1333200]),
                          history([1320000,1326600,1333200]),
                          ['same']*3,session_id='missing-a')
    assert rows[0]['residual_m']==0.
    assert rows[1]['raw_sd_m'] is None and rows[1]['residual_m'] is None
    assert rows[2]['state']=='await_reference'
    np.testing.assert_allclose([r['earliest_evidence_time_s'] for r in rows],
                               [.201,.202,.203],atol=1e-15)


def test_reference_receiver_restart_revokes_pair_even_when_b_does_not_restart():
    a=dict(np.load(DATA/'A.npz'));b=dict(np.load(DATA/'B_clean.npz'))
    ids=['a-old' if n/6.6e6<.46 else 'a-new' for n in a['epoch_start_samples']]
    rows=consume_tracking(a,b,['b-same']*len(b['epoch_start_samples']),
                          session_id='reference-restart',reference_tracking_ids=ids)
    assert any(r['residual_m'] is not None for r in rows if .2<=r['time_s']<.29)
    assert all(r['residual_m'] is None for r in rows if r['time_s']>=.462)
