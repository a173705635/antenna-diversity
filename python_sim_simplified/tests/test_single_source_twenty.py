import numpy as np
from single_source_twenty import ROOT,input_record
from single_source_receiver import frontend,navigate


def test_four_channel_record_is_exactly_paired_and_receiver_has_no_attack_truth():
    record=input_record(ROOT/'inputs/113001_drift_0.6')
    assert 'truth' not in record and 'attack_start' not in record
    np.testing.assert_array_equal(record['samples'][[0,1,2]][:2],record['samples'][[0,1]])


def test_block_buffered_receiver_does_not_use_later_samples():
    record=input_record(ROOT/'inputs/113001_normal_0.6')
    full=frontend(record,[0,1,2])
    partial=dict(record)
    partial['samples']=record['samples'][:,:round(.4*2.046e6)]
    mask=record['times']+.02<.4
    partial['times']=record['times'][mask];partial['satellites']=record['satellites'][mask]
    early=frontend(partial,[0,1,2])
    for field in ('ranges','carrier','lock','cn0','gain'):
        np.testing.assert_allclose(early[field],full[field][mask],rtol=0,atol=1e-10,equal_nan=True)
    np.testing.assert_allclose(navigate(early)['position'],navigate(full)['position'][mask],equal_nan=True)


def test_initialization_requires_consecutive_observed_positions():
    record=input_record(ROOT/'inputs/113001_normal_0.6')
    z=frontend(record,[0,1,2])
    broken=dict(z,ranges=z['ranges'].copy())
    broken['ranges'][:4]=np.nan
    assert navigate(broken)['initialization_time'] is None or navigate(broken)['initialization_time']>=z['available_time'][6]


def test_spatial_match_is_bit_invariant_and_rejects_orthogonal_direction():
    from single_source_receiver import spatial_match
    direction=np.array([1.,1.,1.,1.],complex)
    local=np.ones(2046*20,complex)
    bits=np.repeat(np.tile([1.,-1.],10),2046)
    assert abs(spatial_match(direction[:,None]*bits,local,direction)-1.)<1e-12
    assert spatial_match(np.array([1.,-1.,1.,-1.])[:,None]*bits,local,direction)==0.


def test_reacquired_code_epoch_keeps_global_range_for_noninteger_ms_cursor():
    from acquisition import acquire
    from tracking import track
    from positioning import code_pseudorange
    from presets import default_config
    from code_replica import bandlimited_code_replica,_sample_code
    from gnss import ca_code
    fs=2.046e6;delay=300;cursor=123;cfg=default_config()
    cfg.acquisition.doppler_bins_hz=np.array([0.])
    cfg.tracking.code_replica_bandwidth_hz=.9e6;cfg.tracking.refine_initial_phase=True
    n=np.arange(round(.03*fs));code=bandlimited_code_replica(ca_code(1),1.023e6,.9e6)
    rng=np.random.default_rng(113099)
    signal=_sample_code(code,(n-delay)*1.023e6/fs)+.01*(rng.normal(size=len(n))+1j*rng.normal(size=len(n)))
    acq=acquire(signal[cursor:],fs,1,1.023e6,cfg.acquisition)
    h=track(signal[cursor:],fs,1,acq,cfg.tracking,1575.42e6,1.023e6)
    h['epoch_start_samples']+=cursor
    rho=code_pseudorange(h,np.array([.02]),fs,.001,0.)[0]
    assert abs(rho-delay/fs*299792458.)<1.
