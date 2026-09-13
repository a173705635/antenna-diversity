import numpy as np
import pytest


def test_navigation_is_reusable_without_importing_audit_scripts():
    from gps_lnav import decode_lnav_word, navigation_check
    from recorded_receiver import read_iq8
    assert decode_lnav_word(0x22c00012)==0x8b0000
    assert not navigation_check({'prompt':np.ones(9000,dtype=complex)})['demodulation_evidence']


def test_transmit_time_uses_decoded_tow_and_code_not_a_coarse_range():
    from gps_observables import transmit_time
    fs=1e6
    # Block starts straddle a modulo-code wrap. True residual phase is negative.
    h={'success':True,'epoch_start_samples':np.arange(5)*1000,
       'code_nco_phase_chips':np.array([.02,.01,0.,1022.99,1022.98]),
       'code_nco_rate_hz':np.full(5,1022990.)}
    anchor={'epoch_index':2,'tow_s':123456.}
    t=np.array([.002,.003,.0045,-.001,.005])
    out=transmit_time(h,anchor,t,fs)
    np.testing.assert_allclose(out[:3], [123456.,123456.001-.01/1023000,
       123456.002+(-.02+1022990*.0005)/1023000],atol=1e-10,rtol=0)
    assert np.isnan(out[3:]).all()
    h['epoch_start_samples'][3:]+=1000
    with pytest.raises(ValueError,match='continuous'):
        transmit_time(h,anchor,t,fs)


def test_broadcast_circular_orbit_and_clock_at_toe():
    from gps_broadcast import satellite_state
    eph=dict(sqrt_a=5153.795, e=0., m0=0., omega=0., omega0=0., i0=0.,
       omega_dot=0., idot=0., delta_n=0., cuc=0.,cus=0.,crc=0.,crs=0.,cic=0.,cis=0.,
       toe=0.,toc=0.,af0=1e-4,af1=2e-12,af2=0.,tgd=1e-8)
    xyz,clock=satellite_state(eph,0.)
    np.testing.assert_allclose(xyz,[5153.795**2,0.,0.],atol=1e-6)
    assert clock==pytest.approx(1e-4)
    xyz2,clock2=satellite_state(eph,60.)
    assert np.linalg.norm(xyz2)==pytest.approx(5153.795**2,abs=1e-6)
    assert clock2==pytest.approx(1e-4+60*2e-12)


def test_ephemeris_bit_fields_and_issue_consistency():
    from gps_lnav import decode_ephemeris
    def make(sf,fields):
        value=0x8b << 232
        value|=(100001<<7 | sf<<2) << 192
        for offset,width,x in fields:
            value|=(x&((1<<width)-1)) << (240-offset-width)
        return [(value>>(24*(9-i)))&0xffffff for i in range(10)]
    one=make(1,[(48,10,942),(70,2,1),(160,8,-3),(168,8,42),
        (176,16,37000),(192,8,-2),(200,16,5),(216,22,-1234)])
    two=make(2,[(48,8,42),(56,16,-11),(72,16,10),(88,32,-12345678),
        (136,32,100000000),(184,32,2702000000),(216,16,37000)])
    three=make(3,[(64,32,-23456789),(112,32,600000000),(216,8,42),(224,14,-5)])
    eph=decode_ephemeris({1:one,2:two,3:three})
    assert eph['week_mod1024']==942 and eph['iodc']==298 and eph['iode']==42
    assert eph['af0']==pytest.approx(-1234*2**-31)
    assert eph['tgd']==pytest.approx(-3*2**-31)
    assert eph['m0']==pytest.approx(-12345678*2**-31*np.pi)
    assert eph['sqrt_a']==pytest.approx(2702000000*2**-19)
    assert eph['omega0']==pytest.approx(-23456789*2**-31*np.pi)
    bad=three.copy();bad[9]^=1<<16
    with pytest.raises(ValueError,match='issue'):
        decode_ephemeris({1:one,2:two,3:bad})
    with pytest.raises(ValueError,match='subframes'):
        decode_ephemeris({1:one,2:two})


@pytest.mark.parametrize('bits',[0,8])
def test_chunked_dfe_matches_batch_across_odd_block_boundaries(bits):
    from dfe import process_dfe, process_dfe_blocks
    from config import DFEConfig
    rng=np.random.default_rng(99)
    x=30*(rng.normal(size=(2,999))+1j*rng.normal(size=(2,999)))
    cfg=DFEConfig(decimation=3,cutoff_hz=100e3,num_taps=65,output_bits=bits,output_binary_shift=1)
    expected,_=process_dfe(x,cfg,1e6,23456.)
    chunks=[x[:,:17],x[:,17:201],x[:,201:505],x[:,505:]]
    actual=np.concatenate(list(process_dfe_blocks(chunks,cfg,1e6,23456.)),axis=1)
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-12)


def test_receiver_clock_origin_does_not_set_position():
    from gps_broadcast import satellite_state,reception_frame
    from gps_observables import solve_gps_epoch,C
    receiver=np.array([3900000.,300000.,5000000.]);tr=200000.
    ephs=[];txs=[]
    for omega,m in [(0.,0.),(1.,1.),(2.,2.),(3.,3.),(4.,4.),(5.,5.)]:
        eph=dict(sqrt_a=5153.795,e=.01,m0=m,omega=.3,omega0=omega,i0=.95,
            omega_dot=-8e-9,idot=0.,delta_n=1e-9,cuc=1e-6,cus=2e-6,crc=200.,crs=20.,cic=0.,cis=0.,
            toe=tr,toc=tr,af0=1e-4,af1=2e-12,af2=0.,tgd=0.)
        tau=.075
        for _ in range(10):
            p,clock=satellite_state(eph,tr-tau)
            tau=np.linalg.norm(reception_frame(p,tau)-receiver)/C
        txs.append(tr-tau+clock);ephs.append(eph)
    a=solve_gps_epoch(ephs,txs,tr+.01)
    b=solve_gps_epoch(ephs,txs,tr+.03)
    np.testing.assert_allclose(a['position_m'],receiver,atol=.03)
    np.testing.assert_allclose(b['position_m'],a['position_m'],atol=.03)
    assert b['clock_bias_m']-a['clock_bias_m']==pytest.approx(C*.02,abs=.03)


def _real_navigation_fixture():
    import json
    from pathlib import Path
    f = json.loads((Path(__file__).parent/'data/gps_lnav_prn02.json').read_text())
    prompt = np.concatenate([np.ones(500+f['offset_after_500']),
                             np.repeat(1-2*np.array(f['bits']), 20)]).astype(complex)
    return f, {'prompt': prompt, 'epoch_start_samples': np.arange(len(prompt))*5000}


def test_real_navigation_bits_decode_and_tracking_gap_is_rejected():
    from gps_lnav import decode_lnav
    f, h = _real_navigation_fixture()
    nav = decode_lnav(h, 5e6)
    assert nav['status'] == 'ephemeris_decoded'
    assert len(nav['time_anchors']) == 6
    assert nav['time_anchors'][0]['tow_s'] == f['expected_tow']
    assert nav['ephemeris']['iode'] == f['expected_iode']
    h['epoch_start_samples'][20000:] += 5000
    bad = decode_lnav(h, 5e6)
    assert bad['status'] == 'discontinuous_tracking'
    assert bad['ephemeris'] is None and not bad['time_anchors']


def test_parity_error_in_only_clock_subframe_prevents_ephemeris_admission():
    from gps_lnav import decode_lnav
    _, h = _real_navigation_fixture()
    good = decode_lnav(h, 5e6)
    frame = next(f for f in good['subframes'] if f['subframe_id'] == 1)
    # Flip a payload bit in word 3 of the only complete subframe 1.
    k = frame['epoch_index'] + (60+10)*20
    h['prompt'][k:k+20] *= -1
    bad = decode_lnav(h, 5e6)
    assert bad['status'] == 'time_only' and bad['ephemeris'] is None
