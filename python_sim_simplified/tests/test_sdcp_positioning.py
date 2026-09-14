import numpy as np
from run_gps_spp import build_scene, satellite_setup
from config import MultipathPath, Direction


def test_multisat_echo_is_source_only_and_preserves_preonset_samples():
    satellites = satellite_setup()[:1]
    clean, _ = build_scene(satellites, 30)
    echo, cfg = build_scene(satellites, 30,
        multipath_by_prn={1: [MultipathPath(Direction(110,15), relative_power_db=-6,
                              excess_delay_chips=.25, reflection_phase_rad=np.pi/2)]},
        multipath_start_ms=10)
    count = int(cfg.signal.fs_rf_hz*.01)
    np.testing.assert_allclose(clean[:, :count], echo[:, :count], atol=1e-20)
    assert np.linalg.norm(clean[:, 2*count:]-echo[:, 2*count:]) > 0


def test_sdcp_pvt_geometry_and_unknown_phase_constants_cancel():
    from audit_sdcp_positioning import observables, BASELINE, WAVELENGTH
    from run_gps_spp import receiver_reference
    satellites = satellite_setup()
    starts = np.arange(20,100)*66000
    times = starts/6.6e6
    states = np.array([[s['position_m']+s['velocity_mps']*t for s in satellites] for t in times])
    pair=[]
    for channel in (0,1):
        ranges=np.linalg.norm(states-receiver_reference()[0]-channel*BASELINE,axis=2)+75000
        histories=[]
        for k in range(8):
            histories.append({'acquired':True, 'tracking_status':'tracking', 'epoch_start_samples':starts,
                'carrier_nco_phase_rad':-2*np.pi*(ranges[:,k]+channel*(k+1)*17.13)/WAVELENGTH,
                'carrier_nco_frequency_hz':np.zeros(len(times))})
        pair.append({'time_s':times,'pseudoranges_m':ranges,'fs_dfe_hz':6.6e6,'satellite_tracking':histories})
    _,_,_,geometry,residual=observables(pair,satellites)
    assert np.max(abs(residual)) < 1e-6
    np.testing.assert_allclose(geometry[:,8:],states-BASELINE)
