import numpy as np
from signal_recovery import (N, FS, C, codes, sky, synthesize, observe,
                             array_weights, position_solution, steering, refine_source)
from presets import mate70_phase_centers


def test_iq_delay_to_position_without_position_in_receiver():
    spectrum = codes(8); directions = sky(91001, 8)
    antennas = np.array(list(mate70_phase_centers().values()))
    location = np.array([43., -27., 2.])
    x, satellites, truth = synthesize(spectrum, directions, location, location,
        np.array([1.,0,0]), antennas, 0., np.ones(4), np.zeros((4,N)),
        45., .02, 10., 0)
    desired = steering(directions, antennas, 0.)
    weights, valid, _ = array_weights(x, desired, 'beam')
    rho, _, _, _ = observe(x, spectrum, weights, np.round(truth['ranges']/10000)*10000)
    p, _, _ = position_solution(satellites, rho, valid)
    # Multiple-access C/A interference is present even without thermal noise.
    assert np.linalg.norm(p-location) < 8.


def test_fractional_delay_subsample_accuracy():
    spectrum = codes(1)
    shift = 531.27123
    samples = np.fft.ifft(spectrum*np.exp(-2j*np.pi*np.fft.fftfreq(N)*shift), axis=1)
    _, delay, _, _ = observe(samples, spectrum, np.ones((1,1)), np.array([shift*C/FS]))
    assert abs(delay[0]-shift) < 1e-4


def test_estimated_null_removes_common_source_and_rejects_same_direction():
    rng = np.random.default_rng(8)
    directions = sky(91001, 8)
    antennas = np.array(list(mate70_phase_centers().values()))
    desired = steering(directions, antennas, 0.)
    v = desired[:,0]
    x = 100*v[:,None]*(rng.normal(size=N)+1j*rng.normal(size=N))
    x += rng.normal(size=x.shape)+1j*rng.normal(size=x.shape)
    weights, usable, _ = array_weights(x, desired, 'project')
    assert not usable[0]
    assert np.max(abs(weights.conj().T @ v)) < .01


def test_not_enough_observations_is_unavailable():
    p, _, _ = position_solution(np.ones((3,3)), np.ones(3), np.ones(3,bool))
    assert np.isnan(p).all()


def test_refinement_reestimates_from_iq_and_preserves_position():
    spectrum=codes(8);directions=sky(91002,8)
    antennas=np.array(list(mate70_phase_centers().values()))
    source=np.array([1.,0,0]);location=np.array([17.,23.,0.])
    rng=np.random.default_rng(5)
    noise=(rng.normal(size=(4,N))+1j*rng.normal(size=(4,N)))/np.sqrt(2)
    x,satellites,_=synthesize(spectrum,directions,location,location+[150.,0,0],
        source,antennas,0.,np.ones(4),noise,45.,.02,10.,8)
    desired=steering(directions,antennas,0.)
    coarse=np.round((21e6+np.arange(8)*17000+75000)/10000)*10000
    w,_,_=array_weights(x,desired,'project')
    weights,valid=refine_source(x,spectrum,desired,coarse,w,2,.05)
    rho,_,quality,_=observe(x,spectrum,weights,coarse)
    p,_,_=position_solution(satellites,rho,valid & (quality>=20.))
    assert np.linalg.norm(p[:2]-location[:2]) < 20.


def test_local_tracking_uses_observations_and_reacquires_large_jump():
    spectrum=codes(1)
    def sample(shift):
        return np.fft.ifft(spectrum*np.exp(-2j*np.pi*np.fft.fftfreq(N)*shift),axis=1)
    coarse=np.array([531*C/FS]);weights=np.ones((1,1))
    _,first,_,_=observe(sample(531.2),spectrum,weights,coarse)
    _,second,_,reacq=observe(sample(531.3),spectrum,weights,coarse,first)
    assert not reacq[0]
    assert abs(second[0]-531.3)<1e-4
    _,third,_,reacq=observe(sample(581.3),spectrum,weights,coarse,second)
    assert reacq[0]
    assert abs(third[0]-581.3)<1e-4


def test_future_snapshots_do_not_change_earlier_positions():
    from audit_recovery_fifty import Case, run_case
    _,short,_=run_case(Case(method='refined',duration=12),91004,True)
    _,long,_=run_case(Case(method='refined',duration=15),91004,True)
    for method in ('single','gated','refined'):
        np.testing.assert_array_equal(short[method+'_position'],long[method+'_position'][:13])
