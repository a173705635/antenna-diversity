import numpy as np
import pytest

from sdcp import geometry_offset_m, modified_sdcp, observation_weights


def test_geometry_sign_and_units_match_array_phase_convention():
    g = geometry_offset_m([.0759, 0, 0], 30., 60.)
    assert g == pytest.approx(-.03286566407361945)


def test_constant_ambiguity_removed_without_erasing_later_event():
    time = np.arange(1000)/1000
    g, wavelength = -.03, .19
    length = np.full(time.size, g+7*wavelength)
    length[time >= .5] += .02
    r = modified_sdcp(time, length*2*np.pi/wavelength, wavelength, g)
    assert np.max(abs(r['residual_m'][time < .5])) < 1e-12
    assert np.allclose(r['residual_m'][time >= .5], .02)


def test_static_contamination_is_explicit_reference_blind_spot():
    t = np.arange(1000)/1000
    r = modified_sdcp(t, np.full(t.size, .5), .19, 0.)
    assert np.max(abs(r['residual_m'])) < 1e-12
    assert r['reference_offset_m'] > .01


def test_weights_are_symmetric_and_shared_by_both_antennas():
    r = np.array([0., .01, -.01, .1])
    w = observation_weights(r, .01)
    assert np.allclose(w['soft_pair'][:,0], w['soft_pair'][:,1])
    assert w['soft_pair'][1,0] == pytest.approx(.5)
    assert w['soft_pair'][1,0] == w['soft_pair'][2,0]
    assert w['soft_pair'][0,0] > w['soft_pair'][1,0] > w['soft_pair'][3,0]


def test_common_phase_cancels_in_dual_antenna_observable():
    t = np.arange(1000)/1000
    common = np.where(t >= .5, .5, 0.)
    delta = (.1+common)-(.8+common)
    r = modified_sdcp(t, delta, .19, -.02)
    assert np.max(abs(r['residual_m'])) < 1e-12


def test_time_varying_geometry_is_removed_before_reference_offset():
    t = np.arange(1000)/1000
    geometry = -.04 + .08*t
    phase = (geometry+.19)*2*np.pi/.19
    corrected = modified_sdcp(t, phase, .19, geometry)
    uncompensated = modified_sdcp(t, phase, .19, 0.)
    assert np.max(abs(corrected['residual_m'])) < 1e-12
    assert np.mean(abs(uncompensated['residual_m'][t > .6])) > .02


def test_echo_onset_is_applied_to_source_before_rf(monkeypatch):
    import audit_dual_tracking_phase as audit
    from config import Direction, MultipathPath
    from presets import default_config
    from gnss import generate_scenario
    cfg = default_config()
    cfg.phone.active_antennas = ('TL', 'TR')
    cfg.interferers = []
    cfg.signal.multipath_paths = []
    clean, _ = generate_scenario(cfg)
    class Inspected(Exception):
        pass
    def inspect(samples, *args):
        before = round(.05*cfg.signal.fs_rf_hz)
        after = round(.06*cfg.signal.fs_rf_hz)
        np.testing.assert_allclose(samples[0,:before], clean[0,:before], rtol=1e-12, atol=1e-20)
        assert np.mean(abs(samples[0,after:]-clean[0,after:])) > 1e-10
        raise Inspected
    monkeypatch.setattr(audit, 'process_rf', inspect)
    with pytest.raises(Inspected):
        audit.run_case(multipath_paths=[MultipathPath(Direction(110.,15.))],
                       multipath_start_ms=50)
