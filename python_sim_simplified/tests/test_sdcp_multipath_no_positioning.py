import numpy as np

from experiments.sdcp_multipath import scores, confirmed, TIMES
from core.antenna import build_antenna_model
from core.config import MultipathPath
from core.gnss import generate_scenario, _multipath_scene
from core.presets import default_config
from experiments.sdcp_window_change import change_score, alarm


def test_causal_average_and_confirmation_preserve_large_offsets():
    r = np.zeros(len(TIMES))
    r[TIMES >= .6] = .19
    ends, score = scores(r)
    assert np.all(score[TIMES[ends] < .6] == 0)
    np.testing.assert_allclose(score[TIMES[ends] >= .8], .19)
    alarm = confirmed(score,.1)
    first_exceed = np.flatnonzero(score > .1)[0]
    assert np.flatnonzero(alarm)[0] == first_exceed+2
    changed = r.copy()
    changed[TIMES > 1.] += 10
    _, changed_score = scores(changed)
    np.testing.assert_array_equal(score[TIMES[ends] <= 1.],changed_score[TIMES[ends] <= 1.])


def test_common_direction_echo_has_identical_relative_channel_response():
    cfg = default_config()
    cfg.phone.active_antennas = ('TL','TR')
    cfg.signal.duration_ms = 25
    cfg.signal.multipath_paths = []
    cfg.interferers = []
    cfg.phone.channel_delay_ns = {k:0. for k in cfg.phone.channel_delay_ns}
    scene, truth = generate_scenario(cfg)
    t = np.arange(scene.shape[1])/cfg.signal.fs_rf_hz
    symbols = np.where(t < .020,1.,-1.)
    waveform = truth['desired_waveform']*symbols
    scene *= symbols
    echo = _multipath_scene(MultipathPath(cfg.signal.true_direction,
        relative_power_db=-6, excess_delay_chips=.3,reflection_phase_rad=np.pi/2),
        cfg,build_antenna_model(cfg.phone,cfg.signal.carrier_hz),waveform,t,np.zeros(2)).truth['component']
    np.testing.assert_allclose(echo[0]/scene[0],echo[1]/scene[1],atol=1e-12)
    assert not np.allclose(echo[1]/scene[1], (echo[1]/scene[1])[1000],atol=1e-5)


def test_change_detector_is_blind_to_static_bias_and_sees_transition():
    r=np.full(len(TIMES),.095)
    ends,s=change_score(r)
    np.testing.assert_allclose(s,0,atol=1e-15)
    assert not np.any(alarm(s,.001))
    r[TIMES>=.7]+=.01
    ends,s=change_score(r)
    assert np.all(s[TIMES[ends]<.7]==0)
    assert np.any(alarm(s,.001))
    np.testing.assert_allclose(s[TIMES[ends]>=1.1],0,atol=1e-15)
