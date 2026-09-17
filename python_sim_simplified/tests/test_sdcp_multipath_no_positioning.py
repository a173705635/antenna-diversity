import numpy as np

from experiments.sdcp_multipath import scores, confirmed, TIMES, cases
from core.antenna import build_antenna_model
from core.config import MultipathPath
from core.gnss import generate_scenario, _multipath_scene
from core.presets import default_config


def test_one_entry_runs_simulation_then_verification(monkeypatch, tmp_path):
    from experiments import sdcp_multipath

    stages = []

    def simulate():
        stages.append('simulation')
        return tmp_path

    def verify(output):
        assert output == tmp_path
        assert stages == ['simulation']
        stages.append('verification')

    monkeypatch.setattr(sdcp_multipath, 'run_simulation', simulate)
    monkeypatch.setattr(sdcp_multipath, 'verify_and_plot', verify)
    sdcp_multipath.main()
    assert stages == ['simulation', 'verification']


def test_current_multipath_scope_uses_both_channels_and_distinct_directions():
    scene_cases=cases()
    multipath=[c for c in scene_cases if c['kind']=='multipath']
    los=default_config().signal.true_direction
    assert len(scene_cases)==13 and len(multipath)==10
    assert all(c['channels']=='both' for c in multipath)
    assert all((c['azimuth_deg'],c['elevation_deg']) !=
               (los.azimuth_deg,los.elevation_deg) for c in multipath)
    settings=[tuple(v for k,v in c.items() if k!='name') for c in multipath]
    assert len(set(settings))==len(settings)


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
