"""Short recordings must report unavailable positioning honestly."""

import importlib

import pytest


@pytest.mark.parametrize('module,setting,value', [
    ('gps_spp', 'DURATION_MS', 200),
    ('gps_l1_l5', 'DURATION_MS', 200),
    ('recorded_iq', 'DURATION_S', .7),
])
def test_unsupported_duration_is_explained_before_output_reset(
        monkeypatch, module, setting, value):
    experiment = importlib.import_module('experiments.' + module)
    monkeypatch.setattr(experiment, setting, value)

    def no_reset(*args):
        pytest.fail('Existing results must not be removed for an unevaluable duration')

    monkeypatch.setattr(experiment, 'reset_output', no_reset)
    with pytest.raises(ValueError, match='needs|need|evaluated'):
        experiment.main()


def test_recorded_positioning_without_epochs_is_not_evaluated():
    from core.recorded_receiver import position_channels

    result = position_channels({}, 6.6e6, [])
    assert result['status'] == 'not_evaluated_no_observation_times'
    assert result['epochs'] == []
