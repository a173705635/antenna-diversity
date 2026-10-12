"""Fixed five-degree single-factor RF and saved-observation comparisons."""
from copy import deepcopy

import numpy as np
import pytest

from core.spoofing.recovery_geometry import FS
from core.spoofing import single_source, single_source_scene
from experiments.spoofing.single_source_history_study import generate_new_record
from experiments.spoofing.single_source_close_angle import (
    angle_pair_metrics, remaining_geometry, replay_modes, scene_geometry,
)
from test_single_source_code_blend import patch_geometry, record


def frozen(navigation, prefix):
    saved = {}
    for key, value in navigation.items():
        if key == 'diagnostics':
            for diagnostic, data in value.items():
                saved[f'{prefix}diagnostic_{diagnostic}'] = data.copy()
        else:
            saved[f'{prefix}{key}'] = np.nan if value is None else value.copy() if isinstance(value, np.ndarray) else value
    return saved


@pytest.mark.parametrize('scenario', ('weak','genuine_absent'))
def test_angle_only_changes_attacked_iq_and_preserves_rng_consumption(monkeypatch,
                                                                       scenario):
    plan = dict(seed=2026100301, duration_s=.25, satellites=2,
                startup_cn0_db_hz=45., post_0p3s_cn0_db_hz=30.,
                attack_start_s=.15, spoof_db=10., angle_deg=5.,
                fake_east_m=100.)
    original = np.random.default_rng
    generators = []

    def tracked(seed):
        generator = original(seed)
        generators.append(generator)
        return generator

    monkeypatch.setattr(single_source_scene.np.random, 'default_rng', tracked)
    five, _ = generate_new_record(plan, scenario)
    five_states = [deepcopy(generator.bit_generator.state) for generator in generators]
    generators.clear()
    plan['angle_deg'] = 60.
    sixty, _ = generate_new_record(plan, scenario)
    sixty_states = [deepcopy(generator.bit_generator.state) for generator in generators]
    assert len(five_states) == len(sixty_states) == 3
    assert five_states == sixty_states
    start = round(plan['attack_start_s'] * FS)
    np.testing.assert_array_equal(five['samples'][:, :start],
                                  sixty['samples'][:, :start])
    assert np.any(five['samples'][:, start:] != sixty['samples'][:, start:])


def test_five_degree_source_is_close_to_prn1_but_also_prn14_in_array():
    geometry = scene_geometry(5., 2026100301, 3)
    assert len(geometry['angles_deg']) == 24
    np.testing.assert_allclose(geometry['angles_deg'][0], 5., atol=1e-9)
    np.testing.assert_allclose(geometry['angles_deg'][13], 9.838453194120206,
                               atol=1e-9)
    assert int(np.argmin(geometry['angles_deg'])) == 0
    assert geometry['response_similarity'][0] > .99
    assert geometry['response_similarity'][13] > .98
    np.testing.assert_allclose(geometry['theoretical_ideal_power_retention'],
                               1.-geometry['response_similarity']**2)
    assert geometry['theoretical_ideal_power_retention'][0] < .05
    assert geometry['theoretical_ideal_power_retention'][13] < .05


def test_replay_uses_separate_support_and_exact_historical_references(monkeypatch):
    patch_geometry(monkeypatch)
    monkeypatch.setattr(single_source, 'phase_prediction',
                        lambda *args: np.array([10., 0., 0.]))
    observations = record(5)
    coherent = observations['support'].copy()
    coherent[3, 0] = np.nan
    originals = single_source.navigate(observations, diagnostics=True)
    short_input = {**observations, 'support': coherent}
    shorts = single_source.navigate(short_input, diagnostics=True, code_blend=0.,
                                    phase_leave_one_out=True)
    longs = single_source.navigate(short_input, diagnostics=True, code_blend=0.,
                                   phase_leave_one_out=True, phase_long_anchor=True)
    references = dict(original=(frozen(originals, 'original_'), 'original_'),
                      short=(frozen(shorts, 'on_'), 'on_'),
                      long=(frozen(longs, 'long_'), 'long_'))
    modes = replay_modes(observations, coherent, references)
    np.testing.assert_array_equal(modes['original']['selected'], originals['selected'])
    assert modes['short']['selected'][3, 0] == modes['long']['selected'][3, 0] == False
    assert modes['original']['selected'][3, 0]
    references['long'][0]['long_diagnostic_reason'][3] = 'wrong'
    with pytest.raises(AssertionError):
        replay_modes(observations, coherent, references)


def test_remaining_geometry_uses_actual_selected_stars_excluding_prn1():
    from core.gps_simulation import receiver_reference
    from core.spoofing.recovery_geometry import sky

    reference, rotation = receiver_reference()
    directions = sky(3104, 8)
    satellites = reference + (directions * 21e6) @ rotation
    observations = dict(ranges=np.ones((2, 8)), lock=np.ones((2, 8)),
                        cn0=np.full((2, 8), 30.), gain=np.ones((2, 8)),
                        satellites=np.broadcast_to(satellites, (2, 8, 3)).copy())
    selected = np.ones((2, 8), bool)
    selected[1, 1:5] = False
    geometry = remaining_geometry(observations, selected)
    np.testing.assert_array_equal(geometry['remaining_selected_count'], [7, 3])
    assert np.isfinite(geometry['remaining_condition'][0])
    assert np.isnan(geometry['remaining_condition'][1])
    assert geometry['prn1_quality'].all()
    assert geometry['quality'].shape == (2,8)


def test_angle_pair_uses_only_common_published_epochs():
    five=dict(observation_time=np.arange(3)*.01,
              available_time=np.array([.31,.32,.33]),
              original_position=np.array([[0.,0.,0.],[np.nan]*3,[0.,0.,0.]]),
              original_horizontal_error_m=np.array([1.,np.nan,50.]))
    sixty=dict(observation_time=five['observation_time'].copy(),
               available_time=five['available_time'].copy(),
               original_position=np.array([[0.,0.,0.],[0.,0.,0.],[np.nan]*3]),
               original_horizontal_error_m=np.array([2.,40.,np.nan]))
    for name in ('short','long'):
        for sample in (five,sixty):
            sample[f'{name}_position']=sample['original_position']
            sample[f'{name}_horizontal_error_m']=sample['original_horizontal_error_m']
    metrics=angle_pair_metrics(five,sixty,.3)['original']
    assert metrics['common_published']==1
    assert metrics['five_only']==metrics['sixty_only']==1
    assert metrics['five_degree']['max_error_m']==1.
    assert metrics['sixty_degree']['max_error_m']==2.
