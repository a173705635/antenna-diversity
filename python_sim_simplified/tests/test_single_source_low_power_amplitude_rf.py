import numpy as np
import pytest

from experiments.single_source_low_power_amplitude_rf import (
    navigation_field_map, compare_mapped_navigation,
    selected_carrier_increment_error,
)


def test_navigation_mapping_checks_every_candidate_field_and_rejects_unknown():
    candidate = dict(observation_time=np.array([.25]),
                     original_position=np.array([[1., 2., 3.]]),
                     short_position=np.array([[4., 5., 6.]]),
                     short_support_count=np.array([7]),
                     phase_state_eligible=np.array([True]))
    receiver = dict(observation_time=candidate['observation_time'].copy(),
                    original_position=candidate['original_position'].copy(),
                    short_support_count=candidate['short_support_count'].copy())
    gated = dict(observation_time=candidate['observation_time'].copy(),
                 gated_state_position=candidate['short_position'].copy(),
                 eligible=candidate['phase_state_eligible'].copy())
    mapping = navigation_field_map(candidate, receiver, gated)
    assert set(mapping) == set(candidate)
    compare_mapped_navigation(candidate, receiver, gated)
    with pytest.raises(AssertionError):
        compare_mapped_navigation(dict(candidate, unknown=np.array([1.])),
                                  receiver, gated)
    with pytest.raises(AssertionError):
        compare_mapped_navigation(dict(candidate, short_position=np.zeros((1, 3))),
                                  receiver, gated)


def test_selected_carrier_increment_uses_same_arc_pairs_only():
    from core.gps_simulation import receiver_reference

    reference, rotation = receiver_reference()
    times = np.array([.2, .21, .22, .23])
    receiver = reference+np.column_stack((2*times, np.zeros((4, 2))))@rotation
    satellite = reference+np.array([20_000_000., 0., 0.])
    geometry = np.linalg.norm(satellite-receiver, axis=1)
    tracking = dict(time=times, satellites=np.tile(satellite, (4, 1, 1)),
                    carrier=(geometry+10.)[:, None],
                    arc=np.array([[0], [0], [1], [1]]))
    navigation = dict(original_selected=np.ones((4, 1), bool),
                      short_selected=np.array([[True], [True], [False], [True]]))
    result = selected_carrier_increment_error(tracking, navigation,
                                             dict(velocity_east=2.))
    assert result['original']['pairs'] == 2
    assert result['short']['pairs'] == 1
    assert abs(result['original']['per_prn_signed_increment_sum_m'][0]) < 1e-8


@pytest.mark.local_evidence
def test_real_stage39_and_new_navigation_layout_mapping():
    from experiments.single_source_early_truncation import saved_arrays
    from experiments.single_source_low_power_amplitude_rf import (
        STAGE39, STAGE43, preflight,
    )

    preflight()
    new = saved_arrays(STAGE43.parent/'ramp_transition'/'weak'/'off'/'navigation.npz')
    old = saved_arrays(STAGE39/'weak'/'aided_five_ms'/'navigation.npz')
    gated = saved_arrays(STAGE39/'weak'/'navigation_replay.npz')
    mapping = navigation_field_map(new, old, gated)
    assert len(mapping) == 74
    assert sum(name.startswith('original_') for name in mapping) == 28
    assert sum(name.startswith('short_') for name in mapping) == 41


@pytest.mark.local_evidence
def test_saved_stage39_navigation_replays_with_explicit_mapping(tmp_path):
    import json

    from core.serialization import serializable
    from experiments.single_source_early_truncation import saved_arrays
    from experiments.single_source_low_power_amplitude_rf import (
        STAGE39, check_historical_navigation_layout, check_truth_fields,
    )
    from experiments.single_source_startup_amplitude_rf import replay_navigation

    folder = STAGE39/'weak'
    tracking = saved_arrays(folder/'aided_five_ms'/'tracking.npz')
    trace = saved_arrays(folder/'aided_five_ms'/'trace.npz')
    coherent = {key[len('coherent_'):]: value for key, value in
                saved_arrays(folder/'aided_five_ms'/'support.npz').items()}
    case = json.loads((folder/'input'/'metadata.json').read_text(
        encoding='utf-8'))['case']
    scores, _ = replay_navigation(tracking, trace, coherent, case, tmp_path)
    candidate = saved_arrays(tmp_path/'navigation.npz')
    receiver = saved_arrays(folder/'aided_five_ms'/'navigation.npz')
    gated = saved_arrays(folder/'navigation_replay.npz')
    check_historical_navigation_layout(candidate, receiver, gated)
    compare_mapped_navigation(candidate, receiver, gated)
    check_truth_fields(candidate, receiver, gated, case)
    json.dumps(serializable(scores), allow_nan=False)
