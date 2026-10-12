"""Fixed long replay checks paired startup IQ and causal nav prefixes."""
import numpy as np
from zipfile import ZipFile

from core.spoofing.recovery_geometry import FS
from experiments.spoofing.single_source_history_study import generate_new_record
from experiments.spoofing.single_source_long_coherent import (
    comparative_metrics, navigation_prefix, source_snapshot,
)
from experiments.paths import RESULTS
from experiments.spoofing.single_source_suppression_study import digest


def test_same_seed_generator_has_exact_preattack_iq_across_three_scenes():
    plan = dict(seed=2026100301, duration_s=.25, satellites=2,
                startup_cn0_db_hz=45., post_0p3s_cn0_db_hz=30.,
                attack_start_s=.15, spoof_db=10., angle_deg=60.,
                fake_east_m=100.)
    starts = round(plan['attack_start_s'] * FS)
    records = [generate_new_record(plan, scene)[0] for scene in
               ('weak', 'no_attack_control', 'genuine_absent')]
    for record in records[1:]:
        np.testing.assert_array_equal(records[0]['samples'][:, :starts],
                                      record['samples'][:, :starts])
    assert np.any(records[0]['samples'][:, starts:] !=
                  records[1]['samples'][:, starts:])


def test_navigation_prefix_slices_epoch_arrays_only():
    observations = dict(time=np.arange(4), ranges=np.ones((4, 2)),
                        support=np.ones((4, 2)), startup_prns=np.arange(24),
                        calibration=np.arange(3))
    prefix = navigation_prefix(observations, 2)
    assert len(prefix['time']) == 2
    assert prefix['ranges'].shape == (2, 2)
    assert prefix['startup_prns'].shape == (24,)
    assert prefix['calibration'].shape == (3,)


def test_source_snapshot_contains_source_bytes_named_by_hash(tmp_path):
    name = 'requirements.txt'
    source = RESULTS.parent / name
    archive = tmp_path / 'source_snapshot.zip'
    source_snapshot(archive, {name: digest(source)})
    with ZipFile(archive) as saved:
        assert saved.namelist() == [name]
        assert saved.read(name) == source.read_bytes()


def test_common_publication_and_motion_error_use_same_epoch_truth():
    time = np.array([.2, .3, .4])
    observations = dict(time=time, available_time=time + .02,
                        ranges=np.ones((3, 4)), lock=np.ones((3, 4)),
                        cn0=np.full((3, 4), 30.), gain=np.ones((3, 4)))
    truth_east = 2 * time
    def navigation(bias, missing_first=False, missing_last=False):
        position = np.column_stack((truth_east + bias, np.zeros((3, 2))))
        if missing_first:
            position[0] = np.nan
        if missing_last:
            position[-1] = np.nan
        return dict(position=position, candidate_position=np.column_stack(
            (truth_east + bias, np.zeros((3, 2)))),
            initialization_time=.32 if missing_first else .22)
    modes = dict(original=navigation(1., missing_last=True),
                 **{'coherent_.1': navigation(1., True),
                    'coherent_0': navigation(2.)})
    details, relative, availability = comparative_metrics(
        observations, modes, dict(attack_start=.3, velocity_east=2.))
    assert all(row['three_way_common']['count'] == 1 for row in details.values())
    assert details['coherent_.1']['coherent_pair_common']['count'] == 2
    assert details['coherent_0']['coherent_pair_common']['count'] == 2
    assert details['coherent_0']['initialization_candidate_error_m'] == 2.
    np.testing.assert_allclose(relative['original'][:2], 0., atol=1e-12)
    assert np.isnan(relative['original'][-1])
    assert details['coherent_0']['relative_motion_error']['max_error_m'] < 1e-12
    assert availability['steady_opportunities'] == 0
