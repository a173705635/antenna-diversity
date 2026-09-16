"""Independent physical identities and local uncertainty checks."""
import numpy as np
import doa_hundred as d
from signal_recovery import steering


def test_coordinate_convention_matches_existing_iq_model():
    directions = np.array([d.unit(0, 0), d.unit(90, 0), d.unit(23, -40)])
    np.testing.assert_allclose(steering(directions, d.P, 0), np.exp(1j*d.K*d.P@directions.T))


def test_wrapped_candidates_preserve_truth_and_every_returned_phase():
    generator = np.random.default_rng(91777)
    for u in generator.normal(size=(20, 3)):
        u /= np.linalg.norm(u)
        b = d.P[1:]-d.P[0]
        phi = np.angle(np.exp(1j*d.K*b@u))
        candidates = d.aliases(phi)
        assert np.min(np.linalg.norm(candidates-u[:2], axis=1)) < 1e-9
        for q in candidates:
            for lifted in d.lift(q):
                np.testing.assert_allclose(np.exp(1j*d.K*b@lifted), np.exp(1j*phi), atol=1e-10)


def test_angular_wrap_and_planar_mirror_are_different_operations():
    assert d.angle(d.unit(359, 0), d.unit(1, 0)) < 2.000001
    assert abs(d.angle(d.unit(35, 42), d.unit(35, -42))-84) < 1e-10
    assert d.round062()['candidate_count'] == 2


def test_unobservable_two_channel_and_uncalibrated_cases():
    assert d.round063()['candidate_circle_phase_error'] < 1e-12
    row = d.round068()
    assert row['distinct_direction_deg'] > 100
    assert row['equal_observation_max_difference'] < 1e-12


def test_rotation_diversity_changes_rank_but_dual_band_keeps_mirror():
    yaw = d.round076()
    tilt = d.round077()
    assert yaw['stacked_geometry_rank'] == 2
    assert yaw['mirror_observation_difference'] == 0
    assert tilt['stacked_geometry_rank'] == 3
    assert tilt['best_error_deg'] < 1
    assert d.round079()['dual_full_sphere_count'] == 2


def test_pose_error_coverage_and_reflection_counterexample():
    coverage = d.round078()
    assert coverage['phase_only_95_coverage'] < .1
    assert .90 < coverage['pose_augmented_95_coverage'] < .99
    assert d.round080()['mirror_mix_rank1_identity_error'] < 1e-12
