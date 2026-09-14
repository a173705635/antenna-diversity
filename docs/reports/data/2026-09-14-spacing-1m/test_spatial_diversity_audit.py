import importlib.util
import numpy as np


def audit():
    assert importlib.util.find_spec('audit_spatial_diversity') is not None, 'paired audit missing'
    import audit_spatial_diversity
    return audit_spatial_diversity


def test_spatially_matched_attack_is_observationally_identical_to_los():
    module = audit()
    np.testing.assert_array_equal(module.model_observations(10, 12, 'los'),
                                  module.model_observations(10, 12, 'spatial_matched'))


def test_model_common_source_and_independent_antenna_noise():
    module = audit()
    from spatial_spoofing import spatial_scores
    x = module.model_observations(10, 12, 'all_spoof', noise_m=0.)
    assert x.shape == (12, 10, 4, 8)
    for count in (2, 3, 4):
        assert np.max(spatial_scores(x[:, :, :count], module.W)['global']) < 1e-10
    y = module.model_observations(10, 12, 'all_spoof')
    assert np.std(y - x) > 0


def test_paired_gain_interval_and_identical_predictions():
    module = audit()
    a = np.array([0, 1, 0, 1], bool)
    row = module.paired_gain(a, a)
    assert row['delta'] == 0
    assert row['gained'] == row['lost'] == 0


def test_rf_parameters_preserve_single_source_but_genuine_geometry_varies():
    assert importlib.util.find_spec('audit_spatial_diversity_rf') is not None, 'RF audit missing'
    from audit_spatial_diversity_rf import channel_parameters
    from run_gps_spp import satellite_setup
    sats = satellite_setup()
    a, fa = channel_parameters(sats, 0)
    b, fb = channel_parameters(sats, 2)
    genuine = np.array([v['range_at_zero_m'] for v in b])-np.array([v['range_at_zero_m'] for v in a])
    fake = np.array([v['range_at_zero_m'] for v in fb])-np.array([v['range_at_zero_m'] for v in fa])
    assert np.std(genuine) > .01
    assert np.std(fake) < 1e-8
    assert np.max(abs(np.array([s['range_at_zero_m'] for s in fa])-np.array([s['range_at_zero_m'] for s in a]))) > 20


def test_one_metre_geometry_changes_spatial_delays_without_changing_reference():
    from audit_spatial_diversity_rf import channel_parameters
    from run_gps_spp import satellite_setup
    from presets import mate70_phase_centers
    module = audit()
    geometry = dict(zip(module.NAMES, [(-.5, .5, 0.), (.5, .5, 0.),
                                      (-.5, -.5, 0.), (.5, -.5, 0.)]))
    # Explicit default geometry must leave the previous seeded model unchanged.
    np.testing.assert_array_equal(module.model_observations(14, 5, 'los'),
        module.model_observations(14, 5, 'los', positions_m=mate70_phase_centers()))
    x = module.model_observations(14, 5, 'all_spoof', noise_m=0, positions_m=geometry)
    from spatial_spoofing import spatial_scores
    assert np.max(spatial_scores(x, module.W)['global']) < 1e-10
    sats = satellite_setup()
    reference, fake_ref = channel_parameters(sats, 0, positions_m=geometry)
    original_ref, _ = channel_parameters(sats, 0)
    for actual_sat, original_sat in zip(reference, original_ref):
        for key in original_sat:
            np.testing.assert_array_equal(actual_sat[key], original_sat[key])
    right, fake_right = channel_parameters(sats, 1, positions_m=geometry)
    az = np.deg2rad([10, 55, 100, 145, 190, 235, 280, 325])
    el = np.deg2rad([25, 60, 35, 75, 20, 50, 40, 65])
    actual = np.array([r['range_at_zero_m']-l['range_at_zero_m']
                       for l, r in zip(reference, right)])
    np.testing.assert_allclose(actual, -np.cos(el)*np.sin(az), atol=1e-8)
    fake_actual = [r['range_at_zero_m']-l['range_at_zero_m']
                   for l, r in zip(fake_ref, fake_right)]
    np.testing.assert_allclose(fake_actual, -np.cos(np.deg2rad(10))*np.sin(np.deg2rad(100)), atol=1e-8)


def test_rf_gate_counts_blocked_danger_separately_from_false_rejection():
    assert importlib.util.find_spec('audit_spatial_diversity_rf') is not None, 'RF audit missing'
    from audit_spatial_diversity_rf import policy_metrics
    row = policy_metrics(np.array([5, 10, 100, 90, np.nan]), np.array([-1, 0, 1, 0, -1]))
    assert row['dangerous_available_epochs'] == 2
    assert row['dangerous_retained_epochs'] == 1
    assert row['safe_blocked_epochs'] == 1
    assert row['retained_epochs'] == 2


def test_audit_json_can_archive_config_paths(tmp_path):
    import json
    from dataclasses import asdict
    from presets import default_config
    from run_gps_spp import serializable
    module = audit()
    cfg = default_config()
    module.save_json(tmp_path / 'config.json', serializable(asdict(cfg)))
    assert json.loads((tmp_path / 'config.json').read_text())['plot']['output_dir'] == str(cfg.plot.output_dir)
