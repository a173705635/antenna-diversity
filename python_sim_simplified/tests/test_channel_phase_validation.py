"""DFE 整通道相位误差与准确逆补偿的回归检查。"""

import json

import numpy as np
import pytest

from experiments.interference import channel_phase_validation as experiment
from core.diversity import combine_known


def test_named_channel_rotation_and_inverse_affect_every_component():
    antennas = ('TR', 'TL')
    target = np.array([[1 + 2j, 2 - 1j], [3 + 0j, -1 + 1j]])
    cw = np.array([[2 + 0j, 0 + 1j], [1 - 1j, 3 + 0j]])
    thermal = np.array([[0.5 + 0j, 0.5j], [-0.5j, 0.25]])
    residual = np.array([[0.1j, -0.2], [0.3, 0.1j]])
    whole = target + cw + thermal + residual
    rotated = experiment.apply_channel_phase(whole, antennas, 'TL', 90)
    np.testing.assert_allclose(rotated[0], whole[0])
    np.testing.assert_allclose(rotated[1], 1j * whole[1])
    for component in (target, cw, thermal, residual):
        actual = experiment.apply_channel_phase(component, antennas, 'TL', 90)
        np.testing.assert_allclose(actual[1], 1j * component[1])
        np.testing.assert_allclose(actual[0], component[0])
    np.testing.assert_allclose(
        experiment.apply_channel_phase(rotated, antennas, 'TL', -90), whole
    )
    normal = experiment.apply_channel_phase(whole[[1, 0]], ('TL', 'TR'), 'TR', 90)
    np.testing.assert_allclose(normal[0], whole[1])
    np.testing.assert_allclose(normal[1], 1j * whole[0])


def test_egc_equal_noise_gain_and_antiphase_target_cancellation():
    steering = np.array([1 + 0j, 1 + 0j])
    egc_weights = steering / 2
    reference_target = np.array([1 + 0j, 1 + 0j])
    antiphase_target = np.array([1 + 0j, -1 + 0j])
    assert np.isclose(abs(np.vdot(egc_weights, antiphase_target)), 0)
    assert np.isclose(abs(np.vdot(egc_weights, reference_target)), 1)
    cfg = experiment.build_config(42, 20, ('TL', 'TR'), 'noise_only')
    assert cfg.phone.active_antennas == ('TL', 'TR')
    assert cfg.interferers == []
    assert cfg.diversity.modes == ('single', 'egc', 'mvdr')
    fixture = np.tile(np.array([[1 + 0j], [1 + 0j]]), (1, 128))
    baseline, baseline_info = combine_known(fixture, steering, 'egc', 0.01)
    np.testing.assert_allclose(baseline_info['weights'], [0.5, 0.5])
    assert np.isclose(np.vdot(baseline_info['weights'], baseline_info['weights']).real, 0.5)
    assert np.isclose(
        10 * np.log10(1 / np.vdot(baseline_info['weights'], baseline_info['weights']).real),
        3.010299956639812,
    )
    quarter = experiment.apply_channel_phase(fixture, ('TL', 'BR'), 'BR', 90)
    quarter_output, _ = combine_known(quarter, steering, 'egc', 0.01)
    antiphase = experiment.apply_channel_phase(fixture, ('TL', 'BR'), 'BR', 180)
    cancelled, _ = combine_known(antiphase, steering, 'egc', 0.01)
    np.testing.assert_allclose(baseline, 1)
    np.testing.assert_allclose(quarter_output, 0.5 + 0.5j)
    np.testing.assert_allclose(cancelled, 0, atol=1e-15)


def test_short_paired_cw_run_restores_mvdr_output_and_steering():
    rows, verification = experiment.run_sweep(
        seeds=(42,), duration_ms=20, phases=(90,), trim_ms=2
    )
    assert verification['passed']
    assert verification['white_noise_egc_max_abs_error_db'] < 0.5
    assert all(value < 1e-12 for value in (
        item['target_power_relative_error'] for item in verification['paired_comparisons']
    ))
    indexed = {(row['scene'], row['case'], row['mode']): row for row in rows}
    assert len(indexed) == 18
    for scene in ('noise_only', 'single_cw'):
        for mode in ('single', 'egc', 'mvdr'):
            baseline = indexed[scene, 'baseline', mode]
            calibrated = indexed[scene, 'calibrated', mode]
            assert calibrated['paired_key'] == baseline['paired_key']
            assert calibrated['calibrated_output_relative_error'] < 1e-12
            assert calibrated['dfe_compensation_relative_error'] < 1e-12
            assert calibrated['quadratic_power_relative_error'] < 1e-10
            assert calibrated['steering_unchanged']
            assert np.isclose(calibrated['target_power'], baseline['target_power'])
            assert np.isclose(calibrated['thermal_power'], baseline['thermal_power'])
        mvdr = indexed[scene, 'calibrated', 'mvdr']
        assert mvdr['calibrated_weight_relative_error'] < 1e-12
    cw = indexed['single_cw', 'calibrated', 'mvdr']
    assert cw['cw_suppression_db'] > 0
    assert np.isclose(cw['cw_power'], indexed['single_cw', 'baseline', 'mvdr']['cw_power'])
    assert indexed['noise_only', 'baseline', 'egc']['thermal_snr_gain_db'] > 2
    assert indexed['noise_only', 'uncalibrated', 'egc']['tracking_status']
    tampered = [row.copy() for row in rows]
    next(row for row in tampered if row['case'] == 'calibrated')['target_power'] *= 1.1
    assert not experiment.verify_rows(tampered)['checks']['paired_target_power']
    missing_cn0 = [row.copy() for row in rows]
    calibrated_single = next(row for row in missing_cn0 if row['scene'] == 'noise_only'
                             and row['case'] == 'calibrated' and row['mode'] == 'single')
    baseline_single = indexed['noise_only', 'baseline', 'single']
    assert baseline_single['tracking_cn0_late_median_db_hz'] is not None
    calibrated_single['tracking_cn0_late_median_db_hz'] = None
    calibrated_single['tracking_cn0_late_finite_count'] = 0
    missing_result = experiment.verify_rows(missing_cn0)
    assert not missing_result['passed']
    assert not missing_result['checks']['paired_tracking_cn0_availability']
    assert not missing_result['checks']['paired_tracking_cn0_finite_count']
    changed_success = [row.copy() for row in rows]
    next(row for row in changed_success if row['scene'] == 'noise_only'
         and row['case'] == 'calibrated' and row['mode'] == 'single')['tracking_success'] = False
    assert not experiment.verify_rows(changed_success)['checks']['paired_tracking_success']


def test_thermal_plot_uses_linear_snr_ratio_without_clamping(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    plt.switch_backend('Agg')
    figures = []
    actual_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = actual_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, 'subplots', capture_subplots)
    monkeypatch.setattr(experiment, 'SHOW_PLOTS', False)
    rows = []
    for scene in ('noise_only', 'single_cw'):
        for mode in ('single', 'egc', 'mvdr'):
            for case in ('baseline', 'uncalibrated', 'calibrated'):
                for phase in (0, 180):
                    gain_db = 3.010299956639812
                    if mode == 'egc' and case == 'uncalibrated' and phase == 180:
                        gain_db = -np.inf
                    rows.append({
                        'scene': scene, 'mode': mode, 'case': case,
                        'phase_error_deg': phase,
                        'thermal_snr_gain_db': gain_db,
                        'target_power_retention': 1.0,
                        'cw_suppression_db': 10.0 if scene == 'single_cw' else None,
                        'acquisition_metric': 2.0,
                        'acquisition_threshold': 1.8,
                        'tracking_cn0_late_median_db_hz': (
                            None if scene == 'single_cw' and mode == 'single' else 40.0
                        ),
                    })
    experiment.plot_summary(rows, tmp_path, (0, 180))
    egc_noise_axis = figures[0][1][1, 0]
    baseline_line, uncalibrated_line, calibrated_line = egc_noise_axis.lines
    np.testing.assert_allclose(baseline_line.get_ydata(), [2.0, 2.0])
    np.testing.assert_allclose(uncalibrated_line.get_ydata(), [2.0, 0.0])
    np.testing.assert_allclose(calibrated_line.get_ydata(), [2.0, 2.0])
    empty_tracking_axis = figures[2][1][3, 1]
    np.testing.assert_allclose(empty_tracking_axis.get_xlim(), [0, 180])
    assert any('no finite tracking estimate' in text.get_text()
               for text in empty_tracking_axis.texts)
    for figure, _ in figures:
        plt.close(figure)


def test_short_main_writes_manifest_verification_and_plots(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'OUTPUT', tmp_path / 'channel_phase_validation')
    monkeypatch.setattr(experiment, 'SEEDS', (42,))
    monkeypatch.setattr(experiment, 'DURATION_MS', 20)
    monkeypatch.setattr(experiment, 'POWER_TRIM_MS', 2)
    monkeypatch.setattr(experiment, 'PHASE_ERRORS_DEG', (0, 90))
    monkeypatch.setattr(experiment, 'SHOW_PLOTS', False)
    summary = experiment.main()
    output = experiment.OUTPUT
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    verification = json.loads((output / 'verification.json').read_text(encoding='utf-8'))
    assert len(summary['rows']) == 36
    assert manifest['injection_stage'] == 'DFE output'
    assert manifest['receiver_config_by_scene_and_seed']['single_cw']['42']['interferers']
    assert verification['passed']
    assert (output / 'summary.csv').is_file()
    assert (output / 'summary.json').is_file()
    assert len(list(output.glob('*.png'))) == 3
    failed = {'passed': False, 'checks': {'forced_failure': False}}
    monkeypatch.setattr(experiment, 'run_sweep', lambda *args: (summary['rows'], failed))
    with pytest.raises(AssertionError, match='verification failed'):
        experiment.main()
    assert not json.loads((output / 'verification.json').read_text(encoding='utf-8'))['passed']
