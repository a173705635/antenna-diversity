"""Physical and paired-design checks for the antenna ablation experiment."""

from copy import deepcopy

import numpy as np
import pytest

from core.antenna import antenna_response, build_antenna_model
from core.config import CORNER_NAMES, Direction
from core.pipeline import run_end_to_end
from core.presets import default_config, parametric_config
from core.rf import process_rf
from experiments.interference import antenna_ablation as study


def test_cases_change_one_antenna_feature_without_changing_rf_or_receiver():
    ideal = default_config()
    assert len(study.CASES) == 10
    for case in study.CASES:
        cfg = study.build_case(case, 'default', 42, 100, CORNER_NAMES)
        assert cfg.signal.power_dbm == ideal.signal.power_dbm == -128.5
        assert cfg.acquisition.threshold == ideal.acquisition.threshold == 2.5
        assert cfg.diversity.blind_null.detection_threshold_db == 6.0
        assert cfg.signal.fs_rf_hz == 33e6
        assert [source.power_dbm for source in cfg.interferers] == [-90.0, -90.0]
        if case != 'full_parametric':
            assert cfg.phone.antenna_model.true_state == cfg.phone.antenna_model.assumed_state
    full = study.build_case('full_parametric', 'blockage_active', 42, 100, CORNER_NAMES)
    known = study.build_case('full_known', 'blockage_active', 42, 100, CORNER_NAMES)
    assert full.phone.antenna_model.true_state != full.phone.antenna_model.assumed_state
    assert known.phone.antenna_model.true_state == known.phone.antenna_model.assumed_state
    for cfg, expected_equal in ((full, False), (known, True)):
        model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
        actual, _ = antenna_response(model, cfg.signal.true_direction,
                                     cfg.signal.true_polarization)
        assumed, _ = antenna_response(model, cfg.signal.assumed_direction,
                                      cfg.signal.assumed_polarization, estimated=True)
        assert bool(np.allclose(actual, assumed, atol=1e-12)) is expected_equal


def test_neutral_parametric_matches_ideal_and_blockage_is_contextual():
    ideal = study.build_case('ideal', 'default', 42, 10, CORNER_NAMES)
    neutral = study.build_case('pcv_only', 'default', 42, 10, CORNER_NAMES)
    neutral.phone.antenna_model.true_state.parametric_patterns = {
        name: deepcopy(study.neutral_pattern()) for name in CORNER_NAMES
    }
    neutral.phone.antenna_model.assumed_state = deepcopy(neutral.phone.antenna_model.true_state)
    for cfg in (ideal, neutral):
        model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
        response, _ = antenna_response(model, cfg.signal.true_direction,
                                       cfg.signal.true_polarization)
        if cfg is ideal:
            ideal_response = response
        else:
            np.testing.assert_allclose(response, ideal_response, atol=1e-12)
    blocked = study.build_case('blockage_amplitude_only', 'blockage_active',
                               42, 10, CORNER_NAMES)
    default = study.build_case('blockage_amplitude_only', 'default', 42, 10, CORNER_NAMES)
    for cfg, expected in ((default, False), (blocked, True)):
        model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
        _, diag = antenna_response(model, cfg.signal.true_direction,
                                   cfg.signal.true_polarization)
        assert bool(np.any(np.abs(diag['blockage_voltage']) < 0.999)) is expected


def test_each_single_factor_keeps_other_physical_factors_neutral():
    full = parametric_config().phone.antenna_model.true_state
    neutral = study.build_case('pcv_only', 'default', 42, 10, CORNER_NAMES)
    neutral_state = deepcopy(neutral.phone.antenna_model.true_state)
    neutral_state.parametric_patterns = {
        name: study.neutral_pattern() for name in CORNER_NAMES
    }
    cases = {
        'elevation_gain_only': 'elevation',
        'orientation_polarization_only': 'polarization',
        'azimuth_dependence_only': 'azimuth',
        'pcv_only': 'pcv',
        'coupling_only': 'coupling',
        'blockage_amplitude_only': 'blockage_amplitude',
        'blockage_phase_only': 'blockage_phase',
    }
    for case, chosen in cases.items():
        state = study.build_case(case, 'default', 42, 10,
                                 CORNER_NAMES).phone.antenna_model.true_state
        for name in CORNER_NAMES:
            actual = state.parametric_patterns[name]
            source = full.parametric_patterns[name]
            blank = neutral_state.parametric_patterns[name]
            for factor, fields in {
                'elevation': ('peak_gain_db', 'cosine_exponent', 'back_lobe_gain_db'),
                'polarization': ('nominal_polarization', 'cross_polar_discrimination_db',
                                 'cross_phase_deg'),
                'azimuth': ('azimuth_peak_deg', 'azimuth_attenuation_db'),
                'pcv': ('phase_variation_mm',),
            }.items():
                for field in fields:
                    assert getattr(actual, field) == getattr(source if factor == chosen
                                                            else blank, field)
        assert state.orientations == (full.orientations if chosen == 'polarization'
                                      else neutral_state.orientations)
        assert state.coupling_paths == (full.coupling_paths if chosen == 'coupling' else ())
        if chosen.startswith('blockage'):
            assert len(state.attenuation_regions) == len(full.attenuation_regions)
            for actual, source in zip(state.attenuation_regions, full.attenuation_regions):
                assert actual.attenuation_db == (source.attenuation_db if chosen ==
                                                 'blockage_amplitude' else 0.0)
                assert actual.phase_shift_deg == (source.phase_shift_deg if chosen ==
                                                  'blockage_phase' else 0.0)
        else:
            assert state.attenuation_regions == ()


def test_noise_for_tl_is_identical_in_two_three_and_four_channel_runs():
    cfg = default_config()
    n = 8192
    reference = None
    for names in (('TL', 'TR'), ('TL', 'TR', 'BL'), CORNER_NAMES):
        adc, _ = process_rf(np.zeros((len(names), n), complex), cfg.rf,
                            cfg.reference_impedance_ohm, 42,
                            fs_hz=cfg.signal.fs_rf_hz, if_hz=cfg.signal.if_hz,
                            channel_ids=names)
        if reference is None:
            reference = adc[0]
        else:
            np.testing.assert_array_equal(adc[0], reference)


@pytest.mark.parametrize('output_bits', [8, 0])
def test_short_run_budget_reconstructs_actual_dfe_and_quadratic_power(output_bits):
    cfg = study.build_case('ideal', 'default', 42, 12, ('TL', 'TR'))
    cfg.signal.fs_rf_hz = cfg.rf.noise_bandwidth_hz = 8.184e6
    cfg.signal.if_hz = 2.046e6
    cfg.dfe.decimation = 2
    cfg.dfe.output_bits = output_bits
    cfg.tracking.max_epochs = 12
    results = run_end_to_end(cfg)
    budgets = study.evaluate_budget(results, trim_ms=1)
    for mode, budget in budgets.items():
        assert budget['adc_reconstruction_max_abs_error'] == 0.0
        assert budget['combined_max_abs_error'] < 1e-9
        assert budget['quadratic_power_relative_error'] < 1e-10
        assert budget['reconstruction_max_abs_error'] < 1e-7
        assert budget['thermal_power'] > 0.0
        assert budget['target_power'] > 0.0
        assert budget['cw_power'] > 0.0
        assert budget['wideband_power'] > 0.0
        assert np.isfinite(budget['thermal_oracle_prompt_cn0_db_hz'])


def test_ablations_keep_single_channel_receiver_identical_under_antenna_subsets():
    reference = None
    for names in (('TL', 'TR'), ('TL', 'TR', 'BL'), CORNER_NAMES):
        cfg = study.build_case('ideal', 'default', 42, 12, names)
        cfg.signal.fs_rf_hz = cfg.rf.noise_bandwidth_hz = 8.184e6
        cfg.signal.if_hz = 2.046e6
        cfg.dfe.decimation = 2
        results = run_end_to_end(cfg)
        arrays = (results['scene_input_v'][0], results['adc_codes'][0],
                  results['dfe_channels'][0], results['branches']['single']['combined'])
        if reference is None:
            reference = tuple(array.copy() for array in arrays)
        else:
            for current, expected in zip(arrays, reference):
                np.testing.assert_array_equal(current, expected)


def test_short_entry_writes_calibration_summary_manifest_and_five_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'OUTPUT', tmp_path / 'antenna_ablation')
    monkeypatch.setattr(study, 'SEEDS', (42,))
    monkeypatch.setattr(study, 'DURATION_MS', 12)
    monkeypatch.setattr(study, 'POWER_TRIM_MS', 1)
    monkeypatch.setattr(study, 'CASES', ('ideal', 'pcv_only'))
    monkeypatch.setattr(study, 'CONTEXTS', {'default': Direction(30.0, 60.0)})
    monkeypatch.setattr(study, 'SHOW_PLOTS', False)
    summary = study.main()
    output = tmp_path / 'antenna_ablation'
    assert len(summary['rows']) == 8
    assert len(summary['noise_calibration']) == 1
    for name in ('summary.json', 'summary.csv', 'manifest.json',
                 'noise_calibration.json'):
        assert (output / name).is_file()
    assert len(list(output.glob('*.png'))) == 5
    manifest = (output / 'manifest.json').read_text(encoding='utf-8')
    assert 'Infinity (zero cross voltage)' in manifest


def test_noise_calibration_rejects_wrong_replayed_noise_power(monkeypatch):
    original = study.generate_rf_noise
    monkeypatch.setattr(study, 'generate_rf_noise',
                        lambda *args, **kwargs: 2.0 * original(*args, **kwargs))
    with pytest.raises(AssertionError):
        study.calibrate_noise((42,), 12)
