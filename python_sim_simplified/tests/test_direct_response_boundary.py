"""Direct source coefficients are applied once before the unchanged receiver."""

import numpy as np

from core.config import CORNER_NAMES
from core.gnss import generate_scenario
from core.pipeline import run_end_to_end
from core.presets import default_config
from experiments.interference.direct_response_boundary import (
    DirectResponses,
    SourceResponse,
    generate_direct_scenario,
    build_responses,
)


ZERO = {name: 0.0 for name in CORNER_NAMES}


def response(gain=None, phase=None):
    return SourceResponse({**ZERO, **(gain or {})}, {**ZERO, **(phase or {})})


def test_unity_response_matches_ideal_scene_and_truth():
    cfg = default_config()
    cfg.signal.duration_ms = 2
    original, original_truth = generate_scenario(cfg)
    scene, truth = generate_direct_scenario(cfg, DirectResponses(
        SourceResponse(), SourceResponse(),
        (SourceResponse(), SourceResponse()),
    ))
    np.testing.assert_array_equal(scene, original)
    for key in ('desired_component', 'desired_steering_true',
                'desired_steering_assumed', 'interferer_components'):
        if key == 'interferer_components':
            for actual, expected in zip(truth[key], original_truth[key]):
                np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_array_equal(truth[key], original_truth[key])


def test_independent_source_coefficients_and_assumed_target_only():
    cfg = default_config()
    cfg.signal.duration_ms = 2
    cfg.phone.active_antennas = ('BR', 'TL', 'TR')
    original, base = generate_scenario(cfg)
    responses = DirectResponses(
        response({'TL': 6.020599913279624}, {'BR': 90}),
        response(phase={'TR': 180}),
        (response({'BR': -6.020599913279624}),
         response(phase={'TL': -90})),
    )
    scene, truth = generate_direct_scenario(cfg, responses)
    desired = np.array([1j, 2, 1])
    assumed = np.array([1, 1, -1])
    cw = np.array([0.5, 1, 1])
    wideband = np.array([1, -1j, 1])
    np.testing.assert_allclose(truth['desired_component'],
                               base['desired_component'] * desired[:, None])
    np.testing.assert_allclose(truth['desired_steering_true'],
                               base['desired_steering_true'] * desired)
    np.testing.assert_allclose(truth['desired_steering_assumed'],
                               base['desired_steering_assumed'] * assumed)
    np.testing.assert_allclose(truth['interferer_components'][0],
                               base['interferer_components'][0] * cw[:, None])
    np.testing.assert_allclose(truth['interferer_components'][1],
                               base['interferer_components'][1] * wideband[:, None])
    np.testing.assert_allclose(scene, truth['desired_component'] +
                               sum(truth['interferer_components']))
    np.testing.assert_array_equal(truth['channel_delay_true_ns'],
                                  base['channel_delay_true_ns'])


def test_two_and_four_antenna_orders_follow_names():
    responses = DirectResponses(
        response({'TL': 6.020599913279624}),
        SourceResponse(), (SourceResponse(), SourceResponse()),
    )
    for antennas in (('BR', 'TL'), ('BR', 'BL', 'TR', 'TL')):
        cfg = default_config()
        cfg.signal.duration_ms = 1
        cfg.phone.active_antennas = antennas
        _, base = generate_scenario(cfg)
        _, truth = generate_direct_scenario(cfg, responses)
        ratio = truth['desired_steering_true'] / base['desired_steering_true']
        np.testing.assert_allclose(ratio,
            [2 if name == 'TL' else 1 for name in antennas])


def test_diagnostics_scale_once_but_hardware_stage_does_not():
    cfg = default_config()
    cfg.signal.duration_ms = 1
    _, base = generate_scenario(cfg)
    responses = DirectResponses(
        response({'TR': 6.020599913279624}),
        SourceResponse(), (SourceResponse(), SourceResponse()),
    )
    _, truth = generate_direct_scenario(cfg, responses)
    actual = truth['antenna_model_diagnostics']['desired_true']
    baseline = base['antenna_model_diagnostics']['desired_true']
    for key in ('pattern_polarization_voltage', 'open_circuit_response',
                'coupled_response', 'effective_manifold'):
        np.testing.assert_allclose(actual[key][1], 2 * baseline[key][1])
    np.testing.assert_array_equal(actual['channel_response'], baseline['channel_response'])
    np.testing.assert_allclose(truth['desired_open_circuit_response_true'][1],
                               2 * base['desired_open_circuit_response_true'][1])


def test_true_delay_changes_samples_while_estimated_delay_changes_steering_only():
    cfg = default_config()
    cfg.signal.duration_ms = 2
    responses = DirectResponses(SourceResponse(), SourceResponse(),
                                (SourceResponse(), SourceResponse()))
    baseline, base_truth = generate_direct_scenario(cfg, responses)
    cfg.phone.estimated_channel_delay_ns['TR'] = 100
    estimated, estimated_truth = generate_direct_scenario(cfg, responses)
    np.testing.assert_array_equal(estimated, baseline)
    assert not np.allclose(estimated_truth['desired_steering_assumed'],
                           base_truth['desired_steering_assumed'])
    cfg.phone.channel_delay_ns['TR'] = 1000
    actual, actual_truth = generate_direct_scenario(cfg, responses)
    expected = np.zeros_like(base_truth['desired_component'][1])
    expected[33:] = base_truth['desired_component'][1, :-33] * np.exp(
        -2j * np.pi * (cfg.signal.carrier_hz - cfg.signal.if_hz) * (1000 * 1e-9)
    )
    np.testing.assert_allclose(actual_truth['desired_component'][1], expected,
                               rtol=1e-12, atol=1e-20)
    assert not np.array_equal(actual[1], baseline[1])


def test_source_voltage_applies_once_with_nonzero_hardware_delay():
    cfg = default_config()
    cfg.signal.duration_ms = 2
    _, undelayed = generate_scenario(cfg)
    cfg.phone.channel_delay_ns['TR'] = 1000
    cfg.phone.estimated_channel_delay_ns['TR'] = 50
    base_scene, base = generate_scenario(cfg)
    responses = DirectResponses(
        response(phase={'TR': 90}), SourceResponse(),
        (SourceResponse(), SourceResponse()),
    )
    scene, truth = generate_direct_scenario(cfg, responses)
    expected = np.zeros_like(undelayed['desired_component'][1])
    expected[33:] = 1j * undelayed['desired_component'][1, :-33] * np.exp(
        -2j * np.pi * (cfg.signal.carrier_hz - cfg.signal.if_hz) * (1000 * 1e-9)
    )
    np.testing.assert_allclose(truth['desired_component'][1],
                               expected,
                               rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(truth['desired_steering_true'][1],
                               1j * base['desired_steering_true'][1])
    np.testing.assert_allclose(truth['desired_steering_assumed'],
                               base['desired_steering_assumed'])
    np.testing.assert_allclose(scene[1], base_scene[1] +
                               (1j - 1) * base['desired_component'][1])


def test_paired_assumptions_share_rf_dfe_and_blind_null_but_change_known_weights():
    cfg = default_config()
    cfg.signal.duration_ms = cfg.tracking.max_epochs = 8
    cfg.phone.active_antennas = ('TL', 'TR')
    known, mismatch = build_responses(90, perturbed_antenna='TR')
    known_scene = generate_direct_scenario(cfg, known)
    mismatch_scene = generate_direct_scenario(cfg, mismatch)
    np.testing.assert_array_equal(known_scene[0], mismatch_scene[0])
    np.testing.assert_array_equal(known_scene[1]['desired_steering_true'],
                                  mismatch_scene[1]['desired_steering_true'])
    actual = run_end_to_end(cfg, scene=known_scene)
    paired = run_end_to_end(cfg, scene=mismatch_scene)
    default = run_end_to_end(cfg)
    zero, _ = build_responses(0, perturbed_antenna='TR')
    injected_zero = run_end_to_end(cfg, scene=generate_direct_scenario(cfg, zero))
    for key in ('scene_input_v', 'adc_codes', 'dfe_channels'):
        np.testing.assert_array_equal(default[key], injected_zero[key])
    for mode in default['branches']:
        np.testing.assert_array_equal(default['branches'][mode]['combined'],
                                      injected_zero['branches'][mode]['combined'])
    for key in ('scene_input_v', 'adc_codes', 'dfe_channels'):
        np.testing.assert_array_equal(actual[key], paired[key])
    assert not np.array_equal(actual['scene_input_v'], default['scene_input_v'])
    for mode in ('single', 'blind_null'):
        np.testing.assert_array_equal(actual['branches'][mode]['combined'],
                                      paired['branches'][mode]['combined'])
        np.testing.assert_array_equal(actual['branches'][mode]['combining']['weights'],
                                      paired['branches'][mode]['combining']['weights'])
    assert not np.allclose(actual['branches']['egc']['combining']['weights'],
                           paired['branches']['egc']['combining']['weights'])


def test_replaced_source_and_perturbed_corner_constants_drive_responses(monkeypatch):
    from experiments.interference import direct_response_boundary as study

    monkeypatch.setattr(study, 'DESIRED_GAIN_DB', {**ZERO, 'TL': 3.0})
    monkeypatch.setattr(study, 'DESIRED_PHASE_DEG', {**ZERO, 'BL': 7.0})
    monkeypatch.setattr(study, 'CW_GAIN_DB', {**ZERO, 'TR': -4.0})
    monkeypatch.setattr(study, 'CW_PHASE_DEG', {**ZERO, 'TL': 11.0})
    monkeypatch.setattr(study, 'WIDEBAND_GAIN_DB', {**ZERO, 'BL': 5.0})
    monkeypatch.setattr(study, 'WIDEBAND_PHASE_DEG', {**ZERO, 'BR': 13.0})
    monkeypatch.setattr(study, 'PERTURBED_ANTENNA', 'BL')
    known, mismatch = study.build_responses(10, perturbed_antenna=study.PERTURBED_ANTENNA)
    assert known.desired_true.gain_db['TL'] == 3.0
    assert known.desired_true.phase_deg['BL'] == 17.0
    assert known.desired_assumed.phase_deg['BL'] == 17.0
    assert mismatch.desired_assumed.phase_deg['BL'] == 7.0
    assert known.interferers_true[0].gain_db['TR'] == -4.0
    assert known.interferers_true[0].phase_deg['TL'] == 11.0
    assert known.interferers_true[1].gain_db['BL'] == 5.0
    assert known.interferers_true[1].phase_deg['BR'] == 13.0


def test_short_sweep_writes_audited_summary_manifest_and_three_plots(tmp_path, monkeypatch):
    import json
    import matplotlib
    from experiments.interference import direct_response_boundary as study

    matplotlib.use('Agg')
    from matplotlib.figure import Figure

    saved_xdata = []
    savefig = Figure.savefig

    def capture_axes_and_save(self, *args, **kwargs):
        saved_xdata.extend(np.asarray(line.get_xdata())
                           for axis in self.axes for line in axis.lines
                           if line.get_label() != 'acquisition threshold')
        return savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capture_axes_and_save)
    monkeypatch.setattr(study, 'SHOW_PLOTS', False)
    monkeypatch.setattr(study, 'OUTPUT', tmp_path)
    monkeypatch.setattr(study, 'SEEDS', (42,))
    monkeypatch.setattr(study, 'DURATION_MS', 20)
    monkeypatch.setattr(study, 'PHASE_ERRORS_DEG', (10, 30))
    monkeypatch.setattr(study, 'ANTENNAS', ('TL', 'BL'))
    monkeypatch.setattr(study, 'PERTURBED_ANTENNA', 'BL')
    monkeypatch.setattr(study, 'DESIRED_GAIN_DB', {**ZERO, 'TL': 3.0})
    monkeypatch.setattr(study, 'DESIRED_PHASE_DEG', {**ZERO, 'BL': 7.0})
    monkeypatch.setattr(study, 'POWER_TRIM_MS', 2)
    rows = study.main()['rows']
    assert len(rows) == 16
    assert {row['case'] for row in rows} == {'known_response', 'unmodeled_phase'}
    assert {row['phase_error_deg'] for row in rows} == {10, 30}
    manifest = json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['receiver_config_by_seed']['42']['signal']['power_dbm'] == -128.5
    assert manifest['case_responses_by_phase']['10']['known_response']['desired_true']['phase_deg']['BL'] == 17.0
    assert manifest['case_responses_by_phase']['30']['unmodeled_phase']['desired_assumed']['phase_deg']['BL'] == 7.0
    assert manifest['case_responses_by_phase']['30']['known_response']['desired_true']['gain_db']['TL'] == 3.0
    assert (tmp_path / 'summary.csv').exists()
    assert (tmp_path / 'summary.json').exists()
    paths = list(tmp_path.glob('*.png'))
    assert len(paths) == 3 and all(path.stat().st_size > 1000 for path in paths)
    assert saved_xdata and all(np.array_equal(x, [10, 30]) for x in saved_xdata)
