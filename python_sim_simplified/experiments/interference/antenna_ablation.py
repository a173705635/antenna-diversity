"""Paired antenna-factor study with an independently replayed RF power budget.

Each case runs the unchanged receiver. Truth components are used only after
reception to audit powers under its frozen weights, never to design weights.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.signal import welch

from core.config import AntennaModelConfig, AntennaState, CORNER_NAMES, Direction, ParametricPattern
from core.dfe import process_dfe
from core.metrics import circular_chip_error
from core.pipeline import run_end_to_end
from core.presets import default_config, parametric_config
from core.provenance import source_hashes
from core.rf import BOLTZMANN_J_PER_K, filter_rf_input, generate_rf_noise
from core.serialization import serializable
from experiments.paths import INTERFERENCE_RESULTS, reset_output


OUTPUT = INTERFERENCE_RESULTS / 'antenna_ablation'
SEEDS = (42, 43, 44)
DURATION_MS = 100
ANTENNAS = CORNER_NAMES
SHOW_PLOTS = True
POWER_TRIM_MS = 10
CASES = (
    'ideal', 'elevation_gain_only', 'orientation_polarization_only',
    'azimuth_dependence_only', 'pcv_only', 'coupling_only',
    'blockage_amplitude_only', 'blockage_phase_only',
    'full_known', 'full_parametric',
)
CONTEXTS = {
    'default': Direction(30.0, 60.0),
    'blockage_active': Direction(225.0, 20.0),
}


def neutral_pattern() -> ParametricPattern:
    """Unity co-polar element, independent of elevation and azimuth."""
    return ParametricPattern(
        peak_gain_db=0.0, cosine_exponent=0.0, back_lobe_gain_db=0.0,
        cross_polar_discrimination_db=float('inf'),
        azimuth_peak_deg=0.0, azimuth_attenuation_db=0.0,
        phase_variation_mm=(0.0, 0.0, 0.0),
    )


def build_case(case_name, context, seed, duration_ms, antennas):
    """Construct one receiver configuration; only named antenna factors vary."""
    cfg = default_config()
    cfg.random_seed = seed
    cfg.signal.duration_ms = cfg.tracking.max_epochs = duration_ms
    cfg.signal.true_direction = deepcopy(CONTEXTS[context])
    cfg.signal.assumed_direction = deepcopy(CONTEXTS[context])
    cfg.signal.multipath_paths = []
    cfg.enable_multipath_research = False
    cfg.phone.active_antennas = antennas
    if case_name == 'ideal':
        return cfg

    full_state = parametric_config(antennas).phone.antenna_model.true_state
    if case_name in ('full_known', 'full_parametric'):
        cfg.phone.antenna_model = deepcopy(parametric_config(antennas).phone.antenna_model)
        if case_name == 'full_known':
            cfg.phone.antenna_model.assumed_state = deepcopy(
                cfg.phone.antenna_model.true_state
            )
        return cfg

    state = AntennaState(
        'parametric', {name: neutral_pattern() for name in CORNER_NAMES}
    )
    if case_name == 'elevation_gain_only':
        for name in CORNER_NAMES:
            source = full_state.parametric_patterns[name]
            pattern = state.parametric_patterns[name]
            pattern.peak_gain_db = source.peak_gain_db
            pattern.cosine_exponent = source.cosine_exponent
            pattern.back_lobe_gain_db = source.back_lobe_gain_db
    elif case_name == 'orientation_polarization_only':
        state.orientations = deepcopy(full_state.orientations)
        for name in CORNER_NAMES:
            source = full_state.parametric_patterns[name]
            pattern = state.parametric_patterns[name]
            pattern.nominal_polarization = deepcopy(source.nominal_polarization)
            pattern.cross_polar_discrimination_db = source.cross_polar_discrimination_db
            pattern.cross_phase_deg = source.cross_phase_deg
    elif case_name == 'azimuth_dependence_only':
        for name in CORNER_NAMES:
            source = full_state.parametric_patterns[name]
            pattern = state.parametric_patterns[name]
            pattern.azimuth_peak_deg = source.azimuth_peak_deg
            pattern.azimuth_attenuation_db = source.azimuth_attenuation_db
    elif case_name == 'pcv_only':
        for name in CORNER_NAMES:
            state.parametric_patterns[name].phase_variation_mm = (
                full_state.parametric_patterns[name].phase_variation_mm
            )
    elif case_name == 'coupling_only':
        state.coupling_paths = deepcopy(full_state.coupling_paths)
    elif case_name == 'blockage_amplitude_only':
        state.attenuation_regions = tuple(
            replace(region, phase_shift_deg=0.0)
            for region in full_state.attenuation_regions
        )
    elif case_name == 'blockage_phase_only':
        state.attenuation_regions = tuple(
            replace(region, attenuation_db=0.0)
            for region in full_state.attenuation_regions
        )
    else:
        raise KeyError(case_name)
    cfg.phone.antenna_model = AntennaModelConfig(
        case_name, state, deepcopy(state)
    )
    return cfg


def _power(samples, interval):
    return float(np.mean(np.abs(samples[interval]) ** 2))


def thermal_oracle_prompt_cn0(target, thermal, truth, fs_out_hz, trim_samples):
    """Post-hoc true-code/Doppler 1 ms C/N0 against thermal noise alone.

    This is not a receiver observable. Its level includes RF/DFE passband and
    sampled-code mismatch losses, so it need not equal input C/N0.
    """
    samples_per_ms = round(fs_out_hz * 0.001)
    first = ((trim_samples + samples_per_ms - 1) // samples_per_ms) * samples_per_ms
    last = ((len(target) - trim_samples) // samples_per_ms) * samples_per_ms
    indices = np.arange(first, last)
    chip_phase = (indices * truth['code_rate_hz'] / fs_out_hz -
                  truth['code_delay_chips']) % len(truth['code'])
    replica = truth['code'][np.floor(chip_phase).astype(int)] * np.exp(
        -2j * np.pi * truth['doppler_hz'] * indices / fs_out_hz
    )
    target_prompts = (target[first:last] * replica).reshape(-1, samples_per_ms).mean(axis=1)
    noise_prompts = (thermal[first:last] * replica).reshape(-1, samples_per_ms).mean(axis=1)
    return _db_ratio(np.mean(np.abs(target_prompts) ** 2),
                     np.mean(np.abs(noise_prompts) ** 2) * 0.001)


def evaluate_budget(results, trim_ms=10):
    """Recompute linear component powers in actual DFE-code units.

    Cross terms between linear components and the joint nonlinear/quantization
    residual remain explicit; none is relabelled as thermal noise.
    """
    cfg = results['config']
    info = results['rf_info']
    truth = results['truth']
    fs_hz = cfg.signal.fs_rf_hz
    scale = results['dfe_info']['output_scale']
    lsb_v = info['adc_lsb_v']
    voltage_gain = (10.0 ** (info['channel_total_gain_db'] / 20.0) *
                    np.exp(1j * np.deg2rad(info['phase_error_deg'])))
    float_dfe_cfg = replace(cfg.dfe, output_bits=0)

    def linear_dfe(if_samples):
        if_samples *= voltage_gain[:, None] / lsb_v
        output, _ = process_dfe(if_samples, float_dfe_cfg, fs_hz, cfg.signal.if_hz)
        return output / scale

    components = {}
    for name, source in (
        ('target', truth['desired_component']),
        ('cw', truth['interferer_components'][0]),
        ('wideband', truth['interferer_components'][1]),
    ):
        filtered = filter_rf_input(
            source, info['filter_coefficients'], fs_hz, cfg.signal.if_hz
        )
        components[name] = linear_dfe(filtered)

    noise = generate_rf_noise(
        results['scene_input_v'].shape[1], info['noise_psd_w_per_hz'],
        cfg.reference_impedance_ohm, fs_hz, cfg.signal.if_hz,
        cfg.phone.active_antennas, cfg.random_seed, info['filter_coefficients'],
    )
    # Independent replay of the nonlinear ADC path checks the decomposition's
    # RF source and deterministic noise stream before quantization losses.
    filtered_scene = filter_rf_input(
        results['scene_input_v'], info['filter_coefficients'], fs_hz, cfg.signal.if_hz
    )
    before_limit = (filtered_scene + noise) * 10 ** (info['total_gain_db'] / 20.0)
    before_limit *= (10 ** (info['amplitude_error_db'] / 20.0) *
                     np.exp(1j * np.deg2rad(info['phase_error_deg'])))[:, None]
    limit_v = cfg.rf.adc_vpp / 2.0
    if cfg.rf.enable_soft_limiting:
        adc_input = limit_v * np.tanh(before_limit.real / limit_v) + 1j * (
            limit_v * np.tanh(before_limit.imag / limit_v)
        )
    else:
        adc_input = np.clip(before_limit.real, -limit_v, limit_v) + 1j * (
            np.clip(before_limit.imag, -limit_v, limit_v)
        )
    minimum = -(2 ** (cfg.rf.adc_bits - 1))
    maximum = 2 ** (cfg.rf.adc_bits - 1) - 1
    real_codes = np.clip(np.sign(adc_input.real / lsb_v) *
                         np.floor(np.abs(adc_input.real / lsb_v) + 0.5),
                         minimum, maximum)
    imag_codes = np.clip(np.sign(adc_input.imag / lsb_v) *
                         np.floor(np.abs(adc_input.imag / lsb_v) + 0.5),
                         minimum, maximum)
    adc_reconstruction_error = float(np.max(np.abs(
        real_codes + 1j * imag_codes - results['adc_codes']
    )))
    assert adc_reconstruction_error == 0.0
    del filtered_scene, before_limit, adc_input, real_codes, imag_codes
    components['thermal'] = linear_dfe(noise)
    fs_out_hz = results['dfe_info']['fs_out_hz']
    trim = round(trim_ms * 1e-3 * fs_out_hz)
    interval = slice(trim, results['dfe_channels'].shape[1] - trim)
    channels = results['dfe_channels'][:, interval]
    covariance = channels @ channels.conj().T / channels.shape[1]

    budgets = {}
    for mode, branch in results['branches'].items():
        weights = branch['combining']['weights']
        direct = weights.conj() @ channels
        actual = branch['combined'][interval]
        separated = {
            name: weights.conj() @ value[:, interval]
            for name, value in components.items()
        }
        oracle_target = weights.conj() @ components['target']
        oracle_thermal = weights.conj() @ components['thermal']
        linear_sum = sum(separated.values())
        residual = actual - linear_sum
        actual_power = _power(actual, slice(None))
        quadratic_power = float(np.vdot(weights, covariance @ weights).real)
        component_powers = {
            name + '_power': _power(value, slice(None))
            for name, value in separated.items()
        }
        linear_sum_power = _power(linear_sum, slice(None))
        true = truth['desired_steering_true']
        assumed = truth['desired_steering_assumed']
        budgets[mode] = {
            **component_powers,
            'actual_power': actual_power,
            'linear_sum_power': linear_sum_power,
            'component_cross_power': linear_sum_power - sum(component_powers.values()),
            'joint_residual_power': _power(residual, slice(None)),
            'actual_minus_linear_sum_power': actual_power - linear_sum_power,
            'linear_residual_cross_power': actual_power - linear_sum_power -
                _power(residual, slice(None)),
            'weight_norm_squared': float(np.vdot(weights, weights).real),
            'mean_input_target_antenna_gain_db': _db_ratio(
                np.vdot(true, true).real, len(true)
            ),
            'target_gain': float(abs(np.vdot(weights, true))),
            'assumed_target_gain': float(abs(np.vdot(weights, assumed))),
            'true_assumed_correlation': float(abs(np.vdot(true, assumed)) /
                np.sqrt(np.vdot(true, true).real * np.vdot(assumed, assumed).real)),
            'combined_max_abs_error': float(np.max(np.abs(actual - direct))),
            'quadratic_power_relative_error': abs(actual_power - quadratic_power) / actual_power,
            'reconstruction_max_abs_error': float(np.max(np.abs(
                actual - (linear_sum + residual)
            ))),
            'adc_reconstruction_max_abs_error': adc_reconstruction_error,
            'thermal_oracle_prompt_cn0_db_hz': thermal_oracle_prompt_cn0(
                oracle_target, oracle_thermal, truth, fs_out_hz, trim
            ),
        }
        assert budgets[mode]['combined_max_abs_error'] < 1e-8
        assert budgets[mode]['quadratic_power_relative_error'] < 1e-10
        assert budgets[mode]['reconstruction_max_abs_error'] < 1e-8
    return budgets


def _db_ratio(numerator, denominator):
    return float(10.0 * np.log10(numerator / denominator))


def run_case(cfg, case_name, context, trim_ms=10, *, scene=None):
    """Run one complete receiver, then audit its frozen combining weights."""
    results = run_end_to_end(cfg, scene=scene)
    budgets = evaluate_budget(results, trim_ms)
    single = budgets['single']
    rows = []
    for mode, branch in results['branches'].items():
        budget = budgets[mode]
        acquisition = branch['acquisition']
        tracking = branch['tracking']
        code_estimate = acquisition['code_phase_samples'] * (
            cfg.signal.code_rate_hz / results['dfe_info']['fs_out_hz']
        )
        code_error = float(circular_chip_error(
            code_estimate, cfg.signal.code_delay_chips
        ))
        doppler_error = float(acquisition['doppler_hz'] - cfg.signal.doppler_hz)
        if tracking['success']:
            cn0 = tracking['cn0_db_hz'][len(tracking['cn0_db_hz']) // 2:]
            finite = cn0[np.isfinite(cn0)]
        else:
            finite = np.array([])
        rows.append({
            'context': context, 'case': case_name, 'seed': cfg.random_seed,
            'mode': mode, **budget,
            'target_output_gain_db': _db_ratio(budget['target_power'], single['target_power']),
            'noise_output_gain_db': _db_ratio(budget['thermal_power'], single['thermal_power']),
            'thermal_snr_gain_db': _db_ratio(
                budget['target_power'] / budget['thermal_power'],
                single['target_power'] / single['thermal_power'],
            ),
            'cw_suppression_db': _db_ratio(single['cw_power'], budget['cw_power']),
            'wideband_suppression_db': _db_ratio(
                single['wideband_power'], budget['wideband_power']
            ),
            'acquisition_metric': float(acquisition['metric']),
            'acquired': bool(acquisition['success']),
            'acquired_code_error_chips': code_error,
            'acquired_doppler_error_hz': doppler_error,
            'correct_acquisition': bool(acquisition['success'] and
                                        abs(code_error) <= 0.5 and
                                        abs(doppler_error) <= 125.0),
            'tracking_success': bool(tracking['success']),
            'tracking_cn0_late_median_db_hz': float(np.median(finite)) if finite.size else None,
            'tracking_cn0_late_finite_count': int(finite.size),
        })
    return rows


def calibrate_noise(seeds=SEEDS, duration_ms=DURATION_MS):
    """No-interference ideal TL runs; Welch PSD is independent of RF metadata ENBW."""
    rows = []
    for seed in seeds:
        cfg = build_case('ideal', 'default', seed, duration_ms, ('TL',))
        cfg.interferers = []
        cfg.diversity.modes = ('single',)
        results = run_end_to_end(cfg)
        info = results['rf_info']
        stage_gains = 10 ** (np.array([stage.gain_db for stage in cfg.rf.stages]) / 10.0)
        stage_noise_factors = 10 ** (
            np.array([stage.noise_figure_db for stage in cfg.rf.stages]) / 10.0
        )
        friis_factor = stage_noise_factors[0]
        preceding_gain = stage_gains[0]
        for index in range(1, len(stage_gains)):
            friis_factor += (stage_noise_factors[index] - 1.0) / preceding_gain
            preceding_gain *= stage_gains[index]
        theoretical_psd = BOLTZMANN_J_PER_K * cfg.rf.temperature_k * friis_factor
        filter_enbw_hz = cfg.signal.fs_rf_hz * np.sum(info['filter_coefficients'] ** 2)
        assert np.isclose(theoretical_psd, info['noise_psd_w_per_hz'],
                          rtol=1e-12, atol=0.0)
        assert np.isclose(filter_enbw_hz, cfg.rf.noise_bandwidth_hz,
                          rtol=1e-10, atol=0.0)
        noise = generate_rf_noise(
            results['scene_input_v'].shape[1], info['noise_psd_w_per_hz'],
            cfg.reference_impedance_ohm, cfg.signal.fs_rf_hz, cfg.signal.if_hz,
            ('TL',), seed, info['filter_coefficients'],
        )[0]
        frequencies, spectrum = welch(
            noise, fs=cfg.signal.fs_rf_hz, nperseg=8192,
            return_onesided=False,
        )
        fs_hz = cfg.signal.fs_rf_hz
        offset = (frequencies - cfg.signal.if_hz + fs_hz / 2) % fs_hz - fs_hz / 2
        voltage_psd = float(np.mean(spectrum[np.abs(offset) < 1e6]))
        measured_noise_power_w = float(np.mean(np.abs(noise) ** 2) /
                                       cfg.reference_impedance_ohm)
        measured_psd_w_per_hz = voltage_psd / cfg.reference_impedance_ohm
        assert np.isclose(measured_psd_w_per_hz, theoretical_psd,
                          rtol=0.05, atol=0.0)
        assert np.isclose(measured_noise_power_w,
                          theoretical_psd * cfg.rf.noise_bandwidth_hz,
                          rtol=0.02, atol=0.0)
        signal_power_w = 10 ** ((cfg.signal.power_dbm - 30.0) / 10.0)
        tracking = results['branches']['single']['tracking']
        if tracking['success']:
            late_cn0 = tracking['cn0_db_hz'][len(tracking['cn0_db_hz']) // 2:]
            finite_cn0 = late_cn0[np.isfinite(late_cn0)]
        else:
            finite_cn0 = np.array([])
        rows.append({
            'seed': seed, 'context': 'default', 'case': 'ideal',
            'active_antennas': ('TL',), 'interferers': 0,
            'duration_ms': duration_ms,
            'theoretical_noise_psd_w_per_hz': theoretical_psd,
            'welch_noise_psd_w_per_hz': measured_psd_w_per_hz,
            'configured_enbw_hz': cfg.rf.noise_bandwidth_hz,
            'filter_enbw_hz': filter_enbw_hz,
            'empirical_enbw_hz': measured_noise_power_w /
                theoretical_psd,
            'theoretical_noise_power_w': theoretical_psd *
                cfg.rf.noise_bandwidth_hz,
            'measured_noise_power_w': measured_noise_power_w,
            'ideal_input_cn0_db_hz': _db_ratio(
                signal_power_w, theoretical_psd
            ),
            'acquisition_metric': float(results['branches']['single']['acquisition']['metric']),
            'tracking_cn0_late_median_db_hz': float(np.median(finite_cn0))
                if finite_cn0.size else None,
            'tracking_cn0_late_finite_count': int(finite_cn0.size),
        })
    return rows


def summarize(rows):
    """Report per-seed ideal-paired changes and group means without pooling contexts."""
    ideals = {(row['context'], row['seed'], row['mode']): row
              for row in rows if row['case'] == 'ideal'}
    paired_keys = ('thermal_snr_gain_db', 'cw_suppression_db',
                   'wideband_suppression_db', 'target_output_gain_db',
                   'noise_output_gain_db', 'acquisition_metric',
                   'thermal_oracle_prompt_cn0_db_hz',
                   'mean_input_target_antenna_gain_db')
    for row in rows:
        ideal = ideals[(row['context'], row['seed'], row['mode'])]
        for key in paired_keys:
            row['paired_ideal_delta_' + key] = row[key] - ideal[key]
    groups = []
    for context in CONTEXTS:
        for case in CASES:
            for mode in ('single', 'egc', 'mvdr', 'blind_null'):
                members = [row for row in rows if (row['context'], row['case'], row['mode'])
                           == (context, case, mode)]
                group = {'context': context, 'case': case, 'mode': mode,
                         'seeds': [row['seed'] for row in members],
                         'seed_count': len(members),
                         'acquired_count': sum(row['acquired'] for row in members),
                         'correct_acquisition_count': sum(
                             row['correct_acquisition'] for row in members),
                         'tracked_count': sum(row['tracking_success'] for row in members),
                         'correct_acquisition_rate': float(np.mean([
                             row['correct_acquisition'] for row in members
                         ]))}
                for key in paired_keys:
                    group[key + '_mean'] = float(np.mean([row[key] for row in members]))
                    changes = [row['paired_ideal_delta_' + key] for row in members]
                    group['paired_ideal_delta_' + key + '_mean'] = float(np.mean(changes))
                    group['paired_ideal_delta_' + key + '_min'] = float(np.min(changes))
                    group['paired_ideal_delta_' + key + '_max'] = float(np.max(changes))
                groups.append(group)
    return groups


def plot_summary(groups, output):
    """Five separate figures, each with paired context panels and readable labels."""
    import matplotlib.pyplot as plt

    specs = (
        ('thermal_snr_gain', (('thermal_snr_gain_db_mean',
                              'Thermal SNR gain vs SINGLE (dB)'),)),
        ('interference_suppression', (('cw_suppression_db_mean',
                                      'CW suppression vs SINGLE (dB)'),
                                     ('wideband_suppression_db_mean',
                                      'Wideband suppression vs SINGLE (dB)'))),
        ('target_noise_gain', (('target_output_gain_db_mean',
                               'Target output gain vs SINGLE (dB)'),
                              ('noise_output_gain_db_mean',
                               'Thermal output gain vs SINGLE (dB)'))),
        ('acquisition_metric', (('acquisition_metric_mean', 'Acquisition peak ratio'),)),
    )
    colors = {'single': '#555555', 'egc': '#1f77b4',
              'mvdr': '#d95f02', 'blind_null': '#2ca02c'}
    paths = []
    contexts = tuple(CONTEXTS)
    for filename, metrics in specs:
        fig, axes = plt.subplots(len(metrics), len(contexts),
                                 figsize=(9.5 * len(contexts), 5.5 * len(metrics)),
                                 constrained_layout=True, squeeze=False,
                                 sharey='row')
        for metric_index, (key, ylabel) in enumerate(metrics):
            for context_index, context in enumerate(contexts):
                axis = axes[metric_index, context_index]
                for mode, color in colors.items():
                    subset = {group['case']: group for group in groups
                              if group['context'] == context and group['mode'] == mode}
                    values = [subset[case][key] for case in CASES]
                    axis.plot(range(len(CASES)), values, marker='o', markersize=3,
                              color=color, label=mode)
                axis.set_title(context + (' (LOS outside blockage)' if context == 'default'
                                          else '') + ' — ' + ylabel)
                axis.set_xticks(range(len(CASES)), CASES, rotation=55, ha='right')
                axis.set_ylabel(ylabel)
                axis.grid(alpha=0.25)
                axis.legend(ncol=2, fontsize=8)
        path = output / f'{filename}.png'
        fig.savefig(path, dpi=150)
        paths.append(path)
        if not SHOW_PLOTS:
            plt.close(fig)

    fig, axes = plt.subplots(1, len(contexts),
                             figsize=(9.5 * len(contexts), 5.5),
                             constrained_layout=True, squeeze=False)
    for context_index, context in enumerate(contexts):
        axis = axes[0, context_index]
        counts = np.array([[
            next(group for group in groups if (group['context'], group['case'], group['mode'])
                 == (context, case, mode))['correct_acquisition_count']
            for case in CASES] for mode in colors])
        image = axis.imshow(counts / len(SEEDS), vmin=0, vmax=1,
                            cmap='YlGnBu', aspect='auto')
        for row in range(len(colors)):
            for column in range(len(CASES)):
                axis.text(column, row, f'{counts[row, column]}/{len(SEEDS)}',
                          ha='center', va='center', fontsize=8,
                          color='white' if counts[row, column] / len(SEEDS) > 0.55
                          else 'black')
        axis.set_yticks(range(len(colors)), tuple(colors))
        axis.set_xticks(range(len(CASES)), CASES, rotation=55, ha='right')
        axis.set_title(context + ' — correct acquisition count')
    fig.colorbar(image, ax=axes.ravel().tolist(), label='Correct acquisition rate')
    path = output / 'correct_acquisition_rate.png'
    fig.savefig(path, dpi=150)
    paths.append(path)
    if not SHOW_PLOTS:
        plt.close(fig)
    if SHOW_PLOTS and plt.get_backend().lower() != 'agg':
        plt.show()
    return paths


def _manifest_config(value):
    """Write intentional infinite cross-polar isolation as an explicit model label."""
    if isinstance(value, dict):
        return {key: _manifest_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_manifest_config(item) for item in value]
    if isinstance(value, float) and np.isposinf(value):
        return 'Infinity (zero cross voltage)'
    return value


def main():
    output = reset_output(OUTPUT)
    configurations = {
        context: {case: _manifest_config(asdict(build_case(
            case, context, SEEDS[0], DURATION_MS, ANTENNAS)))
                  for case in CASES}
        for context in CONTEXTS
    }
    manifest = {
        'seeds': SEEDS, 'duration_ms': DURATION_MS, 'antennas': ANTENNAS,
        'contexts': {name: asdict(direction) for name, direction in CONTEXTS.items()},
        'cases': CASES, 'configurations': configurations,
        'source_sha256': source_hashes(),
        'definitions': {
            'power_interval': f'Discard first and last {POWER_TRIM_MS} ms for power only; receiver processes full record.',
            'decomposition': 'Truth target/CW/WB and regenerated thermal noise through linear RF and floating DFE, using actual frozen weights.',
            'cross_terms': 'Linear sum power minus sum of individual component powers; not thermal noise.',
            'joint_residual': 'Actual quantized DFE minus all four linear components; includes limiter, ADC and DFE quantization jointly. Actual-minus-linear power also includes linear/residual cross terms.',
            'gain_baseline': 'Same context, case and seed SINGLE actual component power.',
            'paired_baseline': 'Same context, seed and receiver mode ideal case.',
            'correct_acquisition': 'Acquisition success plus |code bias| <= 0.5 chip and |Doppler bias| <= 125 Hz.',
            'tracking_cn0': 'Late-half finite median from receiver tracking; short-window estimate, not oracle C/N0.',
            'thermal_oracle_prompt_cn0': 'Post-hoc true-code/Doppler 1 ms target-vs-thermal projection; not receiver input and includes RF/DFE/code losses.',
            'default_blockage': 'Default LOS 30/60 does not intersect 225/20 blockage; active context does.',
        },
    }
    (output / 'manifest.json').write_text(
        json.dumps(serializable(manifest), indent=2, allow_nan=False, default=str),
        encoding='utf-8'
    )
    calibration = calibrate_noise(SEEDS, DURATION_MS)
    (output / 'noise_calibration.json').write_text(
        json.dumps(serializable(calibration), indent=2, allow_nan=False), encoding='utf-8'
    )
    rows = []
    for context in CONTEXTS:
        for seed in SEEDS:
            for case in CASES:
                cfg = build_case(case, context, seed, DURATION_MS, ANTENNAS)
                rows.extend(run_case(cfg, case, context, trim_ms=POWER_TRIM_MS))
                print(context, seed, case, 'complete', flush=True)
    groups = summarize(rows)
    summary = {'rows': rows, 'groups': groups, 'noise_calibration': calibration}
    (output / 'summary.json').write_text(
        json.dumps(serializable(summary), indent=2, allow_nan=False), encoding='utf-8'
    )
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot_summary(groups, output)
    return summary


if __name__ == '__main__':
    main()
