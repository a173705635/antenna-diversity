"""理想双天线的 DFE 整通道相位误差、准确补偿和单 CW 复验。

相位因子在 DFE 输出后作用于整行复数样本；不是 RF/ADC 前的硬件模型，
也不表示方向图变化或宽带群时延。truth 仅用于接收完成后的独立评分。
"""

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from core.acquisition import acquire
from core.dfe import process_dfe
from core.diversity import combine_known
from core.gnss import generate_scenario
from core.metrics import circular_chip_error
from core.presets import default_config
from core.provenance import source_hashes
from core.rf import filter_rf_input, generate_rf_noise, process_rf
from core.serialization import serializable
from core.tracking import track
from experiments.paths import INTERFERENCE_RESULTS, reset_output


OUTPUT = INTERFERENCE_RESULTS / 'channel_phase_validation'
ANTENNAS = ('TL', 'BR')
PERTURBED_ANTENNA = 'BR'
SEEDS = (42, 43, 44)
DURATION_MS = 100
PHASE_ERRORS_DEG = (0, 30, 60, 90, 120, 180)
POWER_TRIM_MS = 10
SHOW_PLOTS = True
SCENES = ('noise_only', 'single_cw')
CASES = ('baseline', 'uncalibrated', 'calibrated')
MODES = ('single', 'egc', 'mvdr')
THEORETICAL_EGC_GAIN_DB = 10.0 * np.log10(2.0)


def build_config(seed, duration_ms, antennas, scene):
    """保留默认功率、噪声系数、采样率和捕获门限，只收窄场景。"""
    cfg = default_config()
    cfg.random_seed = seed
    cfg.signal.duration_ms = duration_ms
    cfg.tracking.max_epochs = duration_ms
    cfg.signal.multipath_paths = []
    cfg.enable_multipath_research = False
    cfg.phone.active_antennas = antennas
    cfg.diversity.modes = MODES
    cfg.interferers = [] if scene == 'noise_only' else cfg.interferers[:1]
    return cfg


def apply_channel_phase(channels, antennas, perturbed_antenna, phase_deg):
    """按物理天线名旋转完整 DFE 行，列维度为时间样本。"""
    factors = np.ones(len(antennas), dtype=complex)
    factors[antennas.index(perturbed_antenna)] = np.exp(1j * np.deg2rad(phase_deg))
    return channels * factors[:, None]


def _case_channels(baseline, antennas, phase_deg):
    perturbed = apply_channel_phase(
        baseline, antennas, PERTURBED_ANTENNA, phase_deg
    )
    # 估计值在本轮固定等于真值；逆旋转的是实际受扰序列，不是目标真值。
    calibrated = apply_channel_phase(
        perturbed, antennas, PERTURBED_ANTENNA, -phase_deg
    )
    return {'baseline': baseline, 'uncalibrated': perturbed,
            'calibrated': calibrated}


def _linear_components(cfg, scene_input_v, truth, rf_info, dfe_info):
    """用 RF 元数据独立重放线性目标、CW、热噪声；输出为 DFE 码单位。"""
    fs_hz = cfg.signal.fs_rf_hz
    gain = 10.0 ** (np.asarray(rf_info['channel_total_gain_db']) / 20.0)
    gain = gain * np.exp(1j * np.deg2rad(rf_info['phase_error_deg']))
    float_cfg = replace(cfg.dfe, output_bits=0)

    def linear_dfe(filtered_voltage):
        adc_units = filtered_voltage * gain[:, None] / rf_info['adc_lsb_v']
        values, _ = process_dfe(adc_units, float_cfg, fs_hz, cfg.signal.if_hz)
        return values / dfe_info['output_scale']

    target_filtered = filter_rf_input(
        truth['desired_component'], rf_info['filter_coefficients'],
        fs_hz, cfg.signal.if_hz,
    )
    target = linear_dfe(target_filtered)
    if cfg.interferers:
        cw_filtered = filter_rf_input(
            truth['interferer_components'][0], rf_info['filter_coefficients'],
            fs_hz, cfg.signal.if_hz,
        )
        cw = linear_dfe(cw_filtered)
    else:
        cw = np.zeros_like(target)
    thermal_voltage = generate_rf_noise(
        scene_input_v.shape[1], rf_info['noise_psd_w_per_hz'],
        cfg.reference_impedance_ohm, fs_hz, cfg.signal.if_hz,
        cfg.phone.active_antennas, cfg.random_seed,
        rf_info['filter_coefficients'],
    )
    thermal = linear_dfe(thermal_voltage)
    return {'target': target, 'cw': cw, 'thermal': thermal}


def _receive(channels, steering, cfg, fs_out_hz):
    """仅以未改动的基准 steering 和受扰样本调用公共接收算法。"""
    branches = {}
    for mode in MODES:
        combined, combining = combine_known(
            channels, steering, mode, cfg.diversity.diagonal_loading_factor
        )
        acquisition = acquire(
            combined, fs_out_hz, cfg.signal.prn, cfg.signal.code_rate_hz,
            cfg.acquisition,
        )
        if acquisition['success']:
            tracking = track(
                combined, fs_out_hz, cfg.signal.prn, acquisition,
                cfg.tracking, cfg.signal.carrier_hz, cfg.signal.code_rate_hz,
            )
        else:
            tracking = {'success': False, 'status': 'acquisition_failed',
                        'prompt': np.array([], dtype=complex)}
        branches[mode] = (combined, combining, acquisition, tracking)
    return branches


def _power(values):
    return float(np.mean(np.abs(values) ** 2))


def _db_ratio(numerator, denominator):
    return float(10.0 * np.log10(numerator / denominator))


def _score_branch(channels, components, branch, cfg, fs_out_hz, trim_ms):
    """固定接收器权值后复算样本功率，不复用合并器报告的增益。"""
    combined, combining, acquisition, tracking = branch
    weights = combining['weights']
    trim = round(trim_ms * 1e-3 * fs_out_hz)
    window = slice(trim, channels.shape[1] - trim)
    input_window = channels[:, window]
    output_window = combined[window]
    separated = {
        name: weights.conj() @ samples[:, window]
        for name, samples in components.items()
    }
    covariance = input_window @ input_window.conj().T / input_window.shape[1]
    direct = weights.conj() @ input_window
    quadratic = float(np.vdot(weights, covariance @ weights).real)
    actual_power = _power(output_window)
    linear_sum = sum(separated.values())
    residual = output_window - linear_sum
    code_estimate = (acquisition['code_phase_samples'] *
                     cfg.signal.code_rate_hz / fs_out_hz)
    code_error = float(circular_chip_error(code_estimate, cfg.signal.code_delay_chips))
    if tracking['success']:
        late = tracking['cn0_db_hz'][len(tracking['cn0_db_hz']) // 2:]
        finite = late[np.isfinite(late)]
    else:
        finite = np.array([])
    residual_power = _power(residual)
    return {
        'target_power': _power(separated['target']),
        'cw_power': _power(separated['cw']),
        'thermal_power': _power(separated['thermal']),
        'joint_nonlinear_quantization_residual_power': residual_power,
        'linear_component_cross_power': _power(linear_sum) -
            sum(_power(value) for value in separated.values()),
        'actual_power': actual_power,
        'linear_in_band_sinr_db': _db_ratio(
            _power(separated['target']),
            _power(separated['thermal']) + _power(separated['cw']),
        ),
        'weight_real_imag': [[float(value.real), float(value.imag)] for value in weights],
        'weight_norm_squared': float(np.vdot(weights, weights).real),
        'combined_direct_max_abs_error': float(np.max(np.abs(output_window - direct))),
        'quadratic_power_relative_error': abs(actual_power - quadratic) / actual_power,
        'acquisition_metric': float(acquisition['metric']),
        'acquired': bool(acquisition['success']),
        'acquired_code_error_chips': code_error,
        'acquired_doppler_error_hz': float(
            acquisition['doppler_hz'] - cfg.signal.doppler_hz
        ),
        'tracking_success': bool(tracking['success']),
        'tracking_status': tracking['status'],
        'tracking_cn0_late_median_db_hz': float(np.median(finite)) if finite.size else None,
        'tracking_cn0_late_finite_count': int(finite.size),
    }


def _paired_rows(seed, scene, phase_deg, cfg, truth, baseline, dfe_info,
                 linear_components, baseline_branches, trim_ms):
    channels_by_case = _case_channels(baseline, cfg.phone.active_antennas, phase_deg)
    baseline_norm = np.linalg.norm(baseline)
    compensated_error = float(
        np.linalg.norm(channels_by_case['calibrated'] - baseline) / baseline_norm
    )
    case_branches = {'baseline': baseline_branches}
    case_branches.update({
        case: _receive(channels, truth['desired_steering_assumed'],
                       cfg, dfe_info['fs_out_hz'])
        for case, channels in channels_by_case.items()
        if case != 'baseline'
    })
    rows = []
    for case in CASES:
        components = {
            name: (samples if case == 'baseline' else
                   apply_channel_phase(samples, cfg.phone.active_antennas,
                                       PERTURBED_ANTENNA, phase_deg))
            for name, samples in linear_components.items()
        }
        if case == 'calibrated':
            components = {
                name: apply_channel_phase(samples, cfg.phone.active_antennas,
                                          PERTURBED_ANTENNA, -phase_deg)
                for name, samples in components.items()
            }
        scored = {
            mode: _score_branch(channels_by_case[case], components,
                                case_branches[case][mode], cfg,
                                dfe_info['fs_out_hz'], trim_ms)
            for mode in MODES
        }
        reference = scored['single']
        for mode in MODES:
            row = scored[mode]
            thermal_gain_db = _db_ratio(
                row['target_power'] / row['thermal_power'],
                reference['target_power'] / reference['thermal_power'],
            )
            baseline_branch = case_branches['baseline'][mode]
            current_branch = case_branches[case][mode]
            baseline_output = baseline_branch[0]
            baseline_weights = baseline_branch[1]['weights']
            current_weights = current_branch[1]['weights']
            row.update({
                'seed': seed, 'scene': scene, 'phase_error_deg': phase_deg,
                'case': case, 'mode': mode,
                'paired_key': f'{seed}:{scene}:{phase_deg}:{mode}',
                'antennas': ','.join(cfg.phone.active_antennas),
                'perturbed_antenna': PERTURBED_ANTENNA,
                'acquisition_threshold': cfg.acquisition.threshold,
                'steering_unchanged': bool(np.array_equal(
                    current_branch[1]['constraint_steering'],
                    truth['desired_steering_assumed'])),
                'target_output_gain_db': _db_ratio(
                    row['target_power'], reference['target_power']
                ),
                'target_response_gain': float(np.sqrt(
                    row['target_power'] / reference['target_power']
                )),
                'target_power_retention': float(
                    row['target_power'] / reference['target_power']
                ),
                'thermal_snr_gain_db': thermal_gain_db,
                'cw_suppression_db': _db_ratio(
                    reference['cw_power'], row['cw_power']
                ) if scene == 'single_cw' else None,
                'egc_theoretical_white_noise_gain_db': (
                    THEORETICAL_EGC_GAIN_DB if scene == 'noise_only' and mode == 'egc'
                    else None
                ),
                'egc_white_noise_gain_error_db': (
                    thermal_gain_db - THEORETICAL_EGC_GAIN_DB
                    if scene == 'noise_only' and mode == 'egc' else None
                ),
                'dfe_compensation_relative_error': compensated_error if case == 'calibrated' else None,
                'calibrated_output_relative_error': float(
                    np.linalg.norm(current_branch[0] - baseline_output) /
                    np.linalg.norm(baseline_output)
                ) if case == 'calibrated' else None,
                'calibrated_weight_relative_error': float(
                    np.linalg.norm(current_weights - baseline_weights) /
                    np.linalg.norm(baseline_weights)
                ) if case == 'calibrated' else None,
            })
            rows.append(row)
    return rows


def verify_rows(rows):
    """只约束准确校准闭环和数学恒等式，不要求随机弱信号全部捕获。"""
    white_noise_errors = [row['egc_white_noise_gain_error_db'] for row in rows
                          if row['scene'] == 'noise_only'
                          and row['case'] == 'baseline' and row['mode'] == 'egc']
    maximum_white_noise_error_db = max(abs(value) for value in white_noise_errors)
    baseline_by_key = {row['paired_key']: row for row in rows
                       if row['case'] == 'baseline'}
    paired = []
    for calibrated in (row for row in rows if row['case'] == 'calibrated'):
        baseline = baseline_by_key[calibrated['paired_key']]
        comparison = {'paired_key': calibrated['paired_key']}
        for name in ('target_power', 'thermal_power'):
            comparison[name + '_relative_error'] = abs(
                calibrated[name] - baseline[name]
            ) / baseline[name]
        if calibrated['scene'] == 'single_cw':
            comparison['cw_power_relative_error'] = abs(
                calibrated['cw_power'] - baseline['cw_power']
            ) / baseline['cw_power']
            comparison['cw_suppression_abs_error_db'] = abs(
                calibrated['cw_suppression_db'] - baseline['cw_suppression_db']
            )
        else:
            comparison['cw_power_relative_error'] = None
            comparison['cw_suppression_abs_error_db'] = None
        comparison['acquisition_metric_abs_error'] = abs(
            calibrated['acquisition_metric'] - baseline['acquisition_metric']
        )
        comparison['acquired_equal'] = calibrated['acquired'] == baseline['acquired']
        comparison['code_error_abs_difference_chips'] = abs(
            calibrated['acquired_code_error_chips'] - baseline['acquired_code_error_chips']
        )
        comparison['doppler_error_abs_difference_hz'] = abs(
            calibrated['acquired_doppler_error_hz'] - baseline['acquired_doppler_error_hz']
        )
        comparison['tracking_status_equal'] = (
            calibrated['tracking_status'] == baseline['tracking_status']
        )
        comparison['tracking_success_equal'] = (
            calibrated['tracking_success'] == baseline['tracking_success']
        )
        calibrated_cn0 = calibrated['tracking_cn0_late_median_db_hz']
        baseline_cn0 = baseline['tracking_cn0_late_median_db_hz']
        comparison['tracking_cn0_availability_equal'] = (
            (calibrated_cn0 is None) == (baseline_cn0 is None)
        )
        comparison['tracking_cn0_finite_count_equal'] = (
            calibrated['tracking_cn0_late_finite_count'] ==
            baseline['tracking_cn0_late_finite_count']
        )
        if calibrated_cn0 is None and baseline_cn0 is None:
            comparison['tracking_cn0_comparison'] = 'not_evaluated'
            comparison['tracking_cn0_abs_error_db_hz'] = None
        elif calibrated_cn0 is None or baseline_cn0 is None:
            comparison['tracking_cn0_comparison'] = 'availability_mismatch'
            comparison['tracking_cn0_abs_error_db_hz'] = None
        else:
            comparison['tracking_cn0_comparison'] = 'evaluated'
            comparison['tracking_cn0_abs_error_db_hz'] = abs(
                calibrated_cn0 - baseline_cn0
            )
        paired.append(comparison)
    def within(name, tolerance):
        return all(comparison[name] is None or comparison[name] < tolerance
                   for comparison in paired)

    checks = {
        'complete_case_mode_grid': len(rows) == len({
            (row['seed'], row['scene'], row['phase_error_deg']) for row in rows
        }) * len(CASES) * len(MODES),
        'steering_unchanged': all(row['steering_unchanged'] for row in rows),
        'direct_combining': all(row['combined_direct_max_abs_error'] < 1e-8 for row in rows),
        'covariance_quadratic': all(row['quadratic_power_relative_error'] < 1e-10
                                    for row in rows),
        'accurate_inverse': all(row['dfe_compensation_relative_error'] < 1e-12
                                for row in rows if row['case'] == 'calibrated'),
        'calibrated_output': all(row['calibrated_output_relative_error'] < 1e-12
                                 for row in rows if row['case'] == 'calibrated'),
        'calibrated_weights': all(row['calibrated_weight_relative_error'] < 1e-12
                                  for row in rows if row['case'] == 'calibrated'),
        'white_noise_egc': bool(maximum_white_noise_error_db < 0.5),
        'paired_target_power': within('target_power_relative_error', 1e-12),
        'paired_thermal_power': within('thermal_power_relative_error', 1e-12),
        'paired_cw_power': within('cw_power_relative_error', 1e-12),
        'paired_cw_suppression': within('cw_suppression_abs_error_db', 1e-9),
        'paired_acquisition_metric': within('acquisition_metric_abs_error', 1e-9),
        'paired_acquisition_status': all(item['acquired_equal'] for item in paired),
        'paired_code_error': within('code_error_abs_difference_chips', 1e-9),
        'paired_doppler_error': within('doppler_error_abs_difference_hz', 1e-9),
        'paired_tracking_status': all(item['tracking_status_equal'] for item in paired),
        'paired_tracking_success': all(item['tracking_success_equal'] for item in paired),
        'paired_tracking_cn0_availability': all(
            item['tracking_cn0_availability_equal'] for item in paired
        ),
        'paired_tracking_cn0_finite_count': all(
            item['tracking_cn0_finite_count_equal'] for item in paired
        ),
        'paired_tracking_cn0': within('tracking_cn0_abs_error_db_hz', 1e-7),
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'paired_comparisons': paired,
            'white_noise_egc_errors_db': white_noise_errors,
            'white_noise_egc_max_abs_error_db': maximum_white_noise_error_db,
            'limits': {'relative_closure': 1e-12,
                       'quadratic_power_relative': 1e-10,
                       'white_noise_egc_error_db': 0.5,
                       'paired_metric_absolute': 1e-9,
                       'paired_tracking_cn0_db_hz': 1e-7}}


def run_sweep(seeds=SEEDS, duration_ms=DURATION_MS, phases=PHASE_ERRORS_DEG,
              antennas=ANTENNAS, trim_ms=POWER_TRIM_MS):
    rows = []
    for seed in seeds:
        for scene in SCENES:
            cfg = build_config(seed, duration_ms, antennas, scene)
            scene_input_v, truth = generate_scenario(cfg)
            adc_codes, rf_info = process_rf(
                scene_input_v, cfg.rf, cfg.reference_impedance_ohm, seed,
                fs_hz=cfg.signal.fs_rf_hz, if_hz=cfg.signal.if_hz,
                channel_ids=antennas,
            )
            baseline, dfe_info = process_dfe(
                adc_codes, cfg.dfe, cfg.signal.fs_rf_hz, cfg.signal.if_hz
            )
            components = _linear_components(
                cfg, scene_input_v, truth, rf_info, dfe_info
            )
            baseline_branches = _receive(
                baseline, truth['desired_steering_assumed'], cfg,
                dfe_info['fs_out_hz'],
            )
            for phase_deg in phases:
                rows.extend(_paired_rows(
                    seed, scene, phase_deg, cfg, truth, baseline, dfe_info,
                    components, baseline_branches, trim_ms,
                ))
                print(seed, scene, phase_deg, 'complete', flush=True)
    return rows, verify_rows(rows)


def plot_summary(rows, output, phases):
    import matplotlib.pyplot as plt

    colors = {'baseline': '#333333', 'uncalibrated': '#d95f02',
              'calibrated': '#1f77b4'}
    styles = {'baseline': '-', 'uncalibrated': '--', 'calibrated': ':'}

    def values(scene, case, mode, key):
        result = []
        for phase in phases:
            items = [row[key] for row in rows if row['scene'] == scene
                     and row['case'] == case and row['mode'] == mode
                     and row['phase_error_deg'] == phase and row[key] is not None]
            if key == 'thermal_snr_gain_db':
                items = [10.0 ** (value / 10.0) for value in items]
            result.append(float(np.mean(items)) if items else np.nan)
        return result

    figures = (
        ('thermal_gain', (('thermal_snr_gain_db', 'Linear thermal SNR ratio'),)),
        ('target_and_cw', (
            ('target_power_retention', 'Target power ratio vs single (linear)'),
            ('cw_suppression_db', 'CW suppression vs single (dB)'),
        )),
        ('receiver', (
            ('acquisition_metric', 'Acquisition peak ratio'),
            ('tracking_cn0_late_median_db_hz', 'Late finite tracking C/N0 (dB-Hz)'),
        )),
    )
    paths = []
    for filename, metrics in figures:
        fig, axes = plt.subplots(len(metrics) * len(MODES), len(SCENES),
                                 figsize=(12, 3.5 * len(metrics) * len(MODES)),
                                 squeeze=False, constrained_layout=True)
        if filename == 'thermal_gain':
            fig.suptitle('Linear thermal SNR ratio vs paired SINGLE '
                         '(baseline 2 ≈ 3.01 dB)')
        for metric_index, (metric, ylabel) in enumerate(metrics):
            for mode_index, mode in enumerate(MODES):
                for scene_index, scene in enumerate(SCENES):
                    axis = axes[metric_index * len(MODES) + mode_index, scene_index]
                    if metric == 'cw_suppression_db' and scene == 'noise_only':
                        axis.text(0.5, 0.5, 'No CW in this scene',
                                  ha='center', va='center', transform=axis.transAxes)
                    else:
                        for case in CASES:
                            axis.plot(phases, values(scene, case, mode, metric),
                                      color=colors[case], linestyle=styles[case],
                                      marker='o', markersize=3, label=case)
                        if metric == 'acquisition_metric':
                            axis.axhline(rows[0]['acquisition_threshold'],
                                         color='#888888', linestyle='-.',
                                         label='threshold')
                        axis.text(0.02, 0.02,
                                  'Where available, calibrated matches baseline',
                                  fontsize=7, transform=axis.transAxes)
                        if metric == 'tracking_cn0_late_median_db_hz':
                            axis.text(0.02, 0.09,
                                      'Gaps / no curve = no finite tracking estimate',
                                      fontsize=7, transform=axis.transAxes)
                    axis.set(title=f'{scene} / {mode}',
                             xlabel='DFE phase error (deg)', ylabel=ylabel)
                    axis.set_xlim(phases[0], phases[-1])
                    axis.grid(alpha=0.3)
                    if not (metric == 'cw_suppression_db' and scene == 'noise_only'):
                        axis.legend(fontsize=7, ncol=2)
        path = output / f'{filename}.png'
        fig.savefig(path, dpi=150)
        paths.append(path)
        if not SHOW_PLOTS:
            plt.close(fig)
    if SHOW_PLOTS and plt.get_backend().lower() != 'agg':
        plt.show()
    return paths


def main():
    output = reset_output(OUTPUT)
    manifest = {
        'seeds': SEEDS, 'duration_ms': DURATION_MS, 'phases_deg': PHASE_ERRORS_DEG,
        'power_trim_ms': POWER_TRIM_MS, 'antennas': ANTENNAS,
        'perturbed_antenna': PERTURBED_ANTENNA,
        'injection_stage': 'DFE output',
        'injection_definition': (
            'Constant, frequency-independent complex gain on the entire perturbed '
            'DFE channel including target, CW, thermal noise, and pre-existing '
            'nonlinear/quantization residual; not RF/ADC hardware, pattern or delay.'
        ),
        'estimated_phase_equals_true': True,
        'reference_antenna_unchanged': ANTENNAS[0],
        'metric_definitions': {
            'linear_in_band_sinr_db': 'Post-hoc truth-linear sample-power C/(thermal+CW), excluding the separately reported nonlinear/quantization residual and cross terms; not actual quantized-output C/N0.',
            'thermal_snr_gain_db': 'Independently replayed target/thermal sample-power ratio versus same-case reference single.',
            'thermal_gain_plot': 'Display each row as 10**(thermal_snr_gain_db/10), then average by phase/case/mode/scene; no clamp. Baseline EGC near 2 means 3.0103 dB.',
            'tracking_cn0_late_median_db_hz': 'Finite late-half receiver tracking estimate; null when unavailable. Plotted lines average available seed values only; missing estimates make gaps or no curve, never zeros.',
        },
        'receiver_config_by_scene_and_seed': {
            scene: {str(seed): asdict(build_config(seed, DURATION_MS, ANTENNAS, scene))
                    for seed in SEEDS} for scene in SCENES
        },
        'source_sha256': source_hashes(),
    }
    (output / 'manifest.json').write_text(
        json.dumps(serializable(manifest), indent=2, allow_nan=False, default=str),
        encoding='utf-8',
    )
    rows, verification = run_sweep(SEEDS, DURATION_MS, PHASE_ERRORS_DEG,
                                   ANTENNAS, POWER_TRIM_MS)
    (output / 'verification.json').write_text(
        json.dumps(serializable(verification), indent=2, allow_nan=False),
        encoding='utf-8',
    )
    summary = {'rows': rows, 'verification': verification}
    (output / 'summary.json').write_text(
        json.dumps(serializable(summary), indent=2, allow_nan=False),
        encoding='utf-8',
    )
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot_summary(rows, output, PHASE_ERRORS_DEG)
    if not verification['passed']:
        raise AssertionError(f'channel phase verification failed: {verification["checks"]}')
    return summary


if __name__ == '__main__':
    main()
