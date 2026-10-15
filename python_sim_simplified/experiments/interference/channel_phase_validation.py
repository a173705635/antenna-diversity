"""理想双天线的 DFE 整通道相位误差、准确补偿和单 CW 复验。

相位因子在 DFE 输出后作用于整行复数样本；不是 RF/ADC 前的硬件模型，
也不表示方向图变化或宽带群时延。目标、干扰、噪声和已有前端残差一起旋转。

处理顺序：生成一份基准 RF/DFE 数据 → 构造三组相位条件 → 合并/捕获/跟踪
→ 固定接收权值独立核算功率 → 与同一基准配对核验 → 保存结果和绘图。
接收只使用名义目标导向；真实目标/干扰分量只用于事后评分。
"""

from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    # 直接执行本文件时，将 python_sim_simplified 加入导入路径；-m 运行不需要。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from core.evaluation import linear_components
from core.metrics import circular_chip_error
from core.pipeline import run_frontend, run_receiver
from core.presets import default_config
from core.provenance import source_hashes
from core.serialization import serializable
from experiments.paths import INTERFERENCE_RESULTS_DIR, reset_output


OUTPUT_DIR = INTERFERENCE_RESULTS_DIR / 'channel_phase_validation'
# 行顺序与此元组一致，第一根是 SINGLE 参考；扰动天线应选另一根。
# 更换组合时同时修改 PERTURBED_ANTENNA，几何相位仍由原天线模型计算。
ANTENNAS = ('TL', 'BR')
PERTURBED_ANTENNA = 'BR'
# 不同 seed 是不同噪声实现；同一 seed/场景的全部组别共用同一份输入。
SEEDS = (42, 43, 44)
DURATION_MS = 100
PHASE_ERRORS_DEG = (0, 30, 60, 90, 120, 180)
# 只在事后功率统计中去掉两端瞬态，不截短用于合并、捕获和跟踪的数据。
EVALUATION_EDGE_TRIM_MS = 10
SHOW_PLOTS_FLAG = True
# noise_only 没有外加干扰；single_cw 只保留默认 CW，不包含宽带干扰。
SCENES = ('noise_only', 'single_cw')
# 分别表示无误差、有误差未补偿、同样误差准确补偿。
CASES = ('baseline', 'uncalibrated', 'calibrated')
MODES = ('single', 'egc', 'mvdr')
# 两路等幅目标同相叠加、各路热噪声独立且等功率时，理论 SNR 收益为 3.0103 dB。
# 这里只用于双天线基准核验，不是有干扰时 MVDR 必须达到的收益。
THEORETICAL_EGC_GAIN_DB = 10.0 * np.log10(2.0)


def build_config(seed, duration_ms, antennas, scene):
    """生成一份理想双天线配置；duration_ms 的单位为毫秒。

    default_config 保留真实/名义匹配的理想几何和零通道误差。
    本轮只改变天线组合、记录长度和干扰集合，不改变功率、NF 或门限。
    """
    cfg = default_config()
    cfg.random_seed = seed
    cfg.signal.duration_ms = duration_ms
    # 跟踪每历元为 1 ms；上限随记录时长设置，实际可用历元可能更少。
    cfg.tracking.max_epochs = duration_ms
    cfg.phone.active_antennas = antennas
    cfg.diversity.modes = MODES
    # 默认干扰列表的第一个是 CW；无干扰是独立控制组，不是事后忽略干扰功率。
    cfg.interferers = [] if scene == 'noise_only' else cfg.interferers[:1]
    return cfg


def apply_channel_phase(channels, antennas, perturbed_antenna, phase_deg):
    """按天线名给完整通道乘 exp(jφ)，返回新矩阵，不修改输入。

    channels 形状为 [天线数, 时间样本数]，phase_deg 单位为度。
    单位模因子只旋转相位，不缩放幅度，也不移动样本或改变码包络。
    """
    factors = np.ones(len(antennas), dtype=complex)
    # 用物理天线名找行号，避免换天线组合/顺序后仍误用固定索引。
    factors[antennas.index(perturbed_antenna)] = np.exp(1j * np.deg2rad(phase_deg))
    # [:, None] 将通道因子扩为列向量，广播到该通道的所有时间样本。
    return channels * factors[:, None]


def _case_channels(baseline, antennas, phase_deg):
    """由同一基准构造 X0、D·X0、D_est⁻¹·D·X0 三组接收输入。"""
    perturbed = apply_channel_phase(
        baseline, antennas, PERTURBED_ANTENNA, phase_deg
    )
    # 本轮人为提供准确相位，所以 D_est=D；负相位实现逆旋转。
    # 补偿的是实际受扰序列，包括干扰与噪声，不是只补偿目标或本地码。
    # 这不是自动相位估计实验，也不涉及整序列时延对齐。
    calibrated = apply_channel_phase(
        perturbed, antennas, PERTURBED_ANTENNA, -phase_deg
    )
    return {'baseline': baseline, 'uncalibrated': perturbed,
            'calibrated': calibrated}


def _power(values):
    """样本平均模平方；输入是 DFE 码值时，单位为码值平方，而不是瓦特。"""
    return float(np.mean(np.abs(values) ** 2))


def _db_ratio(numerator, denominator):
    """功率比转 dB，因此使用 10·log10；不是电压比的 20·log10。"""
    return float(10.0 * np.log10(numerator / denominator))


def _score_branch(channels, components, branch, cfg, fs_out_hz, evaluation_edge_trim_ms):
    """接收完成后，以实际求出的固定权值独立核算一个方法的指标。

    channels 是该组实际 DFE 输入；components 是同组线性分量；
    branch 是公共 run_receiver 返回的记录。evaluation_edge_trim_ms 只作用于功率统计窗口。
    """
    combined = branch['combined']
    combining = branch['combining']
    acquisition = branch['acquisition']
    tracking = branch['tracking']
    weights = combining['weights']
    # 两端各去掉 evaluation_edge_trim_ms，降低滤波/补零边缘影响；所有分量采用相同窗口。
    edge_trim_samples = round(evaluation_edge_trim_ms * 1e-3 * fs_out_hz)
    window = slice(edge_trim_samples, channels.shape[1] - edge_trim_samples)
    input_window = channels[:, window]
    output_window = combined[window]
    # 输出定义为 y=wᴴX，conj() 实现共轭；分别投影目标、CW、热噪声用于事后评分。
    separated = {
        name: weights.conj() @ samples[:, window]
        for name, samples in components.items()
    }
    # 从统计窗口独立计算 R=XXᴴ/N。其对应输出功率应等于 wᴴRw。
    # 这里核验固定权值的功率恒等式，不重新用裁剪窗口训练权值。
    covariance = input_window @ input_window.conj().T / input_window.shape[1]
    direct = weights.conj() @ input_window
    quadratic = float(np.vdot(weights, covariance @ weights).real)
    actual_power = _power(output_window)
    linear_sum = sum(separated.values())
    # 包括 RF 非线性、ADC 和 DFE 量化的联合误差，不能全部称为热噪声。
    residual = output_window - linear_sum
    # 捕获码相位由输出采样点换成 chip；循环差避免码周期边界造成假大误差。
    # 若捕获失败，这只是搜索最大点的误差，不代表有效码相位观测。
    code_estimate = (acquisition['code_phase_samples'] *
                     cfg.signal.code_rate_hz / fs_out_hz)
    code_error = float(circular_chip_error(code_estimate, cfg.signal.code_delay_chips))
    if tracking['success_flag']:
        # 后半段降低启动瞬态影响；短窗 C/N₀ 估计中的无效值不参与中位数。
        late = tracking['cn0_db_hz'][len(tracking['cn0_db_hz']) // 2:]
        finite = late[np.isfinite(late)]
    else:
        finite = np.array([])
    residual_power = _power(residual)
    # 以下功率均在同一 DFE 码值尺度下；相对比值可比较，但不能直接当成 dBm。
    return {
        'target_power': _power(separated['target']),
        'cw_power': _power(separated['cw']),
        'thermal_power': _power(separated['thermal']),
        'joint_nonlinear_quantization_residual_power': residual_power,
        # 有限样本的分量交叉项未必为零；它不是额外热噪声。
        'linear_component_cross_power': _power(linear_sum) -
            sum(_power(value) for value in separated.values()),
        'actual_power': actual_power,
        # 仅线性分量 C/(N+I)，不含联合残差/交叉项；不是接收机估计的 C/N₀。
        'linear_in_band_sinr_db': _db_ratio(
            _power(separated['target']),
            _power(separated['thermal']) + _power(separated['cw']),
        ),
        # JSON 不直接保存复数，按每个权值的 [实部, 虚部] 保存以便独立复算。
        'weight_real_imag': [[float(value.real), float(value.imag)] for value in weights],
        # 独立等功率白噪声下，输出噪声功率与 ||w||² 成正比。
        'weight_norm_squared': float(np.vdot(weights, weights).real),
        'combined_direct_max_abs_error': float(np.max(np.abs(output_window - direct))),
        'quadratic_power_relative_error': abs(actual_power - quadratic) / actual_power,
        'acquisition_metric': float(acquisition['metric']),
        'acquired_flag': bool(acquisition['success_flag']),
        'acquired_code_error_chips': code_error,
        'acquired_doppler_error_hz': float(
            acquisition['doppler_hz'] - cfg.signal.doppler_hz
        ),
        'tracking_success_flag': bool(tracking['success_flag']),
        'tracking_status': tracking['status'],
        # 无可用估计保留 None 与计数 0，不用 0 dB-Hz 代替缺测。
        'tracking_cn0_late_median_db_hz': float(np.median(finite)) if finite.size else None,
        'tracking_cn0_late_finite_count': int(finite.size),
    }


def _paired_rows(seed, scene, phase_deg, cfg, truth, baseline, dfe_info,
                 linear_components, baseline_branches, evaluation_edge_trim_ms):
    """构造同一相位点三组×三种方法的九行结果，与同一份基准配对。"""
    channels_by_case = _case_channels(baseline, cfg.phone.active_antennas, phase_deg)
    baseline_norm = np.linalg.norm(baseline)
    # 矩阵的整体相对误差核验全部通道和样本；浮点逆旋转允许舍入误差。
    compensated_error = float(
        np.linalg.norm(channels_by_case['calibrated'] - baseline) / baseline_norm
    )
    # 基准接收已在外层算过一次；只对未补偿/补偿两组重新求权并接收。
    case_branches = {'baseline': baseline_branches}
    case_branches.update({
        case: run_receiver(channels, truth, cfg, dfe_info['fs_out_hz'])
        for case, channels in channels_by_case.items()
        if case != 'baseline'
    })
    rows = []
    for case in CASES:
        # 评分分量经过与实际序列相同的旋转；否则会拿错误相位的真值来评分。
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
                                dfe_info['fs_out_hz'], evaluation_edge_trim_ms)
            for mode in MODES
        }
        # 每组都相对同一根 SINGLE 通道比较；参考通道本轮不施加相位误差。
        reference = scored['single']
        for mode in MODES:
            row = scored[mode]
            # 先算目标/热噪声 SNR，再相对 SINGLE 求收益；没有把 CW 放进分母。
            thermal_gain_db = _db_ratio(
                row['target_power'] / row['thermal_power'],
                reference['target_power'] / reference['thermal_power'],
            )
            baseline_branch = case_branches['baseline'][mode]
            current_branch = case_branches[case][mode]
            baseline_output = baseline_branch['combined']
            baseline_weights = baseline_branch['combining']['weights']
            current_weights = current_branch['combining']['weights']
            # 所有输出/权值恢复误差均与该方法自身的无误差基准比较，而非与 EGC 比。
            row.update({
                'seed': seed, 'scene': scene, 'phase_error_deg': phase_deg,
                'case': case, 'mode': mode,
                # 不含 case，使同一条件的 baseline 和 calibrated 能用同一键配对。
                'paired_key': f'{seed}:{scene}:{phase_deg}:{mode}',
                'antennas': ','.join(cfg.phone.active_antennas),
                'perturbed_antenna': PERTURBED_ANTENNA,
                'acquisition_threshold': cfg.acquisition.threshold,
                'steering_unchanged_flag': bool(np.array_equal(
                    current_branch['combining']['constraint_steering'],
                    truth['desired_steering_assumed'])),
                'target_output_gain_db': _db_ratio(
                    row['target_power'], reference['target_power']
                ),
                # 开方得到电压响应比；未开方的下一项是功率比，二者物理含义不同。
                'target_response_gain': float(np.sqrt(
                    row['target_power'] / reference['target_power']
                )),
                # 相对 SINGLE，不是强制限于 [0,1]；失配 MVDR 也可能放大真目标。
                'target_power_retention': float(
                    row['target_power'] / reference['target_power']
                ),
                'thermal_snr_gain_db': thermal_gain_db,
                # 功率比方向为 SINGLE/CW 输出，因此正 dB 表示干扰减少。
                'cw_suppression_db': _db_ratio(
                    reference['cw_power'], row['cw_power']
                ) if scene == 'single_cw' else None,
                'egc_theoretical_white_noise_gain_db': (
                    THEORETICAL_EGC_GAIN_DB if scene == 'noise_only' and mode == 'egc'
                    else None
                ),
                # 未补组也保留此差值便于查看；verify_rows 只用无误差基准检验 3 dB。
                'egc_white_noise_gain_error_db': (
                    thermal_gain_db - THEORETICAL_EGC_GAIN_DB
                    if scene == 'noise_only' and mode == 'egc' else None
                ),
                'dfe_compensation_relative_error': compensated_error if case == 'calibrated' else None,
                'calibrated_output_relative_error': float(
                    np.linalg.norm(current_branch['combined'] - baseline_output) /
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
    """核验准确补偿是否恢复各自基准，返回通过状态、逐项判据和配对差。

    未补组失捕是研究结果，不是核验失败；基准本就失捕的补偿组也允许失捕。
    这里检查恢复关系，不要求所有方法具有相同抗干扰能力。
    """
    # 3.0103 dB 前提是目标已正确对齐，所以只取 noise_only 的 baseline/EGC。
    white_noise_errors = [row['egc_white_noise_gain_error_db'] for row in rows
                          if row['scene'] == 'noise_only'
                          and row['case'] == 'baseline' and row['mode'] == 'egc']
    maximum_white_noise_error_db = max(abs(value) for value in white_noise_errors)
    # 相位点的基准数据相同，但仍按完整条件键保存，避免跨 seed/场景误配。
    baseline_by_key = {row['paired_key']: row for row in rows
                       if row['case'] == 'baseline'}
    paired = []
    for calibrated in (row for row in rows if row['case'] == 'calibrated'):
        baseline = baseline_by_key[calibrated['paired_key']]
        comparison = {'paired_key': calibrated['paired_key']}
        # 比较功率的相对误差；dB 抑制和捕获/跟踪指标另外用绝对误差比较。
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
        comparison['acquired_equal_flag'] = calibrated['acquired_flag'] == baseline['acquired_flag']
        comparison['code_error_abs_difference_chips'] = abs(
            calibrated['acquired_code_error_chips'] - baseline['acquired_code_error_chips']
        )
        comparison['doppler_error_abs_difference_hz'] = abs(
            calibrated['acquired_doppler_error_hz'] - baseline['acquired_doppler_error_hz']
        )
        comparison['tracking_status_equal_flag'] = (
            calibrated['tracking_status'] == baseline['tracking_status']
        )
        comparison['tracking_success_equal_flag'] = (
            calibrated['tracking_success_flag'] == baseline['tracking_success_flag']
        )
        calibrated_cn0 = calibrated['tracking_cn0_late_median_db_hz']
        baseline_cn0 = baseline['tracking_cn0_late_median_db_hz']
        comparison['tracking_cn0_availability_equal_flag'] = (
            (calibrated_cn0 is None) == (baseline_cn0 is None)
        )
        comparison['tracking_cn0_finite_count_equal_flag'] = (
            calibrated['tracking_cn0_late_finite_count'] ==
            baseline['tracking_cn0_late_finite_count']
        )
        # 先核验跟踪状态、指标可用性和样本数；不能只比较恰好仍有值的数据。
        # 双侧无 C/N₀ 可标为未评价；单侧缺失必须由可用性判据判失败。
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
        # None 只跳过无定义的数值差；单侧 C/N₀ 缺失仍由独立可用性检查拒绝。
        return all(comparison[name] is None or comparison[name] < tolerance
                   for comparison in paired)

    # 严格的舍入级闭环误差与宽松的有限样本白噪声收益误差分开设置。
    # 这些阈值只判研究核验，不修改捕获门限、输入信号或合并权重。
    checks = {
        'complete_case_mode_grid_flag': len(rows) == len({
            (row['seed'], row['scene'], row['phase_error_deg']) for row in rows
        }) * len(CASES) * len(MODES),
        'steering_unchanged_flag': all(row['steering_unchanged_flag'] for row in rows),
        'direct_combining_flag': all(row['combined_direct_max_abs_error'] < 1e-8 for row in rows),
        'covariance_quadratic_flag': all(row['quadratic_power_relative_error'] < 1e-10
                                    for row in rows),
        'accurate_inverse_flag': all(row['dfe_compensation_relative_error'] < 1e-12
                                for row in rows if row['case'] == 'calibrated'),
        'calibrated_output_flag': all(row['calibrated_output_relative_error'] < 1e-12
                                 for row in rows if row['case'] == 'calibrated'),
        'calibrated_weights_flag': all(row['calibrated_weight_relative_error'] < 1e-12
                                  for row in rows if row['case'] == 'calibrated'),
        'white_noise_egc_flag': bool(maximum_white_noise_error_db < 0.5),
        'paired_target_power_flag': within('target_power_relative_error', 1e-12),
        'paired_thermal_power_flag': within('thermal_power_relative_error', 1e-12),
        'paired_cw_power_flag': within('cw_power_relative_error', 1e-12),
        'paired_cw_suppression_flag': within('cw_suppression_abs_error_db', 1e-9),
        'paired_acquisition_metric_flag': within('acquisition_metric_abs_error', 1e-9),
        'paired_acquisition_status_flag': all(item['acquired_equal_flag'] for item in paired),
        'paired_code_error_flag': within('code_error_abs_difference_chips', 1e-9),
        'paired_doppler_error_flag': within('doppler_error_abs_difference_hz', 1e-9),
        'paired_tracking_status_flag': all(item['tracking_status_equal_flag'] for item in paired),
        'paired_tracking_success_flag': all(item['tracking_success_equal_flag'] for item in paired),
        'paired_tracking_cn0_availability_flag': all(
            item['tracking_cn0_availability_equal_flag'] for item in paired
        ),
        'paired_tracking_cn0_finite_count_flag': all(
            item['tracking_cn0_finite_count_equal_flag'] for item in paired
        ),
        'paired_tracking_cn0_flag': within('tracking_cn0_abs_error_db_hz', 1e-7),
    }
    return {'passed_flag': all(checks.values()), 'checks': checks,
            'paired_comparisons': paired,
            'white_noise_egc_errors_db': white_noise_errors,
            'white_noise_egc_max_abs_error_db': maximum_white_noise_error_db,
            'limits': {'relative_closure': 1e-12,
                       'quadratic_power_relative': 1e-10,
                       'white_noise_egc_error_db': 0.5,
                       'paired_metric_absolute': 1e-9,
                       'paired_tracking_cn0_db_hz': 1e-7}}


def run_sweep(seeds=SEEDS, duration_ms=DURATION_MS, phases=PHASE_ERRORS_DEG,
              antennas=ANTENNAS, evaluation_edge_trim_ms=EVALUATION_EDGE_TRIM_MS):
    """逐 seed/场景生成一次前端，遍历相位；返回结果行和配对核验报告。"""
    rows = []
    for seed in seeds:
        for scene in SCENES:
            # RF 加噪声与 ADC 量化、DFE 下变频/抽取只在本层执行，不随相位重抽噪声。
            cfg = build_config(seed, duration_ms, antennas, scene)
            frontend = run_frontend(cfg, retain_analysis_flag=True)
            truth = frontend['truth']
            baseline = frontend['dfe_channels']
            dfe_info = frontend['dfe_info']
            # 事后评分另重建线性分量；它们不会替换实际接收输入。
            shared = linear_components(frontend)
            # 旧报告在无干扰场景仍保留零功率 CW 字段。
            if scene == 'noise_only':
                cw = np.zeros_like(shared['target'])
            else:
                cw = shared['interferer_0_CWInterferer']
            # 保持既有 target/CW/thermal 求和顺序，避免交叉项舍入改变。
            components = {'target': shared['target'], 'cw': cw,
                          'thermal': shared['thermal']}
            # 缓存各方法基准接收，后续相位点共享它们，只重复保存配对结果行。
            baseline_branches = run_receiver(baseline, truth, cfg, dfe_info['fs_out_hz'])
            for phase_deg in phases:
                rows.extend(_paired_rows(
                    seed, scene, phase_deg, cfg, truth, baseline, dfe_info,
                    components, baseline_branches, evaluation_edge_trim_ms,
                ))
                print(seed, scene, phase_deg, 'complete', flush=True)
    return rows, verify_rows(rows)


def plot_summary(rows, output_dir, phases):
    """分方法/场景绘制三张图；横轴为度，曲线不是 Monte Carlo 成功率。"""
    import matplotlib.pyplot as plt

    colors = {'baseline': '#333333', 'uncalibrated': '#d95f02',
              'calibrated': '#1f77b4'}
    styles = {'baseline': '-', 'uncalibrated': '--', 'calibrated': ':'}

    def values(scene, case, mode, key):
        """每个相位点聚合可用 seed；缺测用 NaN 断开曲线，不补成零。"""
        result = []
        for phase in phases:
            items = [row[key] for row in rows if row['scene'] == scene
                     and row['case'] == case and row['mode'] == mode
                     and row['phase_error_deg'] == phase and row[key] is not None]
            if key == 'thermal_snr_gain_db':
                # 先将每个 seed 的 dB 换成倍率再求均值，并非先平均 dB 后再换算。
                # 180° 目标抵消的极小值在线性图中接近零，不压扁其他相位点。
                items = [10.0 ** (value / 10.0) for value in items]
            result.append(float(np.mean(items)) if items else np.nan)
        return result

    # 热噪声收益、目标/CW、实际接收指标分开，避免把“热噪声收益高”当成抗干扰好。
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
    figure_paths = []
    for filename, metrics in figures:
        # 列是场景，行按“指标→方法”排列；每个坐标轴只有三组配对曲线。
        fig, axes = plt.subplots(len(metrics) * len(MODES), len(SCENES),
                                 figsize=(12, 3.5 * len(metrics) * len(MODES)),
                                 squeeze=False, constrained_layout=True)
        if filename == 'thermal_gain':
            fig.suptitle('Linear thermal SNR ratio vs paired SINGLE '
                         '(baseline 2 ≈ 3.01 dB)')
        for metric_index, (metric, ylabel) in enumerate(metrics):
            for mode_index, mode in enumerate(MODES):
                for scene_index, scene in enumerate(SCENES):
                    # 将指标编号和方法编号映射到二维子图的行号。
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
                        # 补偿与基准重合是预期结果；无跟踪曲线则表示缺测，不是 0 C/N₀。
                        axis.text(0.02, 0.02,
                                  'Where available, calibrated matches baseline',
                                  fontsize=7, transform=axis.transAxes)
                        if metric == 'tracking_cn0_late_median_db_hz':
                            axis.text(0.02, 0.09,
                                      'Gaps / no curve = no finite tracking estimate',
                                      fontsize=7, transform=axis.transAxes)
                    axis.set(title=f'{scene} / {mode}',
                             xlabel='DFE phase error (deg)', ylabel=ylabel)
                    # 空跟踪图也保留相同相位轴，防止自动坐标范围被误读成新实验条件。
                    axis.set_xlim(phases[0], phases[-1])
                    axis.grid(alpha=0.3)
                    if not (metric == 'cw_suppression_db' and scene == 'noise_only'):
                        axis.legend(fontsize=7, ncol=2)
        figure_path = output_dir / f'{filename}.png'
        # 先保存，再按设置显示；Agg 是无 GUI 后端，不调用阻塞式弹窗。
        fig.savefig(figure_path, dpi=150)
        figure_paths.append(figure_path)
        if not SHOW_PLOTS_FLAG:
            plt.close(fig)
    if SHOW_PLOTS_FLAG and plt.get_backend().lower() != 'agg':
        plt.show()
    return figure_paths


def main():
    """一次完成生成、核验、结果保存和绘图；核验失败仍保留诊断结果。"""
    # 按项目惯例重建本实验自己的输出目录，不能指向其他实验或已有报告目录。
    output_dir = reset_output(OUTPUT_DIR)
    # 先保存实际参数和当前源码身份，方便后续判断某份结果来自哪个版本。
    # 只保存清单，不复制整套源码或大体积 IQ；输入可由配置与 seed 重建。
    manifest = {
        'seeds': SEEDS, 'duration_ms': DURATION_MS, 'phases_deg': PHASE_ERRORS_DEG,
        'evaluation_edge_trim_ms': EVALUATION_EDGE_TRIM_MS, 'antennas': ANTENNAS,
        'perturbed_antenna': PERTURBED_ANTENNA,
        'injection_stage': 'DFE output',
        'injection_definition': (
            'Constant, frequency-independent complex gain on the entire perturbed '
            'DFE channel including target, CW, thermal noise, and pre-existing '
            'nonlinear/quantization residual; not RF/ADC hardware, pattern or delay.'
        ),
        'estimated_phase_equals_true_flag': True,
        'reference_antenna_unchanged': ANTENNAS[0],
        'metric_definitions': {
            'linear_in_band_sinr_db': 'Post-hoc truth-linear sample-power C/(thermal+CW), excluding the separately reported nonlinear/quantization residual and cross terms; not actual quantized-output C/N0.',
            'thermal_snr_gain_db': 'Independently replayed target/thermal sample-power ratio versus same-case reference single.',
            'thermal_gain_plot': 'Display each row as 10**(thermal_snr_gain_db/10), then average by phase/case/mode/scene; no clamp. Baseline EGC near 2 means 3.0103 dB.',
            'tracking_cn0_late_median_db_hz': 'Finite late-half receiver tracking estimate; null when unavailable. Plotted lines average available seed values only; missing estimates make gaps or no curve, never zeros.',
        },
        # asdict 展开完整配置；仅存顶部参数不足以复现 RF/DFE、功率和门限。
        'receiver_config_by_scene_and_seed': {
            scene: {str(seed): asdict(build_config(seed, DURATION_MS, ANTENNAS, scene))
                    for seed in SEEDS} for scene in SCENES
        },
        'source_sha256': source_hashes(),
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(serializable(manifest), indent=2, allow_nan=False, default=str),
        encoding='utf-8',
    )
    # 显式传当前顶部常量，使 main 使用修改后的条件，而非函数定义时绑定的默认值。
    rows, verification = run_sweep(SEEDS, DURATION_MS, PHASE_ERRORS_DEG,
                                   ANTENNAS, EVALUATION_EDGE_TRIM_MS)
    (output_dir / 'verification.json').write_text(
        json.dumps(serializable(verification), indent=2, allow_nan=False),
        encoding='utf-8',
    )
    # JSON 保留嵌套核验详情，CSV 提供每条件一行的表格；两者来自同一批结果。
    summary = {'rows': rows, 'verification': verification}
    (output_dir / 'summary.json').write_text(
        json.dumps(serializable(summary), indent=2, allow_nan=False),
        encoding='utf-8',
    )
    with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot_summary(rows, output_dir, PHASE_ERRORS_DEG)
    # 失败也先保存结果/图，便于定位问题；非零退出防止把“跑完”误当成“通过”。
    if not verification['passed_flag']:
        raise AssertionError(f'channel phase verification failed: {verification["checks"]}')
    return summary


if __name__ == '__main__':
    main()
