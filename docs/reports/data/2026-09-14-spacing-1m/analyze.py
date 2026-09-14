"""Rebuild the spacing comparison and verify saved numerical evidence."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / '2026-09-13-gpspatron-spatial'
sys.path.insert(0, str(ROOT / 'code'))
from audit_spatial_diversity import LAYOUTS, METHODS, CASES, paired_gain, save_json
from spatial_spoofing import calibrate_threshold, observations_from_records
from audit_spatial_diversity_rf import evaluate_records


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def model_comparison():
    result = []
    thresholds = [read(base/'model/thresholds.json')['thresholds'] for base in (OLD, ROOT)]
    for case, attack in CASES:
        with np.load(OLD/f'model/{case}.npz') as old, np.load(ROOT/f'model/{case}.npz') as new:
            for layout in LAYOUTS:
                for method in METHODS:
                    key = f'{layout}_{method}'
                    a, b = old[key] < thresholds[0][key], new[key] < thresholds[1][key]
                    result.append(dict(case=case, attack=attack, layout=layout, method=method,
                                       old=float(a.mean()), new=float(b.mean()), **paired_gain(a, b)))
    save_json(ROOT/'model_comparison.json', result)
    return result


def rf_rows(base):
    return [row for batch in ('rf_a', 'rf_b') for row in read(base/batch/'summary.json')['rows']]


def verify():
    checks = 0
    # The earlier sealed results are inputs and must remain byte-for-byte intact.
    for rel, digest in read(OLD/'sha256.json').items():
        assert hashlib.sha256((OLD/rel).read_bytes()).hexdigest() == digest, rel
        checks += 1
    payload = read(ROOT/'model/thresholds.json')
    with np.load(ROOT/'model/calibration.npz') as values:
        for key, threshold in payload['thresholds'].items():
            assert calibrate_threshold(values[key]) == threshold
            checks += 1
    for row in read(ROOT/'model/summary.json')['rows']:
        with np.load(ROOT/'model'/f"{row['case']}.npz") as values:
            hits = int(np.sum(values[f"{row['layout']}_{row['method']}"] < row['threshold']))
            assert hits == row['alerts'] and hits / row['events'] == row['fraction']
            checks += 1
    channels = 0
    for batch in ('rf_a', 'rf_b'):
        folder = ROOT/batch
        manifest = read(folder/'manifest.json')
        assert manifest['physical_positions_m'] == payload['positions_m']
        assert manifest['threshold_file_sha256'] == hashlib.sha256((ROOT/'model/thresholds.json').read_bytes()).hexdigest()
        for name, digest in manifest['source_sha256'].items():
            assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest() == digest, name
        rows = read(folder/'summary.json')['rows']
        for case in manifest['cases']:
            for seed in manifest['seeds']:
                records = [read(folder/f'{case}_{seed}_{name}.json') for name in manifest['antenna_names']]
                rebuilt, trace = evaluate_records(records, payload['thresholds'])
                expected = [dict(case=case, seed=seed, **r) for r in rebuilt]
                assert expected == [r for r in rows if r['case'] == case and r['seed'] == seed]
                with np.load(folder/f'{case}_{seed}_observations.npz') as obs:
                    for key, val in zip(('time_s', 'carrier_m', 'valid', 'tl_error_m'), trace):
                        np.testing.assert_array_equal(obs[key], val)
                channels += len(records)
                checks += 6
    save_json(ROOT/'validation.json', dict(passed=True, checks=checks, rf_channels=channels,
                                         model_events=110000, old_archive_unchanged=True))
    return checks


def rf_count(rows, case, layout, method):
    windows = [w for r in rows if (r['case'], r['layout'], r['method']) == (case, layout, method)
               for w in r['windows']]
    return f"{sum(w['state'] == 1 for w in windows)}/{len(windows)}; {sum(w['state'] == -1 for w in windows)}"


def build_report(comparison, checks):
    old_rf, new_rf = rf_rows(OLD), rf_rows(ROOT)
    lines = [
        '# 1 米阵列与手机尺寸阵列：抗欺骗识别对照', '',
        '日期：2026-09-14。仅改变阵列尺寸并重新标定阈值；不加入 IMU，不改变检测公式。', '',
        '## 方法与可比条件', '',
        '- 原阵列：75.9 × 160.9 mm；新阵列：1 × 1 m 正方形。这里“1 米”指相邻天线距离，对角线为 √2 m，不是所有天线对均相距 1 m。长宽比也随之改变，不能把结果解释为保持形状的纯等比例缩放。',
        '- 两天线：TL–TR（旧名 `2_short`，新阵列中边长为 1 m）及 TL–BR（`2_diagonal`）；三天线：TL/TR/BL；四天线：全部四角。',
        '- 每种尺寸、布局、判据分别使用 5000 个无欺骗事件校准名义 1% 虚警率，之后冻结阈值。22 个场景各 5000 个独立评估事件；两种尺寸复用随机种子，在每个场景内保持天空、姿态和噪声抽样配对。',
        '- 判据保持 global 和 cluster3，不进行未校准的 OR 融合。每个事件包含 10 个相关历元；模型不是动态捕获仿真。short_aperture 压力场景在各自基准阵列上再乘 0.1，因此对应绝对尺寸不同。',
        '- RF：相同固定八星天空、单欺骗源方位角 100°/仰角 10°、虚假位置向东偏 100 m；6 场景 × 2 噪声种子 × 4 天线 × 600 ms，共 48 路接收记录。33 MHz RF、6.6 MHz DFE 及原捕获跟踪链保持不变。每场景只有两个独立噪声记录，8 个判决窗口彼此相关。',
        '- RF 的跨天线传播差通过逐星 range_at_zero_m 编码进入信号；单路配置中的默认 phone 坐标不表示此次阵列孔径。实际孔径以 model/thresholds.json 和 RF manifest 的 physical_positions_m 为准。', '',
        '## 观测模型结果', '',
        '表中为旧尺寸 → 1 米阵列的报警率（%）；LOS/clustered_los 是虚警，其余所列为欺骗检出。', '',
        '| 场景 | 判据 | 2 天线边长 | 3 天线 | 4 天线 |',
        '|---|---|---:|---:|---:|']
    selected = ['los', 'all_spoof', 'partial_3', 'partial_4', 'two_sources', 'mixture_0', 'mixture_6', 'clustered_los']
    for case in selected:
        for method in METHODS:
            cells = []
            for layout in ('2_short', '3', '4'):
                row = next(r for r in comparison if (r['case'], r['layout'], r['method']) == (case, layout, method))
                cells.append(f"{row['old']*100:.2f} → {row['new']*100:.2f}")
            lines.append('| ' + ' | '.join([case, method] + cells) + ' |')
    lines += ['', '完整 176 项比较及配对差值区间见 [model_comparison.json](data/2026-09-14-spacing-1m/model_comparison.json)；双天线对角线配置亦包含其中。', '',
              '三 PRN 部分替代的 cluster3 检出率差值（1 米减旧尺寸，单位为百分点）：', '']
    for layout in ('2_short', '3', '4'):
        row = next(r for r in comparison if (r['case'], r['layout'], r['method']) == ('partial_3', layout, 'cluster3'))
        lo, hi = row['normal95']
        lines.append(f"- {layout}：{100*row['delta']:+.2f}，配对正态近似 95% 区间 [{100*lo:+.2f}, {100*hi:+.2f}]。")
    lines += ['', '区间以冻结的校准阈值为条件，没有计入校准集随机性，也没有对跨场景多重比较作校正。', '',
              '![部分同源检测与边界](data/2026-09-14-spacing-1m/spacing_comparison.png)', '',
              '## 数字 RF 接收链结果', '',
              '每格为“报警窗口/总窗口；未知窗口数”，顺序为旧尺寸 → 1 米。未知不算检出；有效且未报警才是无证据。', '',
              '| 场景 | 判据 | 2 天线边长 | 3 天线 | 4 天线 |', '|---|---|---|---|---|']
    for case in ('los', 'all_spoof', 'partial_3', 'partial_4', 'mixture_0', 'mixture_6'):
        for method in METHODS:
            cells = [rf_count(old_rf, case, layout, method)+' → '+rf_count(new_rf, case, layout, method)
                     for layout in ('2_short', '3', '4')]
            lines.append('| '+' | '.join([case, method]+cells)+' |')
    lines += ['', '输出门控另见各 RF summary.json 的 policy 字段：启动未知、检测报警、有效未报警分别保留，门控不代表恢复正确定位。', '',
              'cluster3 下“危险但仍输出”的历元数（3D 误差 >30 m；旧尺寸 → 1 米）：', '',
              '| 场景 | 2 天线边长 | 3 天线 | 4 天线 |', '|---|---:|---:|---:|']
    for case in ('all_spoof', 'partial_3', 'partial_4', 'mixture_0', 'mixture_6'):
        cells = []
        for layout in ('2_short', '3', '4'):
            pair = [sum(r['policy']['dangerous_retained_epochs'] for r in rows
                        if (r['case'], r['layout'], r['method']) == (case, layout, 'cluster3'))
                    for rows in (old_rf, new_rf)]
            cells.append(f'{pair[0]} → {pair[1]}')
        lines.append('| '+' | '.join([case]+cells)+' |')
    lines += ['', '该指标可因无法判决而全部阻断变为零，必须与前表的未知数量一起看；不能把“零危险输出”自动解释为“全部检出”。30 m 是研究示例阈值。', '', '## 分析与边界', '',
        '1. 对三 PRN 同源部分替代，cluster3 的三天线检出率从 79.70% 到 90.92%，四天线从 94.60% 到 98.22%。这是冻结阈值、配对随机观测下的收益；global 全星统计仍不擅长少数 PRN 替代。',
        '2. 真实卫星聚集场景的 cluster3 虚警下降明显，但 1 米三/四天线仍达 6.84%/7.68%，高于名义目标；不能说误报问题已经解决。推断原因是更大孔径提高了近邻方向的相位差异。',
        '3. 真假混合不能概括为随距离增大而改善。模型 mixture_6 中，四天线 global 从 97.00% 到 99.52%，cluster3 却从 35.34% 到 17.00%。两种判据需要分别报告，不能事后挑选效果好的判据并声称系统已经具备该性能。',
        '   RF 也有明确退步：mixture_6 的三天线 cluster3 从 8/8 降到 0/8，全部是有效观测下漏检；同场景 global 仍为 8/8。四天线 cluster3 则新旧均为 0/8。等功率 mixture_0 新阵列双/三天线虽然有 4 个窗口恢复有效，却全部未检出，导致各有 30 个危险历元通过门控。',
        '4. 本判据使用倍相后的圆统计，1 米约为 5.26 个 GPS L1 波长。更大孔径增加方向敏感度，也保留周期折叠和方向混淆。只增加间距不能保证单调改善，更不是方向真实性认证。',
        '5. RF 固定天空只有一个几何样本，模型则随机化天空与姿态；其数值不应相互替代。三 PRN 替代的三/四天线 cluster3 在新旧尺寸均为 8/8，未显示检出次数增加；四 PRN 替代的三天线 global 从 0/8 到 8/8。RF 有效观测上的低检出与无效观测形成的未知必须区分。',
        '6. 仍是假设理想天线、无互耦和共同采样/频率参考的数字仿真；没有加入 1 米硬件所需的电缆差异、同步误差、温漂或真实场地多径。不能外推为实物或 GPSPATRON 商用设备性能。', '',
        '## 验证与复现', '',
        f'保存数据重算验证通过：{checks} 项检查，包括旧归档哈希、冻结阈值、全部模型报警计数、48 路 RF 记录重建后的判决与观测一致性。新增几何检验和相关既有测试共 17 项通过；见 [tests.log](data/2026-09-14-spacing-1m/tests.log)。', '',
        '源代码快照、命令和所有中间记录见 [数据说明](data/2026-09-14-spacing-1m/README.md)。本轮未改写前次报告或冻结数据。', '']
    (ROOT.parent.parent/'2026-09-14-spacing-1m.md').write_text('\n'.join(lines), encoding='utf-8')


def plot(comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = ['partial_3', 'partial_4', 'clustered_los', 'mixture_6']
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout='constrained')
    for ax, case in zip(axes, cases):
        for size, label, style in [('old', 'Phone array', 'o--'), ('new', '1 m square', 's-')]:
            vals = [next(r[size] for r in comparison if (r['case'], r['layout'], r['method']) ==
                         (case, layout, 'cluster3'))*100 for layout in ('2_short', '3', '4')]
            ax.plot([2, 3, 4], vals, style, label=label)
        ax.set_title(case + (' (false alarm)' if case == 'clustered_los' else ''))
        ax.set_xticks([2, 3, 4]); ax.set_ylim(-2, 103); ax.grid(alpha=.25)
        ax.set_xlabel('Antennas (2 = edge baseline)')
    axes[0].set_ylabel('Alert rate (%)'); axes[0].legend()
    fig.suptitle('cluster3; 5000 events per case; separately calibrated nominal 1% H0')
    fig.savefig(ROOT/'spacing_comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    comparison = model_comparison()
    plot(comparison)
    if '--model-only' not in sys.argv:
        checks = verify()
        build_report(comparison, checks)
        print(f'PASS: {checks} checks; report rebuilt')
