"""Render the measured research ledger without turning abstention into success."""
import json
from motion_third_research import ROOT,OUT,CASES


def main():
    data=json.loads((OUT/'motion_validation.json').read_text());s=data['summary']
    lines=[
    '# 旋转与手机差分相位：276–300研究报告',
    '',
    '完成25轮真实数学/估计机制改变和逐轮同输入前后比较。结论是：旋转可使手机平面阵列的时间叠加设计获得三维方向信息，多PRN差分可消共同通道漂移；这些信息不能恢复绝对位置，不能证明信号真实，也不能在任意共同通道漂移下恢复共同源方向。本批只在相位观测域运行，没有真实RF、真实IMU或导航闭环。',
    '',
    f"冻结新种子{s['seeds']}，25工况×4种子＝{s['windows']}个12秒窗口，每窗121样本。宽分离多源8窗中共源误报0；共源92窗中直接判成diverse为0，但indeterminate为20，因此共源未检出率是20/92＝21.74%，不能将弃权记作检测成功。方向可用48/100窗，其方向误差中位数{s['available_direction_median_error_deg']:.5f}°、最大{s['available_direction_max_error_deg']:.5f}°。", 
    '',
    '**必须保留的失败：**新增4个冻结窄角度多源反例全部误判common（4/4）。若与原8个宽分离多源一起计，误报为4/12＝33.33%，不是0%。这一比例是人为工况集合的计数，不代表环境发生概率。所有common输出只表示多数PRN响应在当前分辨率下与共源相容，不是确认攻击或确认全体PRN均共源。',
    '',
    '## 观测模型与条件',
    '',
    '手机名义四角坐标来自presets.py：宽0.0759m、长0.1609m，z=0；研究使用2、3、4天线，最多3条参考差分基线、7个PRN。没有扩大台架或单欺骗源之外的攻击发射机。diverse负例是正常多方向到达；clustered负例仅检验有限角分辨率。',
    '',
    '相位模型为 y(t,b,p)=wrap(k·[R(t)b]ᵀu_p(t)+c_bp+d_bp(t)+noise)。R由测得gyro经偏置/轴比例/时钟读数校正后做SO(3)乘积积分。初始R=I仅定义初始机体系方向坐标，不代表外部航向已知。静态源u恒定；moving源只有一个方向随时间连续转动。估计器通过分段常数＋线性时间联合投影消通道常数、实数相位跳变及线性漂移，再做静态/一阶动态方向拟合。',
    '',
    'gyro bias_readings、scale_readings、clock_offset_readings、measured_baselines均是独立随机流的模拟仪器读数，属于明确的理想辅助，不能声称手机已有这些标定或认证。没有固定手机、零速更新或位置台架；bias读数不被解释成在线静止检测。gyro_certified只是声明是否存在，绝不是密码认证或实物证书。相位噪声主要为±0.025rad均匀噪声，Gaussian工况单独撤销有界域。phase-center读数用高斯扰动生成；给定几何支持是声明值而非无条件硬界。',
    '',
    '二维/三维共同body lever在远场天线间差分中严格相消，因此body_lever与common观测完全相同；本批未建模近场曲率、加速度杆臂、圆锥/划桨、天线方向图随姿态变化、真实相位锁定和多径。转动率积分与差分相位结合也无法发现任意共同绝对平移；static与uniform观测完全相同，均输出indeterminate。',
    '',
    '## 共同漂移及方向可辨识性',
    '',
    '跨PRN减均值：y_p−mean(y)=kRbᵀ(u_p−mean(u))+噪声，共同通道漂移消失，但共同方向也消失。对于两个不同共同方向u和v，只需共享通道漂移d(t,b)=k[R(t)b]ᵀ(u−v)，相位就严格一致。冻结反例最大观测差'+f"{s['common_drift_gauge_max_difference']:.3g}"+'rad。低残差无法发现这种等价变换；因此剩余方向结果始终以通道漂移模型为条件。',
    '',
    '最小奇异值≤0.5时撤销共源决策；方向还要求common、残差RMS<0.08rad、有界声明成立及方向支持诊断<30°。这些门限没有硬件标定或普适误检概率保证。方向支持是把gyro/时间/相位中心声明扰动经过线性设计最小奇异值传播的诊断，未包含全部非线性/动态曲率与选模误差，support_is_certificate始终false。',
    '',
    '## 逐轮同输入结果',
    '',
    '开发种子2760，25固定工况逐轮完整前后评估。表内只是每轮目标工况；不同目标行不可横向比较为性能增益。真实机制改变计迭代，种子、参数、测试和实现纠错不计。',
    '',
    '|轮次|机制|目标|方向误差前→后(°)|RMS前→后(rad)|方向可用前→后|',
    '|---|---|---|---:|---:|---|']
    for n in range(276,301):
        c=json.loads((OUT/f'iteration{n}'/'conclusion.json').read_text());a=c['after'];b=c['before']
        lines.append(f"|{n}|{c['mechanism']}|{c['target']}|{b['direction_error_deg']:.4f}→{a['direction_error_deg']:.4f}|{b['rms']:.4f}→{a['rms']:.4f}|{b['direction_available']}→{a['direction_available']}|")
    lines += ['',
    '277仅换正确旋转组合时，未修复wrap和二维拟合，目标误差恶化并原样保存。281单位球投影没有降低gyro bias误差，增大残差；它是物理约束试验，不是改进证明。290时钟标签校正只有微小变化：该工况仅有标签偏移，原索引配对偶然正确，不能宣称真实异步采样和任意传输延迟已验证。',
    '',
    '292消线性漂移会损失几何信息。296初实现使用分段常数投影后再投影原时间，两投影不交换；同轮纠正为联合最小二乘投影，保留after_uncorrected.py、results_uncorrected.json和conclusion_uncorrected.json；不增加迭代。回归测试验证相位跳变叠加线性漂移不改变共源统计。297 moving方向误差10.46°→1.65°，但支持约79°，最终不可用。298强制方向导数在单位球切平面，残差恶化，未声称提升；299通过有惩罚的数据拟合选择静态/动态模型以限制不必要自由度。',
    '',
    '## 冻结逐工况',
    '',
    '|工况|common/diverse/indeterminate窗数|方向可用窗数|全部候选最大方向误差(°)|',
    '|---|---|---:|---:|']
    for case in CASES:
        rows=[r for r in data['rows'] if r['case']==case]
        counts='/'.join(str(sum(r['status']==status for r in rows)) for status in ('common','diverse','indeterminate'))
        lines.append(f"|{case}|{counts}|{sum(r['direction_available'] for r in rows)}|{max(r['direction_error_deg'] for r in rows):.4f}|")
    lines += ['',
    '未可用的direction数组是内部候选，不能当已输出方向。窄角多源和漂移等价反例另存motion_validation.json的counterexamples。绝对位置输出始终不可用，因此未实现或验证任何“长期数米、全历元≤20m”定位结论。',
    '',
    '## 复现',
    '',
    '工作目录python_sim_simplified，Python D:/Python/Python311/python.exe；OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。',
    '',
    '```text',
    'python -m pytest tests/test_motion_third.py -q',
    'python motion_third_validate.py',
    'python motion_third_report.py',
    '```',
    '',
    '10项针对性测试通过；276–300全部25轮before/after快照逐一重放完全一致。motion_frozen_*.npz保存实际冻结观测。每轮目录包含proposal.json、before.py、after.py、results.json、conclusion.json；motion_summary.json给出机器可读汇总。勿再次执行研究创建命令覆盖历史；record会拒绝已存在目录。',
    '',
    '冻结源码SHA256：`'+s['source_sha256']+'`。']
    second=json.loads((OUT/'motion_second_summary.json').read_text())
    audit=[
        '## 验证流程补充：显式预冻结后的独立结果',
        '',
        '首批验证存在证据缺口：motion_third_validate.py在评估结束后才计算summary源码哈希，没有评估前独立保存的冻结清单，故原工件不能证明预冻结顺序。首批结果曾被检查并用于报告与反例审阅；没有记录表明依据其调模型或门限，但曾移除数值上不起作用的1e-30归一化项并重跑首批验证，因此不把首批称为独立预注册验证。全部181个旧数据/逐轮工件已逐字节核对保持不变，旧报告另存motion_second_previous_report.md。',
        '',
        '此次不改模型，不增加研究轮数。97631已被首批单元测试和反例测试使用，因此从独立集合排除，单列motion_second_reused_diagnostic.json；用97643替代。独立新种子为97621、97627、97639、97643。',
        '',
        f"motion_second_freeze.json于UTC {second['freeze_created_at_utc']}写入，SHA256为`{second['freeze_sha256']}`；验证在UTC {second['evaluation_started_at_utc']}才开始。先检查模型、模拟器、计分器、验证程序、测试、Python执行文件以及NumPy/SciPy源码/二进制依赖哈希，再落盘motion_second_started.json，最后才调用评估。评估后再次核对冻结清单和本地源码哈希。UTC/文件mtime及工具执行顺序是本地审计证据，不是可信第三方时间戳或不可篡改存储。", 
        '',
        f"新的100个12秒窗口：宽分离多源误报{second['false_alarms']}/{second['diverse_windows']}；共源直接误拒{second['missed_common']}/{second['common_windows']}，弃权/未检出{second['common_not_detected']}/{second['common_windows']}＝21.74%。方向可用{second['direction_available_windows']}/100，误差中位数{second['available_direction_median_error_deg']:.5f}°、最大{second['available_direction_max_error_deg']:.5f}°。窄角多源依然4/4误报，合并多源4/12＝33.33%。绝对位置依然不可用，共同漂移不可辨识反例依然成立。", 
        '',
        '**最终新汇总：** `results/research-third-hundred/motion_second_summary.json`；完整新结果为motion_second_validation.json，冻结观测为motion_second_frozen_*.npz，输出哈希见motion_second_output_manifest.json。旧motion_summary.json未覆盖。以下原批统计与逐轮表保留为历史记录，不能替代本段独立新验证。',
        '',
        '补充验证命令是`python motion_third_second_validate.py freeze`后另行运行`python motion_third_second_validate.py validate`。已有工件用独占创建保存，请勿在现有目录重跑写入；先核对冻结清单及输出清单进行只读审计。',
        '']
    lines[2:2]=audit
    lines=[line.replace('冻结新种子[97601', '首批未预注册验证种子[97601').replace('## 冻结逐工况','## 首批验证逐工况（保留记录）') for line in lines]
    (ROOT/'motion_third_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':main()
