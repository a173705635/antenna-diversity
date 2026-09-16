"""Report observed frozen results; never modify a fitted model."""
import json
from pathlib import Path
import numpy as np
from delay_next_campaign import ROOT,write


def main():
    validation=json.loads((ROOT/'delay_validation.json').read_text())
    assert validation['completed']
    audit=json.loads((ROOT/'delay_iteration_audit.json').read_text(encoding='utf-8'))
    freeze=json.loads((ROOT/'delay_freeze.json').read_text())
    tests=json.loads((ROOT/'delay_tests.json').read_text(encoding='utf-8'))
    replay=json.loads((ROOT/'delay_replay.json').read_text())
    result={}
    for side in ('before','after'):
        rows=[p[side] for p in validation['rows']]
        accepted=[r for r in rows if r['usable'] and r['code_error_m'] is not None]
        codes=np.concatenate([np.array(r['code_error_m'])[r['supported_prns']] for r in accepted])
        result[side]=dict(usable_cases=sum(r['usable'] for r in rows),accepted_real_cases=len(accepted),
            accepted_prns=len(codes),accepted_code_rmse_m=float(np.sqrt(np.mean(codes**2))),
            accepted_code_max_abs_m=float(np.max(abs(codes))),accepted_code_over10m=int(np.sum(abs(codes)>10)),
            false_usable_noauth_or_noise=sum(r['usable'] for r in rows if r['kind'] in ('absent','noise')),
            accepted_phase_max_deg=max(r['phase_rmse_deg'] for r in accepted),
            per_kind={kind:dict(usable=sum(r['usable'] for r in rows if r['kind']==kind),total=sum(r['kind']==kind for r in rows),
                phase_deg=[r['phase_rmse_deg'] for r in rows if r['kind']==kind],
                group_rmse_chips=[r['group_delay_rmse_chips'] for r in rows if r['kind']==kind],
                admitted_code_rmse_m=[r['admitted_code_rmse_m'] for r in rows if r['kind']==kind]) for kind in freeze['kinds']})
        matched=np.concatenate([np.array(pair[side]['code_error_m'])[pair['after']['supported_prns']] for pair in validation['rows'] if pair['after']['usable']])
        result[side]['same_new_accepted_cohort_rmse_m']=float(np.sqrt(np.mean(matched**2)))
    write(ROOT/'delay_summary.json',dict(iterations=25,validation_pairs=len(validation['rows']),statistics=result,tests=tests,replay_pairs=len(replay['rows'])))
    text='''# 第101–125轮：空间—时延校准的第二轮研究

完成25次真实连续的机制变更与同输入对照。每轮目录均保留模型、实验脚本、proposal、results、conclusion；25个模型SHA均不同。参数扫点、额外种子、重放和测试没有计入轮次。仅新增delay_next系列文件，历史模型未改。

## 关键修复与模型边界

旧normal/32051的真群时延为零，却输出0.097656chip，相位RMSE26.975°；旧接受PRN还存在31.58m单星码误差。101移除有限窗FFT斜率后相位降到0.169°，但不能处理真实skew；103改为相关域共同群时延/PRN时延联合估计，106用零群时延嵌套证据替代硬件先验。101–103暂未保留信号存在门限，曾错误支持纯噪声；104已恢复绝对噪声似然证据，绝不能将早期版本直接部署。

最终相干模型为：每通道每PRN相关函数由“共享对角复增益×已知天空导向×天空源相关模板”与“公共源复空间向量×公共源相关模板”叠加，二者共享通道群时延与前端信号带宽。每PRN两源复幅由线性最小二乘消去；相位、通道幅度、公共源、时延、带宽进行非线性profile。参考通道群时延固定为0，参考通道复增益/公共源分量固定为1。它不估计绝对接收机时钟、导航位置或未知天空方向。

几何严格使用Mate70四角75.9×160.9mm；2/3天线为原四角的前2/3个。模型假设已知天空、独立PRN/通道噪声、对角接收链路、一个公共干扰源、每PRN最多两相关模板。它不是任意全矩阵耦合模型，不能替代RF硬件校准、GNSS认证或PVT验证。

108起需要额外512个独立无信号相关噪声快拍。125把它们分成256训练、256独立噪声适配性检验。训练用带宽+白噪声比例的实对称平稳相关协方差族及profile噪声方差；信号带宽不会改变该白化矩阵。噪声标定不是由真相位或真时延产生的oracle，但“确实无信号且与目标观测同噪声条件”的假设须在硬件上另行验证。

空间支持检查保留未受天空约束的测得分量，并留一PRN估计方向；没有使用投影后的完美天空分量自证。信号存在、天空分量对仅公共源的条件证据、全局残差、独立噪声适配性、每PRN码Fisher精度预算分别记录。

## 冻结验证

全部旧种子（含32051）均视为开发。冻结SHA及52101/52107/52119写入delay_freeze.json后才运行39组旧/新配对，之后没有调模型。下表只把usable为真且真实天空源存在的PRN计为最终接受；拒绝案例中输出的相位/群时延点估计不算有效结果。

| 指标 | 历史模型 | 第125轮 |
|---|---:|---:|
'''
    labels=[('可用案例','usable_cases'),('接受真实源案例','accepted_real_cases'),('接受PRN','accepted_prns'),('接受码RMSE/m','accepted_code_rmse_m'),('同一组新模型接受PRN上的码RMSE/m','same_new_accepted_cohort_rmse_m'),('接受单星最大绝对码误差/m','accepted_code_max_abs_m'),('接受码误差超过10m的PRN','accepted_code_over10m'),('无真实源或纯噪声误可用案例','false_usable_noauth_or_noise'),('接受案例最大相位RMSE/度','accepted_phase_max_deg')]
    for label,key in labels:text+=f'| {label} | {result["before"][key]:.6g} | {result["after"][key]:.6g} |\n'
    text+='\n| 场景 | 旧可用/3 | 新可用/3 | 新接受码RMSE（逐种子，m） |\n|---|---:|---:|---|\n'
    for kind in freeze['kinds']:
        old=result['before']['per_kind'][kind];new=result['after']['per_kind'][kind]
        text+=f'| {kind} | {old["usable"]}/3 | {new["usable"]}/3 | {new["admitted_code_rmse_m"]} |\n'
    text+='''
## 不能抹去的失败与限制

- 105的0.03chip高斯硬件先验仅部分改善重合源，没有作为默认保留；106转为数据驱动的嵌套比较。
- 113改善重合相位但没有改善码；115–117相干联合才改善码。122允许幅度失配后开发重合相位从0.251°恶化至0.522°，体现释放nuisance的方差代价。
- 112、114、118、123、124中的若干开发判决或指标基本不变。它们是明确的模型/证据机制检验，不宣称每一轮都有数值收益。
- 121拒绝了开发2天线一个局部信息不足PRN，但剩余PRN事后RMSE反而从2.366m升至2.519m。局部标准差不是事后真误差排序。
- 精度预算为局部3σ<10m；124仅对Fisher信息最弱PRN检查±10m两个非线性profile点。没有穷尽其他PRN、所有极值与整个置信区间，也没有多重检验或选择后覆盖率保证。
- 源数初始化仍来自自由通道分量拟合；局部非线性求解可能存在不同极值。没有宣称全局最优，优化器状态也尚未形成独立可用性门限。
- 冻结two/52101被全局残差拒绝（p约2.04e-283），而非精度门限：其第2个PRN被初始化为单源，实际仿真是双源；这说明局部源数初始化仍可失败。保留拒绝结果，没有使用冻结种子反向修复。
- 卡方、BIC及噪声协方差适配性p值是模型内渐近诊断。噪声参数由有限标定数据估计、源数/群时延由数据选择，名义p值不是已标定的实际误警概率。
- 多径、非平稳噪声和低幅信号拒绝是必要代价；即使相位点估计看起来准确，也不能据此继续接受码结果。
- 三个新种子不足以测定小概率误接受上界。该阶段没有使用RF采样、真实手机测量、时变电磁环境或导航闭环。

## 逐轮账本

| 轮次 | 可核对的机制假设 | 实际结论 |
|---|---|---|
'''
    for row in audit['rows']:text+=f'| {row["iteration"]} | {row["hypothesis"]} | {row["decision"]} |\n'
    text+='''
## 验证与重放

4个行为测试通过：旧正常零群时延失败、真实群时延与相位区分、独立256/256噪声证据、全局复相位规范不变性。101/106/115/120/125五快照在normal/overlap/skew/noise的20项重放通过，逐项输入SHA、支持判决、群时延和码时延吻合（绝对容差1e-7chip）。

运行目录为python_sim_simplified，Python为D:/Python/Python311/python.exe，OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。

```text
python -m unittest tests.test_delay_next
python delay_next_verify.py audit
python delay_next_verify.py replay
python delay_next_verify.py validate
python delay_next_report.py
```

验证输出：results/research-next-hundred/delay_validation.json、delay_summary.json、delay_replay.json、delay_tests.json、delay_iteration_audit.json。不要重新运行freeze覆盖现有冻结记录。
'''
    (Path(__file__).parent/'delay_next_REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
