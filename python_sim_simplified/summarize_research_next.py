"""Generate the auditable hundred-round ledger and stratified report."""
import hashlib,json,shutil
from pathlib import Path
from audit_recovery_fifty import save
from research_next_navigation import ROOT

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def ledger():
    lines=['# 第 101–200 轮逐轮账本','',
           '每个编号是一项算法或模型改动；留出种子、测试、重放和实现纠错不计轮次。151–175 的结论格式为 Markdown，其余为 JSON。',
           '', '| 轮次 | 假设 / 实际机制 | 对照与结论 |','|---|---|---|']
    for n in range(101,201):
        folder=ROOT/f'iteration{n:03d}';proposal=read(folder/'proposal.json')
        hypothesis=proposal['hypothesis'];conclusion=next(folder.glob('conclusion.*'))
        assert (folder/'results.json').is_file() or (folder/'record.json').is_file()
        lines.append(f'| {n} | {hypothesis.replace(chr(10)," ").replace("|","/")} | [证据与结论]({conclusion.as_posix()}) |')
    Path('RESEARCH_NEXT_HUNDRED_LEDGER.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def report():
    data=read(ROOT/'navigation_corrected_validation.json');totals=data['totals']
    rf=totals['RF-derived'];obs=totals['observation']
    lines=['# 第二批 100 轮研究：101–200','',
      '**100 轮已闭环，但尚未实现可商用的手机长时间抗欺骗导航。** 本批可确认的收益主要是相关器域时延标定及部分条件 RF 导航；纯噪声错误放行、共同位置偏差和真实硬件误差界仍未解决。',
      '', '## 冻结后的结果','',
      '| 研究线 | 独立验证结果 | 判断 |','|---|---|---|',
      '| 101–125 时延/源分解 | 同一179个接受PRN上，码RMSE 5.762→2.018m；新接受最大误差8.561m；可用案例30→23/39 | 有条件收益，拒绝比例同时增加；不是RF/PVT结果 |',
      '| 126–150 冷启动 | 7组RF导航20m内63/168→63/168；150纯噪声提示4/4错误放行 | 无导航新增收益，拒绝可信门控发布 |',
      '| 151–175 惯性完整性 | 104个600秒模拟案例，正常输出18.0596%；全历元成功8/104；声明域内正常≥30/50m为0 | 以大量不可用换取条件保护，离长期连续导航目标很远 |',
      '', '导航纠错后重新冻结 7 组新 RF（86101–86107）。baseline 是上一批 joint 导航；197 是本批条件可用性比较器；200 corrected 是弧段纠错与矛盾记忆方案。197 保留旧弧段缺陷，仅供比较，不能发布。',
      '', '| RF 方案 | 20m内 / 总历元 | 有输出 | ≥30m | ≥50m |','|---|---|---|---|---|']
    for name,v in rf.items():lines.append(f"| {name} | {v['good20']}/{v['opportunities']} | {v['solved']} | {v['error30']} | {v['error50']} |")
    lines+=['', '| RF 场景 | baseline 20m内 | 197 20m内 | 200 corrected 20m内 | 200最大输出误差(m) |','|---|---|---|---|---|']
    for row in data['rows']:
        if row['domain']!='RF-derived':continue
        s=row['scores'];v=s['200_corrected'];maximum='无输出' if v['max_output_m'] is None else f"{v['max_output_m']:.3f}"
        lines.append(f"| {row['kind']} ({row['seed']}) | {s['baseline']['good20']}/{v['opportunities']} | {s['197']['good20']}/{v['opportunities']} | {v['good20']}/{v['opportunities']} | {maximum} |")
    lines+=['','**不能将上表外推为抗欺骗认证或长期定位保证。** RF仍依赖标称天空、姿态、星历、粗码周期辅助以及capture路线旧启动假设；最长新RF仅1秒。冷攻击下继承启动假设的结果尤其只是条件性能。没有真实台架数据。',
      '', '另外39组独立观测域反例（87101/17/39×13场景），不是射频生成的攻击检测率：','',
      '| 方案 | 20m内 / 总历元 | ≥30m | ≥50m |','|---|---|---|---|']
    for name,v in obs.items():lines.append(f"| {name} | {v['good20']}/{v['opportunities']} | {v['error30']} | {v['error50']} |")
    lines+=['','共同几何码偏差、共同牵引仍可使输出一致但错误。观测域设定保留正向质量指标，用来证明下游估计器的不可辨识性；不声称单个真实RF发射器必然能制造全部这些指标。',
      '', '## 实际改动与失败','',
      '- 101–125：从移除伪FFT群时延开始，加入联合相关函数拟合、独立噪声白化、源存在性、条件信息量和噪声失配检验。部分两天线源数初始化仍失败，冻结后保留拒绝结果。',
      '- 126–150：稳健相位/增益、跨块相干积累、重捕获与模板更新。相位对齐引入噪声选择偏差，逆增益导致异方差噪声；即使拟合更好也不能放行可信位置。',
      '- 151–175：独立标定、偏置/温漂/姿态/杆臂/时间误差集合、独立位置速度约束与过期/失效状态。600秒是三轴离散平动仿真，未实现完整手机旋转动力学。实际手机无证书输入时正常率0%。',
      '- 176–200：双路线可用性、冲突、融合；参考星差分相关协方差；共同端点载波弧段；原始码更新及鲁棒创新；持续矛盾记忆。取消平均、码域融合、经验协方差下限均有退步，没有自动采用后编号方案。',
      '', '两项实现纠错没有增加迭代计数：175修正延迟锚点历史积分，保留第一次冻结80个正常≥50m和133次漏包；200修正失锁后的相位弧段，保留约126.7m虚假步长回归。两者都重新冻结到新种子，原始快照与失败结果保留。',
      '', '名义协方差与实际观测筛选支持仍不完全匹配，不能输出为保护级；不规则采样和缺失时间区间尚不在当前10ms导航模型域。未知或漏报IMU偏置、错误零速证明、假冒独立源均有严重反例。',
      '', '## 可复现性与工程状态','',
      '[逐轮账本](RESEARCH_NEXT_HUNDRED_LEDGER.md) · [数学与验证方法](RESEARCH_NEXT_HUNDRED_METHODS.md) · [结果目录](results/research-next-hundred)',
      '', '测试、快照重放、哈希和独立复核详见结果目录verification.json、navigation_final_review.json、navigation_replay.json。测试中保留一个严格预期失败，代表未修复的150纯噪声门控问题，不能把它描述为算法已通过抗欺骗验收。',
      '', '尚需真实同步RF/IMU台架、可追溯通道/噪声标定、相位弧段与质量协方差的一致建模，以及攻击下仍可信的独立绝对约束。当前适合继续研究，不能承诺客户位置不被带偏或长时间20米内连续输出。']
    Path('RESEARCH_NEXT_HUNDRED_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':ledger();report()
