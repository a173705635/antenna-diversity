"""Evidence-first report and 100-row ledger, with disjoint validation domains."""
import json
from pathlib import Path
ROOT=Path('results/research-third-hundred')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def ledger():
    lines=['# 第201–300轮逐轮账本','','每轮均有实际模型/算法修改、父模型同输入结果与结论。种子、优化器内部步、测试、回放及实现纠错不计轮数。','',
           '| 轮次 | 假设 / 实际改动 | 结果与结论 |','|---|---|---|']
    for n in range(201,301):
        folder=ROOT/f'iteration{n}';proposal=read(folder/'proposal.json')
        h=(proposal['mechanism']+': '+proposal['rationale'] if 226<=n<=250 or n>=276 else proposal['hypothesis'])
        h=h.replace('\n',' ').replace('|','/')
        assert (folder/'results.json').is_file() and (folder/'conclusion.json').is_file()
        lines.append(f'| {n} | {h} | [证据]({folder.as_posix()}/conclusion.json) |')
    Path('RESEARCH_THIRD_HUNDRED_LEDGER.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def report():
    data=read(ROOT/'navigation_validation.json');rf=data['totals']['RF-derived'];obs=data['totals']['observation']
    lines=['# 第三批100轮研究：201–300','',
      '**100轮已闭环，仍未实现手机攻击下长期可信定位，本批导航方案不替换现有方案。** 新RF中20米内历元从172/204降至154/204；独立观测反例中≥50米错误从baseline的770增加到217的1003、225的990。有进展的是分层源存在性检验；各分支尚未组成一套端到端验证的新接收机。',
      '', '| 研究线 | 最终独立验证 | 未解决的问题 |','|---|---|---|',
      '| 226–250 源存在性 | 16类×100=1600条提示记录；11类无持续四源条件各0/100误放行；弱源95/100、强源97/100、幅度失配64/100、混合70/100、近公共方向86/100检出 | 单类0/100的双侧95%上界仍3.62%；有限噪声/AR/公共方向估计与训练边界耦合；未接RF/PVT |',
      '| 251–275 真信号恢复 | 33个相关器案例；真实PRN仅14/162接受；同接受集合码RMSE110.085→8.853m、最大18.717m；无真/纯噪0/36误接受 | 弱星、重合源、2/3天线等全部拒绝，恢复目标未解决；基线是未校准sky beam，不是上批RF导航 |',
      '| 276–300 IMU转动相位 | 补充预冻结100窗，方向可用48窗；中位0.06744°、最大0.22435°；共源未检出20/92 | 窄角多源4/4误报，合并多源4/12；绝对位置不可用；条件方向精度不能替代分类可靠性 |',
      '', '运动原验证缺少可独立核验的预冻结记录，未据此宣称完成严格留出。保留旧工件后，先写motion_second_freeze.json再用97621/27/39/43新种子运行；97631因已用于测试而排除。模型未改，这不是新增研究轮。',
      '', '## 导航最终RF验证','',
      '101001–101006六组新RF，baseline200为上批已修正失锁弧段的导航。217是本批鲁棒联合GLS参考；225是未在开发上改善的协方差交集提案，同时预登记验证。没有根据留出结果重调阈值。',
      '', '| 方案 | 20m内 / 总历元 | 有输出 | ≥30m | ≥50m |','|---|---|---|---|---|']
    for name,v in rf.items():lines.append(f"| {name} | {v['good20']}/{v['opportunities']} | {v['solved']} | {v['error30']} | {v['error50']} |")
    lines+=['','| RF场景 | baseline200 | 217 | 225 |','|---|---|---|---|']
    for row in data['rows']:
        if row['domain']=='RF-derived':
            scores=row['scores'];values=[f"{scores[n]['good20']}/{scores[n]['opportunities']}" for n in ('baseline200','217','225')]
            lines.append('| '+row['kind']+' | '+' | '.join(values)+' |')
    lines+=['','最长RF仅1秒，保留旧标称天空、姿态、星历、粗码周期及capture启动假设，有100ms启动与20ms块延迟。无真实RF/IMU台架。不能外推为长期商用能力。',
      '', '## 独立观测域反例','',
      '102001/13/37三个种子。共同伪位置、码偏差、牵引、断弧、双故障及有色噪声单独评分，不混入RF成功率。采样缺口另存unobserved_time_s，未观测时间不能算成功。',
      '', '| 方案 | 20m内 / 总历元 | ≥30m | ≥50m |','|---|---|---|---|']
    for name,v in obs.items():lines.append(f"| {name} | {v['good20']}/{v['opportunities']} | {v['error30']} | {v['error50']} |")
    lines+=['','这些观测反例证明下游估计器仍有共同偏差不可辨识性，不声称单个物理RF发射器能生成全部正向质量指标。正常输出少、拒绝多不等于抗欺骗目标实现。',
      '', '## 实际演进与保留的失败','',
      '- 201–209：实际码筛选几何、共享载波端点协方差、弧段/采样缺口契约及单/双周跳假设。单纯增加故障假设不等于有统计保证。',
      '- 210–217：统一路线曾使开发RF160→132且新增3个≥30m输出；恢复capture原始/校正一致性后改善。最终共同起点GLS显式保留跨时相位相关项，鲁棒码权重恢复部分离群场景。',
      '- 218–225：有色噪声、未知相关性、持续偏差、非线性几何、输出交叉协方差和路线协方差交集。多项没有收益；225开发观察域新增一个≥50m输出，未采用为改进模型。',
      '- 226–250：独立训练/检测、噪声参考、公共子空间、源模型竞争、四源与双窗合取、残差失配和移动源选择。开发失败和拒绝真实源的代价均保留。',
      '- 251–275：空间抑制、码profile、增益/噪声标定、两源重构、独立块支持与信息量审计；275显式抵消后残差误差反升，最终选271。',
      '- 276–300：SO(3)陀螺积分、跨PRN差分、时间/偏置/尺度/相位中心及信息秩、漂移与动态方向。共同漂移和方向有严格等价反例。',
      '', '名义协方差假定的噪声分布、相关性和零均值未由真实硬件证明；数据驱动筛选/稳健权重使统计解释进一步受限。协方差交集无法消除两路线共同错误。',
      '', '## 代码、复现和下一步','',
      '研究参考入口：navigation_third_selected.py（217）、source_third.py（250）、recovery_third_selected.py（271）、motion_third.py（300）。navigation_third.py与recovery_third.py保留225/275末轮失败提案，不能按文件名当作发布版。原默认接收机入口未替换。',
      '', '[100轮账本](RESEARCH_THIRD_HUNDRED_LEDGER.md) · [数学与方法](RESEARCH_THIRD_HUNDRED_METHODS.md) · [源检测报告](source_third_report.md) · [恢复报告](recovery_third_report.md) · [运动报告](motion_third_report.md)',
      '', '测试、源代码AST/哈希审计、冻结验证及历史重放见results/research-third-hundred/verification.json、branch_final_audit.json、navigation_replay.json。保留旧150模型的严格预期失败，不表示本批已完成整机安全认证。',
      '', '下一阶段应把通过独立提示检验的源支持真正接入同一RF/PVT链，检查标定和噪声输入在攻击中是否可取得，并用同步台架验证。当前仍不能保证客户位置不被带偏，也不能保证长期20米内连续输出。']
    Path('RESEARCH_THIRD_HUNDRED_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':ledger();report()
