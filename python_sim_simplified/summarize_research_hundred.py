"""Build the final ledger from completed evidence, never planned iteration counts."""
import json,hashlib
from pathlib import Path
import numpy as np
from research_hundred import ROOT
from audit_recovery_fifty import save


TITLES=['启动证据来源约束','连续锚点速度拟合','以载波运动运输锚点','同弧非连续锚缓存','载波失效时绝对锚输出',
    '空间—延迟双峰模型','非线性延迟与线性幅度消元','单峰/双峰模型竞争','跨星共同增益标签选择','剩余公共源一致性',
    '幅度协方差标签加权','仅最终增益加权','包含延迟的完整局部协方差','天空/公共源模型竞争','带限BPSK模板',
    '相关噪声白化','未知带宽拟合','共享带宽重拟合','带宽轮廓似然','通道幅度形状对齐',
    '跨通道频谱相位斜率','分离公共源后估计群延迟','群延迟进入前向模板','全部通道对的延迟图拟合',
    '多天线非相干捕获','跨毫秒差分捕获','带限本地捕获码','自归一化差分检测','频率行背景归一化',
    '短时差分频率细化','长时差分相位解模糊','平方提示消除数据符号','二次相位动态频率','全局啁啾似然初始化',
    '静态/动态相位模型竞争','阵列捕获接入连续RF/PVT','捕获与联合启动标定集成','动态频率参考时刻运输',
    '逐星天空/公共源支持','增加纯噪声零假设']


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def summary():
    ledger=[]
    for n in range(1,101):
        folder=ROOT/f'iteration{n:03d}'
        proposal=read(folder/'proposal.json');conclusion=read(folder/'conclusion.json')
        result=read(folder/'results.json')
        ledger.append(dict(number=n,title=TITLES[n-1] if n<=40 else proposal.get('title',proposal['hypothesis']),
            proposal=proposal,conclusion=conclusion,result_path=str(folder/'results.json'),
            source_files=[str(p) for p in folder.rglob('*.py')]))
    save(ROOT/'ledger.json',dict(actual_completed_iterations=len(ledger),iterations=ledger))
    lines=['# 100次逐轮索引','',
           '每行链接到实际已执行的提案、源码、结果与结论。种子、内部求解步骤、纠错和最终验证不增加计数。','']
    for row in ledger:
        n=row['number'];lines.append(f"{n:03d}. [{row['title']}](results/research-hundred/iteration{n:03d}/proposal.json) — [结果](results/research-hundred/iteration{n:03d}/results.json)；[结论](results/research-hundred/iteration{n:03d}/conclusion.json)")
    Path('RESEARCH_HUNDRED_LEDGER.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    rf=read(ROOT/'rf_validation.json')['rows'];lag=read(ROOT/'lag_validation.json')['rows']
    methods=('baseline','navigation','capture','joint')
    totals={m:{k:sum(r['scores'][m][k] for r in rf) for k in ('opportunities','solved','good20','error30','error50')} for m in methods}
    complete={m:sum(r['scores'][m]['good20']==r['scores'][m]['opportunities'] for r in rf) for m in methods}
    save(ROOT/'final_totals.json',dict(rf_totals=totals,rf_complete_cases=complete,
        rf_records=len(rf),lag_records=len(lag),actual_iterations=len(ledger)))
    lines=['# 100次实际研究迭代结果','',
      '完成100个算法/模型研究闭环，仍未实现手机在攻击后长期保持数米定位。新增方案改善了低信噪比捕获、部分连续导航、源选择和条件误差边界；低信噪比冷攻击、弱分离延迟、真实硬件误差包络仍未解决。','',
      '本批为40次主线、20次矩阵标定、20次测向、20次惯性完整性研究。不是100种场景，也不是100次参数扫描。失败方案及纠错均保留，纠错不另计次数。详见[逐轮100项索引](RESEARCH_HUNDRED_LEDGER.md)与[方法、输入前提及限制](RESEARCH_HUNDRED_METHODS.md)。','',
      '## 冻结后连续RF留出','',
      '8个未用于选择的种子43001–43008，包含0.6s和2s采样RF。所有位置比值按全部评分历元计，停止输出仍计未成功。基线是前批20轮导航，nav001额外阻止重复启动证据；capture使用新100ms捕获；joint进一步使用测量型联合标定。后三者仍依赖理想姿态/星历/粗距离辅助。','',
      '| 工况 | 基线 ≤20m/全部 | nav001 | capture | joint | joint最大输出误差m |',
      '|---|---:|---:|---:|---:|---:|']
    for r in rf:
        cells=[f"{r['scores'][m]['good20']}/{r['scores'][m]['opportunities']}" for m in methods]
        maximum=r['scores']['joint']['max_output_m'];maximum='无输出' if maximum is None else f'{maximum:.3f}'
        lines.append('| '+r['kind']+' | '+' | '.join(cells)+' | '+maximum+' |')
    lines+=['','| 方案 | ≤20m/全部历元 | 输出历元 | ≥30m | ≥50m | 全程≤20m的记录 |','|---|---:|---:|---:|---:|---:|']
    for m,t in totals.items():lines.append(f"| {m} | {t['good20']}/{t['opportunities']} | {t['solved']} | {t['error30']} | {t['error50']} | {complete[m]}/8 |")
    lines+=['','这些是整个接收流程的差值：基线4ms，新捕获100ms，且替换了原始启动与波束后捕获。026另有同100ms普通阵列非相干对照，不能把整表收益全部归因于天线数。2s RF也不是长期导航验收。','',
      '## 主线得到的具体改进','',
      '- 001：在12条旧RF派生观测中，攻击≥30m从6历元降至0，但正常有效历元1884→1881；这是排除重复证据，不是证明认证。','- 026：CN0=30的开发RF，100ms差分阵列捕获8/8；同100ms普通非相干0/8。该轮20次缺席PRN搜索均未触发，但不等于确定虚警率为零。','- 027：匹配带限本地码，所测非攻击PRN粗码相位RMSE0.452→0.046chip。','- 032：平方提示消除未知数据符号，CN0=30频率RMSE3.15→0.505Hz；035支持动态模型，但冷攻击一颗发生多余动态拟合。','- 038：修正1ms积分中心与缓冲区起点的时标偏差，无噪声±500Hz/s例的约0.25Hz偏差降至0.00005Hz以内。','- 039：两天线一个错误码峰被逐星模型竞争剔除，保留7/8颗，该记录所选码距RMSE79.47→2.10m。不是PVT误差。','- 040：显式加入纯噪声零假设，开发纯噪声可用判定1/4→0/4。它仍是条件BIC启发式，未给出硬件虚警率或完整性保证。','',
      '## 空间—延迟独立留出','',
      '最终源选择代码冻结后使用32051/32052，每类2个相关域样本。以下“可用”只是该相关模型支持，不是可输出可信位置。','',
      '| 工况 | 模型可用 | 保留PRN | 保留单星最大码距误差m |','|---|---:|---:|---:|']
    for kind in dict.fromkeys(r['kind'] for r in lag):
        subset=[r for r in lag if r['kind']==kind]
        errors=[abs(e) for r in subset if r['code_error_m'] is not None for e,a in zip(r['code_error_m'],r['supported_prns']) if a]
        maximum=f'{max(errors):.3f}' if errors else '不适用'
        lines.append(f"| {kind} | {sum(r['usable'] for r in subset)}/2 | {sum(sum(r['supported_prns']) for r in subset)}/16 | {maximum} |")
    lines+=['','冻结留出正常种子32051失败：通道相位RMSE26.97°，一个被保留的单星码距误差31.58m，虚假通道延迟约0.0977chip。因此该空间—延迟方案不应作为可信标定/测距模块放行，也未据留出结果重新调参。开发已发现：重叠延迟仍有25.42m单星码距误差；0.06chip末通道偏差只估出约0.025–0.028chip，正常场景还可能估出虚假偏差。准确相位不能代替准确群延迟或码距。单峰历史延迟第二项可能是旧双峰残留，选取接口已按单双峰分开处理。','',
      '## 三个独立研究分支','',
      '- [矩阵041–060](matrix_hundred_report.md)：盲全矩阵即使零残差仍不可辨识；独立多端口注入及相干参考监测有条件有效。手机几何与方向采样错误已纠正并重跑，更换新种子；部分旧强结论已撤回。','- [测向061–080](doa_hundred_report.md)：保留两天线连续多解、平面镜像与未知通道相位歧义。带模拟多轴姿态的新种子角误差0.198–0.276°，声明姿态噪声模型下95%集合覆盖94.0–96.6%；没有实测测向验收。','- [惯性081–100](integrity_hundred_inertial_report.md)：84个120s留出/无GNSS场景，正常输出31.24%，全程≤20m仅8/84。声明误差界内正常≥30/50m和漏包为0，但误差支持直接基于模拟大小，不能当作真实手机保证。未声明加速度偏差反例仍会正常输出110.5m错误。','',
      '## 验证和复现','',
      '完整测试298项通过，保留1条既有无噪声用例的tracking除零警告，见[完整测试日志](hundred_all_tests.log)。001–005共60条输入/方案位置和分数独立复现；矩阵、测向、惯性各20轮已重放。006–040均实际执行过，但没有宣称全部重新独立回放；主线最终方案另做冻结留出。共享依赖哈希记录于冻结清单，源码快照不是单独锁定的运行环境。','',
      '旧20轮748份、旧200编号归档1343份文件哈希再次全部匹配。总计100个proposal/results/conclusion具备源码快照。最终证据见results/research-hundred/manifest.json、ledger.json、rf_freeze.json、rf_validation.json、lag_validation.json及各分支验证文件。','',
      '```powershell',"$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'",'D:\\Python\\Python311\\python.exe -m pytest tests -q','D:\\Python\\Python311\\python.exe summarize_research_hundred.py','```','',
      '重新执行实验应在副本中使用逐轮源码与相应生成器，避免覆盖已归档证据；validate_research_hundred.py记录了最终RF/相关域验证流程。全部新增工作位于python_sim_simplified，未提交或推送。','',
      '下一步优先针对仍失败的低CN0冷攻击做捕获后真伪峰分离、群延迟可辨识性和持续模板更新；同时准备独立标定输入及RF/IMU台架。当前证据不支持承诺攻击后长期数米或商业完整性指标。']
    Path('RESEARCH_HUNDRED_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def manifest():
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
    save(ROOT/'manifest.json',dict(actual_iterations=100,sha256=files))
    print('archived hashes',len(files))


if __name__=='__main__':summary();manifest()
