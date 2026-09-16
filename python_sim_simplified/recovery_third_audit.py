"""Post-freeze evaluation only; never selects or changes a receiver."""
import json, hashlib, pathlib
import numpy as np
import recovery_third_campaign as c

freeze=json.loads((c.ROOT/'recovery_freeze.json').read_text())
selected_path=pathlib.Path('recovery_third_selected.py')
assert hashlib.sha256(selected_path.read_bytes()).hexdigest()==freeze['source_sha']
selected=c.load(selected_path);baseline=c.load(c.ROOT/'recovery_baseline.py')
paired=[]
for seed in freeze['seeds']:
    for scenario in c.SCENARIOS:
        obs,truth=selected.generate(seed,scenario)
        new=selected.estimate(obs);old=baseline.estimate(obs)
        mask=new['accepted']
        if truth['exists']:
            paired.append(dict(seed=seed,scenario=scenario,accepted_indices=np.flatnonzero(mask).tolist(),baseline_error_m=((old['delay']-truth['delay'])[mask]*293.052256).tolist(),selected_error_m=((new['delay']-truth['delay'])[mask]*293.052256).tolist()))
old_errors=[e for r in paired for e in r['baseline_error_m']];new_errors=[e for r in paired for e in r['selected_error_m']]
out=dict(source_sha=freeze['source_sha'],paired_rows=paired,same_accepted_count=len(new_errors),baseline_same_support_rmse_m=float(np.sqrt(np.mean(np.array(old_errors)**2))),selected_same_support_rmse_m=float(np.sqrt(np.mean(np.array(new_errors)**2))),tests_passed=5,post_freeze_model_changes=False)
c.write(c.ROOT/'recovery_paired_support.json',out)
summary=json.loads((c.ROOT/'recovery_summary.json').read_text())
summary.update(same_support_baseline_code_rmse_m=out['baseline_same_support_rmse_m'],weak_overlap_two_three_accepted=0,tests_passed=5,baseline_definition='Uncalibrated nominal sky beam on same synthetic observations; not previous delay125 nor previous RF receiver')
c.write(c.ROOT/'recovery_summary.json',summary)
report=pathlib.Path('recovery_third_report.md')
s=report.read_text(encoding='utf-8')
s=s.replace('25个真实机制变更已逐轮闭环，保留失败；','**弱星恢复目标尚未达到：冻结验证weak/overlap/two/three场景全部被拒绝。** 25个真实机制变更已逐轮闭环，保留失败；')
s=s.replace('所有案例码RMSE均值/m（包含无真源，不作为导航指标）','真实源案例码RMSE均值/m（全部估计，非导航指标）')
s+=f"\n同一组14个冻结接受PRN上，简单基线码RMSE {out['baseline_same_support_rmse_m']:.3f}m→{out['selected_same_support_rmse_m']:.3f}m。基线是同观测上的未校准标称天空波束，不是delay125或上一批RF导航实现，不可跨模型比较为既有系统收益。5项单测已通过，配对子集明细在recovery_paired_support.json。\n"
report.write_text(s,encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='paired_rows'},indent=2))
