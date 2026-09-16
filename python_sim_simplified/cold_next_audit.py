"""Evidence inventory and deterministic prompt-domain reproduction, no iteration count."""
import hashlib,json
from pathlib import Path
import numpy as np
from cold_next import calibrate
from cold_next_research import ROOT,save,score


def main():
    frozen=json.loads((ROOT/'cold_freeze.json').read_text(encoding='utf-8'))
    for name,digest in frozen['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    checks=[];ledger=[]
    for number in range(126,151):
        folder=ROOT/f'iteration{number:03d}'
        proposal=json.loads((folder/'proposal.json').read_text(encoding='utf-8'))
        results=json.loads((folder/'results.json').read_text(encoding='utf-8'))
        assert (folder/'conclusion.json').exists()
        ledger.append(dict(number=number,hypothesis=proposal['hypothesis'],layer=proposal['layer'],
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.py')}))
        if number<=140:
            for row in results['rows']:
                with np.load(ROOT/'cold_inputs'/f"{row['kind']}.npz") as data:
                    reproduced=score(*calibrate(data['z'],data['a'],number),data)
                np.testing.assert_allclose(reproduced['phase_rmse_deg'],row['proposal']['phase_rmse_deg'],atol=1e-9)
                assert reproduced['usable']==row['proposal']['usable']
                checks.append(dict(number=number,kind=row['kind'],passed=True))
    save(ROOT/'cold_ledger.json',ledger)
    save(ROOT/'cold_reproduction.json',dict(prompt_checks=checks,rf='Executed same-seeded comparisons and independent frozen RF holdouts; not all RF rounds reexecuted.'))
    validation=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'cold_validation').glob('*_result.json'))]
    save(ROOT/'cold_validation.json',dict(rows=validation))
    noise=[]
    for row in validation:
        with np.load(ROOT/'cold_validation'/f"{row['seed']}_proposal.npz") as data:
            diagonal=abs(data['startup_phase_estimate'])**2
        noise.append(dict(seed=row['seed'],kind=row['kind'],relative_noise_variance=diagonal.tolist(),
            condition_number=float(np.max(diagonal)/np.min(diagonal)),
            noise_only_eigen_ratio=float(np.max(diagonal)/((sum(diagonal)-np.max(diagonal))/(len(diagonal)-1)))))
    save(ROOT/'cold_noise_covariance.json',dict(assumption='input IID channel noise; inverse gain transforms noise into a diagonal nonwhite covariance',
        limitation='source projector, spatial_evidence and geometric gain gates are not whitened; signal gain inversion is not complete hardware calibration',rows=noise))
    report=Path('cold_next_report.md');body=report.read_text(encoding='utf-8').split('<!-- FROZEN RESULTS -->')[0]
    table=['<!-- FROZEN RESULTS -->','\n## 冻结后实测汇总（自动读取证据）\n',
        '|工况|125导航≤20m/全部|150导航≤20m/全部|125码定位≤20m/全部|150码定位≤20m/全部|150码定位最大误差m|',
        '|---|---|---|---|---|---|']
    for row in validation:
        fields=[row['kind']]
        for label,metric in [('baseline','navigation'),('proposal','navigation'),('baseline','calibrated_spp'),('proposal','calibrated_spp')]:
            value=row[label][metric];fields.append(f"{value['good20']}/{value['opportunities']}")
        maximum=row['proposal']['calibrated_spp']['max_output_m'];fields.append('无输出' if maximum is None else f'{maximum:.3f}')
        table.append('|'+'|'.join(fields)+'|')
    table.append('\n该表使用原导航候选，150 `usable` 的已知统计缺陷并未被修复；导航输出不因此得到认证。')
    table.extend(['\n|工况|逆增益后通道噪声方差条件数|纯噪声最大/其余平均特征值比|','|---|---|---|'])
    table.extend(f"|{r['kind']}|{r['condition_number']:.3f}|{r['noise_only_eigen_ratio']:.3f}|" for r in noise)
    report.write_text(body+'\n'.join(table)+'\n',encoding='utf-8')
    print(f'{len(ledger)} archived closures; {len(checks)} prompt replays; {len(validation)} frozen RF cases')


if __name__=='__main__':main()
