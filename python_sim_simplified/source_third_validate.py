"""Frozen independent receiver-observation validation and historical replay."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import beta
from source_third_research import ROOT,KINDS,observation,load,save
from source_third import detect

def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]

def freeze():
    paths=['source_third.py','source_third_research.py','source_third_validate.py','presets.py','signal_recovery.py']
    save(ROOT/'source_freeze.json',dict(revision=250,files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},numpy=np.__version__,scipy=scipy.__version__,cases=KINDS,seeds='92601+1000*case_index+repeat_index; 100 repeats each',claim='Prompt-model source support only, no RF/PVT certification',predeclared=True))

def validate():
    manifest=json.loads((ROOT/'source_freeze.json').read_text())
    for p,h in manifest['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    rows=[];summary=[]
    present={'weak','strong','gain','mixed','near_common'}
    for c,kind in enumerate(KINDS):
        count=0
        for i in range(100):
            seed=92601+1000*c+i;z,a,noise=observation(seed,kind);info=detect(z,a,noise)
            count+=int(info['usable']);rows.append(dict(case=kind,seed=seed,input_sha256=hashlib.sha256(z.tobytes()+a.tobytes()+noise.tobytes()).hexdigest(),diagnostics=info))
        item=dict(case=kind,accepted=count,total=100,rate=count/100,clopper_pearson_95=interval(count,100),meaning='detection' if kind in present else 'false persistent-four-source admission')
        summary.append(item);print(item,flush=True)
    save(ROOT/'source_validation.json',dict(rows=rows,summary=summary,freeze_sha256=hashlib.sha256((ROOT/'source_freeze.json').read_bytes()).hexdigest()))

def replay():
    checks=[]
    for revision in range(226,251):
        folder=ROOT/f'iteration{revision:03d}';data=json.loads((folder/'results.json').read_text());after=load(folder/'after.py');before=load(folder/'before.py')
        proposal=json.loads((folder/'proposal.json').read_text())
        for i,row in enumerate(data['rows']):
            z,a,noise=observation(82600+i,row['case'])
            assert hashlib.sha256(z.tobytes()+a.tobytes()+noise.tobytes()).hexdigest()==row['input_sha256']
            for side,model in [('before',before),('after',after)]:
                info=model.complex_calibrate(z,a)[1] if side=='before' and proposal['parent']==150 else model.detect(z,a,noise)
                assert info['usable']==row[side]['usable']
                for key in ('p','statistic','four_source_p','residual_noise_p','innovation_p','local_residual_p','motion_bic'):
                    if key in info:np.testing.assert_allclose(info[key],row[side][key],rtol=1e-10,atol=1e-12)
                checks.append(dict(iteration=revision,case=row['case'],side=side,matched=True))
    save(ROOT/'source_replay.json',dict(checks=checks,matched=len(checks),result='all receiver inputs and stored decisions/numerics reproduced'))
    print('Replay matched',len(checks))

if __name__=='__main__':{'freeze':freeze,'validate':validate,'replay':replay}[sys.argv[1]]()
