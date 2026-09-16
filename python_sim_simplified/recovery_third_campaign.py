"""Run ONE real source revision at a time; no parameter/switch sweep."""
import hashlib, importlib.util, json, pathlib, sys, shutil
import numpy as np

ROOT = pathlib.Path('results/research-third-hundred')
SCENARIOS = ['weak','overlap','two','three','colored','bandwidth','multipath','drift','cal_outlier','absent','noise']

def load(path):
    spec = importlib.util.spec_from_file_location('candidate',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def evaluate(model,seeds):
    rows=[]
    for seed in seeds:
        for scenario in SCENARIOS:
            obs,truth=model.generate(seed,scenario)
            result=model.estimate(obs)
            accepted=result['accepted']
            error=abs(result['wave']-truth['wave'])**2
            norm=np.sum(abs(truth['wave'])**2)
            rows.append(dict(seed=seed,scenario=scenario,input_sha=hashlib.sha256(obs['z'].tobytes()+obs['noise'].tobytes()+obs['calibration'].tobytes()).hexdigest(),
                mse=float(error.mean()),nmse=float(error.sum()/norm) if truth['exists'] else None,
                code_rmse_m=float(np.sqrt(np.mean((result['delay']-truth['delay'])**2))*293.052256),
                accepted=int(accepted.sum()),accepted_code_squared_m=[float(v) for v in ((result['delay']-truth['delay'])*293.052256)[accepted]**2] if truth['exists'] else [],
                recovery_power=float(np.mean(abs(result['wave'])**2)),genuine=truth['exists']))
    present=[r for r in rows if r['genuine']]; absent=[r for r in rows if not r['genuine']]
    return dict(rows=rows,mean_nmse=float(np.mean([r['nmse'] for r in present])),false_accepts=sum(r['accepted'] for r in absent),accepted=sum(r['accepted'] for r in present),
                mean_code_rmse_m=float(np.mean([r['code_rmse_m'] for r in present])))

def run(number,hypothesis):
    folder=ROOT/f'iteration{number}';folder.mkdir(parents=True,exist_ok=True)
    src=pathlib.Path('recovery_third.py'); previous=ROOT/f'iteration{number-1}'/'source.py' if number>251 else ROOT/'recovery_baseline.py'
    shutil.copyfile(src,folder/'source.py')
    shutil.copyfile(__file__,folder/'campaign.py')
    write(folder/'proposal.json',dict(iteration=number,hypothesis=hypothesis,comparison='same seed, scenario and observation bytes',development_seeds=[94101],model_scope='assisted correlator recovery; not PVT',previous_source_sha=hashlib.sha256(previous.read_bytes()).hexdigest(),source_sha=hashlib.sha256(src.read_bytes()).hexdigest()))
    before=evaluate(load(previous),[94101]);after=evaluate(load(src),[94101])
    assert [r['input_sha'] for r in before['rows']]==[r['input_sha'] for r in after['rows']]
    write(folder/'results.json',dict(before=before,after=after))
    conclusion=dict(iteration=number,hypothesis=hypothesis,delta_nmse=after['mean_nmse']-before['mean_nmse'],delta_false_accepts=after['false_accepts']-before['false_accepts'],delta_accepted=after['accepted']-before['accepted'],interpretation='Recovery error improved' if after['mean_nmse']<before['mean_nmse'] else 'No recovery error improvement; retained failed experiment',selection='Cumulative experimental lineage; final selection uses development only, no automatic adoption')
    write(folder/'conclusion.json',conclusion)
    print(json.dumps(conclusion)); print(json.dumps({k:v for k,v in after.items() if k!='rows'}))

if __name__=='__main__':run(int(sys.argv[1]),sys.argv[2])
