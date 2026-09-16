"""Archive actual proposals on identical RF-derived startup observations."""
import hashlib,json,shutil,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import continuous_improvement as engine
from array_acquisition import acquire_array
from presets import default_config
from tracking import track
from cold_next import calibrate

ROOT=Path('results/research-next-hundred')
SOURCE='https://gssc.esa.int/navipedia/index.php/Data_Demodulation_and_Processing'


def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def inputs():
    folder=ROOT/'cold_inputs';folder.mkdir(exist_ok=True)
    for seed,kind,changes in ((43002,'low',dict(cn0=35.)),(43007,'phase',dict(phase_deg=20.)),
                              (43001,'normal',dict(cn0=35.)),(48001,'absent',dict(cn0=35.,genuine=False))):
        case=engine.Case(duration=.1,attack_start=0.,**changes)
        x,a,codes,*_=engine.generate(case,seed,kind!='normal')
        cfg=default_config();cfg.acquisition.doppler_bins_hz=np.arange(-4000.,4001.,250.)
        cfg.tracking.max_epochs=100;cfg.tracking.code_replica_bandwidth_hz=.9e6
        cfg.tracking.refine_initial_phase=True;cfg.tracking.carrier_discriminator='costas'
        locals=[];prns=[];records=[]
        for i in range(case.stars):
            acq=acquire_array(x,engine.FS,i+1,1.023e6,cfg.acquisition)
            records.append(dict(prn=i+1,success=acq['success'],metric=acq['metric']))
            if not acq['success']:continue
            h=track(x[0],engine.FS,i+1,acq,cfg.tracking,1575.42e6,1.023e6)
            if not h['success']:continue
            template=(h['epoch_start_samples'][-1]/engine.FS,h['code_nco_phase_chips'][-1],h['code_nco_rate_hz'][-1],h['carrier_nco_frequency_hz'][-1])
            locals.append(engine.local_signal(codes[i],template,np.arange(x.shape[1])/engine.FS));prns.append(i)
        local=np.array(locals)
        z=np.einsum('mkn,skn->msk',x.reshape(len(x),100,2046),local.conj().reshape(len(local),100,2046))/2046
        rng=np.random.default_rng(seed);g=np.exp(1j*np.radians(case.phase_deg*rng.normal(size=case.antennas)))
        g*=10**(case.amplitude_db*rng.normal(size=case.antennas)/20)
        np.savez_compressed(folder/f'{kind}.npz',z=z,a=a[:,prns],truth_gain=g,prns=prns)
        save(folder/f'{kind}.json',dict(case=asdict(case),seed=seed,records=records))
        print('input',kind,z.shape,flush=True)


def score(g,info,data):
    error=np.degrees(np.angle(g/data['truth_gain']));error-=error[0]
    error=np.degrees(np.angle(np.exp(1j*np.radians(error))))
    return dict(phase_rmse_deg=float(np.sqrt(np.mean(error**2))),usable=bool(info['usable']),diagnostics=info)


def run(number,parent,hypothesis):
    folder=ROOT/f'iteration{number:03d}';folder.mkdir()
    save(folder/'proposal.json',dict(number=number,parent=parent,hypothesis=hypothesis,source=SOURCE,
        layer='RF-derived tracked prompts, not full receiver output',truth='gain used only for scoring',bits='unknown'))
    for name in ('cold_next.py','cold_next_research.py'):shutil.copy2(name,folder/name)
    rows=[]
    for path in sorted((ROOT/'cold_inputs').glob('*.npz')):
        with np.load(path) as data:
            before=score(*calibrate(data['z'],data['a'],parent),data)
            after=score(*calibrate(data['z'],data['a'],number),data)
        rows.append(dict(kind=path.stem,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parent=before,proposal=after))
    save(folder/'results.json',dict(rows=rows))
    delta=sum(r['proposal']['phase_rmse_deg']-r['parent']['phase_rmse_deg'] for r in rows if r['kind']!='absent')/3
    false_support=any(r['kind']=='absent' and r['proposal']['usable'] for r in rows)
    decision='retain for comparison' if delta<-1e-6 and not false_support else 'reject as replacement'
    save(folder/'conclusion.json',dict(decision=decision,mean_phase_delta_deg=delta,
        evidence='Same input parent comparison; usability is algebraic, not authenticity.',next_step='Investigate low-CN0 residual and no-auth controls.'))
    print(number,decision,round(delta,4),[(r['kind'],round(r['proposal']['phase_rmse_deg'],2),r['proposal']['usable']) for r in rows],flush=True)


if __name__=='__main__':
    if sys.argv[1]=='inputs':inputs()
    else:run(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
