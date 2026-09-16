"""Fixed-receiver paired 8/24-star study and no-attack projection ablation."""
import sys,time,shutil
from pathlib import Path
from dataclasses import asdict
from types import FunctionType
import numpy as np
import single_source_twenty as campaign
from single_source_twenty import save,read,digest,load
from continuous_improvement import Case
from satellite_count_generator import generate
from signal_recovery import sky

ROOT=Path('results/satellite-count-24')
OLD=Path('results/single-source-twenty')


def freeze():
    ROOT.mkdir(parents=True)
    shutil.copy2('single_source_receiver_fixed.py',ROOT/'receiver.py')
    save(ROOT/'plan.json',dict(seeds=[114001,114013,114037],kinds=['normal','near','cluster'],
         stars=24,duration=2.,subsets=['three','four'],source_sha256=digest(ROOT/'receiver.py'),
         generator_sha256=digest(Path('satellite_count_generator.py')),
         design='Keep original eight PRNs, sky, Dopplers, signal/noise level; add sixteen genuine AND sixteen same-source forged PRNs',
         independent_holdout=False,thresholds_changed=False,
         ablation='Normal 114001 only: disable projection, retain calibration/tracking/navigation; diagnostic only'))


def inputs():
    plan=read(ROOT/'plan.json')
    for seed in plan['seeds']:
        for kind in plan['kinds']:
            folder=ROOT/'inputs'/f'{seed}_{kind}_24';folder.mkdir(parents=True)
            case=Case(stars=24,duration=2.,cn0=45.,phase_deg=10.,amplitude_db=1.,velocity_east=2.,attack_start=.3,
                      angle=60. if kind=='normal' else 0.)
            generator=generate
            if kind=='cluster':
                def clustered(seed,count):
                    u=sky(seed,count);tangent=np.cross(u[0],[0.,0.,1.]);tangent/=np.linalg.norm(tangent)
                    for i,angle in ((1,.5),(2,-.5)):u[i]=u[0]*np.cos(np.radians(angle))+tangent*np.sin(np.radians(angle))
                    return u
                generator=FunctionType(generate.__code__,dict(generate.__globals__,sky=clustered))
            x,a,codes,times,satellites,radii=generator(case,seed,kind!='normal')
            old=OLD/'inputs'/f'{seed}_{kind}_2.0'
            with np.load(old/'observations.npz') as z:
                np.testing.assert_array_equal(a[:,:8],z['nominal'])
                np.testing.assert_array_equal(satellites[:,:8],z['satellites'])
                np.testing.assert_array_equal(np.array(codes)[:8],z['codes'])
            np.save(folder/'samples.npy',x)
            np.savez(folder/'observations.npz',nominal=a,codes=np.array(codes),times=times,satellites=satellites,
                     coarse=np.round((radii+75000)/10000)*10000)
            save(folder/'metadata.json',dict(seed=seed,kind=kind,case=asdict(case),attacked=kind!='normal',
                 samples_sha256=digest(folder/'samples.npy'),observations_sha256=digest(folder/'observations.npz'),
                 physical_spoof_emitters=int(kind!='normal'),paired_eight=str(old)))
            print('created',seed,kind,flush=True)


def run(worker):
    campaign.ROOT=ROOT;plan=read(ROOT/'plan.json')
    assert digest(ROOT/'receiver.py')==plan['source_sha256']
    jobs=[(seed,kind,s) for seed in plan['seeds'] for kind in plan['kinds'] for s in plan['subsets']]
    for seed,kind,subset in jobs[worker::4]:
        folder=ROOT/'inputs'/f'{seed}_{kind}_24'
        while not (folder/'metadata.json').exists():time.sleep(1)
        after,identity=campaign.evaluate(ROOT/'receiver.py',folder,subset)
        before=next(r for r in read(OLD/f'corrective_{subset}.json') if r['case']==dict(seed=seed,kind=kind,duration=2.))
        row=dict(seed=seed,kind=kind,subset=subset,before=before['score'],before_id=before['identity'],after=after,after_id=identity)
        save(ROOT/f'{seed}_{kind}_{subset}.json',row)
        print(seed,kind,subset,{k:after[k] for k in ('solved','good20','error30','error50')},flush=True)


def ablation():
    source=(ROOT/'receiver.py').read_text(encoding='utf-8')
    source=source.replace('return projected/(response+.01)[None, :], response/len(a), ratio','return a/len(a), np.ones(a.shape[1]), ratio')
    path=ROOT/'no_projection.py';path.write_text(source,encoding='utf-8');campaign.ROOT=ROOT
    rows=[]
    for s in ('three','four'):
        score,identity=campaign.evaluate(path,OLD/'inputs/114001_normal_2.0',s)
        rows.append(dict(subset=s,score=score,identity=identity));print(s,{k:score[k] for k in ('solved','good20','error30','error50')},flush=True)
        save(ROOT/'normal_ablation.json',rows)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='freeze':freeze()
    elif mode=='inputs':inputs()
    elif mode=='ablation':ablation()
    else:run(int(mode))
