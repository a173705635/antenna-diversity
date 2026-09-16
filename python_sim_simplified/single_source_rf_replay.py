"""Raw-IQ replay and prefix causality across loss and actual reacquisition."""
import numpy as np
from single_source_twenty import ROOT,read,save,load,input_record


model=load(ROOT/'selected.py')
record=input_record(ROOT/'inputs/113001_return_2.0')
full=model.frontend(record,[0,1,2])
row=next(r for r in read(ROOT/'round020/results.json')['rows'] if r['kind']=='return' and r['subset']=='three')
with np.load(ROOT/'evaluations'/row['after_id']/'tracking.npz') as stored:
    for field in stored.files:np.testing.assert_allclose(full[field],stored[field],rtol=0,atol=1e-9,equal_nan=True)
comparisons=[]
for cutoff in (.9,1.3):
    partial=dict(record)
    partial['samples']=record['samples'][:,:round(cutoff*2.046e6)]
    mask=record['times']+.02<cutoff
    partial['times']=record['times'][mask];partial['satellites']=record['satellites'][mask]
    early=model.frontend(partial,[0,1,2])
    for field in ('ranges','carrier','lock','cn0','gain','support','arc'):
        np.testing.assert_allclose(early[field],full[field][mask],rtol=0,atol=1e-9,equal_nan=True)
    np.testing.assert_allclose(model.navigate(early)['position'],model.navigate(full)['position'][mask],rtol=0,atol=1e-9,equal_nan=True)
    comparisons.append(dict(cutoff_s=cutoff,compared_epochs=int(mask.sum())))
    print(comparisons[-1],flush=True)
save(ROOT/'rf_replay.json',dict(raw_record='113001_return_2.0',subset='three',full_replay_matches=True,
     prefix_checks=comparisons,checks='range/carrier/lock/CN0/gain/spatial support/arc/PVT'))
