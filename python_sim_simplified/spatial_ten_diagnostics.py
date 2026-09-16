"""Replay prompt-domain rounds independently of final navigation validation."""
import sys
import numpy as np
from spatial_ten_study import ROOT,OLD,read,save,load,input_record,campaign
from signal_recovery import FS
from doppler_recovery import prompt_projector


def replay(number):
    rows=[]
    for path in sorted((ROOT/'prompts').glob('*.npz')):
        z=np.load(path);a=z['desired'];post=z['time']>=.3
        if number in (2,3):
            model=load(ROOT/f'round{number:02d}/model.py');scores=[]
            stem,subset=path.stem.rsplit('_',1);record=input_record(OLD/'inputs'/stem)
            for t,p in zip(z['time'],z['prompts']):
                noise=1.
                if number==3:
                    x=record['samples'][campaign.SUBSETS[subset],round(t*FS):round((t+.02)*FS)]
                    noise=np.mean(abs(x)**2)/2046
                _,q=model.residual_trigger(p.reshape(len(a),-1),np.repeat(a,p.shape[-1],axis=1),noise)
                scores.append(q)
            q=np.array(scores)
            row=dict(input=path.name,pre_trigger=int((q[~post]>=3).sum()),post_trigger=int((q[post]>=3).sum()))
            if number==2:
                row.update(pre_count=int((~post).sum()),post_count=int(post.sum()),range=[float(q.min()),float(q.max())])
            else:
                row.update(normal_range=[float(q[~post].min()),float(q[~post].max())],post_range=[float(q[post].min()),float(q[post].max())])
        else:
            differences=[];splits=[]
            for p in z['prompts'][post]:
                desired=np.repeat(a,p.shape[-1],axis=1);flat=p.reshape(len(a),-1)
                p2,_=prompt_projector(flat,desired,2);p6,_=prompt_projector(flat,desired,6)
                differences.append(float(np.linalg.norm(p2-p6)));pp=[]
                for idx in (np.arange(a.shape[1])[::2],np.arange(a.shape[1])[1::2]):
                    pi,_=prompt_projector(p[:,idx].reshape(len(a),-1),np.repeat(a[:,idx],p.shape[-1],axis=1),2);pp.append(pi)
                splits.append(float(np.linalg.norm(pp[0]-pp[1])))
            row=dict(input=path.name,iteration_change_median=float(np.median(differences)),iteration_change_max=float(np.max(differences)),
                     split_disagreement_median=float(np.median(splits)),split_disagreement_max=float(np.max(splits)))
        rows.append(row)
    # Round 2 was inspected while the last extractor was still finishing.
    expected=read(ROOT/f'round{number:02d}/results.json');names={r['input'] for r in expected}
    actual=[r for r in rows if r['input'] in names]
    assert actual==expected
    save(ROOT/f'replayed_round{number:02d}_diagnostics.json',rows)
    print('reproduced',number,len(expected),'archived rows;',len(rows),'available rows')


if __name__=='__main__':replay(int(sys.argv[1]))
