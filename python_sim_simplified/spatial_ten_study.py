"""Ten sequential research loops; keep RF, candidates and evidence immutable."""
import sys
from pathlib import Path
import numpy as np
import single_source_twenty as campaign
from single_source_twenty import save,read,digest,load,input_record

ROOT=Path('results/spatial-ten')
OLD=Path('results/single-source-twenty')
EXPANDED=Path('results/satellite-count-24')


def proposal(number,problem,hypothesis):
    folder=ROOT/f'round{number:02d}';folder.mkdir(parents=True)
    save(folder/'proposal.json',dict(number=number,problem=problem,hypothesis=hypothesis))


def conclusion(number,decision,evidence,next_problem):
    save(ROOT/f'round{number:02d}'/'conclusion.json',dict(decision=decision,evidence=evidence,next_problem=next_problem))


def extract(folder,subset):
    """Run the unchanged startup/calibration; capture beam inputs before tracking."""
    source=Path('single_source_receiver_fixed.py').read_text(encoding='utf-8')
    begin=source.index('    stream=np.empty')
    end=source.index('\n\ndef navigate')
    body='''    observed=[];clock=[]
    for start in range(startup,x.shape[1],width):
        block=calibrated[:,start:start+width]
        t=np.arange(start,start+len(block[0]))/FS
        local=np.stack([local_signal(codes[i],template,t) for i,template in zip(prns,templates)])
        chunks=block.shape[1]//2046
        z=np.einsum('mkn,skn->msk',block.reshape(len(a),chunks,2046),local.conj().reshape(len(local),chunks,2046))/2046
        observed.append(z);clock.append(start/FS)
    return dict(prompts=np.array(observed),time=np.array(clock),desired=a[:,prns],prns=np.array(prns))
'''
    namespace={};exec(compile(source[:begin]+body+source[end:],'<prompt-extractor>','exec'),namespace)
    result=namespace['frontend'](input_record(folder),campaign.SUBSETS[subset])
    target=ROOT/'prompts';target.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target/f'{folder.name}_{subset}.npz',**result)
    print('extracted',folder.name,subset,flush=True)


def run_pair(number,model,folder,subset):
    base=EXPANDED if folder.is_relative_to(EXPANDED) else OLD if folder.is_relative_to(OLD) else ROOT
    if base==EXPANDED and subset=='two':base=ROOT
    campaign.ROOT=base
    before,bid=campaign.evaluate(ROOT/'baseline.py',folder,subset)
    campaign.ROOT=ROOT
    after,aid=campaign.evaluate(Path(model),folder,subset)
    row=dict(input=str(folder),subset=subset,before=before,after=after,before_id=bid,after_id=aid,before_root=str(base))
    save(ROOT/f'round{number:02d}'/f'{folder.name}_{subset}.json',row)
    print(folder.name,subset,{s:{k:row[s][k] for k in ('solved','good20','error30','error50')} for s in ('before','after')},flush=True)


def run_revision(number,model,parent,folder,subset):
    run_pair(number,model,folder,subset)
    score,identity=campaign.evaluate(Path(parent),folder,subset)
    path=ROOT/f'round{number:02d}'/f'{folder.name}_{subset}.json'
    row=read(path);row.update(parent=score,parent_id=identity);save(path,row)
    print('parent',folder.name,subset,score['good20'],flush=True)


if __name__=='__main__':
    if sys.argv[1]=='extract':extract(Path(sys.argv[2]),sys.argv[3])
    elif sys.argv[1]=='pair':run_pair(int(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),sys.argv[5])
