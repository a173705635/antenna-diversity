"""Replay every counted proposal from saved measured tracking, and check hashes."""
import ast
import numpy as np
from single_source_twenty import ROOT,read,save,digest,load


def audit():
    replayed=set();rows=0;changes=[]
    for number in range(1,21):
        folder=ROOT/f'round{number:03d}'
        proposal=read(folder/'proposal.json');read(folder/'conclusion.json')
        parent=ROOT/'baseline.py' if proposal['parent']==0 else ROOT/f"round{proposal['parent']:03d}/receiver.py"
        current=folder/'receiver.py'
        assert ast.dump(ast.parse(parent.read_text(encoding='utf-8')))!=ast.dump(ast.parse(current.read_text(encoding='utf-8')))
        changes.append(dict(round=number,parent=proposal['parent'],source_sha256=digest(current)))
        results=read(folder/'results.json')
        for row in results['rows']:
            assert row['before']['input_sha256']==row['after']['input_sha256']
            input_folder=ROOT/'inputs'/f"{proposal['development_seed']}_{row['kind']}_{proposal['duration']}"
            metadata=read(input_folder/'metadata.json')
            assert metadata['samples_sha256']==row['after']['input_sha256']
            for side,model_path in (('before',parent),('after',current)):
                assert row[side]['source_sha256']==digest(model_path)
                identity=row[side+'_id'];rows+=1
                if identity in replayed:continue
                cache=ROOT/'evaluations'/identity
                model=load(model_path)
                with np.load(cache/'tracking.npz') as z:actual=model.navigate({k:z[k] for k in z.files})
                with np.load(cache/'navigation.npz') as stored:
                    for key in stored.files:np.testing.assert_allclose(actual[key],stored[key],rtol=0,atol=1e-9,equal_nan=True)
                assert actual['initialization_time']==row[side]['initialization_time']
                replayed.add(identity)
        print('replayed round',number,flush=True)
    previous=ROOT.parent/'research-warm-hundred'
    old=read(previous/'manifest.json')['sha256']
    for name,sha in old.items():assert digest(previous/name)==sha
    frozen=read(ROOT/'freeze.json')
    for name,sha in frozen['dependencies'].items():assert digest(__import__('pathlib').Path(name))==sha
    assert digest(ROOT/'selected.py')==frozen['selected_sha256']==digest(__import__('pathlib').Path('single_source_receiver.py'))
    inputs=[]
    for metadata_file in (ROOT/'inputs').glob('*/metadata.json'):
        meta=read(metadata_file)
        for filename,key in (('samples.npy','samples_sha256'),('observations.npz','observations_sha256')):
            assert digest(metadata_file.parent/filename)==meta[key]
        inputs.append(str(metadata_file.parent))
    result=dict(completed_rounds=20,paired_side_rows=rows,unique_navigation_replays=len(replayed),
                round_source_changes=changes,previous_archive_verified_files=len(old),
                frozen_dependencies_verified=len(frozen['dependencies']),input_records_verified=len(inputs),
                scope='all counted PVT replays from immutable RF-derived tracking; not a rerun of every RF frontend')
    correction=read(ROOT/'post_freeze_correction.json');corrected=load(ROOT/'audit_corrected.py')
    assert digest(ROOT/'audit_corrected.py')==correction['corrected_sha256']
    corrected_count=0
    for subset in ('two','three','four'):
        for row in read(ROOT/f'corrective_{subset}.json'):
            assert row['score']['frontend_reused']
            assert row['score']['frontend_ast_sha256']==row['before']['frontend_ast_sha256']
            cache=ROOT/'evaluations'/row['identity']
            with np.load(cache/'tracking.npz') as z:actual=corrected.navigate({k:z[k] for k in z.files})
            with np.load(cache/'navigation.npz') as stored:
                for key in stored.files:np.testing.assert_allclose(actual[key],stored[key],rtol=0,atol=1e-9,equal_nan=True)
            assert actual['initialization_time']==row['score']['initialization_time']
            corrected_count+=1
    assert corrected_count==108
    result['post_freeze_corrective_replays']=corrected_count
    result['corrective_frontend_ast_unchanged']=True
    save(ROOT/'audit.json',result);print(result,flush=True)


if __name__=='__main__':audit()
