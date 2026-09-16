"""Independent invariance and archived-noise counterexample checks."""
import hashlib,json
from pathlib import Path
import numpy as np
from cold_next import complex_calibrate
from source_third import detect
from source_third_research import observation


def test_original_four_noise_counterexamples():
    _,a,_=observation(1,'iid');old=[];new=[]
    for seed in range(48020,48024):
        rng=np.random.default_rng(seed)
        z=(rng.normal(size=(4,8,100))+1j*rng.normal(size=(4,8,100)))/np.sqrt(2)
        old.append(complex_calibrate(z,a)[1]['usable'])
        # Original 100 samples are retained verbatim; new observation contract needs 240.
        extra=(rng.normal(size=(4,8,140))+1j*rng.normal(size=(4,8,140)))/np.sqrt(2)
        reference=(rng.normal(size=(4,8,240))+1j*rng.normal(size=(4,8,240)))/np.sqrt(2)
        new.append(detect(np.concatenate((z,extra),axis=2),a,reference)['usable'])
    assert old==[True]*4
    assert new==[False]*4


def test_receiver_scale_invariance():
    z,a,noise=observation(95711,'mixed')
    x=detect(z,a,noise);y=detect((2-3j)*z,a,(2-3j)*noise)
    assert x['usable']==y['usable']
    np.testing.assert_allclose(x['p'],y['p'],rtol=1e-9,atol=1e-12)


def test_no_detection_data_in_motion_fit():
    z,a,noise=observation(95712,'moving_common')
    first=detect(z,a,noise);z[:,:,80:]*=7
    second=detect(z,a,noise)
    assert first['motion_bic']==second['motion_bic']


def test_four_source_null_and_persistence():
    for kind in ('three_sources','transient','noise_drift','two_common'):
        z,a,noise=observation(95713,kind)
        assert not detect(z,a,noise)['usable']


def test_frozen_module_and_all_round_artifacts():
    root=Path(__file__).resolve().parents[1]
    result=root/'results/research-third-hundred'
    manifest=json.loads((result/'source_freeze.json').read_text())
    assert hashlib.sha256((root/'source_third.py').read_bytes()).hexdigest()==manifest['files']['source_third.py']
    for revision in range(226,251):
        folder=result/f'iteration{revision:03d}'
        for name in ('proposal.json','before.py','after.py','results.json','conclusion.json'):assert (folder/name).is_file()
        assert (folder/'before.py').read_bytes()!=(folder/'after.py').read_bytes()
