import numpy as np

from config import TrackingConfig, AcquisitionConfig
from gnss import ca_code
from tracking import track
from autonomous_tracking import receive


def samples():
    fs=2.046e6
    n=np.arange(round(fs*.36))
    code=ca_code(1)[np.floor(np.mod(n*1.023e6/fs,1023)).astype(int)]
    rng=np.random.default_rng(921)
    noise=.15*(rng.normal(size=n.size)+1j*rng.normal(size=n.size))
    clean=code*np.exp(2j*np.pi*500*n/fs)+noise
    broken=clean.copy()
    broken[round(.16*fs):round(.24*fs)]=noise[round(.16*fs):round(.24*fs)]
    return fs,clean,broken


def test_loss_stops_at_causal_prefix_and_default_still_tracks_full_record():
    fs,clean,broken=samples();cfg=TrackingConfig(max_epochs=400)
    acq=dict(code_phase_samples=0,doppler_hz=500.)
    full=track(broken,fs,1,acq,cfg,1575.42e6,1.023e6)
    stopped=track(broken,fs,1,acq,cfg,1575.42e6,1.023e6,loss_confirm_epochs=5)
    assert full['status']=='tracking' and stopped['status']=='tracking_quality_lost'
    k=len(stopped['prompt'])
    for key,value in stopped.items():
        if isinstance(value,np.ndarray):
            np.testing.assert_array_equal(value,full[key][:k])
    end=int(stopped['epoch_start_samples'][-1])+round(fs*.001)
    prefix=track(broken[:end],fs,1,acq,cfg,1575.42e6,1.023e6,loss_confirm_epochs=5)
    np.testing.assert_array_equal(prefix['prompt'],stopped['prompt'])
    nominal=track(clean,fs,1,acq,cfg,1575.42e6,1.023e6,loss_confirm_epochs=5)
    assert nominal['status']=='tracking'


def test_real_acquisition_retries_without_outage_time_input():
    fs,_,broken=samples()
    result=receive(broken,fs,1,1.023e6,1575.42e6,
                   AcquisitionConfig(doppler_bins_hz=np.array([-500.,0.,500.,1000.])),
                   TrackingConfig(max_epochs=400))
    assert len(result['segments'])>=2
    assert any(e['kind']=='acquisition' and not e['success'] for e in result['events'])
    assert result['segments'][-1]['history']['epoch_start_samples'][0]/fs >= .24
    previous_end=0
    for segment in result['segments']:
        h=segment['history']
        assert h['epoch_start_samples'][0]>=previous_end
        previous_end=h['epoch_start_samples'][-1]+round(.001*fs)
