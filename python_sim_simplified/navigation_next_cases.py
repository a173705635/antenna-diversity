"""Observation-domain controls, not sampled RF; truth never passed to estimator."""
import numpy as np
from run_gps_spp import receiver_reference
from signal_recovery import sky

KINDS=('normal','route_bias','common_step','common_drag','dropout','phase_offset','route_drift','common_bias','handoff_slip','fixed_geometry_bias','ionosphere_ramp','code_impulse','conflict_then_common')


def generate(seed,kind):
    rng=np.random.default_rng(seed);n=120;time=np.arange(n)*.01
    ref,rotation=receiver_reference();directions=sky(seed,8)
    satellites=ref+directions@rotation*21e6
    truth=np.column_stack((2*time,np.zeros((n,2))))
    common=rng.normal(0,2.,(n,8));pair={}
    ionosphere=time[:,None]*rng.normal(0,3.,(1,8)) if kind=='ionosphere_ramp' else 0.
    shared_bias=rng.normal(0,80.,(1,8)) if kind=='common_bias' else 0.
    if kind=='fixed_geometry_bias':shared_bias=np.linalg.norm(satellites-(ref+np.array([200.,0.,0.])@rotation),axis=1)-np.linalg.norm(satellites-ref,axis=1)
    for route in ('capture','joint'):
        apparent=truth.copy()
        if kind=='common_step':apparent[time>=.4,0]+=200.
        if kind=='common_drag':apparent[:,0]+=80.*time
        if route=='joint' and kind=='route_bias':apparent[:,0]+=60.
        if route=='joint' and kind=='route_drift':apparent[:,0]+=30.*time
        if kind=='conflict_then_common':
            if route=='joint':apparent[(time>=.3)&(time<.7),0]+=60.
            apparent[time>=.7,0]+=200.
        position=ref+apparent@rotation
        ranges=np.linalg.norm(satellites[None,:,:]-position[:,None,:],axis=2)+75000
        route_noise=rng.normal(0,2.,(n,8))
        bias=shared_bias
        z=dict(time_s=time.copy(),satellite_positions=np.tile(satellites,(n,1,1)),startup_prns=np.arange(8),joint_usable=np.array(True))
        for channel in ('raw','calibrated'):
            z[channel+'_ranges']=ranges+ionosphere+common+route_noise+bias+rng.normal(0,.5,(n,8))
            if kind=='code_impulse':
                displaced=ref+(truth[30]+np.array([1500.,0.,0.]))@rotation
                z[channel+'_ranges'][30]+=np.linalg.norm(satellites-displaced,axis=1)-np.linalg.norm(satellites-(ref+truth[30]@rotation),axis=1)
            z[channel+'_carrier']=ranges-ionosphere+32100.+rng.normal(0,.0003,(n,8))
            if kind=='phase_offset' and route=='capture':z[channel+'_carrier']+=10000*rng.integers(-5,6,(1,8))
            z[channel+'_lock']=np.ones((n,8));z[channel+'_cn0']=np.full((n,8),45.)
            z[channel+'_gain']=np.ones((n,8));z[channel+'_spatial']=np.full((n,8),20.)
            if kind=='dropout' and route=='joint':
                for field in ('ranges','carrier','lock'):z[channel+'_'+field][40:50]=np.nan
        if kind=='handoff_slip':
            begin,end=(30,40) if route=='capture' else (40,50)
            for channel in ('raw','calibrated'):
                if route=='capture':z[channel+'_carrier'][40:]+=299792458/1575.42e6*rng.integers(-300,301,(1,8))
                for field in ('ranges','carrier','lock'):z[channel+'_'+field][begin:end]=np.nan
        pair[route]=z
    return pair,truth
