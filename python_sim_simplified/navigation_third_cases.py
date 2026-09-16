"""Observed stress controls; not claims about realizable RF spoofers."""
import numpy as np
from navigation_next_cases import generate as previous,KINDS as PREVIOUS

KINDS=PREVIOUS+('lock_ambiguity','two_slips','weak_subset','time_gap','small_slip','code_fault','two_code_faults','phase_colored','code_colored')

def generate(seed,kind):
    pair,truth=previous(seed,kind if kind in PREVIOUS else 'normal')
    rng=np.random.default_rng(seed+70000)
    colored=np.zeros((120,8))
    for k in range(1,120):colored[k]=.8*colored[k-1]+rng.normal(0,.0038*np.sqrt(1-.8**2),8)
    for z in pair.values():
        if kind=='code_colored':z['calibrated_ranges']+=colored*(5./.0038)
        if kind=='phase_colored':z['calibrated_carrier']+=colored
        if kind=='lock_ambiguity':
            z['calibrated_lock'][30,0]=0.
            z['calibrated_carrier'][30:,0]+=200.
        if kind=='two_code_faults':z['calibrated_ranges'][:,:2]+=np.array([80.,-100.])
        if kind=='code_fault':z['calibrated_ranges'][:,0]+=80.
        if kind=='small_slip':z['calibrated_carrier'][30:,0]+=.08
        if kind=='two_slips':z['calibrated_carrier'][30:,:2]+=np.array([3.,-5.])*299792458/1575.42e6
        if kind=='weak_subset':
            z['calibrated_cn0'][:,:1]=23.;z['calibrated_gain'][:,1]=.005
    if kind=='time_gap':
        from run_gps_spp import receiver_reference
        ref,rotation=receiver_reference();sat=pair['joint']['satellite_positions'][0]
        offset=np.linalg.norm(sat-(ref+np.array([200.,0.,0.])@rotation),axis=1)-np.linalg.norm(sat-ref,axis=1)
        keep=np.r_[0:40,50:120]
        for route,z in pair.items():
            z['calibrated_carrier'][50:]+=offset
            pair[route]={k:(v[keep] if v.ndim and len(v)==120 else v) for k,v in z.items()}
        truth=truth[keep]
    return pair,truth
