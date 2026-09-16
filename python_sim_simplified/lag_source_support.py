"""Per-PRN measured-component support, not GNSS authentication."""
import numpy as np
from lag_calibration import estimate as calibrate,shape


def support(z,a,lag,gain,info):
    pairs=np.array(info['components']);pairs=pairs[...,0]+1j*pairs[...,1]
    labels=np.array(info['labels']);double=np.array(info['double'])
    selected=pairs[np.arange(len(labels)),labels].T
    source=pairs[np.flatnonzero(double),1-labels[double]].T if double.sum()>=3 else selected
    source=source/np.linalg.norm(source,axis=0)
    _,vectors=np.linalg.eigh(source@source.conj().T);v=vectors[:,-1]
    desired=gain[:,None]*a;desired/=np.linalg.norm(desired,axis=0)
    sky=np.sum(abs(selected-desired*np.sum(desired.conj()*selected,axis=0))**2,axis=0)/np.sum(abs(selected)**2,axis=0)
    common=np.sum(abs(selected-v[:,None]*(v.conj()@selected))**2,axis=0)/np.sum(abs(selected)**2,axis=0)
    admitted=sky<common
    delays=np.array(info['delays'])
    selected_delay=np.array([delays[j,labels[j] if double[j] else 0] for j in range(len(labels))])
    return dict(info,global_usable=info['usable'],usable=bool(info['usable'] and admitted.sum()>=3),
        selected_delay_chips=selected_delay.tolist(),sky_component_residual=sky.tolist(),
        common_component_residual=common.tolist(),supported_prns=admitted.tolist())


def estimate(z,a,lag):
    gain,info=calibrate(z,a,lag)
    info=support(z,a,lag,gain,info)
    pairs=np.array(info['components']);pairs=pairs[...,0]+1j*pairs[...,1]
    shifts=np.array(info['channel_delay_chips']);evidence=[]
    m,s,n=z.shape
    for star in range(s):
        bandwidth=info['bandwidths'][star];double=info['double'][star]
        covariance=.99*shape(lag[:,None],lag[None,:],bandwidth)+.01*np.eye(n)
        chol=np.linalg.cholesky(covariance)
        predicted=np.zeros((m,n),complex)
        for component in range(2 if double else 1):
            predicted+=pairs[star,component,:,None]*shape(lag[None,:],
                info['delays'][star][component]+shifts[:,None],bandwidth)
        null=np.linalg.solve(chol,z[:,star].T)
        residual=np.linalg.solve(chol,(z[:,star]-predicted).T)
        count=2*m*n;parameters=(4*m+3 if double else 2*m+2)+(m-1)/s
        evidence.append(float(count*np.log(np.sum(abs(null)**2)/np.sum(abs(residual)**2))-parameters*np.log(count)))
    signal=np.array(evidence)>0
    spatial=np.array(info['supported_prns']);admitted=spatial&signal
    info.update(spatial_supported_prns=spatial.tolist(),signal_bic=evidence,
        signal_supported_prns=signal.tolist(),supported_prns=admitted.tolist(),
        usable=bool(info['global_usable'] and admitted.sum()>=3))
    return gain,info
