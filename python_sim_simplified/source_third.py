"""Source existence from receiver prompt and independent off-code noise observations."""
import numpy as np
from scipy.stats import chi2, f

def detect(z,a,noise):
    lagged=noise[:,:,:79];following=noise[:,:,1:80]
    rho=np.vdot(lagged,following)/np.vdot(lagged,lagged)
    z=z[:,:,1:]-rho*z[:,:,:-1]
    noise=noise[:,:,1:]-rho*noise[:,:,:-1]
    calibration=noise[:,:,:79].reshape(z.shape[0],-1)
    covariance=calibration@calibration.conj().T/calibration.shape[1]
    white=np.linalg.inv(np.linalg.cholesky(covariance))
    z=np.einsum('ij,jsk->isk',white,z);a=white@a
    noise=np.einsum('ij,jsk->isk',white,noise[:,:,79:])
    train=z[:,:,:79];test=z[:,:,79:]
    pooled=train.reshape(z.shape[0],-1)
    values,vectors=np.linalg.eigh(pooled@pooled.conj().T/pooled.shape[1])
    descending=values[::-1];m=len(values);count=pooled.shape[1]
    mdl=[count*(m-r)*(np.log(np.mean(descending[r:]))-np.mean(np.log(descending[r:])))+.5*r*(2*m-r)*np.log(count) for r in range(m)]
    common_rank=int(np.argmin(mdl))
    nuisance=vectors[:,-1]
    projector=np.eye(z.shape[0])-np.outer(nuisance,nuisance.conj())
    local=np.column_stack([np.linalg.eigh(train[:,s]@train[:,s].conj().T)[1][:,-1] for s in range(z.shape[1])])
    # Compare whitened sky directions against a common emitter using training only.
    nominal=a/np.linalg.norm(a,axis=0)
    sky_alignment=np.mean(abs(np.sum(nominal.conj()*local,axis=0))**2)
    common_alignment=np.mean(abs(nuisance.conj()@local)**2)
    projector=projector if common_alignment>sky_alignment and common_rank>0 else np.eye(z.shape[0])
    directions=projector@a
    directions/=np.linalg.norm(directions,axis=0)
    training_scalar=np.einsum('ms,msk->sk',directions.conj(),train)
    frequency_grid=np.linspace(-.02,.02,161)
    doubled=np.exp(-4j*np.pi*frequency_grid[:,None]*np.arange(train.shape[2])[None,:])
    frequency=frequency_grid[np.argmax(abs(training_scalar**2@doubled.T),axis=1)]
    rotation=np.exp(-2j*np.pi*frequency[:,None]*np.arange(test.shape[2])[None,:])
    early=np.linalg.eigh(train[:,:,:39].reshape(m,-1)@train[:,:,:39].reshape(m,-1).conj().T)[1][:,-1]
    late=np.linalg.eigh(train[:,:,39:].reshape(m,-1)@train[:,:,39:].reshape(m,-1).conj().T)[1][:,-1]
    early*=np.exp(-1j*np.angle(np.vdot(nuisance,early)));late*=np.exp(-1j*np.angle(np.vdot(nuisance,late)))
    tangent=(late-early)/39.5
    train_axes=nuisance[:,None]+tangent[:,None]*(np.arange(train.shape[2])[None,:]-39.)
    train_axes/=np.linalg.norm(train_axes,axis=0)
    constant_error=train-nuisance[:,None,None]*np.einsum('m,msk->sk',nuisance.conj(),train)[None,:,:]
    linear_error=train-train_axes[:,None,:]*np.einsum('mk,msk->sk',train_axes.conj(),train)[None,:,:]
    motion_bic=float(2*(np.sum(abs(constant_error)**2)-np.sum(abs(linear_error)**2))-2*(m-1)*np.log(2*train.size))
    tangent*=int(motion_bic>0)
    predicted=nuisance[:,None]+tangent[:,None]*(np.arange(test.shape[2])[None,:]+39.5)
    predicted/=np.linalg.norm(predicted,axis=0)
    moving=a[:,:,None]-predicted[:,None,:]*np.einsum('mt,ms->st',predicted.conj(),a)[None,:,:]
    moving/=np.linalg.norm(moving,axis=0)
    filters=moving if common_alignment>sky_alignment and common_rank>0 else directions[:,:,None]
    projected=np.sum(filters.conj()*test,axis=0)
    reference=np.sum(filters.conj()*noise,axis=0)
    projected*=rotation;reference*=rotation
    projected=projected.reshape(z.shape[1],-1,10).sum(axis=2)/np.sqrt(10)
    reference=reference.reshape(z.shape[1],-1,10).sum(axis=2)/np.sqrt(10)
    halves=[train[:,:,:39],train[:,:,39:]]
    axes=[np.linalg.eigh(q.reshape(m,-1)@q.reshape(m,-1).conj().T)[1][:,-1] for q in halves]
    axes=[v*np.exp(-1j*np.angle(np.vdot(nuisance,v))) for v in axes]
    leakage=10*values[-1]*abs(directions.conj().T@(axes[0]-axes[1]))**2/4
    leakage*=int(common_alignment>sky_alignment and common_rank>0)
    variance=np.mean(abs(reference)**2,axis=1)+leakage
    statistic=2*np.sum(abs(projected)**2,axis=1)/variance
    p=f.sf(statistic/(2*projected.shape[1]),2*projected.shape[1],2*reference.shape[1])
    first=f.sf(np.mean(abs(projected[:,:8])**2,axis=1)/variance,16,32)
    second=f.sf(np.mean(abs(projected[:,8:])**2,axis=1)/variance,16,32)
    p=np.maximum(first,second)
    residual_signal=[];residual_noise=[];local_fit=[]
    for s in range(z.shape[1]):
        basis=np.column_stack((directions[:,s],nuisance))
        q=np.linalg.qr(basis)[0]
        orthogonal=np.eye(m)-q@q.conj().T
        on_power=np.sum(abs(orthogonal@test[:,s])**2)
        off_power=np.sum(abs(orthogonal@noise[:,s])**2)
        ratio=on_power/off_power;df=2*(m-2)*test.shape[2]
        local_fit.append(float(2*min(f.cdf(ratio,df,df),f.sf(ratio,df,df))))
        residual_signal.extend(abs(orthogonal@test[:,s]).ravel()**2)
        residual_noise.extend(abs(orthogonal@noise[:,s]).ravel()**2)
    residual_ratio=float(np.mean(residual_signal)/np.mean(residual_noise))
    residual_df=2*(m-2)*test.shape[2]*z.shape[1]
    fit_p=float(2*min(f.cdf(residual_ratio,residual_df,residual_df),f.sf(residual_ratio,residual_df,residual_df)))
    correlations=[]
    for lag in range(1,6):
        left=noise[:,:,:-lag];right=noise[:,:,lag:]
        correlations.append(abs(np.vdot(left,right)/np.sqrt(np.vdot(left,left).real*np.vdot(right,right).real))**2)
    temporal_stat=2*noise.size*sum(correlations)
    temporal_p=float(chi2.sf(temporal_stat,10))
    p=np.where(np.array(local_fit)>.001,p,1.)
    conjunction=float(min(1.,(z.shape[1]-3)*np.sort(p)[3]))
    return dict(usable=bool(conjunction<.01 and temporal_p>.001),p=p.tolist(),statistic=statistic.tolist(),four_source_p=conjunction,residual_noise_p=fit_p,innovation_p=temporal_p,local_residual_p=local_fit,motion_bic=motion_bic)




















