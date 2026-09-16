"""Development space-delay calibration; synthetic known-sky diagonal receiver."""
import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.stats import chi2
from lag_calibration import shape
from lag_source_support import support


def estimate(z,a,lag,noise):
    training=noise[:len(noise)//2];validation=noise[len(noise)//2:]
    sample=(training.conj().T@training).real/len(training)
    def noise_model(parameters):
        bw,floor=parameters
        return (1-floor)*shape(lag[:,None],lag[None,:],bw)+floor*np.eye(len(lag))
    def noise_likelihood(parameters):
        cov=noise_model(parameters)
        scale=np.trace(np.linalg.solve(cov,sample))/len(lag)
        return len(lag)*np.log(scale)+np.linalg.slogdet(cov)[1]
    noise_fit=minimize(noise_likelihood,[.9,.01],bounds=((.5,1.2),(.001,.1)))
    covariance=noise_model(noise_fit.x)
    noise_variance=np.trace(np.linalg.solve(covariance,sample))/(2*len(lag))
    _,initial=fit_components(z,a,lag,None,np.zeros(len(z)),covariance,noise_variance)
    bandwidth=float(np.median(initial['bandwidths']))
    gain,info=fit_components(z,a,lag,bandwidth,np.zeros(len(z)),covariance,noise_variance)
    delays=np.array(info['delays']);double=np.array(info['double'])
    whiten=np.linalg.inv(np.linalg.cholesky(covariance))
    whitened=validation@whiten.T
    covariance_check=(whitened.conj().T@whitened).real/(2*noise_variance*len(validation))
    noise_statistic=len(validation)*np.sum((covariance_check-np.eye(len(lag)))**2)
    noise_p=float(chi2.sf(noise_statistic,len(lag)*(len(lag)+1)/2))
    observation=z@whiten.T
    frequency=np.linspace(0,bandwidth/1.023,65)
    power=np.sinc(frequency)**2;power[[0,-1]]*=.5;power/=power.sum()
    cosine=whiten@(np.cos(2*np.pi*lag[:,None]*frequency)*power)
    sine=whiten@(np.sin(2*np.pi*lag[:,None]*frequency)*power)
    def residual(parameters):
        shifts=np.r_[0.,parameters[:len(z)-1]]
        local_delays=parameters[len(z)-1:-1].reshape(delays.shape)
        local_bandwidth=parameters[-1]
        errors=[]
        for star in range(z.shape[1]):
            for channel in range(len(z)):
                template=whiten@shape(lag[:,None],local_delays[star,:2 if double[star] else 1][None,:]+shifts[channel],local_bandwidth)
                coefficient=np.linalg.lstsq(template,observation[channel,star],rcond=None)[0]
                errors.append(observation[channel,star]-template@coefficient)
        error=np.array(errors).ravel()
        return np.r_[error.real,error.imag]
    start=np.r_[np.zeros(len(z)-1),delays.ravel(),bandwidth]
    lower=np.r_[np.full(len(start)-1,-np.inf),.5];upper=np.r_[np.full(len(start)-1,np.inf),1.2]
    fit=least_squares(residual,start,bounds=(lower,upper),max_nfev=30)
    shifts=np.r_[0.,fit.x[:len(z)-1]]
    zero=least_squares(lambda t:residual(np.r_[np.zeros(len(z)-1),t]),fit.x[len(z)-1:],bounds=(lower[len(z)-1:],upper[len(z)-1:]),max_nfev=30)
    shift_evidence=(np.sum(zero.fun**2)-np.sum(fit.fun**2))/noise_variance-(len(z)-1)*np.log(2*z.size)
    if shift_evidence<0:shifts=np.zeros(len(z))
    bandwidth=float(zero.x[-1] if shift_evidence<0 else fit.x[-1])
    gain,info=fit_components(z,a,lag,bandwidth,shifts,covariance,noise_variance)
    gain,info=coherent_refine(z,a,lag,whiten,gain,info,shifts,bandwidth,noise_variance)
    bandwidth=info['bandwidths'][0]
    shifts=np.array(info['channel_delay_chips'])
    info['prefit_group_delay_bic']=float(shift_evidence)
    info['noise_variance']=float(noise_variance)
    info['noise_bandwidth_mhz']=float(noise_fit.x[0])
    info['noise_white_fraction']=float(noise_fit.x[1])
    info.update(noise_training_snapshots=len(training),noise_validation_snapshots=len(validation),noise_model_p=noise_p,noise_model_statistic=float(noise_statistic))
    info=support(z,a,lag,gain,info)
    pairs=np.array(info['components']);pairs=pairs[...,0]+1j*pairs[...,1]
    labels=np.array(info['labels'])
    measured=np.array(info['measured_components']);measured=measured[...,0]+1j*measured[...,1]
    selected=measured[:,0].T
    directions=selected/a;directions/=np.linalg.norm(directions,axis=0)
    common=measured[:,1].T;common/=np.linalg.norm(common,axis=0)
    loo_sky=[];loo_common=[]
    for star in range(z.shape[1]):
        others=np.arange(z.shape[1])!=star
        _,vectors=np.linalg.eigh(directions[:,others]@directions[:,others].conj().T)
        sky_vector=np.exp(1j*np.angle(vectors[:,-1]/vectors[0,-1]))*a[:,star]
        sky_vector/=np.linalg.norm(sky_vector)
        _,vectors=np.linalg.eigh(common[:,others]@common[:,others].conj().T)
        common_vector=vectors[:,-1]
        observed=selected[:,star]
        loo_sky.append(float(np.sum(abs(observed-sky_vector*np.vdot(sky_vector,observed))**2)/np.sum(abs(observed)**2)))
        loo_common.append(float(np.sum(abs(observed-common_vector*np.vdot(common_vector,observed))**2)/np.sum(abs(observed)**2)))
    info['supported_prns']=(np.array(loo_sky)<np.array(loo_common)).tolist()
    info['loo_sky_residual']=loo_sky;info['loo_common_residual']=loo_common
    evidence=[];desired_evidence=[]
    for star in range(z.shape[1]):
        predicted=np.zeros((len(z),len(lag)),complex)
        count=2 if info['double'][star] else 1
        for component in range(count):
            predicted+=pairs[star,component,:,None]*shape(lag[None,:],info['delays'][star][component]+shifts[:,None],bandwidth)
        null=z[:,star]@whiten.T;error=(z[:,star]-predicted)@whiten.T
        no_sky=null.copy()
        if count==2:
            nuisance=pairs[star,1,:,None]*shape(lag[None,:],info['delays'][star][1]+shifts[:,None],bandwidth)
            nuisance=nuisance@whiten.T
            no_sky-=nuisance*np.vdot(nuisance,null)/np.vdot(nuisance,nuisance)
        desired_evidence.append(float((np.sum(abs(no_sky)**2)-np.sum(abs(error)**2))/noise_variance-3*np.log(2*len(z)*len(lag))))
        parameters=2*len(z)*count+count+(len(z)-1)/z.shape[1]
        evidence.append(float((np.sum(abs(null)**2)-np.sum(abs(error)**2))/noise_variance-parameters*np.log(2*len(z)*len(lag))))
    admitted=np.array(info['supported_prns'])&(np.array(evidence)>0)
    admitted&=np.array(desired_evidence)>0
    info['desired_source_bic']=desired_evidence
    source_count=1+np.array(info['double'],dtype=int)
    parameters=3*source_count.sum()+4*(len(z)-1)+1+(len(z)-1)*int(info['group_delay_bic']>0)
    fit_p=float(chi2.sf(info['coherent_rss']/noise_variance,2*z.size-parameters))
    admitted&=fit_p>.001
    admitted&=noise_p>.001
    info['model_fit_p']=fit_p
    admitted&=3*np.array(info['code_std_m'])<10.
    admitted[info['profile_prn']]&=min(info['profile_delta_chi2'])>9.
    info.update(signal_bic=evidence,supported_prns=admitted.tolist(),usable=bool(info['global_usable'] and admitted.sum()>=3))
    return gain,info


def coherent_refine(z,a,lag,whiten,gain,info,shifts,bandwidth,noise_variance):
    m,s,n=z.shape
    pairs=np.array(info['components']);pairs=pairs[...,0]+1j*pairs[...,1]
    labels=np.array(info['labels']);double=np.array(info['double'])
    nuisance=pairs[np.arange(s),1-labels].T
    nuisance/=np.linalg.norm(nuisance,axis=0)
    _,vectors=np.linalg.eigh(nuisance@nuisance.conj().T)
    source=vectors[:,-1]/vectors[0,-1]
    delays=np.array(info['delays'])
    starts=np.array([[delays[k,labels[k] if double[k] else 0],delays[k,1-labels[k]]] for k in range(s)])
    measured=np.array([pairs[k,[labels[k],1-labels[k]]] for k in range(s)])
    observation=z@whiten.T
    frequency=np.linspace(0,bandwidth/1.023,65)
    power=np.sinc(frequency)**2;power[[0,-1]]*=.5;power/=power.sum()
    cosine=whiten@(np.cos(2*np.pi*lag[:,None]*frequency)*power)
    sine=whiten@(np.sin(2*np.pi*lag[:,None]*frequency)*power)
    def project(parameters):
        local_delays=parameters[:2*s].reshape(s,2)
        local_gain=np.exp(np.r_[0.,parameters[2*s+4*(m-1):-1]]+1j*np.r_[0.,parameters[2*s:2*s+m-1]])
        local_source=np.r_[1.,parameters[2*s+m-1:2*s+2*(m-1)]+1j*parameters[2*s+2*(m-1):2*s+3*(m-1)]]
        local_shifts=np.r_[0.,parameters[2*s+3*(m-1):2*s+4*(m-1)]]
        local_bandwidth=parameters[-1]
        frequency=np.linspace(0,local_bandwidth/1.023,65)
        power=np.sinc(frequency)**2;power[[0,-1]]*=.5;power/=power.sum()
        cosine=whiten@(np.cos(2*np.pi*lag[:,None]*frequency)*power)
        sine=whiten@(np.sin(2*np.pi*lag[:,None]*frequency)*power)
        errors=[];components=[]
        for star in range(s):
            count=2 if double[star] else 1
            arguments=2*np.pi*(local_delays[star,:count][None,:,None]+local_shifts[:,None,None])*frequency
            forms=(np.cos(arguments)@cosine.T+np.sin(arguments)@sine.T).transpose(0,2,1)
            steering=np.column_stack((local_gain*a[:,star],local_source))[:,:count]
            template=(forms*steering[:,None,:]).reshape(m*n,count)
            coefficients=np.linalg.lstsq(template,observation[:,star].ravel(),rcond=None)[0]
            errors.append(observation[:,star].ravel()-template@coefficients)
            fitted=(steering*coefficients).T
            components.append(np.stack((fitted[0],fitted[-1])))
        return np.array(errors),np.array(components)
    def residual(parameters):
        errors,_=project(parameters)
        return np.r_[errors.real.ravel(),errors.imag.ravel()]
    start=np.r_[starts.ravel(),np.angle(gain[1:]),source[1:].real,source[1:].imag,shifts[1:],np.zeros(m-1),bandwidth]
    lower=np.r_[np.full(2*s,-2.),np.full(5*(m-1),-np.inf),.5]
    upper=np.r_[np.full(2*s,2.),np.full(5*(m-1),np.inf),1.2]
    fit=least_squares(residual,start,bounds=(lower,upper),max_nfev=40)
    base=2*s+3*(m-1)
    expand=lambda p:np.r_[p[:base],np.zeros(m-1),p[base:]]
    zero=least_squares(lambda p:residual(expand(p)),np.r_[fit.x[:base],fit.x[base+m-1:]],bounds=(np.r_[lower[:base],lower[base+m-1:]],np.r_[upper[:base],upper[base+m-1:]]),max_nfev=40)
    evidence=(np.sum(zero.fun**2)-np.sum(fit.fun**2))/noise_variance-(m-1)*np.log(2*z.size)
    if evidence<0:fit.x=expand(zero.x)
    selected=fit.x.copy()
    jacobian=np.column_stack([(residual(selected+np.eye(len(selected))[j]*1e-6)-residual(selected-np.eye(len(selected))[j]*1e-6))/2e-6 for j in range(len(selected))])
    # Unselected hardware-delay parameters are fixed by the nested model.
    active=np.r_[np.arange(base),np.arange(base+m-1,len(selected))] if evidence<0 else np.arange(len(selected))
    code_std=[]
    for star in range(s):
        derivative=jacobian[:,2*star]
        nuisance=jacobian[:,active[active!=2*star]]
        projected=derivative-nuisance@np.linalg.lstsq(nuisance,derivative,rcond=1e-9)[0]
        code_std.append(float(np.sqrt(noise_variance/np.sum(projected**2))*299792458/1.023e6))
    profile_prn=int(np.argmax(code_std))
    column=2*profile_prn;free=active[active!=column]
    delta=[]
    reference_rss=np.sum(residual(selected)**2)
    for sign in (-1,1):
        fixed=selected.copy();fixed[column]+=sign*10*1.023e6/299792458
        def profile_residual(values):
            parameters=fixed.copy();parameters[free]=values
            return residual(parameters)
        probe=least_squares(profile_residual,selected[free],bounds=(lower[free],upper[free]),max_nfev=30)
        delta.append(float((np.sum(probe.fun**2)-reference_rss)/noise_variance))
    errors,components=project(fit.x)
    gain=np.exp(np.r_[0.,fit.x[base+m-1:-1]]+1j*np.r_[0.,fit.x[2*s:2*s+m-1]])
    info.update(delays=fit.x[:2*s].reshape(s,2).tolist(),labels=np.zeros(s,dtype=int).tolist(),
        channel_delay_chips=np.r_[0.,fit.x[base:base+m-1]].tolist(),group_delay_bic=float(evidence),channel_amplitudes=abs(gain).tolist(),
        code_std_m=code_std,
        profile_prn=profile_prn,profile_delta_chi2=delta,
        bandwidths=np.full(s,fit.x[-1]).tolist(),
        components=np.stack((components.real,components.imag),axis=-1).tolist(),
        measured_components=np.stack((measured.real,measured.imag),axis=-1).tolist(),coherent_rss=float(np.sum(abs(errors)**2)))
    return gain,info


def fit_components(z,a,lag,fixed_bandwidth,channel_delays,noise_cov,noise_variance):
    m,s,n=z.shape;flat=z.reshape(m,-1)
    whiten=np.linalg.inv(np.linalg.cholesky(noise_cov))
    _,vectors=np.linalg.eigh(flat@flat.conj().T);v=vectors[:,-1]
    p=np.eye(m)-np.outer(v,v.conj());pairs=[];delays=[];double=[];bandwidths=[]
    for k in range(s):
        observation=whiten@z[:,k].T
        projected=p@z[:,k]
        true_delay=lag[np.argmax(np.sum(abs(projected)**2,axis=0))]
        fake_delay=lag[np.argmax(np.sum(abs(z[:,k])**2,axis=0))]
        def linear_fit(delays,bandwidth):
            coefficients=[];predictions=[]
            for channel in range(m):
                template=whiten@shape(lag[:,None],np.array(delays)[None,:]+channel_delays[channel],bandwidth)
                coefficient=np.linalg.lstsq(template,observation[:,channel],rcond=None)[0]
                coefficients.append(coefficient);predictions.append(template@coefficient)
            return np.array(coefficients).T,observation-np.array(predictions).T
        def residual(parameters):
            bandwidth=parameters[2] if fixed_bandwidth is None else fixed_bandwidth
            _,error=linear_fit(parameters[:2],bandwidth);error=error.ravel()
            return np.r_[error.real,error.imag]
        initial=[true_delay,fake_delay,.9] if fixed_bandwidth is None else [true_delay,fake_delay]
        bounds=([-2.,-2.,.5],[2.,2.,1.2]) if fixed_bandwidth is None else (-2.,2.)
        fit=least_squares(residual,initial,bounds=bounds,max_nfev=80)
        true_delay,fake_delay=fit.x[:2]
        bandwidth=fit.x[2] if fixed_bandwidth is None else fixed_bandwidth
        coefficients,_=linear_fit([true_delay,fake_delay],bandwidth)
        def single_residual(delay):
            _,error=linear_fit([delay[0]],delay[1] if fixed_bandwidth is None else fixed_bandwidth);error=error.ravel()
            return np.r_[error.real,error.imag]
        initial_one=[fake_delay,bandwidth] if fixed_bandwidth is None else [fake_delay]
        bounds_one=([-2.,.5],[2.,1.2]) if fixed_bandwidth is None else (-2.,2.)
        one=least_squares(single_residual,initial_one,bounds=bounds_one,max_nfev=80)
        count=2*m*n
        bic_one=np.sum(one.fun**2)/noise_variance+(1+2*m+int(fixed_bandwidth is None))*np.log(count)
        bic_two=np.sum(fit.fun**2)/noise_variance+(2+4*m+int(fixed_bandwidth is None))*np.log(count)
        if bic_one<bic_two:
            true_delay=one.x[0]
            bandwidth=one.x[1] if fixed_bandwidth is None else fixed_bandwidth
            coefficients,_=linear_fit([true_delay],bandwidth)
        bandwidths.append(float(bandwidth))
        double.append(bic_two<=bic_one)
        pairs.append(np.stack((coefficients[0],coefficients[-1])));delays.append((true_delay,fake_delay))
    pairs=np.array(pairs);double=np.array(double);best=-np.inf
    for choice in range(2**s):
        labels=(choice>>np.arange(s))&1
        directions=pairs[np.arange(s),labels].T/a
        directions/=np.linalg.norm(directions,axis=0)
        values,vectors=np.linalg.eigh(directions@directions.conj().T)
        evidence=values[-1]/np.sum(values)
        if double.sum()>=3:
            source=pairs[np.arange(s)[double],1-labels[double]].T
            source/=np.linalg.norm(source,axis=0)
            power=np.linalg.eigvalsh(source@source.conj().T)
            evidence+=power[-1]/np.sum(power)
        if evidence>best:
            best=evidence;chosen=labels.copy()
    directions=pairs[np.arange(s),chosen].T/a
    directions/=np.linalg.norm(directions,axis=0)
    values,vectors=np.linalg.eigh(directions@directions.conj().T)
    raw_directions=directions*a
    common=np.linalg.eigvalsh(raw_directions@raw_directions.conj().T)
    usable=values[-1]>common[-1]
    gain=vectors[:,-1];gain/=gain[0]
    information=[]
    for star in range(s):
        selected=chosen[star] if double[star] else 0
        templates=whiten@shape(lag[:,None],np.array(delays[star])[None,:],bandwidths[star])
        column=templates[:,selected]
        if double[star]:
            nuisance=templates[:,1-selected]
            column=column-nuisance*np.vdot(nuisance,column)/np.vdot(nuisance,nuisance)
        information.append(float(np.vdot(column,column).real*np.sum(abs(pairs[star,chosen[star]])**2)))
    _,weighted_vectors=np.linalg.eigh((directions*np.array(information))@directions.conj().T)
    gain=weighted_vectors[:,-1];gain/=gain[0]
    return np.exp(1j*np.angle(gain)),dict(delays=np.array(delays).tolist(),components=np.stack((pairs.real,pairs.imag),axis=-1).tolist(),labels=chosen.tolist(),double=double.tolist(),bandwidths=bandwidths,coherence=float(best),usable=bool(usable),sky_residual=float(s-values[-1]),common_residual=float(s-common[-1]))
