"""Space-delay calibration: synthetic correlation-shape model, not RF verification."""
import numpy as np
from scipy.optimize import least_squares,minimize_scalar


def shape(lag,delay,bandwidth=.9):
    # Fourier correlation of rectangular BPSK chips through a 0.9 MHz front end.
    frequency=np.linspace(0,bandwidth/1.023,65)
    power=np.sinc(frequency)**2;power[[0,-1]]*=.5
    return np.sum(np.cos(2*np.pi*(lag-delay)[...,None]*frequency)*power,axis=-1)/power.sum()


def fit_components(z,a,lag,fixed_bandwidth,channel_delays):
    m,s,n=z.shape;flat=z.reshape(m,-1)
    noise_cov=.99*shape(lag[:,None],lag[None,:],.9 if fixed_bandwidth is None else fixed_bandwidth)+.01*np.eye(n)
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
        bic_one=count*np.log(np.sum(one.fun**2)/count)+(2+2*m)*np.log(count)
        bic_two=count*np.log(np.sum(fit.fun**2)/count)+(3+4*m)*np.log(count)
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
    return np.exp(1j*np.angle(gain)),dict(delays=np.array(delays).tolist(),components=np.stack((pairs.real,pairs.imag),axis=-1).tolist(),labels=chosen.tolist(),double=double.tolist(),bandwidths=bandwidths,coherence=float(best),usable=bool(usable),sky_residual=float(s-values[-1]),common_residual=float(s-common[-1]))


def estimate(z,a,lag):
    _,initial=fit_components(z,a,lag,None,np.zeros(len(z)))
    common=float(np.median(initial['bandwidths']))
    _,model=fit_components(z,a,lag,common,np.zeros(len(z)))
    components=np.array(model['components']);components=components[...,0]+1j*components[...,1]
    isolated=z.copy();double=np.array(model['double'])
    if double.sum()>=3:
        for s in np.flatnonzero(double):
            label=model['labels'][s]
            isolated[:,s]-=components[s,label,:,None]*shape(lag,model['delays'][s][label],common)
        isolated=isolated[:,double]
    spectrum=np.fft.fft(isolated,axis=2)
    frequency=np.fft.fftfreq(len(lag),lag[1]-lag[0]);selected=np.flatnonzero(abs(frequency)<.7)
    selected=selected[np.argsort(frequency[selected])];shifts=[0.]
    for m in range(1,len(z)):
        cross=np.sum(spectrum[m]*spectrum[0].conj(),axis=0)[selected]
        h=np.column_stack((np.ones(len(selected)),frequency[selected]))
        weight=np.sqrt(abs(cross))
        phase=np.unwrap(np.angle(cross))
        slope=np.linalg.lstsq(h*weight[:,None],phase*weight,rcond=None)[0][1]
        shift=-slope/(2*np.pi);shifts.append(float(shift))
    gain,info=fit_components(z,a,lag,common,np.array(shifts));info['channel_delay_chips']=shifts
    return gain,info
