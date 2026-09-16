"""Array acquisition proposals: measured power only, no transmitter truth."""
import numpy as np
from acquisition import acquire
from gnss import ca_code
from code_replica import bandlimited_code_replica,_sample_code


def select(power,record,cfg):
    row,code=np.unravel_index(np.argmax(power),power.shape)
    distance=abs(np.arange(power.shape[1])-code)
    distance=np.minimum(distance,power.shape[1]-distance)
    other=power[row].copy();other[distance<=record['exclusion_samples']]=-np.inf
    second=int(np.argmax(other));metric=power[row,code]/other[second]
    result=dict(record)
    result.update(success=bool(metric>=cfg.threshold),code_phase_samples=int(code),
        doppler_hz=float(cfg.doppler_bins_hz[row]),peak=float(power[row,code]),
        second_peak=float(other[second]),second_peak_code_phase_samples=second,
        metric=float(metric),search_power=power)
    return result


def acquire_array(x,fs,prn,rate,cfg):
    record=acquire(x[0],fs,prn,rate,cfg);unit=round(fs*.001)
    samples=np.arange(unit);code=ca_code(prn)
    local=_sample_code(bandlimited_code_replica(code,rate,.9e6),samples*rate/fs)
    replica=np.fft.fft(local).conj();blocks=x.reshape(len(x),-1,unit)
    power=[];differentials=[]
    for frequency in cfg.doppler_bins_hz:
        correlation=np.fft.ifft(np.fft.fft(blocks*np.exp(-2j*np.pi*frequency*samples/fs),axis=2)*replica,axis=2)
        differential=np.sum(correlation[:,1:]*correlation[:,:-1].conj(),axis=(0,1))
        power.append(abs(differential));differentials.append(differential)
    power=np.array(power);result=select(power,record,cfg)
    row,code=np.unravel_index(np.argmax(power),power.shape);coarse=result['doppler_hz']
    phase=np.angle(differentials[row][code]*np.exp(-2j*np.pi*coarse*.001))
    fine=coarse+phase/(2*np.pi*.001)
    frequency=cfg.doppler_bins_hz[row]
    corr=np.fft.ifft(np.fft.fft(blocks*np.exp(-2j*np.pi*frequency*samples/fs),axis=2)*replica,axis=2)[:,:,code]
    from scipy.optimize import minimize_scalar
    time=np.arange(corr.shape[1])*.001
    squared=corr*corr
    def objective(frequency):
        return -np.sum(abs(squared@np.exp(-4j*np.pi*frequency*time))**2)
    grid=fine+np.arange(-40,41,2.)
    center=grid[np.argmin([objective(f) for f in grid])]
    fit=minimize_scalar(objective,bounds=(center-2,center+2),method='bounded')
    from scipy.optimize import least_squares
    def residual(parameters):
        model=np.exp(4j*np.pi*(parameters[0]*time+5*parameters[1]*time*time))
        error=squared-(squared@model.conj()/len(time))[:,None]*model
        return np.r_[error.real.ravel(),error.imag.ravel()]/np.linalg.norm(squared)
    best=np.inf;initial=None
    for drift in np.arange(-75.,76.,5.):
        for frequency in fit.x+np.arange(-40.,41.,2.):
            cost=np.sum(residual([frequency,drift])**2)
            if cost<best:best=cost;initial=[frequency,drift]
    dynamic=least_squares(residual,initial,bounds=([fit.x-42,-100],[fit.x+42,100]))
    result['constant_doppler_hz']=float(fit.x)
    count=2*squared.size
    constant_rss=np.sum(residual([fit.x,0.])**2)
    dynamic_rss=np.sum(dynamic.fun**2)
    use_dynamic=count*np.log(dynamic_rss/constant_rss)+np.log(count)<0
    result['doppler_hz']=float(dynamic.x[0] if use_dynamic else fit.x)
    result['doppler_rate_hz_s']=float(dynamic.x[1]*10 if use_dynamic else 0.)
    result['dynamic_model']=bool(use_dynamic)
    result['doppler_hz']-=result['doppler_rate_hz_s']*(unit-1)/(2*fs)
    result['doppler_reference_time_s']=0.
    return result
