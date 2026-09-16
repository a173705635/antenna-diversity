"""W5 numeric calibration for supplied bench references; not hardware certification.

RF input is a frequency-ordered complex sweep, condition/frequency/channel.
External fixture transfer and coarse relative phase anchors are explicit.
IMU reference force/rate must come from independent metrology, not its solution.
"""
import numpy as np


def rf_features(temperature_c, gain_db, boot_id, frequency_hz, *, boots):
    temp=np.asarray(temperature_c)[:,None]-25.
    gain=np.asarray(gain_db)[:,None]
    frequency=(np.asarray(frequency_hz)[None,:]-1575.42e6)/1e6
    shape=(len(temperature_c),len(frequency_hz))
    columns=[np.ones(shape),np.broadcast_to(temp,shape),np.broadcast_to(gain,shape),
             np.broadcast_to(frequency,shape),temp*frequency]
    columns += [np.broadcast_to((np.asarray(boot_id)==boot)[:,None],shape) for boot in boots[1:]]
    return np.stack(columns,axis=-1)


def rf_observations(transfer, fixture_transfer, phase_anchor_rad):
    corrected=np.asarray(transfer)/fixture_transfer
    ratio=corrected[:,:,1:]/corrected[:,:,:1]
    phase=np.unwrap(np.angle(ratio),axis=1)
    middle=phase.shape[1]//2
    phase+=2*np.pi*np.rint((phase_anchor_rad-phase[:,middle,:])/(2*np.pi))[:,None,:]
    return phase,20*np.log10(abs(ratio))


def fit_rf(data):
    boots=np.unique(data['boot_id'])
    x=rf_features(data['temperature_c'],data['gain_db'],data['boot_id'],data['frequency_hz'],boots=boots)
    phase,gain=rf_observations(data['transfer'],data['fixture_transfer'],data['phase_anchor_rad'])
    design=x.reshape(-1,x.shape[-1]);y=phase.reshape(-1,phase.shape[-1])
    coefficient,_,rank,_=np.linalg.lstsq(design,y,rcond=None)
    gain_coefficient=np.linalg.lstsq(design,gain.reshape(y.shape),rcond=None)[0]
    residual=y-design@coefficient
    return dict(phase_coefficient=coefficient,gain_coefficient=gain_coefficient,
                relative_delay_s=-coefficient[3]/(2*np.pi*1e6),
                delay_temperature_s_per_c=-coefficient[4]/(2*np.pi*1e6),
                feature_covariance=np.linalg.inv(design.T@design),
                residual_covariance=residual.T@residual/(len(y)-rank),rank=int(rank),
                boots=boots,temperature_range_c=[min(data['temperature_c']),max(data['temperature_c'])],
                gain_values_db=np.unique(data['gain_db']),
                frequency_range_hz=[min(data['frequency_hz']),max(data['frequency_hz'])])


def evaluate_rf(model,data):
    x=rf_features(data['temperature_c'],data['gain_db'],data['boot_id'],data['frequency_hz'],boots=model['boots'])
    phase,gain=rf_observations(data['transfer'],data['fixture_transfer'],data['phase_anchor_rad'])
    inside=((data['temperature_c']>=model['temperature_range_c'][0])
            &(data['temperature_c']<=model['temperature_range_c'][1])
            &np.isin(data['gain_db'],model['gain_values_db'])&np.isin(data['boot_id'],model['boots']))
    frequencies=((data['frequency_hz']>=model['frequency_range_hz'][0])
                 &(data['frequency_hz']<=model['frequency_range_hz'][1]))
    return dict(phase_residual_rad=phase-x@model['phase_coefficient'],
                gain_residual_db=gain-x@model['gain_coefficient'],
                calibration_valid=inside[:,None]&frequencies[None,:])


def fit_imu(reference, measured, temperature_c):
    design=np.column_stack((np.ones(len(reference)),np.asarray(temperature_c)-25.,reference))
    coefficient,_,rank,_=np.linalg.lstsq(design,measured,rcond=None)
    residual=measured-design@coefficient
    return dict(bias=coefficient[0],temperature_coefficient=coefficient[1],
                matrix=coefficient[2:].T,rank=int(rank),
                feature_covariance=np.linalg.inv(design.T@design),
                residual_covariance=residual.T@residual/(len(reference)-rank))


def correct_imu(model, measured, temperature_c):
    centered=measured-model['bias']-(np.asarray(temperature_c)-25.)[:,None]*model['temperature_coefficient']
    return np.linalg.solve(model['matrix'],centered.T).T


def time_offset_profile(time_s, measured, reference_time_s, reference, offsets_s):
    """Dynamic bench alignment, positive lag means measured(t)=reference(t-lag).

Caller selects an interior interval covered by the reference for every lag.
Profile flatness/excitation must be inspected: an argmin alone is not proof.
"""
    dynamic=bool(np.any(np.ptp(measured,axis=0)) and np.any(np.ptp(reference,axis=0)))
    measured=measured-measured.mean(axis=0);scores=[]
    for lag in offsets_s:
        predicted=np.column_stack([np.interp(time_s-lag,reference_time_s,reference[:,i]) for i in range(3)])
        predicted-=predicted.mean(axis=0)
        scores.append(float(np.mean((measured-predicted)**2)))
    return dict(offset_s=float(offsets_s[np.argmin(scores)]) if dynamic else None,
                status='profile_minimum_only' if dynamic else 'unobservable_constant_signal',scores=scores,
                offsets_s=np.asarray(offsets_s),excitation=float(np.mean(measured**2)))


def long_run_covariance(residual, dt_s, lags):
    """Bartlett lag-window estimate of continuous noise intensity, not bias variance."""
    x=residual-residual.mean(axis=0);n=len(x)
    covariance=x.T@x/n
    for lag in range(1,lags+1):
        cross=x[lag:].T@x[:-lag]/n
        covariance+=(1-lag/(lags+1))*(cross+cross.T)
    return dt_s*covariance
