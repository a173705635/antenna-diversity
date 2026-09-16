"""Explicit Doppler time-reference correction; truth only in external scoring."""
import shutil
import numpy as np
from array_dynamic_hundred import acquire_array
from research_hundred import ROOT,load
from code_replica import bandlimited_code_replica,_sample_code
from gnss import ca_code
from presets import default_config
from continuous_improvement import Case,generate
from audit_recovery_fifty import save


def run():
    folder=ROOT/'iteration038';rows=[];fs=2046000.
    old=load(ROOT/'iteration035/array_acquisition.py').acquire_array
    cfg=default_config().acquisition;cfg.doppler_bins_hz=np.arange(-500,501,250.)
    t=np.arange(round(fs*.1))/fs
    code=_sample_code(bandlimited_code_replica(ca_code(1),1.023e6,.9e6),t*1.023e6-23.)
    bits=np.where(np.floor(t/.02).astype(int)%2,1.,-1.)
    for rate in (-500.,0.,500.):
        x=(np.exp(1j*np.array([.2,-1.,2.,.8]))[:,None]*code*bits
           *np.exp(2j*np.pi*(123.*t+.5*rate*t*t)))
        before=old(x,fs,1,1.023e6,cfg);after=acquire_array(x,fs,1,1.023e6,cfg)
        rows.append(dict(kind='analytic_noiseless',true_initial_hz=123.,true_rate=rate,
            old_frequency=before['doppler_hz'],new_frequency=after['doppler_hz'],
            new_rate=after['doppler_rate_hz_s']))
    cfg.doppler_bins_hz=np.arange(-4000,4001,250.)
    x,*_=generate(Case(duration=.1,cn0=35.,doppler_rate=500.),41007,False)
    for prn in (1,3,6,8):
        before=old(x,fs,prn,1.023e6,cfg);after=acquire_array(x,fs,prn,1.023e6,cfg)
        rows.append(dict(kind='dynamic_multisatellite_RF',prn=prn,true_initial_hz=-1500+(prn-1)*500.,
            true_rate=500*(-1+2*(prn-1)/7),old_frequency=before['doppler_hz'],
            new_frequency=after['doppler_hz'],new_rate=after['doppler_rate_hz_s']))
    for name in ('array_dynamic_hundred.py','audit_dynamic_hundred.py'):shutil.copy2(name,folder/name)
    save(folder/'results.json',dict(rows=rows));print(rows,flush=True)


if __name__=='__main__':run()
