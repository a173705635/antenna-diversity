"""Independent-event dual-polarization correlator model, not RF/tracking/PVT."""
import hashlib
import json
from pathlib import Path
import numpy as np
from antenna import jones_vector
from config import WavePolarization


def wilson(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-half),float(mid+half)]


def main():
    out=Path(__file__).resolve().parents[1]/'docs/reports/data/2026-09-12-dual-polarization-model'
    out.mkdir(parents=True,exist_ok=False)
    rh=jones_vector(WavePolarization(kind='rhcp')); lh=jones_vector(WavePolarization(kind='lhcp'))
    def samples(seed,n,kind,xpd=20.,gain_l_db=0.):
        rng=np.random.default_rng(seed); leak=10**(-xpd/20)
        ports=np.column_stack([rh+leak*lh,lh+leak*rh])/np.sqrt(1+leak**2)
        phase=rng.uniform(-np.pi,np.pi,n)
        # Echo after prompt correlation: 0.5 voltage * triangular C/A R(.25)=.75.
        if kind=='los': field=np.tile(rh,(n,1))
        elif kind=='nlos_lhcp': field=np.tile(lh,(n,1))
        elif kind=='nlos_rhcp': field=np.tile(rh,(n,1))
        else:
            echo=lh if kind=='mp_lhcp' else rh
            field=rh[None,:]+(.375*np.exp(1j*phase))[:,None]*echo
        signal=field@ports.conj()
        # Ideal 100-ms shared-reference coherent correlator, C/N0=45 dB-Hz
        # complex noise variance normalized to unit co-pol signal before port loss.
        var=1/(10**4.5*.1)
        noise=np.sqrt(var/2)*(rng.normal(size=(n,2))+1j*rng.normal(size=(n,2)))
        measured=signal+noise
        measured[:,1]*=10**(gain_l_db/20)
        ratio=20*np.log10(np.maximum(abs(measured[:,0]),1e-15)/np.maximum(abs(measured[:,1]),1e-15))
        phase_error=np.angle(signal[:,0])
        return ratio,phase_error
    calibration,_=samples(8100,1000,'los')
    threshold=float(np.quantile(calibration,.01,method='lower'))
    cases=[('los',20.,0.),('mp_lhcp',20.,0.),('mp_rhcp',20.,0.),
           ('nlos_lhcp',20.,0.),('nlos_rhcp',20.,0.),('los',10.,0.),('los',20.,3.)]
    rows=[]
    for i,(kind,xpd,gain) in enumerate(cases):
        ratio,phase=samples(8200+i,2000,kind,xpd,gain)
        hits=int(np.count_nonzero(ratio<threshold)); n=len(ratio)
        rows.append({'case':kind,'xpd_db':xpd,'unreported_lhcp_gain_db':gain,'seed':8200+i,
            'events':n,'anomaly_events':hits,'anomaly_fraction':hits/n,'wilson95':wilson(hits,n),
            'ratio_db_median':float(np.median(ratio)),
            'noiseless_prompt_phase_abs_max_rad':float(abs(phase).max())})
        np.savez_compressed(out/f'events_{i}.npz',ratio_db=ratio,noiseless_phase_rad=phase)
    np.savez_compressed(out/'calibration.npz',ratio_db=calibration)
    result={'scope':__doc__,'calibration_seed':8100,'calibration_events':1000,
        'threshold_db':threshold,'decision':'ratio below frozen LOS 1st percentile => polarization anomaly, NOT NLOS proof',
        'assumptions':{'coherent_s':.1,'cn0_db_hz':45,'xpd_calibration_db':20,
                       'mp_voltage':.5,'mp_delay_chips':.25,'reference':'ideal common aligned reference; no PLL/DLL'},
        'rows':rows,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('antenna.py'),Path(__file__).with_name('config.py')]}}
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (out/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir() if p.name!='sha256.json'},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
