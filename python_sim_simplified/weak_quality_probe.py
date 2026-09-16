"""Known continuously present signal: investigate quality-estimator false loss."""
import numpy as np
from tracking import estimate_cn0_m2m4_db_hz
from weak_ten_study import ROOT,save


def run():
    rng=np.random.default_rng(116006)
    signals=1+(rng.normal(size=(20,2000))+1j*rng.normal(size=(20,2000)))/np.sqrt(2)
    rows=[]
    for width in (10,50,100):
        lost=[];missing=[]
        for signal in signals:
            cn=[];lock=[]
            for k in range(len(signal)):
                window=signal[max(0,k-width+1):k+1]
                cn.append(estimate_cn0_m2m4_db_hz(window,.001))
                lock.append(abs(np.sum(window**2))/np.sum(abs(window)**2))
            cn=np.array(cn);snr=10**(cn/10)*.001
            good=(np.array(lock)>=.8*snr/(1+snr))&(cn>=25)
            bad_runs=np.convolve((~good[100:]).astype(int),np.ones(5,int),mode='valid')
            lost.append(bool(np.any(bad_runs==5)));missing.append(float(np.mean(~np.isfinite(cn[100:]))))
        rows.append(dict(window_ms=width,trials=20,trials_with_false_loss=sum(lost),nonfinite_cn0_fraction=float(np.mean(missing))))
    save(ROOT/'round05/quality_feasibility.json',dict(seed=116006,true_cn0=30.,scope='constant-phase correlator simulation; not RF navigation',rows=rows))
    print(rows)


if __name__=='__main__':run()
