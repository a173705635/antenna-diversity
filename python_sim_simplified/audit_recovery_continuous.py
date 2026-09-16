"""Short continuous baseband check with existing acquisition/DLL/PLL/PVT.

Independent rectangular-pulse Fourier source, 0 Hz Doppler, +1 navigation bits.
No stop-and-hop noise reduction. This is not a long-duration RF hardware test.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from acquisition import acquire
from code_replica import bandlimited_code_replica, _sample_code
from gnss import ca_code
from positioning import code_pseudorange
from presets import default_config, mate70_phase_centers
from run_gps_spp import receiver_reference, serializable
from signal_recovery import C, FS, WAVELENGTH, sky, steering, array_weights, position_solution, refine_source
from tracking import track


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=.6)
    parser.add_argument('--iterations',type=int,default=2)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(850001);directions=sky(850001,8)
    antennas=np.array(list(mate70_phase_centers().values()))
    a=steering(directions,antennas,0.)
    source=np.array([1.,0.,0.]);v=steering(source[None],antennas,0.)[:,0]
    count=round(FS*args.duration);time=np.arange(count)/FS
    radii=21e6+np.arange(8)*17000.
    rho=radii+75000.
    forged=np.linalg.norm(directions*radii[:,None]-np.array([100.,0.,0.]),axis=1)+75000.
    clean=np.zeros((4,count),complex);fake=np.zeros_like(clean)
    spectra=[]
    for i in range(8):
        code=bandlimited_code_replica(ca_code(i+1),1.023e6,.9e6)
        spectra.append(np.fft.fft(_sample_code(code,np.arange(2046)*.5)))
        for delay,vector,target in ((rho[i],a[:,i],clean),(forged[i],v,fake)):
            signal=_sample_code(code,(time-delay/C)*1.023e6)*np.exp(-2j*np.pi*delay/WAVELENGTH)
            target+=vector[:,None]*signal
    noise=(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))/np.sqrt(2)
    cfg=default_config();cfg.acquisition.doppler_bins_hz=np.array([0.])
    cfg.tracking.max_epochs=round(args.duration*1000);cfg.tracking.code_replica_bandwidth_hz=.9e6
    cfg.tracking.refine_initial_phase=True;cfg.tracking.carrier_discriminator='costas'
    reference,rotation=receiver_reference();satellites=reference+(directions*radii[:,None])@rotation
    summaries=[]
    for attack in (False,True):
        x=clean+np.sqrt(FS/10**4.5)*noise
        if attack:x[:,time>=.15]+=np.sqrt(10.)*fake[:,time>=.15]
        y=np.empty((8,count),complex);refined=np.empty_like(y);ratios=[]
        width=round(.02*FS)
        for start in range(0,count,width):
            block=x[:,start:start+width]
            weights,_,info=array_weights(block,a,'gated')
            y[:,start:start+width]=weights.conj().T@block
            ratios.append(info['eigen_ratio'])
            coherent=block.reshape(4,20,2046).mean(axis=1)
            rw,_,ri=array_weights(coherent,a,'gated')
            if ri['eigen_ratio']>=3.:
                rw,_=refine_source(coherent,np.array(spectra),a,np.round(rho/10000)*10000,rw,args.iterations,.05)
            refined[:,start:start+width]=rw.conj().T@block
        # A whole block's covariance is available at its END. Keep this latency.
        evaluation=np.arange(.35,args.duration-.01,.01);histories={}
        for method,signals in (('single',np.tile(x[0],(8,1))),('gated',y),('refined_coherent',refined)):
            ranges=np.full((len(evaluation),8),np.nan);locked=[]
            for i in range(8):
                acq=acquire(signals[i],FS,i+1,1.023e6,cfg.acquisition)
                if acq['success']:
                    h=track(signals[i],FS,i+1,acq,cfg.tracking,1575.42e6,1.023e6)
                    ranges[:,i]=code_pseudorange(h,evaluation,FS,.001,round(rho[i]/10000)*10000)
                    histories[f'{method}_prn{i+1}']={k:h[k] for k in ('status','success')}
                    locked.append(bool(h['success']))
                else:locked.append(False)
            p=np.array([position_solution(satellites,r,np.isfinite(r))[0] for r in ranges])
            error=np.linalg.norm(p[:,:2],axis=1);valid=np.isfinite(error)
            summaries.append(dict(attack=attack,method=method,acquired_tracking=locked,
                solved=int(valid.sum()),opportunities=len(valid),
                horizontal_p95_m=float(np.quantile(error[valid],.95)) if valid.any() else None,
                good20=int((error<=20).sum()),minimum_evidence_latency_s=.02,
                refinement_iterations=args.iterations,duration_s=args.duration,
                late_horizontal_p95_m=float(np.quantile(error[valid & (evaluation>=args.duration-.2)],.95))
                    if np.any(valid & (evaluation>=args.duration-.2)) else None))
            np.savez_compressed(args.output/f'{int(attack)}_{method}.npz',time_s=evaluation,
                                position=p,ranges=ranges,eigen_ratios=ratios)
        (args.output/f'{int(attack)}_tracking.json').write_text(json.dumps(histories,indent=2))
    (args.output/'summary.json').write_text(json.dumps(serializable(summaries),indent=2,allow_nan=False))
    print(json.dumps(serializable(summaries)),flush=True)


if __name__=='__main__':main()
