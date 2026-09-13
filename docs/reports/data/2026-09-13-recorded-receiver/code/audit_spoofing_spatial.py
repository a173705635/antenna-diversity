"""Known-direction dual-antenna LCMV capability, no RF/PVT claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from diversity import combine_lcmv

W = 299792458/1575.42e6


def response(az, el):
    # ENU azimuth clockwise from north; east baseline, received carrier sign.
    projection = .0759*np.cos(np.deg2rad(el))*np.sin(np.deg2rad(az))
    return np.array([1, np.exp(2j*np.pi*projection/W)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=False)
    azs = np.arange(0, 360, 5); els = np.arange(-10, 91, 5)
    targets = list(zip([10,55,100,145,190,235,280,325], [25,60,35,75,20,50,40,65]))
    rows = []
    for j,(az,el) in enumerate(targets):
        desired = response(az,el)
        for se in els:
            for sa in azs:
                source = response(sa,se)
                # Identity covariance: isolates geometry, not a measured covariance.
                _, info = combine_lcmv(np.eye(2,dtype=complex), desired, [source], .001, .99)
                w = info['weights']; accepted = not info['rejected_constraint_indices']
                for mode in range(4):
                    actual_target = response(az+3,el) if mode==1 else desired.copy()
                    actual_source = response(sa+3,se) if mode==2 else source.copy()
                    if mode==3:
                        error=10**(1/20)*np.exp(1j*np.deg2rad(5))
                        actual_target[1]*=error;actual_source[1]*=error
                    gain = np.vdot(w,actual_target); leak = abs(np.vdot(w,actual_source))
                    rows.append([j+1,sa,se,mode,int(accepted),
                        -20*np.log10(max(leak,1e-12)),20*np.log10(max(abs(gain),1e-12)),
                        np.angle(gain)*W/(2*np.pi)*1000,10*np.log10(np.vdot(w,w).real)])
    x = np.array(rows); np.savez_compressed(out/'scan.npz', rows=x)
    summary=[]
    for mode in range(4):
        a=x[x[:,3]==mode];v=a[a[:,4]==1]
        summary.append({'mode':mode,'grid_points':len(a),'accepted_points':len(v),
            'accepted_fraction':len(v)/len(a),'suppression_db_p05_median':np.quantile(v[:,5],[.05,.5]).tolist(),
            'target_abs_gain_error_db_p95':float(np.quantile(abs(v[:,6]),.95)),
            'target_phase_abs_mm_p95':float(np.quantile(abs(v[:,7]),.95)),
            'noise_gain_db_p95_max':np.quantile(v[:,8],[.95,1]).tolist(),
            'usable_illustrative_fraction_all_grid':float(np.mean((a[:,4]==1)&(a[:,5]>=20)&(abs(a[:,6])<=1)&(abs(a[:,7])<=3)&(a[:,8]<=3)))})
    # A different direction with precisely the same east projection.
    assert np.allclose(response(10,25),response(170,25))
    result={'scope':__doc__,'baseline_m':.0759,'correlation_rejection_threshold':.99,
        'columns':['prn','source_az_deg','source_el_deg','mode','accepted','suppression_db',
                   'target_gain_db','target_phase_mm','noise_gain_db_vs_single'],
        'modes':['ideal','target_az_error_3deg','source_az_error_3deg','B_gain_1dB_phase_5deg'],
        'rows':summary,'same_projection_counterexample':{'target':[10,25],'other':[170,25]},
        'interpretation':'Grid fractions are not sky-area probabilities. 240dB is numerical display floor, not hardware rejection.',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('diversity.py')]}}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(mode,col,title,lo,hi) in zip(axs.flat,[(0,8,'Ideal: noise gain vs one antenna (dB)',-3,15),(0,4,'Ideal: null constraint accepted',0,1),(2,5,'Source azimuth error 3 deg: suppression (dB)',0,40),(3,7,'B mismatch 1 dB / 5 deg: target phase (mm)',-10,10)]):
        q=x[(x[:,0]==1)&(x[:,3]==mode)];z=q[:,col].reshape(len(els),len(azs))
        if col!=4:z=np.where(q[:,4].reshape(z.shape)==1,z,np.nan)
        im=ax.pcolormesh(azs,els,z,shading='auto',vmin=lo,vmax=hi,cmap='viridis');fig.colorbar(im,ax=ax)
        ax.set_title(title);ax.set_xlabel('Source azimuth (deg)');ax.set_ylabel('Source elevation (deg)');ax.plot(10,25,'rx')
    fig.suptitle('75.9 mm east baseline; target az=10, el=25; blank = rejected null')
    fig.savefig(out/'capability.png',dpi=160);plt.close(fig)
    (out/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
