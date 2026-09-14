"""C/A waveform joint temporal/spatial separation; not RF/DFE/PLL validation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from gnss import ca_code
from audit_spoofing_spatial import response
from diversity import combine_lcmv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    fs=4.092e6;n=40920;t=np.arange(n)/fs;code=ca_code(1)
    period=np.repeat(code,4);spectrum=np.fft.fft(period)
    def wave(delay,hz):
        # Periodic bandlimited interpolation retains off-grid fractional delay.
        shifted=np.fft.ifft(spectrum*np.exp(-2j*np.pi*np.fft.fftfreq(len(period))*delay*4))
        return np.tile(shifted,10)*np.exp(2j*np.pi*hz*t)
    grid=[(d,f) for d in [0,.25,.5,1.,2.] for f in [0.,50.,250.,500.]]
    bank=np.array([wave(*p) for p in grid]);bc=bank.conj()
    # Fixed known target array response, but no true temporal parameters to search.
    a=response(10,25);b=response(100,10);q=np.array([-np.conj(a[1]),1])/np.sqrt(2)
    true=wave(0,0);rng=np.random.default_rng(9730);noise_power=fs/10**4.5
    rows=[];raw=[]
    cases=[(0,0),(.25,0),(.5,0),(1,0),(0,50),(0,250),(.25,50),(.37,80)]
    for delay,hz in cases:
        fake=wave(delay,hz)*np.exp(.7j)
        correlation=float(abs(np.vdot(true,fake))/n)
        oracle_condition=float(np.linalg.cond(np.array([[1,np.vdot(true,fake)/n],[np.vdot(fake,true)/n,1]])))
        values=[]
        for trial in range(20):
            x=a[:,None]*true+b[:,None]*fake
            x+=np.sqrt(noise_power/2)*(rng.normal(size=x.shape)+1j*rng.normal(size=x.shape))
            z=q.conj()@x
            fi=int(np.argmax(abs(bc@z)**2));v=bank[fi]
            # Remove the selected waveform in time, then find genuine waveform.
            residual=x-(x@v.conj()/n)[:,None]*v
            y=a.conj()@residual/np.vdot(a,a)
            cross=bc@v/n;den=1-abs(cross)**2
            score=abs(bc@y/n)**2/np.maximum(den,1e-12);score[den<1e-8]=-1
            ti=int(np.argmax(score));u=bank[ti]
            design=np.column_stack([u,v]);gram=design.conj().T@design/n
            # Fit both complex antenna responses, only after the data-driven search.
            coefficients=np.linalg.solve(gram,design.conj().T@x.T/n).T
            est=coefficients[:,1]
            _,info=combine_lcmv(x,a,[est],.001,.99);w=info['weights']
            suppression=-20*np.log10(max(abs(np.vdot(w,b)),1e-12))
            # Oracle separation knows both true waveforms; singular case stays unavailable.
            oracle=None
            if oracle_condition<1e10:
                td=np.column_stack([true,fake]);g=td.conj().T@td/n
                c=np.linalg.solve(g,td.conj().T@x.T/n).T
                _,oi=combine_lcmv(x,a,[c[:,1]],.001,.99)
                oracle=-20*np.log10(max(abs(np.vdot(oi['weights'],b)),1e-12))
            values.append({'trial':trial,'fake_grid':grid[fi],'true_grid':grid[ti],
                'search_suppression_db':float(suppression),'oracle_suppression_db':oracle,
                'null_accepted':not bool(info['rejected_constraint_indices']),
                'fitted_gram_condition':float(np.linalg.cond(gram)),
                'noise_gain_db':float(10*np.log10(np.vdot(w,w).real))})
        raw.append({'delay_chips':delay,'doppler_hz':hz,'trials':values})
        oracle_vals=[v['oracle_suppression_db'] for v in values if v['oracle_suppression_db'] is not None]
        rows.append({'delay_chips':delay,'doppler_hz':hz,'waveform_correlation':correlation,
            'oracle_gram_condition':oracle_condition,'trials':len(values),
            'both_grid_correct':sum(tuple(v['fake_grid'])==(delay,hz) and tuple(v['true_grid'])==(0,0) for v in values),
            'accepted':sum(v['null_accepted'] for v in values),
            'search_suppression_db_p05_median':np.quantile([v['search_suppression_db'] for v in values],[.05,.5]).tolist(),
            'oracle_suppression_db_median':float(np.median(oracle_vals)) if oracle_vals else None})
    result={'scope':__doc__,'seed':9730,'sample_rate_hz':fs,'duration_s':n/fs,'prn':1,
        'cn0_db_hz_per_component':45,'noise_power':noise_power,'power_ratio_db':0,
        'grid':grid,'rows':rows,'assumptions':'Periodic bandlimited interpolation of 4-sample/chip C/A; constant nav bit; exact known target response; calibrated equal channel gains; no RF/PLL. No reliability classifier.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name,data in [('summary.json',result),('trials.json',raw)]:
        (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    (out/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
