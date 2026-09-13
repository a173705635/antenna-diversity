"""Recompute audit metrics from saved arrays and produce the report figure."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]/'python_sim_simplified'))
from audit_real_gnss import navigation_check


def main():
    checked=0
    for summary_path in ROOT.glob('*/summary.json'):
        rows=json.loads(summary_path.read_text())['rows']
        manifest=json.loads((summary_path.parent/'manifest.json').read_text())
        fs=manifest['dfe_info']['fs_out_hz']
        for row in rows:
            if 'tracking' not in row: continue
            z=np.load(summary_path.parent/f"prn{row['prn']:02d}.npz")
            m=z['epoch_start_samples']/fs>=.5
            p=z['prompt'][m]
            expected={
                'pll_rms_rad':np.sqrt(np.mean(z['pll_error_rad'][m]**2)),
                'bit_insensitive_phase_rms_rad':np.sqrt(np.mean((np.angle(p*p)/2)**2)),
                'bit_insensitive_lock':abs(np.sum(p*p))/np.sum(abs(p)**2),
                'doppler_std_hz':np.std(z['carrier_nco_frequency_hz'][m]),
                'doppler_median_hz':np.median(z['carrier_nco_frequency_hz'][m]),
            }
            for key,value in expected.items():
                assert np.isclose(value,row['tracking'][key],rtol=1e-12,atol=1e-12),(summary_path,key)
                checked+=1
            # Accumulated phase must preserve all turns. Verify state update,
            # rather than wrapping the observable into a wavelength interval.
            dt=np.diff(z['epoch_start_samples'])/fs
            increment=2*np.pi*z['carrier_nco_frequency_hz'][1:]*dt+manifest['tracking']['pll_kp']*z['pll_error_rad'][:-1]
            assert np.max(abs(np.diff(z['carrier_nco_phase_rad'])-increment))<1e-8
    a=np.load(ROOT/'legacy_start1s_9s'/'prn07.npz')
    b=np.load(ROOT/'costas_start1s_9s'/'prn07.npz')
    t=b['epoch_start_samples']/5e6+1
    fig,ax=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
    for label,z in [('Original atan2',a),('Costas',b)]:
        ax[0].plot(t,z['carrier_nco_frequency_hz'],label=label,lw=.7)
    ax[0].set(ylabel='Carrier NCO frequency (Hz)',title='Live-sky GPS PRN 7: identical IQ, independent tracking runs')
    ax[0].legend();ax[0].grid(alpha=.3)
    ax[1].plot(t,(a['carrier_nco_phase_rad']-b['carrier_nco_phase_rad'])/(2*np.pi),lw=.8)
    ax[1].set(ylabel='Original minus Costas (cycles)',title='Direct accumulated phase difference; no modulo (not truth error)')
    ax[1].grid(alpha=.3)
    sel=(t>=2)&(t<2.2)
    p=b['prompt']; scale=np.median(abs(p[sel]))
    ax[2].plot(t[sel],p.real[sel]/scale,label='Prompt I')
    ax[2].plot(t[sel],p.imag[sel]/scale,label='Prompt Q')
    ax[2].set(xlabel='Time from recording start (s)',ylabel='Normalized correlator',title='Navigation bit changes remain in prompt I, not absorbed by the carrier loop')
    ax[2].legend();ax[2].grid(alpha=.3)
    fig.savefig(ROOT/'tracking_comparison.png',dpi=150);plt.close(fig)
    nav_rows=[]
    for label in ('costas_start1s_9s','legacy_start1s_9s','costas_quant8_start1s_9s'):
        for path in (ROOT/label).glob('prn*.npz'):
            d=navigation_check(np.load(path))
            nav_rows.append({'run':label,'prn':int(path.stem[3:]),
                'demodulation_evidence':d['demodulation_evidence'],
                'valid_words':d['valid_words'],'tested_words':d['tested_words'],
                'longest_chain':d['longest_consecutive_valid_words'],'preambles':d['preambles'],
                'unique_payload_count':len(set(w['payload'] for w in d['words'] if w['payload'] is not None))})
    expected=json.loads((ROOT/'navigation_acceptance.json').read_text())['rows']
    assert nav_rows==expected,'navigation acceptance differs from retained evidence'
    checks={'recomputed_scalar_metrics':checked,'nco_increment_checks':'all passed',
        'navigation_records_recomputed':len(nav_rows),
        'scope':'Internal recomputation; no absolute phase or position truth'}
    (ROOT/'verification.json').write_text(json.dumps(checks,indent=2))
    print(checks)


if __name__=='__main__': main()
