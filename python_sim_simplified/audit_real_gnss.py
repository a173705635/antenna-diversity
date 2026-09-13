"""Replay signed 8-bit interleaved live-sky IQ through the existing receiver.

Input is already hardware sampled. No synthetic RF, noise, truth or antenna.
The default input parameters describe the ION HackRF L1 recording.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from recorded_receiver import read_iq8, receive_channels
from gps_lnav import decode_lnav_word, navigation_check
from config import AcquisitionConfig, DFEConfig, TrackingConfig
from dfe import process_dfe


def save_json(path, obj):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, default=convert, indent=2, allow_nan=False), encoding='utf-8')


def summarize(h, fs):
    if not h.get('success'): return {'status':h['status']}
    t = h['epoch_start_samples']/fs
    use = t >= 0.5
    z = h['prompt'][use]
    # This bit-insensitive diagnostic applies to BOTH before/after runs.
    phase_error = .5*np.angle(z*z)
    m2 = np.mean(abs(z)**2); m4 = np.mean(abs(z)**4)
    signal = np.sqrt(max(0., 2*m2*m2-m4))
    noise = m2-signal
    cn0 = 10*np.log10(signal/noise/.001) if signal>0 and noise>0 else None
    return {'status':h['status'], 'epochs':len(t), 'start_s':float(t[0]), 'end_s':float(t[-1]),
        'evaluated_epochs':int(use.sum()), 'evaluation_start_s':.5,
        'finite_nco':bool(np.isfinite(h['carrier_nco_phase_rad']).all()),
        'pll_rms_rad':float(np.sqrt(np.mean(h['pll_error_rad'][use]**2))),
        'bit_insensitive_phase_rms_rad':float(np.sqrt(np.mean(phase_error**2))),
        'bit_insensitive_lock':float(abs(np.sum(z*z))/np.sum(abs(z)**2)),
        'configured_lock_median':float(np.median(h['phase_lock_metric'][use])),
        'm2m4_cn0_db_hz':None if cn0 is None else float(cn0),
        'dll_discriminator_rms':float(np.sqrt(np.mean(h['dll_error'][use]**2))),
        'doppler_median_hz':float(np.median(h['carrier_nco_frequency_hz'][use])),
        'doppler_std_hz':float(np.std(h['carrier_nco_frequency_hz'][use]))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iq',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--start',type=float,default=0.)
    p.add_argument('--fs',type=float,default=10e6)
    p.add_argument('--if-hz',type=float,default=420e3)
    p.add_argument('--prns',type=int,nargs='+',default=list(range(1,33)))
    p.add_argument('--mode',choices=['legacy','costas'],default='costas')
    p.add_argument('--output-bits',type=int,default=0)
    p.add_argument('--acquisition-only',action='store_true')
    args=p.parse_args()
    if args.seconds<=.5: p.error('need more than 0.5 seconds for evaluation')
    args.output.mkdir(parents=True,exist_ok=False)
    x=read_iq8(args.iq,round(args.start*args.fs),round(args.seconds*args.fs))
    raw_stats={'mean_i':float(x.real.mean()),'mean_q':float(x.imag.mean()),
        'rms':float(np.sqrt(np.mean(abs(x)**2))),
        'rail_fraction':float(np.mean((x.real==-128)|(x.real==127)|(x.imag==-128)|(x.imag==127)))}
    dcfg=DFEConfig(decimation=2,cutoff_hz=1.8e6,output_bits=args.output_bits,output_binary_shift=0)
    baseband,di=process_dfe(x[None,:],dcfg,args.fs,args.if_hz)
    del x
    fs=di['fs_out_hz']; x=baseband[0]
    acfg=AcquisitionConfig(doppler_bins_hz=np.arange(-10000.,10000.1,250.))
    tcfg=TrackingConfig(max_epochs=round(args.seconds*1000))
    if args.mode=='costas': tcfg.carrier_discriminator='costas'
    manifest={'arguments':vars(args),'dfe':asdict(dcfg),'acquisition':asdict(acfg),
        'tracking':asdict(tcfg),'input_stats':raw_stats,
        'input_start_sample':round(args.start*args.fs),'input_samples':round(args.seconds*args.fs),
        'dfe_info':di,'receiver_source_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in ('audit_real_gnss.py','recorded_receiver.py','gps_lnav.py','gps_broadcast.py',
                      'gps_observables.py','positioning.py','acquisition.py','tracking.py','config.py','dfe.py','gnss.py','code_replica.py')}}
    digest=hashlib.sha256()
    with args.iq.open('rb') as f:
        f.seek(round(args.start*args.fs)*2)
        remaining=round(args.seconds*args.fs)*2
        while remaining:
            block=f.read(min(remaining,1024*1024))
            if not block: raise ValueError('input truncated during hashing')
            digest.update(block); remaining-=len(block)
    manifest['input_window_sha256']=digest.hexdigest()
    save_json(args.output/'manifest.json',manifest)
    rows=[]
    for channel in receive_channels(x,fs,args.prns,acfg,tcfg,acquisition_only=args.acquisition_only):
        prn=channel['prn'];a=channel['acquisition']
        row={'prn':prn,'acquisition':a}
        print('ACQ',prn,round(a['metric'],3),a['doppler_hz'],a['success'],flush=True)
        if a['success'] and not args.acquisition_only:
            h=channel['history']
            row['tracking']=summarize(h,fs)
            if h.get('success'):
                nav=navigation_check(h)
                save_json(args.output/f'prn{prn:02d}_navigation.json',nav)
            np.savez_compressed(args.output/f'prn{prn:02d}.npz',
                **{k:v for k,v in h.items() if isinstance(v,np.ndarray)})
            print('TRACK',prn,row['tracking'],flush=True)
        rows.append(row)
        save_json(args.output/'summary.json',{'rows':rows})
    print('DONE',args.output,flush=True)


if __name__=='__main__': main()
