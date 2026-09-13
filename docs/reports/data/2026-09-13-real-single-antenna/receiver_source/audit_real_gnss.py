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

from acquisition import acquire
from config import AcquisitionConfig, DFEConfig, TrackingConfig
from dfe import process_dfe
from tracking import track


def read_iq8(path, start_sample, count):
    size = Path(path).stat().st_size
    if size % 2:
        raise ValueError('IQ file must contain complete byte pairs')
    if start_sample < 0 or count < 1 or (start_sample+count)*2 > size:
        raise ValueError('requested sample window is outside the recording')
    raw = np.fromfile(path, dtype=np.int8, offset=start_sample*2, count=count*2)
    return raw[::2].astype(np.float32) + 1j*raw[1::2].astype(np.float32)


def save_json(path, obj):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, default=convert, indent=2, allow_nan=False), encoding='utf-8')


def decode_lnav_word(word):
    """IS-GPS-200 user parity algorithm, 2 previous + 30 current bits.

    Protocol parity masks cross-checked against RTKLIB src/rtkcmn.c decode_word.
    This is an audit diagnostic, not an ephemeris/telemetry decoder.
    """
    masks = (0xBB1F3480,0x5D8F9A40,0xAEC7CD00,0x5763E680,0x6BB1F340,0x8B7A89C0)
    corrected = word ^ (0x3FFFFFC0 if word & 0x40000000 else 0)
    parity = 0
    for mask in masks:
        parity = (parity << 1) | ((corrected & mask).bit_count() & 1)
    return (corrected >> 6) & 0xFFFFFF if parity == (word & 63) else None


def navigation_check(h):
    """Offline 20-ms bit synchronization, word parity and TLM/HOW evidence.

    No truth bits supplied, no NCO modification. A long consecutive parity
    chain is evidence of demodulation, not authentication or BER truth.
    """
    z = h['prompt'][500:]  # discard startup, same for every PRN/mode
    if len(z) < 700: return {'status':'too_short'}
    scores=[]
    for offset in range(20):
        n=(len(z)-offset)//20
        scores.append(float(np.mean(abs(z[offset:offset+n*20].real.reshape(n,20).sum(axis=1)))))
    offset=int(np.argmax(scores)); n=(len(z)-offset)//20
    sums=z[offset:offset+n*20].real.reshape(n,20).sum(axis=1)
    bits=(sums<0).astype(int).tolist()
    candidates=[]
    for polarity in (0,1):
        b=[v^polarity for v in bits]
        for start in range(2,32):
            words=[]; longest=run=0
            for i in range(start,len(b)-29,30):
                word=int(''.join(map(str,b[i-2:i+30])),2)
                payload=decode_lnav_word(word)
                run=run+1 if payload is not None else 0
                longest=max(longest,run)
                words.append({'bit_index':i,'payload':payload})
            valid=sum(w['payload'] is not None for w in words)
            candidates.append((longest,valid,polarity,start,words))
    longest,valid,polarity,start,words=max(candidates,key=lambda c:(c[0],c[1]))
    frames=[]
    for j,w in enumerate(words[:-1]):
        d=w['payload']; how=words[j+1]['payload']
        if d is not None and d>>16==0x8b and how is not None:
            frames.append({'bit_index':w['bit_index'],'next_subframe_tow_s':(how>>7)*6,
                'subframe_id':(how>>2)&7,
                'ten_words_parity_ok':len(words[j:j+10])==10 and all(v['payload'] is not None for v in words[j:j+10])})
    return {'status':'evaluated','bit_offset_epochs_after_500':offset,
        'bit_sync_scores':scores,'bit_count':n,'bits':bits,'polarity':polarity,
        'word_start_bit':start,'words':words,'valid_words':valid,'tested_words':len(words),
        'longest_consecutive_valid_words':longest,'preambles':frames,
        'half_cycle_ambiguity_resolved_for_nco':False,
        'demodulation_evidence':longest>=4 and len(set(bits))==2 and any(
            1<=f['subframe_id']<=5 and 0<=f['next_subframe_tow_s']<604800 for f in frames)}


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
            for n in ('audit_real_gnss.py','acquisition.py','tracking.py','config.py','dfe.py','gnss.py','code_replica.py')}}
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
    for prn in args.prns:
        a=acquire(x,fs,prn,1.023e6,acfg)
        compact={k:v for k,v in a.items() if k not in ('search_power','doppler_bins_hz')}
        row={'prn':prn,'acquisition':compact}
        print('ACQ',prn,round(a['metric'],3),a['doppler_hz'],a['success'],flush=True)
        if a['success'] and not args.acquisition_only:
            h=track(x,fs,prn,a,tcfg,1575.42e6,1.023e6)
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
