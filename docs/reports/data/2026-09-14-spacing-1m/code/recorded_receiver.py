"""Reusable offline recorded GPS L1 receiver, independent of audit scripts."""
from pathlib import Path
from dataclasses import asdict
import json
import hashlib
import numpy as np
from config import AcquisitionConfig, DFEConfig, TrackingConfig
from dfe import process_dfe_blocks, design_decimator
from acquisition import acquire
from tracking import track
from gps_lnav import decode_lnav
from gps_observables import transmit_time, solve_gps_epoch
from positioning import carrier_range
from config import SPEED_OF_LIGHT_MPS as C


def read_iq8(path, start_sample, count):
    size = Path(path).stat().st_size
    if size % 2:
        raise ValueError('IQ file must contain complete byte pairs')
    if start_sample < 0 or count < 1 or (start_sample+count)*2 > size:
        raise ValueError('requested sample window is outside the recording')
    raw = np.fromfile(path, dtype=np.int8, offset=start_sample*2, count=count*2)
    return raw[::2].astype(np.float32) + 1j*raw[1::2].astype(np.float32)


def prepare_recording(path, cache_path, *, start_sample, sample_count,
                      fs_hz, if_hz, dfe_config, block_samples=1000000):
    """Read once into a disk-backed DFE output; input sample units stay explicit."""
    path=Path(path);cache_path=Path(cache_path)
    if cache_path.exists():raise ValueError('refusing to overwrite an existing DFE cache')
    if path.stat().st_size%2 or start_sample<0 or sample_count<1 or (start_sample+sample_count)*2>path.stat().st_size:
        raise ValueError('invalid IQ file window')
    h,info=design_decimator(fs_hz,dfe_config.decimation,dfe_config.cutoff_hz,dfe_config.num_taps)
    delay=info['group_delay_input_samples']
    count=len(range(delay,sample_count,dfe_config.decimation))
    if count<1:raise ValueError('recording shorter than DFE group delay')
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    out=np.memmap(cache_path,dtype='<c16',mode='w+',shape=(1,count))
    digest=hashlib.sha256();sum_i=sum_q=sum_power=0.;rails=0
    def blocks():
        nonlocal sum_i,sum_q,sum_power,rails
        with path.open('rb') as f:
            f.seek(start_sample*2);remaining=sample_count
            while remaining:
                n=min(block_samples,remaining);data=f.read(n*2)
                if len(data)!=n*2:raise ValueError('truncated IQ while reading')
                digest.update(data);raw=np.frombuffer(data,dtype='i1')
                i=raw[::2].astype(float);q=raw[1::2].astype(float)
                sum_i+=i.sum();sum_q+=q.sum();sum_power+=np.sum(i*i+q*q)
                rails+=int(np.sum((i==-128)|(i==127)|(q==-128)|(q==127)))
                yield (i+1j*q)[None,:];remaining-=n
    index=0
    for block in process_dfe_blocks(blocks(),dfe_config,fs_hz,if_hz):
        out[:,index:index+block.shape[1]]=block;index+=block.shape[1]
    assert index==count
    out.flush()
    info.update({'if_hz':if_hz if dfe_config.nco_frequency_hz is None else dfe_config.nco_frequency_hz,
        'output_bits':dfe_config.output_bits,'output_length':count,'filter_coefficients':h,
        'input_window_sha256':digest.hexdigest(),'input_start_sample':start_sample,
        'input_samples':sample_count,'input_stats':{'mean_i':sum_i/sample_count,'mean_q':sum_q/sample_count,
        'rms':np.sqrt(sum_power/sample_count),'rail_fraction':rails/sample_count}})
    return out,info


def receive_channels(baseband, fs_hz, prns, acquisition_config, tracking_config, *, acquisition_only=False):
    """Yield independent GPS L1 channels from one common baseband stream."""
    if np.asarray(baseband).ndim!=1:raise ValueError('one antenna stream required')
    for prn in prns:
        a=acquire(baseband,fs_hz,prn,1.023e6,acquisition_config)
        row={'prn':prn,'acquisition':{k:v for k,v in a.items() if k not in ('search_power','doppler_bins_hz')}}
        if a['success'] and not acquisition_only:
            row['history']=track(baseband,fs_hz,prn,a,tracking_config,1575.42e6,1.023e6)
        yield row


def position_channels(channels, fs_hz, times_s, *, common_travel_s=.075):
    """Offline GPS code PVT, using broadcast data decoded from these channels.

    This is retrospective processing of one continuous arc; ephemeris decoded
    near its end can be used earlier. No receiver reference position is input.
    """
    times=np.asarray(times_s,float)
    if len(times)==0:return {'status':'no_query_times','epochs':[]}
    usable={};navigation={};carriers={}
    for prn,h in channels.items():
        nav=decode_lnav(h,fs_hz);navigation[prn]=nav
        eph=nav['ephemeris']
        if eph is not None and eph['health']==0 and not eph['tgd_unavailable']:
            usable[prn]=(eph,transmit_time(h,nav['time_anchors'][0],times,fs_hz))
            carriers[prn]=carrier_range(h,times,fs_hz,.001,C/1575.42e6)
    if len(usable)<4:return {'status':'insufficient_decoded_ephemerides','navigation':navigation,'epochs':[]}
    first=[tx[0] for _,tx in usable.values() if np.isfinite(tx[0])]
    if len(first)<4:return {'status':'insufficient_first_epoch','navigation':navigation,'epochs':[]}
    origin=max(first)+common_travel_s
    epochs=[]
    for k,t in enumerate(times):
        prns=[];ephemerides=[];txs=[]
        for prn,(eph,tx) in usable.items():
            h=channels[prn]
            i=np.searchsorted(h['epoch_start_samples']/fs_hz,t,side='right')-1
            if i<0 or not np.isfinite(tx[k]):continue
            if h['phase_lock_metric'][i]<.8 or abs(h['pll_error_rad'][i])>.7:continue
            prns.append(prn);ephemerides.append(eph);txs.append(tx[k])
        if len(prns)<4:epochs.append({'sample_time_s':float(t),'status':'insufficient_tracking_quality'});continue
        try:solution=solve_gps_epoch(ephemerides,txs,origin+t-times[0])
        except ValueError as e:epochs.append({'sample_time_s':float(t),'status':str(e)});continue
        epochs.append({'sample_time_s':float(t),'status':'position','prns':prns,
                       'transmit_tow_s':txs,
                       'carrier_range_m':[float(carriers[prn][k]) for prn in prns],
                       'carrier_half_cycle_resolved':False,**solution})
    return {'status':'evaluated','navigation':navigation,'epochs':epochs,
            'common_clock_origin_tow_s':origin,'common_clock_origin_sample_s':float(times[0]),
            'mode':'offline GPS L1 code SPP; decoded clocks/TGD, Sagnac; no atmosphere correction'}


def _json_write(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,default=convert,indent=2,allow_nan=False),encoding='utf-8')


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iq',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=float,default=1.);p.add_argument('--seconds',type=float,default=39.)
    p.add_argument('--fs',type=float,default=1e7);p.add_argument('--if-hz',type=float,default=420e3)
    p.add_argument('--prns',nargs='+',type=int,default=list(range(1,33)))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    dcfg=DFEConfig(decimation=2,output_bits=0)
    acfg=AcquisitionConfig(doppler_bins_hz=np.arange(-10000.,10000.1,250.))
    tcfg=TrackingConfig(carrier_discriminator='costas',max_epochs=round(args.seconds*1000))
    bb,info=prepare_recording(args.iq,args.output/'baseband.c128',start_sample=round(args.start*args.fs),
        sample_count=round(args.seconds*args.fs),fs_hz=args.fs,if_hz=args.if_hz,dfe_config=dcfg)
    fs=info['fs_out_hz'];channels={};acquisitions=[]
    _json_write(args.output/'manifest.json',{'arguments':vars(args),'dfe':info,'tracking':asdict(tcfg),
        'acquisition':asdict(acfg),'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(__file__).parent.glob('*.py')}})
    for row in receive_channels(bb[0],fs,args.prns,acfg,tcfg):
        print('PRN',row['prn'],'acquired',row['acquisition']['success'],flush=True)
        acquisitions.append({'prn':row['prn'],'acquisition':row['acquisition']})
        h=row.get('history')
        if h and h['success']:
            channels[row['prn']]=h
            np.savez_compressed(args.output/f"prn{row['prn']:02d}.npz",**{k:v for k,v in h.items() if isinstance(v,np.ndarray)})
    _json_write(args.output/'acquisition.json',acquisitions)
    solution=position_channels(channels,fs,np.arange(1.,args.seconds-.01,.1))
    _json_write(args.output/'position.json',solution)
    print(solution['status'],'positions',sum(e['status']=='position' for e in solution['epochs']),flush=True)


if __name__=='__main__':main()
