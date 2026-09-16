"""Offline recorded GPS L1 receiver experiment."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
from dataclasses import asdict
import numpy as np
from core.provenance import source_hashes, freeze_source
from core.serialization import serializable
from core.config import AcquisitionConfig, DFEConfig, TrackingConfig
from core.recorded_receiver import prepare_recording, receive_channels, position_channels

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
    p.add_argument('--freeze-source',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.freeze_source:freeze_source(args.output)
    dcfg=DFEConfig(decimation=2,output_bits=0)
    acfg=AcquisitionConfig(doppler_bins_hz=np.arange(-10000.,10000.1,250.))
    tcfg=TrackingConfig(carrier_discriminator='costas',max_epochs=round(args.seconds*1000))
    bb,info=prepare_recording(args.iq,args.output/'baseband.c128',start_sample=round(args.start*args.fs),
        sample_count=round(args.seconds*args.fs),fs_hz=args.fs,if_hz=args.if_hz,dfe_config=dcfg)
    fs=info['fs_out_hz'];channels={};acquisitions=[]
    _json_write(args.output/'manifest.json',{'arguments':vars(args),'dfe':info,'tracking':asdict(tcfg),
        'acquisition':asdict(acfg),'source_sha256':source_hashes()})
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


if __name__ == "__main__":
    main()
