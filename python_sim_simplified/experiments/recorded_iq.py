"""Offline recorded GPS L1 receiver experiment."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
from dataclasses import asdict
import numpy as np
from core.provenance import source_hashes, freeze_source
from core.serialization import serializable
from core.config import AcquisitionConfig, DFEConfig, TrackingConfig
from core.recorded_receiver import prepare_recording, receive_channels, position_channels
from experiments.paths import ROOT, RESULTS, reset_output

IQ = RESULTS/'real_gnss_hackrf/first40s.int8'
OUTPUT = RESULTS/'recorded_iq'
START_S = 1.
DURATION_S = 39.
SAMPLE_RATE_HZ = 1e7
IF_HZ = 420e3
PRNS = list(range(1, 33))
FREEZE_SOURCE = False

def _json_write(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,default=convert,indent=2,allow_nan=False),encoding='utf-8')


def main():
    output = reset_output(OUTPUT.name)
    if FREEZE_SOURCE:freeze_source(output)
    dcfg=DFEConfig(decimation=2,output_bits=0)
    acfg=AcquisitionConfig(doppler_bins_hz=np.arange(-10000.,10000.1,250.))
    tcfg=TrackingConfig(carrier_discriminator='costas',max_epochs=round(DURATION_S*1000))
    bb,info=prepare_recording(IQ,output/'baseband.c128',start_sample=round(START_S*SAMPLE_RATE_HZ),
        sample_count=round(DURATION_S*SAMPLE_RATE_HZ),fs_hz=SAMPLE_RATE_HZ,if_hz=IF_HZ,dfe_config=dcfg)
    fs=info['fs_out_hz'];channels={};acquisitions=[]
    _json_write(output/'manifest.json',{'input':str(IQ),'start_s':START_S,'duration_s':DURATION_S,
        'sample_rate_hz':SAMPLE_RATE_HZ,'if_hz':IF_HZ,'prns':PRNS,'dfe':info,'tracking':asdict(tcfg),
        'acquisition':asdict(acfg),'source_sha256':source_hashes()})
    for row in receive_channels(bb[0],fs,PRNS,acfg,tcfg):
        print('PRN',row['prn'],'acquired',row['acquisition']['success'],flush=True)
        acquisitions.append({'prn':row['prn'],'acquisition':row['acquisition']})
        h=row.get('history')
        if h and h['success']:
            channels[row['prn']]=h
            np.savez_compressed(output/f"prn{row['prn']:02d}.npz",**{k:v for k,v in h.items() if isinstance(v,np.ndarray)})
    _json_write(output/'acquisition.json',acquisitions)
    solution=position_channels(channels,fs,np.arange(1.,DURATION_S-.01,.1))
    _json_write(output/'position.json',solution)
    print(solution['status'],'positions',sum(e['status']=='position' for e in solution['epochs']),flush=True)


if __name__ == "__main__":
    main()
