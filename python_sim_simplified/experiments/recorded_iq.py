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
from experiments.paths import RESULTS_DIR, reset_output

IQ_PATH = RESULTS_DIR/'real_gnss_hackrf/first40s.int8'
OUTPUT_DIR = RESULTS_DIR/'recorded_iq'
START_S = 1.
DURATION_S = 39.
SAMPLE_RATE_HZ = 1e7
IF_HZ = 420e3
PRNS = list(range(1, 33))
FREEZE_SOURCE_FLAG = False

def _json_write(output_path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    Path(output_path).write_text(json.dumps(value,default=convert,indent=2,allow_nan=False),encoding='utf-8')


def main():
    if DURATION_S <= 1.01:
        raise ValueError('Recorded positioning is not evaluated: observations start at 1 s with a 10 ms end margin; use more than 1.01 s. Decoded ephemerides are also required.')
    output_dir = reset_output(OUTPUT_DIR)
    if FREEZE_SOURCE_FLAG:freeze_source(output_dir)
    dfe_cfg=DFEConfig(decimation=2,output_bits=0)
    acquisition_cfg=AcquisitionConfig(doppler_bins_hz=np.arange(-10000.,10000.1,250.))
    tracking_cfg=TrackingConfig(carrier_discriminator='costas',max_epochs=round(DURATION_S*1000))
    baseband_samples,dfe_info=prepare_recording(
        IQ_PATH,output_dir/'baseband.c128',start_sample=round(START_S*SAMPLE_RATE_HZ),
        sample_count=round(DURATION_S*SAMPLE_RATE_HZ),fs_hz=SAMPLE_RATE_HZ,
        if_hz=IF_HZ,dfe_config=dfe_cfg,
    )
    fs_out_hz=dfe_info['fs_out_hz']
    tracking_histories_by_prn={}
    acquisitions=[]
    _json_write(output_dir/'manifest.json',{'input':str(IQ_PATH),'start_s':START_S,'duration_s':DURATION_S,
        'sample_rate_hz':SAMPLE_RATE_HZ,'if_hz':IF_HZ,'prns':PRNS,'dfe':dfe_info,
        'tracking':asdict(tracking_cfg),'acquisition':asdict(acquisition_cfg),
        'source_sha256':source_hashes()})
    for channel_result in receive_channels(
        baseband_samples[0],fs_out_hz,PRNS,acquisition_cfg,tracking_cfg,
    ):
        print('PRN',channel_result['prn'],'acquired',
              channel_result['acquisition']['success_flag'],flush=True)
        acquisitions.append({'prn':channel_result['prn'],
                             'acquisition':channel_result['acquisition']})
        tracking_history=channel_result.get('history')
        if tracking_history and tracking_history['success_flag']:
            tracking_histories_by_prn[channel_result['prn']]=tracking_history
            np.savez_compressed(
                output_dir/f"prn{channel_result['prn']:02d}.npz",
                **{key:value for key,value in tracking_history.items()
                   if isinstance(value,np.ndarray)},
            )
    _json_write(output_dir/'acquisition.json',acquisitions)
    solution=position_channels(tracking_histories_by_prn,fs_out_hz,np.arange(1.,DURATION_S-.01,.1))
    _json_write(output_dir/'position.json',solution)
    print(solution['status'],'positions',sum(e['status']=='position' for e in solution['epochs']),flush=True)


if __name__ == "__main__":
    main()
