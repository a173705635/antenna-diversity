"""Audit NCO phase origin without fitting/subtracting a reference constant.

Acquisition is measured. Delayed tracking starts and NCO initial values are
test instrumentation only; receiver production code and SDCP remain unchanged.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from acquisition import acquire
from config import SPEED_OF_LIGHT_MPS
from dfe import process_dfe
from gnss import ca_code, generate_scenario
from positioning import carrier_range
from presets import default_config
from rf import process_rf
from run_gps_spp import serializable
from sdcp import geometry_offset_m
import tracking


def make_streams(layer, common_phase, doppler=1375.25):
    cfg=default_config()
    cfg.signal.duration_ms=400
    cfg.signal.initial_phase_rad=common_phase
    cfg.signal.doppler_hz=doppler
    cfg.signal.multipath_paths=[]
    cfg.interferers=[]
    cfg.phone.active_antennas=('TL','TR')
    for name in cfg.phone.channel_delay_ns:
        cfg.phone.channel_delay_ns[name]=0.
        cfg.phone.channel_phase_error_deg[name]=0.
        cfg.phone.channel_amplitude_error_db[name]=0.
    cfg.rf.temperature_k=0.
    cfg.tracking.max_epochs=400
    baseline=np.asarray(cfg.phone.phase_centers_m['TR'])-cfg.phone.phase_centers_m['TL']
    direction=cfg.signal.true_direction
    g=geometry_offset_m(baseline,direction.azimuth_deg,direction.elevation_deg)
    if layer=='ideal':
        fs=cfg.signal.fs_rf_hz/cfg.dfe.decimation
        t=np.arange(round(.4*fs))/fs
        phase=(t*cfg.signal.code_rate_hz*(1+doppler/cfg.signal.carrier_hz)-cfg.signal.code_delay_chips)%1023
        wave=ca_code(cfg.signal.prn)[np.floor(phase).astype(int)]*np.exp(1j*(common_phase+2*np.pi*doppler*t))
        x=np.array([wave,wave*np.exp(2j*np.pi*g/cfg.signal.wavelength_m)])
    else:
        if layer=='quantized_visible':
            cfg.signal.power_dbm=-90.
        antenna,truth=generate_scenario(cfg)
        del truth
        if layer=='linear_dfe':
            cfg.dfe.output_bits=0
            x,info=process_dfe(antenna,cfg.dfe,cfg.signal.fs_rf_hz,cfg.signal.if_hz)
        else:
            adc,_=process_rf(antenna,cfg.rf,cfg.reference_impedance_ohm,7000)
            del antenna
            x,info=process_dfe(adc,cfg.dfe,cfg.signal.fs_rf_hz,cfg.signal.if_hz)
        fs=info['fs_out_hz']
    return x,fs,g,cfg


def run_variant(x,fs,g,cfg,acquisitions,label,start_ms=(4,4),nco_phase=(0.,0.),channel_bias=0.):
    histories=[]
    original=tracking._TrackingState
    for channel in (0,1):
        tc=type(cfg.tracking)(**asdict(cfg.tracking))
        tc.acquisition_ms=start_ms[channel]
        def initial_state(*args,**kwargs):
            state=original(*args,**kwargs)
            state.carrier_phase_rad=nco_phase[channel]
            return state
        stream=x[channel] if channel==0 or channel_bias==0 else x[channel]*np.exp(1j*channel_bias)
        with patch.object(tracking,'_TrackingState',side_effect=initial_state):
            h=tracking.track(stream,fs,cfg.signal.prn,acquisitions[channel],tc,
                             cfg.signal.carrier_hz,cfg.signal.code_rate_hz)
        if not h['success']:
            return {'label':label,'status':h['status']}
        histories.append(h)
    times=np.arange(200,390)*.001+.000123
    wavelength=cfg.signal.wavelength_m
    ranges=[carrier_range(h,times,fs,.001,wavelength) for h in histories]
    # Positive NCO phase convention used by SDCP, NOT carrier-range convention.
    cycles=-(ranges[1]-ranges[0])/wavelength-g/wavelength
    if not np.isfinite(cycles).all():
        raise ValueError('missing common-time observations')
    fraction=cycles-np.rint(cycles)
    corrected=cycles-channel_bias/(2*np.pi)
    corrected-=np.rint(corrected)
    tail=times>=.3
    return {'label':label,'status':'evaluated','time_s':times,
        'start_ms':start_ms,'nco_initial_rad':nco_phase,'injected_channel_bias_rad':channel_bias,
        'geometry_m':g,'cycles_after_geometry':cycles,'fractional_cycles':fraction,
        'tail_integer_branches':np.unique(np.rint(cycles[tail])).tolist(),
        'tail_fractional_mean_cycles':float(fraction[tail].mean()),
        'tail_fractional_max_abs_m':float(abs(fraction[tail]).max()*wavelength),
        'tail_known_bias_corrected_max_abs_m':float(abs(corrected[tail]).max()*wavelength),
        'tail_phase_lock_min':float(min(np.min(h['phase_lock_metric'][h['epoch_start_samples']/fs>=.3]) for h in histories)),
        'histories':[{k:h[k] for k in ('epoch_start_samples','carrier_nco_phase_rad','carrier_nco_frequency_hz',
                                     'pll_error_rad','phase_lock_metric')} for h in histories]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results/phase_origin')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,data):
        (args.output/name).write_text(json.dumps(serializable(data),indent=2,allow_nan=False),encoding='utf-8')
    rows=[]
    manifest={'scope':__doc__,'layers':['ideal','linear_dfe','quantized_nominal','quantized_visible'],
        'common_initial_phases_rad':[.35,2.4,-2.2], 'doppler_hz':1375.25,
        'evaluation_s':[.3,.39],'no_reference_constant_fitted':True,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    save('manifest.json',manifest)
    for layer in manifest['layers']:
        for index,common in enumerate(manifest['common_initial_phases_rad']):
            print(f'RUN {layer} phase={common}',flush=True)
            x,fs,g,cfg=make_streams(layer,common)
            if not all(np.any(v) for v in x):
                row={'layer':layer,'phase_rad':common,'status':'zero_dfe_output'}
                rows.append(row)
                print(row,flush=True)
                continue
            acquisitions=[acquire(v,fs,cfg.signal.prn,cfg.signal.code_rate_hz,cfg.acquisition) for v in x]
            if not all(a['success'] for a in acquisitions):
                rows.append({'layer':layer,'phase_rad':common,'status':'acquisition_failed'})
                continue
            variants=[('baseline',(4,4),(0.,0.),0.),('different_start',(4,17),(0.,0.),0.),
                      ('different_nco',(4,17),(1.7,-2.1),0.),('known_channel_bias',(4,17),(1.7,-2.1),.4)]
            for label,starts,initial,bias in variants:
                result=run_variant(x,fs,g,cfg,acquisitions,label,starts,initial,bias)
                result.update(layer=layer,phase_rad=common,fs_hz=fs,signal_power_dbm=cfg.signal.power_dbm)
                save(f'{layer}_{index}_{label}.json',result)
                row={k:v for k,v in result.items() if k not in ('histories','time_s','cycles_after_geometry','fractional_cycles')}
                rows.append(row)
                print(json.dumps(row),flush=True)
            del x
    save('summary.json',rows)
    save('sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.json') if p.name!='sha256.json'})


if __name__=='__main__':
    main()
