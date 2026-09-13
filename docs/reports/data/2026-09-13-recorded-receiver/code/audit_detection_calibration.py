"""Freeze a record-level H0 threshold, then evaluate disjoint seed sets."""

import argparse
import json
from pathlib import Path

import numpy as np

from acquisition import acquire
from antenna import antenna_response, build_antenna_model
from audit_multipath_baseline import baseline_config
from config import Direction, MultipathPath
from detection_calibration import calibrate_threshold, residual_peak_score_db, binomial_interval
from dfe import process_dfe
from diversity import combine_known
from gnss import generate_scenario
from multipath import _correlation_vectors, project_out_los
from rf import process_rf


def score_trial(seed,kind,*,echo_power_db=-6.,echo_delay_chips=.5,echo_direction=None):
    cfg=baseline_config()
    cfg.signal.duration_ms=6
    cfg.signal.code_bandwidth_hz=cfg.dfe.cutoff_hz
    nuisance=np.random.default_rng(seed+900000)
    cfg.signal.code_delay_chips=137.+nuisance.random()
    cfg.signal.initial_phase_rad=2*np.pi*nuisance.random()
    cfg.signal.doppler_hz=float(nuisance.choice([-1500.,1500.]))
    if kind!='los_only':
        direction={'echo_grid':Direction(110.,15.),'echo_offgrid':Direction(113.,17.),
                   'echo_same_direction':cfg.signal.true_direction}[kind]
        if echo_direction is not None:
            direction=Direction(*echo_direction)
        cfg.signal.multipath_paths=[MultipathPath(direction,relative_power_db=echo_power_db,
            excess_delay_chips=echo_delay_chips,reflection_phase_rad=2*np.pi*nuisance.random())]
    channels,_=generate_scenario(cfg)
    channels,_=process_rf(channels,cfg.rf,cfg.reference_impedance_ohm,seed)
    channels,info=process_dfe(channels,cfg.dfe,cfg.signal.fs_rf_hz,cfg.signal.if_hz)
    model=build_antenna_model(cfg.phone,cfg.signal.carrier_hz)
    los,_=antenna_response(model,cfg.signal.assumed_direction,cfg.signal.assumed_polarization,estimated=True)
    pilot,_=combine_known(channels,los,'egc',.01)
    acq=acquire(pilot,info['fs_out_hz'],cfg.signal.prn,cfg.signal.code_rate_hz,cfg.acquisition)
    row={'seed':seed,'kind':kind,'acquired':bool(acq['success']),'score_db':None,
         'code_delay_chips':cfg.signal.code_delay_chips,'doppler_hz':cfg.signal.doppler_hz,
         'initial_phase_rad':cfg.signal.initial_phase_rad}
    if not acq['success']: return row
    axis=np.arange(.25,3.001,.25)
    vectors=_correlation_vectors(channels,info['fs_out_hz'],cfg.signal.prn,
        cfg.signal.code_rate_hz,acq,cfg.acquisition,axis)
    residual=project_out_los(vectors,los)
    row['score_db']=residual_peak_score_db(np.sum(abs(residual)**2,axis=0))
    return row


def evaluate(rows,threshold):
    eligible=[r for r in rows if r['score_db'] is not None]
    positives=sum(r['score_db']>threshold for r in eligible)
    return {'records':len(rows),'acquisition_failed':sum(not r['acquired'] for r in rows),
            'eligible':len(eligible),'positive':positives,
            'rate_per_record':positives/len(rows),
            'rate_given_eligible':positives/len(eligible) if eligible else None,
            'interval95_per_record':binomial_interval(positives,len(rows)),
            'interval95_given_eligible':binomial_interval(positives,len(eligible))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',type=int,default=500)
    parser.add_argument('--evaluation',type=int,default=1000)
    parser.add_argument('--alternatives',type=int,default=200)
    args=parser.parse_args()
    if min(args.calibration,args.evaluation,args.alternatives)<1 or max(args.calibration,args.evaluation,args.alternatives)>9000:
        parser.error('counts must be between 1 and 9000 to keep seed sets disjoint')
    dest=Path(__file__).resolve().parent/'results'/'detection_calibration'
    dest.mkdir(parents=True,exist_ok=True)
    def batch(kind,start,count):
        rows=[]
        for i in range(count):
            rows.append(score_trial(start+i,kind))
            if (i+1)%100==0: print(kind,start,i+1,flush=True)
        return rows
    calibration=batch('los_only',10000,args.calibration)
    threshold=calibrate_threshold([r['score_db'] for r in calibration if r['score_db'] is not None],.01)
    frozen={'threshold_db':threshold,'target_alpha':.01,'decision':'strict score > threshold',
            'calibration':calibration,'seed_range':[10000,10000+args.calibration-1],
            'statistic':'10log10(max residual correlation power / median power); 12 delays .25: .25: 3 chips',
            'scope':'6 ms; 4 ms rectangular local-code correlation; ideal calibrated four antennas; thermal noise and quantization; continuous source 1.8 MHz; random code delay [137,138), phase [0,2pi), Doppler +/-1500 Hz; no external interference; fixed LOS geometry and power'}
    # Persist and reload the frozen cutoff BEFORE opening any evaluation data.
    (dest/'threshold.json').write_text(json.dumps(frozen,indent=2,allow_nan=False),encoding='utf-8')
    threshold=json.loads((dest/'threshold.json').read_text(encoding='utf-8'))['threshold_db']
    print('FROZEN threshold_db',threshold,flush=True)
    datasets={'los_only':batch('los_only',20000,args.evaluation)}
    for kind,start in [('echo_grid',30000),('echo_offgrid',40000),('echo_same_direction',50000)]:
        datasets[kind]=batch(kind,start,args.alternatives)
    summaries={kind:evaluate(rows,threshold) for kind,rows in datasets.items()}
    result={'threshold_db':threshold,'summary':summaries,'datasets':datasets,
            'interpretation':'H1 positive is evidence of residual, not validated delay/DOA or path identification; confidence intervals are scenario-specific; no threshold retuning'}
    (dest/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__': main()
