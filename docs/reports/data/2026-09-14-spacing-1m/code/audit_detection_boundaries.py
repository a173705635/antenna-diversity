"""One-factor detection boundary slices using an unchanged frozen threshold."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audit_detection_calibration import score_trial,evaluate


def cells():
    yield 'control',0,{}
    for power in (-30,-24,-18,-12,-6,0):
        yield 'power_db',power,{'echo_power_db':power}
    for delay in (0,.05,.1,.25,.5,1,2,3):
        yield 'delay_chips',delay,{'echo_delay_chips':delay}
    for angle in (0,2,5,10,20,40,60):
        # A single meridian: LOS az=30, el=60; angular distance is exactly angle.
        yield 'angle_deg',angle,{'echo_direction':(30.,60.-angle)}


def plot_summary(cases,destination):
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    labels=['Echo power relative to LOS (dB)','Excess delay (chip)',
            'LOS/echo angular separation (deg)']
    for ax,axis,label in zip(axes,('power_db','delay_chips','angle_deg'),labels):
        selected=[c for c in cases if c['axis']==axis]
        x=[c['value'] for c in selected]
        p=[c['summary']['rate_per_record'] for c in selected]
        lower=[v-c['summary']['interval95_per_record'][0] for v,c in zip(p,selected)]
        upper=[c['summary']['interval95_per_record'][1]-v for v,c in zip(p,selected)]
        ax.errorbar(x,p,yerr=[lower,upper],fmt='o-',capsize=3)
        ax.set(xlabel=label,ylabel='Positive records / all records',ylim=(-.03,1.03))
        ax.grid(alpha=.25)
    fig.suptitle('Frozen threshold; one-factor slices; pointwise 95% intervals')
    fig.savefig(destination/'boundaries.png',dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=100)
    args=parser.parse_args()
    if not 1<=args.trials<=9000: parser.error('trials must be in 1..9000')
    root=Path(__file__).resolve().parent/'results'
    threshold_path=root/'detection_calibration'/'threshold.json'
    frozen_bytes=threshold_path.read_bytes()
    threshold=json.loads(frozen_bytes)['threshold_db']
    digest=hashlib.sha256(frozen_bytes).hexdigest()
    destination=root/'detection_boundaries'
    destination.mkdir(parents=True,exist_ok=True)
    cases=[]
    for axis,value,parameters in cells():
        path=destination/f'{axis}_{value}_{args.trials}.json'
        if path.exists():
            record=json.loads(path.read_text(encoding='utf-8'))
            if record['threshold_sha256']!=digest:
                raise ValueError('checkpoint belongs to a different frozen threshold')
        else:
            rows=[score_trial(60000+i,'los_only' if axis=='control' else 'echo_grid',**parameters)
                  for i in range(args.trials)]
            record={'axis':axis,'value':value,'parameters':parameters,'threshold_db':threshold,
                    'threshold_sha256':digest,'rows':rows,'summary':evaluate(rows,threshold)}
            path.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
        cases.append(record)
        print(axis,value,record['summary'],flush=True)
    if threshold_path.read_bytes()!=frozen_bytes: raise ValueError('threshold changed during study')
    result={'threshold_db':threshold,'threshold_sha256':digest,'cases':cases,
            'scope':'same RF/DFE and nuisance distribution as calibration; no external interference; one-factor slices, not a joint performance map',
            'seed_policy':'60000 onward; common random seeds across cells for paired comparison; disjoint from earlier calibration/evaluation',
            'base':'-6 dB, .5 chip, echo 110/15 deg; angle slice is az=30 deg and el=60-angle',
            'interpretation':'positive spatial-residual statistic, not validated path estimation; intervals are pointwise, not simultaneous'}
    (destination/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    plot_summary(cases,destination)


if __name__=='__main__': main()
