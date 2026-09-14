"""Code-only per-satellite selection baseline; no carrier switching or truth weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_sdcp_positioning import BASELINE
from positioning import solve_position
from run_gps_spp import satellite_setup, receiver_reference, serializable


def evaluate(pair, metric):
    t = np.asarray(pair[0]['time_s'])
    if not np.array_equal(t, pair[1]['time_s']): raise ValueError('unaligned times')
    sats = satellite_setup()
    expected = [s['prn'] for s in sats]
    if any([h['prn'] for h in r['satellite_tracking']] != expected for r in pair):
        raise ValueError('unexpected PRN order')
    p = np.asarray([r['pseudoranges_m'] for r in pair], float)
    q = np.full(p.shape, -np.inf)
    for c,r in enumerate(pair):
        for j,h in enumerate(r['satellite_tracking']):
            if metric not in h: raise ValueError('requested quality history missing')
            starts = np.asarray(h['epoch_start_samples'])/r['fs_dfe_hz']
            i = np.searchsorted(starts,t,side='right')-1
            safe = np.clip(i,0,len(starts)-1)
            v = np.asarray(h[metric],float)[safe]
            good = (i>=0)&(t<starts[safe]+.001)&np.isfinite(v)&np.isfinite(p[c,:,j])
            q[c,good,j] = v[good]
    chosen = np.argmax(q,axis=0)
    out = {}
    for mode in ['fixed_A','fixed_B','quality_selection']:
        pick = chosen if mode=='quality_selection' else np.full(chosen.shape,int(mode=='fixed_B'))
        positions = []
        for k,time in enumerate(t):
            values = p[pick[k],k,np.arange(len(sats))]
            valid = np.isfinite(values)
            if mode=='quality_selection': valid &= np.isfinite(q[pick[k],k,np.arange(len(sats))])
            states = np.array([s['position_m']+s['velocity_mps']*time for s in sats])-pick[k,:,None]*BASELINE
            try: positions.append(solve_position(states[valid],values[valid])['position_m'])
            except ValueError: positions.append([np.nan]*3)
        # Truth enters after selection and solution, only for evaluation.
        error = np.linalg.norm(np.asarray(positions)-receiver_reference()[0],axis=1)
        valid = np.isfinite(error)
        out[mode] = {'epochs':len(t),'solved':int(valid.sum()),
                     'switches':int(np.count_nonzero(np.diff(pick,axis=0))),
                     'B_fraction':float(pick.mean())}
        if valid.any(): out[mode].update(rms_m=float(np.sqrt(np.mean(error[valid]**2))),
                                         p95_m=float(np.quantile(error[valid],.95)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a',type=Path,required=True)
    p.add_argument('--b',type=Path,required=True)
    p.add_argument('--metric',choices=['phase_lock_metric','cn0_db_hz'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = {'scope':__doc__,'metric':args.metric,
              'inputs':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [args.a,args.b]},
              'modes':evaluate([json.loads(f.read_text(encoding='utf-8')) for f in [args.a,args.b]],args.metric)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(serializable(result),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result['modes']))


if __name__=='__main__': main()
