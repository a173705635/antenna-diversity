"""Frozen self-check replay and observation-domain stress; no new RF generation."""
from pathlib import Path
import hashlib
import json
import argparse
from dataclasses import asdict
from collections import Counter
import numpy as np
from sdcp_self_check import detect, SelfCheckConfig
from audit_sdcp_windows import window_mean
from audit_sdcp_positioning import WAVELENGTH, observables, statistics
from positioning import solve_position, solve_code_carrier_float

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'docs/reports/data/2026-09-11-sdcp-long'
OUT = ROOT/'docs/reports/data/2026-09-12-sdcp-self-check'


def read(name):
    return json.loads((SOURCE/name).read_text(encoding='utf-8'))


def save(name, value):
    def convert(x):
        if isinstance(x, np.ndarray): return convert(x.tolist())
        if isinstance(x, np.generic): return convert(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        if isinstance(x, dict): return {k: convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [convert(v) for v in x]
        return x
    (OUT/name).write_text(json.dumps(convert(value), ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def load(seed, kind):
    d = read(f'residual_{kind}_{seed}.json')
    return np.asarray(d['time_s']), np.asarray(d['residual_m'])


def score(t, result, affected, onset):
    state = result['state']
    late = t >= 1.5
    truth = np.zeros_like(state, bool)
    truth[np.ix_(t >= onset, affected)] = True
    active = np.broadcast_to(late[:, None], state.shape)
    known = state >= 0
    alarm = state == 1
    def rate(a, mask): return float(a[mask].mean()) if mask.any() else None
    per_prn = []
    for i in affected:
        hits = np.flatnonzero((t >= onset) & alarm[:, i])
        per_prn.append({'column': i, 'detected': bool(len(hits)),
                        'delay_s': float(t[hits[0]]-onset) if len(hits) else None})
    return {'coverage': rate(known, active),
            'false_alarm_all': rate(alarm, active & ~truth),
            'false_alarm_judged': rate(alarm, active & ~truth & known),
            'detection_all': rate(alarm, active & truth),
            'detection_judged': rate(alarm, active & truth & known),
            'fixed_false_alarm': rate(result['fixed_alarm'], active & ~truth),
            'fixed_detection': rate(result['fixed_alarm'], active & truth),
            'fixed_confirmed_false_alarm': rate(result['fixed_confirmed'], active & ~truth),
            'fixed_confirmed_detection': rate(result['fixed_confirmed'], active & truth),
            'per_prn_coverage': known[late].mean(axis=0).tolist(),
            'per_prn_alarm': alarm[late].mean(axis=0).tolist(),
            'whole_record_coverage': float(known.mean()),
            'reasons': dict(Counter(result['reason'][active].ravel())),
            'events': per_prn}


def scenarios(t, base, seed):
    n, m = base.shape
    onset = t >= .5
    for kind in ['los','offset_10mm','drift_10mm_s','noise_x3','geometry_unreported',
                 'geometry_reported','loss_flag','missing','half_cycle','skew',
                 'one_static_25mm','three_static_25mm','five_static_25mm',
                 'all_static_25mm','combined','integer_jump_unreported']:
        r = base.copy(); valid = np.ones_like(r, bool)
        uncertainty = np.zeros_like(r); skew = np.zeros(n); affected = []
        if kind == 'offset_10mm': r += .01
        if kind == 'drift_10mm_s': r += (np.maximum(t-.5,0)*.01)[:,None]
        if kind == 'noise_x3':
            # Additional observed phase noise, not a physical CN0/RF experiment.
            r += np.random.default_rng(seed+100000).normal(0,.012,r.shape)
        if kind.startswith('geometry_'):
            r += np.linspace(-.015,.015,m)[None,:]
            if kind == 'geometry_reported': uncertainty[:] = .015
        if kind == 'loss_flag': valid[(t >= .8)&(t < 1.2),0] = False
        if kind == 'missing': r[(t >= .8)&(t < 1.2),0] = np.nan
        if kind == 'half_cycle': r[onset,0] += WAVELENGTH/2
        if kind == 'integer_jump_unreported': r[onset,0] += WAVELENGTH
        if kind == 'skew': skew[onset] = .0001
        counts = {'one_static_25mm':1,'three_static_25mm':3,
                  'five_static_25mm':5,'all_static_25mm':8}
        if kind in counts:
            affected = list(range(counts[kind])); r[:,affected] += .025
        if kind == 'combined':
            r += (np.maximum(t-.5,0)*.01)[:,None]
            r[onset,0] += .025; affected = [0]
        yield kind,r,valid,uncertainty,skew,affected,0. if kind in counts else .5


def pvt(seed, kind, result, satellites):
    satellites = [dict(s, position_m=np.asarray(s['position_m']),
                       velocity_mps=np.asarray(s['velocity_mps'])) for s in satellites]
    pair = [read(f'{"common" if kind=="common" else "los"}_{seed}_0.json'),
            read(f'{kind}_{seed}_1.json')]
    t, code, phase, states, _ = observables(pair, satellites)
    # Offline float solve at 50-ms spacing, identical epochs for all methods.
    ix = np.arange(0,len(t),5)
    modes = {}
    weights = {'plain':np.ones(result['state'].shape),
               'fixed':np.where(result['fixed_alarm'],.1,1.),
               'fixed_confirmed':np.where(result['fixed_confirmed'],.1,1.),
               'self_check':np.where(result['state']==1,.1,1.)}
    for mode,w in weights.items():
        w = np.concatenate([w,w],axis=1)[ix]
        scale = 1/np.sqrt(w)
        cp = np.array([solve_position(s,p,sigma_m=30*z)['position_m']
                       for s,p,z in zip(states[ix],code[ix],scale)])
        fp = solve_code_carrier_float(states[ix],code[ix],phase[ix],
                        code_sigma_m=30*scale,carrier_sigma_m=.01*scale)['position_m']
        # Match detector late interval; statistics() selects >= .6 within these rows.
        late = t[ix] >= 1.5
        modes[mode] = {'code':statistics(cp[late],t[ix][late]),
                       'float':statistics(fp[late],t[ix][late]),
                       'time_s':t[ix], 'code_position_m':cp,'float_position_m':fp}
    return modes


def main():
    global SOURCE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--seeds', nargs='+', type=int, default=[6300,6302])
    args = parser.parse_args()
    SOURCE, OUT = args.source, args.output
    OUT.mkdir(parents=True, exist_ok=False)
    hashes = read('sha256.json')
    for name,digest in hashes.items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == digest,name
    configs = {n:SelfCheckConfig(window=n) for n in [50,100]}
    floors = {}
    diagnostics = []
    for n in configs:
        values = []
        for seed in [6200,6202,6204]:
            t,r = load(seed,'los'); v = window_mean(r,n)[t[n-1:]>=1.5]
            values.extend(abs(v).ravel())
            diagnostics.append({'seed':seed,'window':n,'mean_mm':v.mean(axis=0)*1000,
                                'std_mm':v.std(axis=0,ddof=1)*1000})
        floors[n] = float(np.quantile(values,.99,method='higher'))
    save('manifest.json', {'scope':__doc__, 'source_hashes':hashes,
         'calibration_seeds':[6200,6202,6204], 'evaluation_seeds':args.seeds,
         'configs':{str(n):asdict(c) for n,c in configs.items()},'floors_m':floors,
         'code_hashes':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
                        for n in ['sdcp_self_check.py','audit_sdcp_self_check.py']},
         'metadata':'RF replay assumes aligned locked integer-coherent arcs and zero geometry uncertainty; stresses override these explicitly',
         'pvt':'exploratory offline float, 50ms subset, pair code+phase weight .1 for alarms; unknown falls back to 1',
         'limitations':'two RF seed pairs; injected observation-domain stresses share base records, not independent RF trials; no physical CN0, attitude or reacquisition simulation'})
    save('los_bias_diagnostics.json',diagnostics)
    rows=[];pvts=[]
    for seed in args.seeds:
        t,base = load(seed,'los')
        for n,c in configs.items():
            def evaluate(r,valid=True,uncertainty=0.,skew=0.):
                return detect(t,r,WAVELENGTH,floors[n],valid=valid,
                         geometry_uncertainty_m=uncertainty,channel_skew_s=skew,config=c)
            for kind,r,valid,u,skew,affected,onset in scenarios(t,base,seed):
                o = evaluate(r,valid,u,skew)
                rows.append(dict(seed=seed,window_ms=n*10,domain='observation',case=kind,
                                 **score(t,o,affected,onset)))
                save(f'observation_{kind}_{seed}_w{n}.json',o)
            for kind in ['los','strong','weak','static','common']:
                t,r = load(seed,kind);o=evaluate(r)
                rows.append(dict(seed=seed,window_ms=n*10,domain='RF_replay',case=kind,
                                 **score(t,o,[] if kind=='los' else [0],0. if kind=='static' else .5)))
                save(f'rf_{kind}_{seed}_w{n}.json',o)
                if n == 50:
                    print('PVT',seed,kind,flush=True)
                    modes=pvt(seed,kind,o,read('manifest.json')['satellites'])
                    save(f'pvt_{kind}_{seed}.json',modes)
                    pvts.append({'seed':seed,'case':kind,'modes':{k:{q:v[q] for q in ['code','float']} for k,v in modes.items()}})
            save('summary.json',rows);save('pvt_summary.json',pvts)
    save('sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.json')})
    print('Done',len(rows),'detection cases;',len(pvts),'PVT cases',flush=True)


if __name__ == '__main__': main()
