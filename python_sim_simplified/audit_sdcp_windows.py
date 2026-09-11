"""Causal circular SDCP means, frozen independent LOS quantile thresholds."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_calibrated_sdcp import SOURCE, ROOT, read, residuals
from audit_sdcp_positioning import WAVELENGTH
from sdcp import calibrated_sdcp


def window_mean(values, count):
    """Only complete trailing windows; no future samples or zero padding."""
    values = np.asarray(values)
    if count < 1 or count > len(values) or not np.isfinite(values).all():
        raise ValueError('finite data and valid window required')
    phasors = np.exp(2j*np.pi*values/WAVELENGTH)
    return np.array([np.angle(phasors[k-count+1:k+1].mean(axis=0))*WAVELENGTH/(2*np.pi)
                     for k in range(count-1, len(values))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/reports/data/2026-09-11-sdcp-windows')
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    def save(name, data):
        (out/name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    for name, digest in read('sha256.json').items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == digest
    satellites = read('manifest.json')['satellites']
    def load(seed, kind, calibration=False):
        names = [f'cal_{seed}_{ch}.json' for ch in (0,1)] if calibration else [
            f'{"common" if kind == "common" else "los"}_{seed}_0.json', f'{kind}_{seed}_1.json']
        t, s, g, _ = residuals([read(n) for n in names], satellites)
        return t, calibrated_sdcp(s, g, WAVELENGTH, 0)
    calibration = [load(s, 'los', True) for s in (6000,6002,6004)]
    evaluation = {(s,k): load(s,k) for s in (6100,6102) for k in ('los','strong','weak','common','static')}
    rows = []
    for count in (1,5,10,20):
        scores = np.concatenate([abs(window_mean(r,count)[t[count-1:]>=.6]).ravel() for t,r in calibration])
        threshold = float(np.quantile(scores,.99,method='higher'))
        row = {'window_ms': count*10, 'threshold_mm': threshold*1000, 'cases': []}
        noise = []
        for (seed,kind), (times,raw) in evaluation.items():
            r = window_mean(raw,count)
            t = times[count-1:]
            late = t>=.6
            alarm = abs(r)>threshold
            after = np.flatnonzero((t>=.5)&alarm[:,0])
            item = {'seed':seed,'kind':kind,'late_prn1_fraction':float(alarm[late,0].mean()),
                    'late_other_fraction':float(alarm[late,1:].mean()),
                    'first_available_s':float(t[0]),
                    'first_alarm_s':float(t[np.flatnonzero(alarm[:,0])[0]]) if alarm[:,0].any() else None,
                    'first_post_onset_delay_ms':float((t[after[0]]-.5)*1000) if len(after) and kind in ('strong','weak','common') else None,
                    'pre_onset_prn1_alarms':int(alarm[t<.5,0].sum())}
            if kind=='los':
                item['late_all_false_fraction']=float(alarm[late].mean())
                noise.append(r[late])
            row['cases'].append(item)
            save(f'w{count}_{kind}_{seed}.json', {'time_s':t.tolist(),'residual_m':r.tolist(),'alarm':alarm.tolist()})
        row['los_mean_temporal_std_mm']=float(np.mean([np.std(r,axis=0,ddof=1).mean()*1000 for r in noise]))
        rows.append(row)
    save('summary.json', rows)
    save('manifest.json', {'scope':__doc__, 'source_hashes':read('sha256.json'),
        'code_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'calibration_seeds':[6000,6002,6004], 'evaluation_seeds':[6100,6102],
        'rule':'99th percentile higher abs circular mean, calibration t>=.6; complete trailing windows, 10ms update',
        'limitations':'overlapping correlated epochs; two noise pairs; ideal calibrated hardware; no new RF or PVT; first crossing is not persistent alarm'})
    save('sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')})
    for row in rows:
        print(row)


if __name__=='__main__':
    main()
