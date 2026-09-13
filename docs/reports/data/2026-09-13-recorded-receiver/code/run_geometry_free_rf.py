"""Four independent RF/DFE/tracking streams, common scene and epochs; raw GF only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_sdcp_positioning import channel_satellites
from positioning import carrier_range
from run_gps_spp import satellite_setup, build_scene, run_record, serializable
from sdcp import geometry_free_sdcp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--duration-ms', type=int, default=400)
    args = p.parse_args()
    if args.duration_ms < 300: p.error('duration must be >=300 ms')
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, data):
        (args.output/name).write_text(json.dumps(serializable(data), indent=2, allow_nan=False), encoding='utf-8')
    sats = satellite_setup()
    records, phases = {}, {}
    save('manifest.json', {'scope': __doc__, 'scene': 'LOS only, shared ideal clock, zero hardware bias',
        'duration_ms': args.duration_ms, 'satellites': sats, 'seeds': [7100,7101,7102,7103],
        'source_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in Path(__file__).parent.glob('*.py')}})
    for band, kind, freq in [(0,'L1CA',1575.42e6),(1,'L5Q',1176.45e6)]:
        for ch in (0,1):
            key = f'{kind}_{ch}'
            shifted = channel_satellites(sats,ch)
            scene,cfg = build_scene(shifted,args.duration_ms,signal_kind=kind)
            r = run_record(7100+2*band+ch,shifted,scene,cfg)
            del scene
            save(key+'.json',r)
            records[key] = r
            phases[key] = np.column_stack([carrier_range(dict(h,success=h.get('tracking_status')=='tracking'),
                r['time_s'],r['fs_dfe_hz'],.001,299792458/freq) for h in r['satellite_tracking']])
    t = np.asarray(records['L1CA_0']['time_s'])
    if not all(np.array_equal(t,r['time_s']) for r in records.values()):
        raise ValueError('time axes differ')
    valid = np.logical_and.reduce([np.isfinite(v) for v in phases.values()])
    valid &= np.logical_and.reduce([np.isfinite(r['pseudoranges_m']) for r in records.values()])
    columns = []
    for j,s in enumerate(sats):
        mask = valid[:,j]
        item = {'prn': s['prn'], 'valid_epochs': int(mask.sum()), 'total_epochs': len(t)}
        if mask.any():
            v = [phases[k][mask,j:j+1] for k in ['L1CA_0','L1CA_1','L5Q_0','L5Q_1']]
            gf = geometry_free_sdcp(t[mask],v[0],v[1],t[mask],v[2],v[3])[:,0]
            item.update(time_s=t[mask],raw_gf_m=gf,mean_m=float(gf.mean()),
                        std_m=float(gf.std()),max_abs_m=float(abs(gf).max()))
        columns.append(item)
    save('summary.json', {'scope': 'LOS RF baseline, raw uncalibrated combination; no detector/no PVT benefit claim',
                          'satellites': columns})
    save('sha256.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.output.glob('*.json')})
    print('Saved raw GF RF baseline',flush=True)


if __name__ == '__main__': main()
