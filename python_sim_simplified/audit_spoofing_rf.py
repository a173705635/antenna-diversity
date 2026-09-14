"""Digital-only dual RF spoofing defense validation; no transmission hardware.

Full replacement models startup already tracking a single source. Mixtures are
static acquisition challenges, not a dynamic carry-off or navigation decoder.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_sdcp_positioning import channel_satellites
from positioning import carrier_range
from run_gps_spp import satellite_setup, receiver_reference, build_scene, run_record, serializable
from spoofing import window_evidence

ROOT = Path(__file__).resolve().parents[1]
W = 299792458/1575.42e6


def evaluate(pair, threshold):
    t = np.asarray(pair[0]['time_s']); phase = []; valid = np.ones((len(t), 8), bool)
    for r in pair:
        if not np.array_equal(t, r['time_s']): raise ValueError('time mismatch')
        if [h['prn'] for h in r['satellite_tracking']] != list(range(1, 9)): raise ValueError('PRN mismatch')
        phase.append(np.column_stack([carrier_range(dict(h, success=(h['acquired'] and h['tracking_status'] == 'tracking')),
            t, r['fs_dfe_hz'], .001, W) for h in r['satellite_tracking']]))
        valid &= np.isfinite(np.asarray(r['pseudoranges_m'], float))
    windows = window_evidence(t, *phase, valid, threshold, W)
    states = np.array([r['state'] for r in windows]); usable = states != -1
    # Integrity policy: only suppress an output on evidence/unknown; retained is not authenticated.
    return {'windows': windows, 'window_count': len(windows), 'valid_windows': int(usable.sum()),
            'alerts': int((states == 1).sum()), 'unknown': int((states == -1).sum()),
            'alert_fraction_valid': float((states[usable] == 1).mean()) if usable.any() else None,
            'coverage': float(usable.mean()), 'unprotected_A_pvt': pair[0]['summary'],
            'policy': 'alert or unknown => do not certify this GPS solution; no recovered position claim'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    args = p.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=False)
    threshold = json.loads(args.model.read_text(encoding='utf-8'))['threshold']
    def save(name, value):
        (out/name).write_text(json.dumps(serializable(value), indent=2, allow_nan=False), encoding='utf-8')
    sats = satellite_setup(); rx, rotation = receiver_reference()
    fake_rx = rx + rotation.T @ np.array([100., 0., 0.])
    fake = copy.deepcopy(sats)
    for s in fake:
        s['range_at_zero_m'] = float(np.linalg.norm(s['position_m']-fake_rx)+75000.)
    save('manifest.json', {'scope': __doc__, 'duration_ms': 400, 'threshold': threshold,
         'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
         'false_position_enu_m': [100, 0, 0], 'common_B_minus_A_m': -.035,
         'baseline_m': .0759, 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in Path(__file__).parent.glob('*.py')}})
    rows = []
    for label, seed in [('all_spoof', 9300), ('all_spoof', 9400), ('partial_2', 9500), ('mixture_equal', 9600)]:
        pair = []
        for ch in (0, 1):
            genuine = channel_satellites(sats, ch)
            forged = copy.deepcopy(fake)
            for s in forged: s['range_at_zero_m'] += ch*(-.035)
            chosen = forged if label == 'all_spoof' else forged[:2]+genuine[2:]
            if label == 'mixture_equal':
                scene, cfg = build_scene(genuine, 400)
                extra, _ = build_scene(forged, 400)
                scene += extra; del extra
            else: scene, cfg = build_scene(chosen, 400)
            # Metadata used by receiver has original satellite states/time aid, not spoof truth.
            record = run_record(seed+ch, sats, scene, cfg); del scene
            save(f'{label}_{seed}_{ch}.json', record); pair.append(record)
        rows.append({'case': label, 'seed': seed, **evaluate(pair, threshold)})
        save('summary.json', {'threshold': threshold, 'rows': rows})
    source = ROOT/'docs/reports/data/2026-09-12-geometry-free-rf'
    a = json.loads((source/'L1CA_0.json').read_text(encoding='utf-8'))
    b = json.loads((source/'L1CA_1.json').read_text(encoding='utf-8'))
    for label, rb, path in [('archived_los', b, source/'L1CA_1.json'),
            ('archived_single_mp', json.loads((ROOT/'docs/reports/data/2026-09-12-gf-startup-rf/L1CA_1_static.json').read_text(encoding='utf-8')),
             ROOT/'docs/reports/data/2026-09-12-gf-startup-rf/L1CA_1_static.json')]:
        rows.append({'case': label, 'reused_sha256': {str(source/'L1CA_0.json'): hashlib.sha256((source/'L1CA_0.json').read_bytes()).hexdigest(),
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()}, **evaluate([a, rb], threshold)})
    save('summary.json', {'threshold': threshold, 'rows': rows})
    save('sha256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')})
    print(json.dumps(serializable({'rows': rows}), indent=2))


if __name__ == '__main__': main()
