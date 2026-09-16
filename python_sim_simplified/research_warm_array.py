"""Frozen proposals and paired four-channel sampled-code experiments."""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from signal_recovery import codes, sky, steering, synthesize, N
from presets import mate70_phase_centers
from research_hundred import load

ROOT = Path('results/research-warm-hundred/array')
SUBSETS = {'two_short': [0, 1], 'two_long': [0, 2], 'three': [0, 1, 2], 'four': [0, 1, 2, 3]}


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def generate(seed, kind, epochs=8):
    rng = np.random.default_rng(seed)
    count = 8
    directions = sky(seed, count)
    positions = np.array(list(mate70_phase_centers().values()))
    spectrum = codes(count)
    source = np.array([np.cos(.7), np.sin(.7), 0.])
    if kind == 'near':
        source = directions[0]
    phase = rng.normal(0, .25, 4)
    gain = np.exp(1j*phase + rng.normal(0, .12, 4))
    cn0 = 35. if kind == 'weak' else 45.
    sample_list, sat_list, truth_list = [], [], []
    for k in range(epochs):
        attack = k >= 3 and kind != 'normal'
        actual = np.array([k*.2, 0., 0.])
        fake = actual + np.array([100., 0., 0.])
        noise = (rng.normal(size=(4, N)) + 1j*rng.normal(size=(4, N)))/np.sqrt(2)
        current_gain = gain*np.exp(1j*k*.025*np.arange(4)) if kind == 'drift' else gain
        x, satellites, truth = synthesize(spectrum, directions, actual, fake, source,
            positions, 0., current_gain, noise, cn0, .02, 10., count if attack else 0,
            genuine=not (kind == 'absent' and attack))
        sample_list.append(x); sat_list.append(satellites); truth_list.append(actual)
    record = dict(samples=np.array(sample_list), nominal=steering(directions, positions, 0.),
                  spectrum=spectrum, coarse=np.round((21e6+np.arange(count)*17000+75000)/10000)*10000,
                  satellites=np.array(sat_list), startup_count=3,
                  snapshot_end_s=np.arange(1, epochs+1)*.02,
                  initialization_end_s=.06, attack_start_s=.06,
                  initialization_to_attack_gap_s=0.)
    return record, np.array(truth_list)


def score(result, truth):
    error = np.linalg.norm((result['position']-truth)[:, :2], axis=1)[3:]
    return dict(initialized=result['initialized'], opportunities=len(error),
                solved=int(np.isfinite(error).sum()), good20=int((error <= 20).sum()),
                error30=int((error >= 30).sum()), error50=int((error >= 50).sum()),
                errors_m=[float(e) if np.isfinite(e) else None for e in error],
                admitted=int(result['admitted'][3:].sum()),
                mean_weight_norm=float(np.mean(result['weight_norm'][3:])))


def propose(number, parent, hypothesis):
    folder = ROOT/f'iteration{number:03d}'
    folder.mkdir(parents=True)
    save(folder/'proposal.json', dict(number=number, parent=parent, hypothesis=hypothesis,
         time_utc=datetime.now(timezone.utc).isoformat()))


def finish(number, decision, reason, next_step):
    folder = ROOT/f'iteration{number:03d}'
    save(folder/'conclusion.json', dict(decision=decision, reason=reason, next_step=next_step))


def run(number):
    folder = ROOT/f'iteration{number:03d}'
    proposal = json.loads((folder/'proposal.json').read_text(encoding='utf-8'))
    shutil.copy2('warm_array.py', folder/'warm_array.py')
    parent = ROOT/'baseline.py' if proposal['parent'] == 0 else ROOT/f"iteration{proposal['parent']:03d}/warm_array.py"
    models = dict(before=load(parent), after=load(folder/'warm_array.py'))
    rows = []
    for kind in ('normal', 'attack', 'weak', 'near', 'absent', 'drift'):
        record, truth = generate(111001, kind)
        digest = hashlib.sha256(record['samples'].tobytes()).hexdigest()
        for name, subset in SUBSETS.items():
            result = {v: module.process(record, subset) for v, module in models.items()}
            np.savez_compressed(folder/f'{kind}_{name}.npz', **{v: r['position'] for v, r in result.items()})
            rows.append(dict(kind=kind, subset=name, input_sha256=digest,
                             **{v: score(r, truth) for v, r in result.items()}))
    totals = {v: {key: sum(r[v][key] for r in rows) for key in ('solved', 'good20', 'error30', 'error50')}
              for v in models}
    save(folder/'results.json', dict(rows=rows, totals=totals,
         source_sha256=hashlib.sha256((folder/'warm_array.py').read_bytes()).hexdigest()))
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    run(int(sys.argv[1]))
