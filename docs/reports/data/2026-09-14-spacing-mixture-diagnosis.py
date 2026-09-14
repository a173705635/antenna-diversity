"""Read-only diagnosis of archived mixture_6 scores; no RF or threshold refit."""
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
W = 299792458 / 1575.42e6
PAIRS = np.array(list(itertools.combinations(range(3), 2)))
TRIPLES = np.array(list(itertools.combinations(range(8), 3)))
results = []
thresholds = []
for archive in ('2026-09-13-gpspatron-spatial', '2026-09-14-spacing-1m'):
    base = ROOT/archive
    payload = json.loads((base/'model/thresholds.json').read_text(encoding='utf-8'))
    threshold = payload['thresholds']['3_cluster3']
    thresholds.append(threshold)
    rows = json.loads((base/'rf_b/summary.json').read_text(encoding='utf-8'))['rows']
    for seed in (610000, 620000):
        for case, batch in (('mixture_6', 'rf_b'), ('all_spoof', 'rf_a')):
            with np.load(base/batch/f'{case}_{seed}_observations.npz') as data:
                x, valid = data['carrier_m'][:, :3], data['valid'][:, :3]
            assert valid.all()
            for k in range(4):
                block = x[k*10:(k+1)*10]
                d = block[:, PAIRS[:, 1], :] - block[:, PAIRS[:, 0], :]
                z = np.exp(4j*np.pi*d/W)
                # Baseline x candidate triple, averaged over the ten epochs.
                dispersion = (1-abs(z[..., TRIPLES].mean(axis=-1))).mean(axis=0)
                scores = dispersion.mean(axis=0)
                best = int(scores.argmin())
                score = float(scores[best])
                if case == 'mixture_6':
                    saved = next(r for r in rows if (r['case'], r['seed'], r['layout'], r['method']) ==
                                 (case, seed, '3', 'cluster3'))['windows'][k]['score']
                    np.testing.assert_allclose(score, saved, atol=1e-12, rtol=0)
                results.append(dict(archive=archive, seed=seed, case=case, window=k,
                    best_prns=(TRIPLES[best]+1).tolist(), score=score, threshold=threshold,
                    score_over_threshold=score/threshold, alert=bool(score<threshold),
                    baseline_names=['TL-TR', 'TL-BL', 'TR-BL'],
                    baseline_dispersion=dispersion[:, best].tolist()))

for row in results:
    row['alerts_under_old_and_new_threshold'] = [bool(row['score'] < t) for t in thresholds]
out = ROOT/'2026-09-14-spacing-mixture-diagnosis.json'
out.write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
for archive in dict.fromkeys(r['archive'] for r in results):
    for case in ('mixture_6', 'all_spoof'):
        selected = [r for r in results if r['archive']==archive and r['case']==case]
        print(archive, case, 'range=', min(r['score'] for r in selected), max(r['score'] for r in selected),
              'alerts=', sum(r['alert'] for r in selected), 'of', len(selected))
print('PASS: 16 saved mixture scores independently reconstructed; all 32 windows valid')
