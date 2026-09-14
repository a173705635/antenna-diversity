"""Independent-event spatial spoofing evidence; model only, not receiver data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from spoofing import spatial_dispersion
from audit_dual_polarization import wilson

W = 299792458/1575.42e6


def samples(seed, n, kind, baseline=.0759, noise_m=.003):
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 2*np.pi, (n, 8))
    up = rng.uniform(np.sin(np.deg2rad(15)), 1, (n, 8))
    if kind == 'concentrated_los':
        az = rng.normal(0, .015, (n, 8)); up = rng.uniform(.69, .71, (n, 8))
    dirs = np.stack([np.sqrt(1-up**2)*np.cos(az), np.sqrt(1-up**2)*np.sin(az), up], axis=-1)
    baseline_dir = rng.normal(size=(n, 3))
    baseline_dir /= np.linalg.norm(baseline_dir, axis=1)[:, None]
    d = -baseline*np.einsum('nsc,nc->ns', dirs, baseline_dir)
    common = rng.uniform(-baseline, baseline, (n, 1))
    if kind in ('all_spoof', 'bias_spoof'): d[:] = common
    if kind.startswith('partial_'): d[:, :int(kind.split('_')[1])] = common
    if kind == 'two_sources':
        d[:, :4] = common; d[:, 4:] = rng.uniform(-baseline, baseline, (n, 1))
    if kind == 'bias_spoof': d += rng.normal(0, .02, (n, 8))
    if kind == 'single_mp_los': d[:, 0] += rng.uniform(-.03, .03, n)
    # Common receiver offset, arbitrary integer and half-cycle ambiguities.
    d += rng.uniform(-1, 1, (n, 1)) + rng.integers(-10, 11, (n, 8))*W/2
    observed = d[:, None, :] + rng.normal(0, noise_m, (n, 10, 8))
    return spatial_dispersion(observed, W).mean(axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=False)
    calibration = samples(9100, 5000, 'los')
    threshold = float(np.quantile(calibration, .01, method='lower'))
    np.savez_compressed(out/'calibration.npz', score=calibration)
    rows = []
    cases = [('los', .0759, .003), ('all_spoof', .0759, .003),
             ('partial_2', .0759, .003), ('partial_4', .0759, .003), ('partial_6', .0759, .003),
             ('two_sources', .0759, .003), ('bias_spoof', .0759, .003),
             ('concentrated_los', .0759, .003), ('los', .01, .003),
             ('single_mp_los', .0759, .003), ('all_spoof', .0759, .02)]
    for i, (kind, length, noise) in enumerate(cases):
        score = samples(9200+i, 5000, kind, length, noise)
        hits = int((score < threshold).sum())
        rows.append(dict(case=kind, baseline_m=length, noise_m=noise, seed=9200+i,
                         events=len(score), alerts=hits, fraction=hits/len(score),
                         wilson95=wilson(hits, len(score)), median_score=float(np.median(score))))
        np.savez_compressed(out/f'events_{i}.npz', score=score)
    result = dict(scope=__doc__, threshold=threshold, calibration_seed=9100,
                  calibration_events=5000, rows=rows,
                  decision='100ms block mean circular dispersion below frozen threshold; suspected common source only',
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__), Path(__file__).with_name('spoofing.py')]})
    (out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out/'sha256.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in out.iterdir()}, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
