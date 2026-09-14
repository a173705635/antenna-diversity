"""Multi-baseline common-source evidence inspired by public GPSPATRON principles.

The statistics and thresholds are local research choices, NOT GP-Cloud's
proprietary algorithm. No geometry/truth/clean-start reference is consumed.
Distances must refer to synchronized epochs in ONE carrier frequency band.


Inputs use aligned increasing epochs, identical PRN order, boolean quality masks, 2-4 antennas and at least 3 PRNs.
"""
from itertools import combinations

import numpy as np


def spatial_scores(carrier_m, wavelength_m):
    """Return global and minimum-three-PRN circular dispersion per input block.

    Input shape (..., epochs, antennas, PRNs), with 2–4 antennas and >=3 PRNs.
    Lower scores suggest a common source. All baseline pairs are averaged,
    not treated as statistically independent. Doubled phase removes half-cycle
    ambiguities but also creates spatial aliases. NaNs must be gated upstream.
    """
    x = np.asarray(carrier_m, float)
    pairs = np.array(list(combinations(range(x.shape[-2]), 2)))
    d = x[..., pairs[:, 1], :] - x[..., pairs[:, 0], :]
    z = np.exp(4j * np.pi * d / wavelength_m)
    global_score = (1 - abs(z.mean(axis=-1))).mean(axis=(-2, -1))
    triples = np.array(list(combinations(range(x.shape[-1]), 3)))
    # The SAME PRN subset must remain coherent over the whole block.
    subset_scores = (1 - abs(z[..., triples].mean(axis=-1))).mean(axis=(-3, -2))
    return {'global': np.clip(global_score, 0., 1.),
            'cluster3': np.clip(subset_scores.min(axis=-1), 0., 1.)}


def calibrate_threshold(h0_scores, alpha=.01):
    """Freeze the empirical lower-tail quantile; alert uses strict '<'.

    Independent holdout validation is still required. The complete cluster
    scan is calibrated, including its multiple comparisons, in each layout.
    """
    values = np.asarray(h0_scores, float)
    return float(np.quantile(values, alpha, method='lower'))


def detect_windows(times_s, carrier_m, valid, threshold, wavelength_m, *,
                   method='cluster3', window_epochs=10, step_s=.01,
                   observation_latency_s=.001):
    """Complete causal blocks: 1 evidence, 0 no evidence, -1 unknown.

    Caller supplies aligned epochs and PRN order with positive wavelength.
    No evidence is NOT authentication. Invalid data is never silently dropped
    or imputed. A configuration/PRN-count change requires separate calibration.
    decision_s includes the caller's observation production latency.
    """
    t, x, v = np.asarray(times_s, float), np.asarray(carrier_m, float), np.asarray(valid)
    rows = []
    for start in range(0, len(t), window_epochs):
        end = min(start + window_epochs, len(t))
        complete = end - start == window_epochs
        regular = np.allclose(np.diff(t[start:end]), step_s, rtol=0, atol=1e-9)
        good = complete and regular and v[start:end].all() and np.isfinite(x[start:end]).all()
        score = float(spatial_scores(x[start:end], wavelength_m)[method]) if good else None
        rows.append({'start_index': start, 'end_index': end,
                     'start_s': float(t[start]), 'decision_s': float(t[end - 1] + observation_latency_s),
                     'score': score, 'state': int(score < threshold) if good else -1,
                     'reason': 'evaluated' if good else ('incomplete' if not complete else
                               'time_gap' if not regular else 'invalid_observation')})
    return rows


def observations_from_records(records, wavelength_m):
    """Adapt existing run_gps_spp records without consulting truth or PVT errors."""
    from positioning import carrier_range
    t = np.asarray(records[0]['time_s'], float)
    carriers, masks = [], []
    for record in records:
        x = np.column_stack([carrier_range(
            dict(h, success=(h['acquired'] and h['tracking_status'] == 'tracking')),
            t, record['fs_dfe_hz'], .001, wavelength_m) for h in record['satellite_tracking']])
        # Existing pseudorange mask already carries current PLL quality screening.
        mask = np.isfinite(np.asarray(record['pseudoranges_m'], float)) & np.isfinite(x)
        carriers.append(x); masks.append(mask)
    return t, np.stack(carriers, axis=1), np.stack(masks, axis=1)


def output_gate(times_s, windows):
    """Causal example policy; -1 startup/unknown, 1 alert, 0 provisionally retain.

    Outputs before the first decision are held. Completed decisions are used
    only at/after their availability time; no retrospective gain is credited.
    One new no-evidence window releases the gate (not certified reacquisition).
    """
    t = np.asarray(times_s, float)
    if not windows:
        return np.full(t.shape, -1, int)
    decisions = np.array([r['decision_s'] for r in windows])
    states = np.array([r['state'] for r in windows], int)
    index = np.searchsorted(decisions, t, side='right') - 1
    return np.where(index >= 0, states[np.maximum(index, 0)], -1)
