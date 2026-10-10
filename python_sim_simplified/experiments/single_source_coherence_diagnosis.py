"""Decompose saved receiver prompts into 5 ms coherent and residual energy.

This is an offline diagnostic of the Stage 14 output-window trace. It does not
change the receiver, use a new decision threshold, or publish navigation.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.recovery_geometry import FS
from core.serialization import serializable
from experiments.paths import RESULTS
from experiments.single_source_float_window_study import CASE_INPUTS
from experiments.single_source_suppression_study import OUTPUT as STUDY_OUTPUT, digest


STAGE14 = RESULTS / STUDY_OUTPUT / 'evidence_history'
OUTPUT = RESULTS / STUDY_OUTPUT / 'coherence_diagnosis'
CASES = ('pll_weak', 'genuine_absent', 'control')
PROMPT_SAMPLES = 2046
GROUP_SIZE = 5
BEAM_SAMPLES = round(.02 * FS)


def decompose_group(prompts, steering, projector, dimension):
    """Energy of five calibrated complex prompts under one spatial projector."""
    projected_steering = projector @ steering
    gain = np.vdot(steering, projected_steering).real
    if dimension < 2 or gain / len(steering) < .05:
        return dict(gain=float(gain), on_noncoh_sum=np.nan, on_coh=np.nan,
                    off_noncoh_sum=np.nan, off_coh=np.nan,
                    off_res_direct=np.nan, total_concentration=np.nan,
                    on_concentration=np.nan, off_concentration=np.nan)

    on_each = abs(prompts @ projected_steering.conj()) ** 2 / gain
    projected_total = np.einsum('tm,mn,tn->t', prompts.conj(), projector, prompts).real
    on_noncoh_sum = float(on_each.sum())
    off_noncoh_sum = float((projected_total - on_each).sum() / (dimension - 1))

    coherent = prompts.sum(axis=0) / np.sqrt(GROUP_SIZE)
    on_coh = float(abs(np.vdot(projected_steering, coherent)) ** 2 / gain)
    off_coh = float((np.vdot(coherent, projector @ coherent).real - on_coh)
                    / (dimension - 1))

    off_projector = projector - np.outer(projected_steering, projected_steering.conj()) / gain
    residual = prompts - prompts.mean(axis=0)
    off_res_direct = float(np.einsum('tm,mn,tn->', residual.conj(),
                                    off_projector, residual).real / (dimension - 1))
    np.testing.assert_allclose(off_noncoh_sum - off_coh, off_res_direct,
                               rtol=1e-10, atol=1e-10)
    with np.errstate(divide='ignore', invalid='ignore'):
        total_concentration = np.divide(on_coh + (dimension - 1) * off_coh,
                                        on_noncoh_sum + (dimension - 1) * off_noncoh_sum)
        on_concentration = np.divide(on_coh, on_noncoh_sum)
        off_concentration = np.divide(off_coh, off_noncoh_sum)
    return dict(gain=float(gain), on_noncoh_sum=on_noncoh_sum, on_coh=on_coh,
                off_noncoh_sum=off_noncoh_sum, off_coh=off_coh,
                off_res_direct=off_res_direct,
                total_concentration=float(total_concentration),
                on_concentration=float(on_concentration),
                off_concentration=float(off_concentration))


def analyze_groups(trace):
    """Deduplicate trace ticks and emit absolute 5 ms groups within one arc/P."""
    prompts = {}
    rows = zip(trace['trace_star'], trace['trace_arc'], trace['trace_start_sample'],
               trace['trace_available_time'], trace['trace_prompts'])
    for star, arc, start, available, window in rows:
        for offset, prompt in enumerate(window):
            tick_start = int(start + offset * PROMPT_SAMPLES)
            key = (int(star), int(arc), tick_start)
            if key in prompts:
                previous, first_available = prompts[key]
                np.testing.assert_allclose(prompt, previous, rtol=1e-12, atol=1e-12)
                prompts[key] = previous, min(first_available, available)
            else:
                prompts[key] = prompt, available

    fields = ('star', 'arc', 'start_sample', 'end_sample', 'start_time_s', 'end_time_s',
              'available_time', 'q',
              'gain', 'on_noncoh_sum', 'on_coh', 'off_noncoh_sum', 'off_coh',
              'off_res_direct', 'total_concentration', 'on_concentration',
              'off_concentration')
    output = {field: [] for field in fields}
    for star, arc, start in sorted(prompts):
        if start % (GROUP_SIZE * PROMPT_SAMPLES):
            continue
        keys = [(star, arc, start + offset * PROMPT_SAMPLES)
                for offset in range(GROUP_SIZE)]
        if any(key not in prompts for key in keys):
            continue
        block = start // BEAM_SAMPLES
        assert all(key[2] // BEAM_SAMPLES == block for key in keys)
        group_prompts = np.array([prompts[key][0] for key in keys])
        available = max(prompts[key][1] for key in keys)
        dimension = int(trace['trace_dimensions'][block])
        energy = decompose_group(group_prompts, trace['trace_nominal'][:, star],
                                 trace['trace_projectors'][block], dimension)
        row = dict(star=star, arc=arc, start_sample=start,
                   end_sample=start + GROUP_SIZE * PROMPT_SAMPLES,
                   start_time_s=start / FS,
                   end_time_s=(start + GROUP_SIZE * PROMPT_SAMPLES) / FS,
                   available_time=available, q=dimension, **energy)
        for field in fields:
            output[field].append(row[field])
    return {field: np.array(values) for field, values in output.items()}


def sample_phase_masks(table):
    """Classify groups by their actual sample interval, not availability."""
    start = table['start_time_s']
    end = table['end_time_s']
    return dict(pre=end <= .3,
                transition=(start >= .3) & (end <= .5),
                steady=start >= .5)


def stratified_summary(table):
    """Use sample intervals, separating pre, transition and steady by q."""
    metrics = ('total_concentration', 'on_concentration', 'off_concentration')
    summary = {}
    for phase, period in sample_phase_masks(table).items():
        summary[phase] = {}
        for q in (2, 3):
            selected = period & (table['q'] == q)
            row = dict(count=int(selected.sum()))
            for metric in metrics:
                values = table[metric][selected]
                finite = values[np.isfinite(values)]
                row[metric] = dict(valid=int(len(finite)),
                                   median=None if not len(finite) else float(np.median(finite)),
                                   p95=None if not len(finite) else float(np.percentile(finite, 95)))
            summary[phase][str(q)] = row
    return summary


def validate_saved_sources(stage14, case_inputs):
    """Match all saved trace files and RF inputs to the Stage 14 summary."""
    summary = json.loads((stage14 / 'summary.json').read_text(encoding='utf-8'))
    for name in CASES:
        _, folder = case_inputs[name]
        row = summary['cases'][name]
        for filename in ('samples.npy', 'observations.npz', 'metadata.json'):
            assert digest(folder / filename) == row['input_sha256'][filename], f'{name} {filename} changed'
        for kind in ('tracking', 'navigation', 'trace', 'history'):
            assert digest(stage14 / name / f'{kind}.npz') == row[f'{kind}_sha256'], f'{name} {kind} changed'
    return summary


def main():
    stage14_summary = validate_saved_sources(STAGE14, CASE_INPUTS)
    sources = source_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = {}
    plots = {}
    for name in CASES:
        with np.load(STAGE14 / name / 'trace.npz') as saved:
            trace = dict(saved)
        table = analyze_groups(trace)
        path = OUTPUT / f'{name}.npz'
        np.savez_compressed(path, **table)
        reports[name] = dict(stage14_trace_sha256=stage14_summary['cases'][name]['trace_sha256'],
                             table_sha256=digest(path), groups=len(table['star']),
                             strata=stratified_summary(table))
        plots[name] = table
    fig, axes = plt.subplots(2, len(CASES), figsize=(12, 7), sharex='col',
                             constrained_layout=True)
    colors = dict(pre='tab:blue', transition='tab:orange', steady='tab:green')
    for column, name in enumerate(CASES):
        table = plots[name]
        for row, metric in enumerate(('on_concentration', 'off_concentration')):
            axis = axes[row, column]
            for phase, period in sample_phase_masks(table).items():
                for q, marker in ((2, 'o'), (3, 'x')):
                    selected = period & (table['q'] == q) & np.isfinite(table[metric])
                    axis.scatter(table['end_time_s'][selected], table[metric][selected],
                                 s=5, alpha=.3, color=colors[phase], marker=marker,
                                 label=f'{phase} q{q} n={selected.sum()}')
            axis.set(ylabel=metric.replace('_', ' '))
            axis.grid(alpha=.3)
        axes[0, column].set_title(name)
        axes[1, column].set_xlabel('Group sample end (s)')
        axes[0, column].legend(fontsize=6, ncol=2)
    fig.savefig(OUTPUT / 'concentration.png', dpi=150)
    plt.close(fig)
    assert source_hashes() == sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources,
        stage14_summary_sha256=digest(STAGE14 / 'summary.json'),
        group_ms=5, cases=reports)), indent=2, allow_nan=False), encoding='utf-8')
    return reports


if __name__ == '__main__':
    main()
