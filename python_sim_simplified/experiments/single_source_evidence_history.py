"""Diagnose causal, noncoherent direction evidence from saved RF output windows.

Only the prompts covered by receiver output windows are observed here. The
20/50/100 ms histories are offline candidates and never enter navigation.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.recovery_geometry import FS
from core.serialization import serializable
from core.single_source import frontend, navigate
from experiments.paths import RESULTS
from experiments.single_source_beam_history_rf import score
from experiments.single_source_float_window_study import CASE_INPUTS
from experiments.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, RECEIVER_PLL, digest,
)


OUTPUT = RESULTS / STUDY_OUTPUT / 'evidence_history'
WINDOW_MS = (20, 50, 100)
PROMPT_SAMPLES = 2046
BEAM_SAMPLES = round(.02 * FS)
THRESHOLD = 3.
CASES = ('pll_weak', 'genuine_absent', 'control')


def prompt_energy(prompt, steering, projector, dimension):
    """One calibrated 1 ms prompt's on/off energy in its actual spatial plane."""
    projected_steering = projector @ steering
    gain = np.vdot(steering, projected_steering).real
    if dimension < 2 or gain / len(steering) < .05:
        return np.nan, np.nan
    on = abs(np.vdot(steering, projector @ prompt)) ** 2 / gain
    off = (np.vdot(prompt, projector @ prompt).real - on) / (dimension - 1)
    return float(on), float(off)


def analyze_trace(observations):
    """Verify original 20 ms evidence, then form available contiguous histories.

    A prompt becomes available at the first original output window containing
    it. Later overlapping windows must reproduce the same complex prompt.
    """
    shape = (*observations['support_split'].shape, len(WINDOW_MS))
    result = dict(ratio=np.full(shape, np.nan), on=np.full(shape, np.nan),
                  off=np.full(shape, np.nan), sample_count=np.zeros(shape, int),
                  start_time_s=np.full(shape, np.nan), end_time_s=np.full(shape, np.nan),
                  data_age_s=np.full(shape, np.nan),
                  oldest_sample_age_s=np.full(shape, np.nan))
    prompts = {}
    rows = list(zip(observations['trace_window_k'], observations['trace_star'],
                    observations['trace_arc'], observations['trace_start_sample'],
                    observations['trace_available_time'], observations['trace_prompts']))
    for k, star, arc, start, available, window_prompts in rows:
        np.testing.assert_allclose(available, observations['available_time'][k])
        for offset, prompt in enumerate(window_prompts):
            tick_start = int(start + offset * PROMPT_SAMPLES)
            key = (int(star), int(arc), tick_start)
            if key in prompts:
                previous, first_available = prompts[key]
                np.testing.assert_allclose(prompt, previous, rtol=1e-12, atol=1e-12)
                prompts[key] = (previous, min(first_available, available))
            else:
                prompts[key] = (prompt, available)

    energy_by_key = {}
    for key, (prompt, _) in prompts.items():
        star, _, tick_start = key
        block = tick_start // BEAM_SAMPLES
        energy_by_key[key] = prompt_energy(
            prompt, observations['trace_nominal'][:, star],
            observations['trace_projectors'][block],
            observations['trace_dimensions'][block])

    for k, star, arc, start, available, window_prompts in rows:
        end = int(start + 20 * PROMPT_SAMPLES)
        for index, duration_ms in enumerate(WINDOW_MS):
            begins = range(end - duration_ms * PROMPT_SAMPLES, end, PROMPT_SAMPLES)
            keys = [(int(star), int(arc), tick_start) for tick_start in begins]
            if any(key not in prompts or prompts[key][1] > available for key in keys):
                continue
            energy = [energy_by_key[key] for key in keys]
            if not np.isfinite(energy).all():
                continue
            on, off = np.sum(energy, axis=0)
            result['on'][k, star, index] = on
            result['off'][k, star, index] = off
            result['ratio'][k, star, index] = on / off
            result['sample_count'][k, star, index] = duration_ms
            result['start_time_s'][k, star, index] = (end - duration_ms * PROMPT_SAMPLES) / FS
            result['end_time_s'][k, star, index] = end / FS
            result['data_age_s'][k, star, index] = available - end / FS
            result['oldest_sample_age_s'][k, star, index] = (
                available - result['start_time_s'][k, star, index])
        np.testing.assert_allclose(result['on'][k, star, 0],
                                   observations['split_on_energy'][k, star], rtol=1e-10)
        np.testing.assert_allclose(result['off'][k, star, 0],
                                   observations['split_off_energy'][k, star], rtol=1e-10)
        np.testing.assert_allclose(result['ratio'][k, star, 0],
                                   observations['support_split'][k, star], rtol=1e-10)
    return result


def support_counts(observations, history_ratio):
    """Count directional support after the receiver's unchanged quality gates."""
    quality = (np.isfinite(observations['ranges']) & (observations['lock'] >= .5)
               & (observations['cn0'] >= 25.) & (observations['gain'] >= .05))
    ratios = np.concatenate((observations['support'][:, :, None], history_ratio), axis=2)
    return (quality[:, :, None] & (ratios >= THRESHOLD)).sum(axis=1)


def support_statistics(counts, evaluation, tail):
    """Summarize support epochs; each output epoch spans 10 ms."""
    six = evaluation & (counts >= 6)
    edges = np.diff(np.r_[False, six, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return dict(epochs_ge4=int((evaluation & (counts >= 4)).sum()),
                epochs_ge6=int(six.sum()),
                longest_ge6_s=0. if not len(starts) else float((ends - starts).max() * .01),
                tail50_ge4=int((tail & (counts >= 4)).sum()),
                tail50_ge6=int((tail & (counts >= 6)).sum()),
                tail50_mean_count=float(counts[tail].mean()))


def main():
    """Replay each existing RF once and save trace, navigation, and diagnostics."""
    sources = source_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepared = {}
    for name in CASES:
        historical_tracking, folder = CASE_INPUTS[name]
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        hashes = {filename: digest(folder / filename)
                  for filename in ('samples.npy', 'observations.npz', 'metadata.json')}
        assert hashes['samples.npy'] == metadata['samples_sha256']
        assert hashes['observations.npz'] == metadata['observations_sha256']
        prepared[name] = (historical_tracking, folder, metadata, hashes)
    pll_root = RESULTS / STUDY_OUTPUT / 'pll'
    pll_summary = json.loads((pll_root / 'summary.json').read_text(encoding='utf-8'))
    assert digest(prepared['pll_weak'][0]) == pll_summary['tracking_sha256']
    assert digest(pll_root / 'navigation.npz') == pll_summary['navigation_sha256']
    reference_initialization = pll_summary['pll']['initialization_available_time_s']

    reports = {}
    plots = {}
    for name in CASES:
        historical_tracking, folder, metadata, hashes = prepared[name]
        with np.load(folder / 'observations.npz') as saved:
            record = dict(saved)
        record['samples'] = np.load(folder / 'samples.npy', mmap_mode='r')
        print(name, 'frontend start', flush=True)
        observations = frontend(record, ANTENNAS, split_support=True,
                                receiver_pll=RECEIVER_PLL, evidence_trace=True)
        navigation = navigate(observations)
        history = analyze_trace(observations)
        if name == 'pll_weak':
            reference = historical_tracking
            with np.load(reference) as saved:
                for key in saved.files:
                    np.testing.assert_array_equal(observations[key], saved[key], err_msg=key)
            with np.load(pll_root / 'navigation.npz') as saved:
                for key in saved.files:
                    np.testing.assert_array_equal(navigation[key], saved[key], err_msg=key)
            assert navigation['initialization_time'] == reference_initialization
        output = OUTPUT / name
        output.mkdir(exist_ok=True)
        trace_keys = [key for key in observations if key.startswith('trace_')]
        counts = support_counts(observations, history['ratio'])
        np.savez_compressed(output / 'tracking.npz',
                            **{key: value for key, value in observations.items()
                               if key not in trace_keys})
        np.savez_compressed(output / 'navigation.npz', **navigation)
        np.savez_compressed(output / 'trace.npz',
                            **{key: observations[key] for key in trace_keys})
        np.savez_compressed(output / 'history.npz', **history, support_counts=counts)
        post = observations['available_time'] >= metadata['case']['attack_start']
        tail = np.arange(len(post)) >= len(post) - 50
        strata = {}
        for index, duration_ms in enumerate((0, *WINDOW_MS)):
            label = 'original' if index == 0 else str(duration_ms)
            strata[label] = support_statistics(counts[:, index], post, tail)
        for index, duration_ms in enumerate(WINDOW_MS):
            finite = post[:, None] & np.isfinite(history['ratio'][:, :, index])
            strata[str(duration_ms)].update(
                evaluated=int(finite.sum()),
                above_threshold=int((finite & (history['ratio'][:, :, index] >= THRESHOLD)).sum()),
                latest_window_end_s=None if not finite.any() else
                    float(np.nanmax(history['end_time_s'][:, :, index][finite])),
                maximum_data_age_s=None if not finite.any() else
                    float(np.nanmax(history['data_age_s'][:, :, index][finite])),
                maximum_oldest_sample_age_s=None if not finite.any() else
                    float(np.nanmax(history['oldest_sample_age_s'][:, :, index][finite])),
                samples_per_evaluation=duration_ms)
        reports[name] = dict(input_sha256=hashes,
                             historical_tracking_sha256=digest(historical_tracking),
                             historical_tracking_pll=RECEIVER_PLL if name == 'pll_weak' else None,
                             tracking_sha256=digest(output / 'tracking.npz'),
                             navigation_sha256=digest(output / 'navigation.npz'),
                             trace_sha256=digest(output / 'trace.npz'),
                             history_sha256=digest(output / 'history.npz'),
                             trace_windows=len(observations['trace_window_k']),
                             evaluation_start_s=metadata['case']['attack_start'],
                             evaluation_kind='control' if name == 'control' else 'post_attack',
                             navigation=score(navigation, observations, metadata['case']),
                             history=strata)
        plots[name] = (observations['available_time'], counts)
        print(name, 'done', reports[name]['navigation'], flush=True)
    fig, axes = plt.subplots(len(CASES), 1, figsize=(10, 8), sharex=True,
                             constrained_layout=True)
    for axis, name in zip(axes, CASES):
        available, counts = plots[name]
        for index, label in enumerate(('original', '20 ms', '50 ms', '100 ms')):
            axis.plot(available, counts[:, index], label=label)
        axis.axhline(6, color='gray', linestyle=':', linewidth=1)
        axis.set(ylabel=f'{name}\nsupport count')
        axis.grid(alpha=.3)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel('Output available time (s)')
    fig.savefig(OUTPUT / 'support_count.png', dpi=150)
    plt.close(fig)
    assert source_hashes() == sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources, receiver_pll=RECEIVER_PLL,
        antennas=ANTENNAS, threshold=THRESHOLD, window_ms=WINDOW_MS,
        cases=reports)), indent=2, allow_nan=False), encoding='utf-8')
    return reports


if __name__ == '__main__':
    main()
