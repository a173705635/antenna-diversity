"""Offline shape of weak-scene short-path drift around actual fit switches.

The fixed windows characterize timing only. They do not reconstruct all
navigation history, estimate a causal projection effect, or run new RF.
"""
import json
from pathlib import Path

import numpy as np

from experiments.spoofing import single_source_nominal_yaw as study
from experiments.spoofing.single_source_projection_long import OUTPUT as STAGE47
from experiments.spoofing.single_source_suppression_study import digest
from tools.spoofing.nominal_yaw_mechanism import (
    STAGE48_SUMMARY_SHA256, assert_frozen_source_identity, read_npz,
    verified_path,
)


METRICS = ('before_velocity_en_mps', 'after20_excess_en_m',
           'after80_velocity_change_en_mps')


def evidence_blocks(end_sample, width):
    """Return block/sample overlaps for the half-open RF window [end-width,end)."""
    start = end_sample-width
    first = start//width
    stop = (end_sample-1)//width+1
    return [(block, min(end_sample, (block+1)*width)-max(start, block*width))
            for block in range(first, stop)]


def _vectors(steps, boundary, half_step, fs):
    before = np.sum([steps[boundary+k*half_step] for k in range(-9, 1)], axis=0)
    after20 = np.sum([steps[boundary+k*half_step] for k in range(1, 3)], axis=0)
    after80 = np.sum([steps[boundary+k*half_step] for k in range(3, 11)], axis=0)
    before_velocity = before/(10*half_step/fs)
    return (before_velocity,
            after20-before_velocity*(2*half_step/fs),
            after80/(8*half_step/fs)-before_velocity)


def _component_range(events, key):
    if not events:
        return None
    values = np.asarray([event[key] for event in events])
    return dict(median_en=np.median(values, axis=0).tolist(),
                minimum_en=np.min(values, axis=0).tolist(),
                maximum_en=np.max(values, axis=0).tolist())


def transition_shape(samples, positions, active, fs, width):
    """Score fixed 100 ms windows on three-way common adjacent 10 ms epochs."""
    half_step = width//2
    score_start = int(np.rint(.3*fs))
    common = (samples >= score_start) & np.logical_and.reduce([
        np.isfinite(value).all(axis=1) for value in positions.values()])
    steps = {}
    for yaw, value in positions.items():
        increments = {}
        for index in range(1, len(samples)):
            if (common[index-1] and common[index]
                    and samples[index]-samples[index-1] == half_step):
                delta = value[index, :2]-value[index-1, :2]
                increments[int(samples[index])] = delta-np.array(
                    [2*half_step/fs, 0.])
        steps[yaw] = increments
    switches = {yaw: np.flatnonzero(state[1:] != state[:-1])+1
                for yaw, state in active.items()}
    result = {}
    for yaw, boundaries in switches.items():
        counts = dict(on=0, off=0, boundary=0, dense=0, missing_common=0)
        events = []
        rejected = []
        for block in boundaries:
            boundary = int(block*width)
            direction = 'on' if active[yaw][block] else 'off'
            if boundary-5*width < samples[0] or boundary+5*width > samples[-1]:
                reason = 'boundary'
            elif any(other*width >= boundary-5*width
                     and other*width <= boundary+5*width
                     and (other_yaw != yaw or other != block)
                     for other_yaw, other_blocks in switches.items()
                     for other in other_blocks):
                reason = 'dense'
            elif any(boundary+k*half_step not in steps[yaw]
                     for k in range(-9, 11)):
                reason = 'missing_common'
            else:
                reason = None
            if reason is not None:
                counts[reason] += 1
                rejected.append(dict(boundary_sample=boundary,
                                     direction=direction, reason=reason))
                continue
            own = _vectors(steps[yaw], boundary, half_step, fs)
            baseline = _vectors(steps['+0'], boundary, half_step, fs)
            event = dict(boundary_sample=boundary, direction=direction)
            for key, vector, reference in zip(METRICS, own, baseline):
                event[key] = vector.tolist()
                event['paired_'+key] = (vector-reference).tolist()
            events.append(event)
            counts[direction] += 1
        by_direction = {}
        for direction in ('on', 'off'):
            subset = [event for event in events if event['direction'] == direction]
            by_direction[direction] = dict(count=len(subset), **{
                key: _component_range(subset, key) for key in
                (*METRICS, *(f'paired_{name}' for name in METRICS))})
        result[yaw] = dict(counts=counts, events=events, rejected=rejected,
                           by_direction=by_direction)
    return dict(common_published_epochs=int(common.sum()), paths=result)


def analyze():
    assert digest(STAGE47/'summary.json') == study.STAGE47_SUMMARY_SHA256
    assert digest(study.OUTPUT/'recovery_summary.json') == STAGE48_SUMMARY_SHA256
    stage47 = json.loads((STAGE47/'summary.json').read_text(encoding='utf-8'))
    stage48 = json.loads((study.OUTPUT/'recovery_summary.json').read_text(encoding='utf-8'))
    plan = json.loads((study.OUTPUT/'plan.json').read_text(encoding='utf-8'))
    assert_frozen_source_identity(plan, stage48)
    samples = None
    positions = {}
    active = {}
    for yaw in study.YAWS_DEG:
        folder, navigation_name = verified_path('weak', yaw, stage47, stage48)
        tracking = read_npz(folder/'tracking.npz')
        navigation = read_npz(folder/navigation_name)
        fit = read_npz(folder/'fit.npz')
        np.testing.assert_array_equal(tracking['time'], navigation['observation_time'])
        current = np.rint(tracking['time']*study.FS).astype(np.int64)
        if samples is None:
            samples = current
        else:
            np.testing.assert_array_equal(current, samples)
        label = f'{yaw:+g}'
        positions[label] = navigation['short_position']
        active[label] = fit['raw'] >= 3.
    result = transition_shape(samples, positions, active, study.FS, study.WIDTH)
    return dict(scope=__doc__, analysis_source_sha256=digest(Path(__file__)),
                stage47_summary_sha256=study.STAGE47_SUMMARY_SHA256,
                stage48_recovery_summary_sha256=STAGE48_SUMMARY_SHA256,
                timing=dict(sample_rate_hz=study.FS, fit_block_samples=study.WIDTH,
                            half_open_evidence=True, pre_steps=10,
                            post20_steps=2, post80_steps=8), **result)


def main():
    destination = study.OUTPUT/'transition_shape_summary.json'
    assert not destination.exists()
    destination.write_text(json.dumps(analyze(), indent=2, allow_nan=False),
                           encoding='utf-8')


if __name__ == '__main__':
    main()
