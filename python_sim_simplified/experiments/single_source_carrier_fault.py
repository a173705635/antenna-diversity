"""Inject two fixed PRN-14 carrier faults into saved four-second tracking.

Carrier feeds both smooth_history and phase_prediction. This is a saved-
observation fault study, not an RF replay or a phase-only perturbation.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.mechanism_smoothing import smooth_history
from core.provenance import source_hashes
from core.recovery_geometry import WAVELENGTH
from core.serialization import serializable
from core.single_source import navigate
from experiments.paths import RESULTS
from experiments.single_source_code_blend import (
    OUTPUT as STAGE19, assert_saved_navigation, published_metrics,
)
from experiments.single_source_coherent_float import preflight_sources
from experiments.single_source_coherent_navigation import (
    quality_support_counts, score_mode,
)
from experiments.single_source_coherent_transfer import INPUT, OUTPUT as STAGE17
from experiments.single_source_suppression_study import OUTPUT as STUDY_OUTPUT, digest


OUTPUT = RESULTS / STUDY_OUTPUT / 'carrier_fault'
FAULT_TIME_S = 2.
FAULT_PRN = 14
DRIFT_M_PER_S = .5
BLENDS = (('.1', .1), ('0', 0.))
FAULTS = ('cycle', 'drift')


def inject_carrier_fault(observations, kind):
    """Change only PRN 14, arc zero, at observation time >= 2 s."""
    star = FAULT_PRN - 1
    assert star in observations['startup_prns']
    time = observations['time']
    selected = ((time >= FAULT_TIME_S) & (observations['arc'][:, star] == 0)
                & np.isfinite(observations['carrier'][:, star]))
    delta = np.zeros(len(time))
    if kind == 'cycle':
        delta[selected] = WAVELENGTH
    else:
        delta[selected] = DRIFT_M_PER_S * (time[selected] - FAULT_TIME_S)
    carrier = observations['carrier'].copy()
    carrier[:, star] += delta
    return {**observations, 'carrier': carrier}, delta


def assert_prefault_equal(baseline, fault, before):
    """Every output and diagnostic before the target observation is unchanged."""
    for key, value in baseline.items():
        if key == 'diagnostics':
            for diagnostic, data in value.items():
                np.testing.assert_array_equal(data[before],
                                              fault[key][diagnostic][before],
                                              err_msg=diagnostic)
        elif np.ndim(value) == 0:
            np.testing.assert_array_equal(value, fault[key], err_msg=key)
        else:
            np.testing.assert_array_equal(value[before], fault[key][before],
                                          err_msg=key)


def smoothing_ranges(observations):
    finite = np.isfinite(observations['ranges'])
    valid = (finite & (observations['lock'] >= .5)
             & (observations['cn0'] >= 25.) & (observations['gain'] >= .05)
             & (observations['support'] >= 3.))
    return smooth_history(observations, valid)[0], valid


def compare_fault(baseline, fault, observations, metadata, fault_window):
    time = observations['time']
    base_published = np.isfinite(baseline['position']).all(axis=1)
    fault_published = np.isfinite(fault['position']).all(axis=1)
    common = fault_window & base_published & fault_published
    truth = np.column_stack((metadata['case']['velocity_east'] * time,
                             np.zeros(len(time))))
    base_error = np.linalg.norm(baseline['position'][:, :2] - truth, axis=1)
    fault_error = np.linalg.norm(fault['position'][:, :2] - truth, axis=1)
    displacement = np.linalg.norm(fault['position'][:, :2]
                                  - baseline['position'][:, :2], axis=1)
    return dict(fault_window_opportunities=int(fault_window.sum()),
                common_published=int(common.sum()),
                newly_published=int((fault_window & ~base_published & fault_published).sum()),
                lost_publication=int((fault_window & base_published & ~fault_published).sum()),
                baseline_common=published_metrics(base_error, common),
                fault_common=published_metrics(fault_error, common),
                position_displacement_m=published_metrics(displacement, common)), fault_error


def main():
    preflight_sources()
    stage17 = json.loads((STAGE17 / 'summary.json').read_text(encoding='utf-8'))
    stage19 = json.loads((STAGE19 / 'summary.json').read_text(encoding='utf-8'))
    assert stage19['manifest_sha256'] == digest(STAGE19 / 'manifest.json')
    assert stage19['cases']['transfer_weak']['result_sha256'] == digest(
        STAGE19 / 'transfer_weak.npz')
    for name in ('tracking.npz', 'support.npz'):
        assert stage17['output_sha256'][name] == digest(STAGE17 / name)
    metadata = json.loads((INPUT / 'metadata.json').read_text(encoding='utf-8'))
    sources = source_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources,
        stage17_summary_sha256=digest(STAGE17 / 'summary.json'),
        stage19_summary_sha256=digest(STAGE19 / 'summary.json'),
        tracking_sha256=digest(STAGE17 / 'tracking.npz'),
        support_sha256=digest(STAGE17 / 'support.npz'),
        metadata_sha256=digest(INPUT / 'metadata.json'),
        fault_prn=FAULT_PRN, fault_arc=0, fault_observation_time_s=FAULT_TIME_S,
        cycle_step_m=WAVELENGTH, drift_m_per_s=DRIFT_M_PER_S,
        blend_values={name: value for name, value in BLENDS})),
        indent=2, allow_nan=False), encoding='utf-8')

    with np.load(STAGE17 / 'tracking.npz') as saved:
        tracking = dict(saved)
    with np.load(STAGE17 / 'support.npz') as saved:
        support = saved['coherent_support']
    observations = {**tracking, 'support': support}
    star = FAULT_PRN - 1
    assert star in tracking['startup_prns']
    counts = quality_support_counts(tracking, support)
    before = tracking['time'] < FAULT_TIME_S
    baseline = {label: navigate(observations, diagnostics=True, code_blend=blend)
                for label, blend in BLENDS}
    with np.load(STAGE19 / 'transfer_weak.npz') as saved:
        for label in baseline:
            assert_saved_navigation(baseline[label], saved, prefix=f'{label}_')
    baseline_ranges, prehistory_valid = smoothing_ranges(observations)

    reports = {}
    plots = {}
    for kind in FAULTS:
        changed, delta = inject_carrier_fault(observations, kind)
        actual_delta = changed['carrier'][:, star] - observations['carrier'][:, star]
        fault_ranges, fault_prehistory_valid = smoothing_ranges(changed)
        np.testing.assert_array_equal(prehistory_valid, fault_prehistory_valid)
        np.testing.assert_array_equal(baseline_ranges[before], fault_ranges[before])
        fault = {label: navigate(changed, diagnostics=True, code_blend=blend)
                 for label, blend in BLENDS}
        arrays = dict(observation_time=tracking['time'],
                      available_time=tracking['available_time'],
                      prn14_requested_carrier_delta_m=delta,
                      prn14_actual_carrier_delta_m=actual_delta,
                      prn14_prehistory_valid=prehistory_valid[:, star],
                      smooth_range_delta_m=fault_ranges - baseline_ranges)
        mask = ((tracking['time'] >= FAULT_TIME_S) & (tracking['arc'][:, star] == 0)
                & np.isfinite(tracking['carrier'][:, star]))
        modes = {}
        errors = {}
        for label in fault:
            assert_prefault_equal(baseline[label], fault[label], before)
            comparison, error = compare_fault(baseline[label], fault[label],
                                              tracking, metadata, mask)
            errors[label] = error
            modes[label] = dict(
                baseline=score_mode(baseline[label], tracking, metadata['case'],
                                    counts, support),
                fault=score_mode(fault[label], tracking, metadata['case'],
                                 counts, support), **comparison,
                prn14_selected_after_fault=int((mask & fault[label]['selected'][:, star]).sum()))
            arrays[f'{label}_prn14_selected'] = fault[label]['selected'][:, star]
            arrays[f'{label}_horizontal_error_m'] = error
            for key, value in fault[label].items():
                if key == 'diagnostics':
                    for diagnostic, data in value.items():
                        arrays[f'{label}_diagnostic_{diagnostic}'] = data
                else:
                    arrays[f'{label}_{key}'] = np.nan if value is None else value
        path = OUTPUT / f'{kind}.npz'
        np.savez_compressed(path, **arrays)
        reports[kind] = dict(result_sha256=digest(path),
                             first_target_available_time_s=float(tracking['available_time'][mask][0]),
                             prn14_target_epochs=int(mask.sum()),
                             prn14_requested_nonzero_epochs=int(np.count_nonzero(delta)),
                             prn14_actual_changed_epochs=int(np.count_nonzero(actual_delta)),
                             first_actual_changed_available_time_s=float(
                                 tracking['available_time'][actual_delta != 0][0]),
                             modes=modes)
        plots[kind] = errors
        print(kind, 'done', flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             constrained_layout=True)
    for axis, kind in zip(axes, FAULTS):
        for label, error in plots[kind].items():
            axis.plot(tracking['available_time'], error,
                      label=f'{kind}, code_blend={label}')
        axis.axhline(20., color='gray', linestyle=':', linewidth=1)
        axis.set(ylabel='Published error (m)')
        axis.grid(alpha=.3)
        axis.legend()
    axes[-1].set_xlabel('Output available time (s)')
    fig.savefig(OUTPUT / 'overview.png', dpi=150)
    plt.close(fig)
    assert source_hashes() == sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources,
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        figure_sha256=digest(OUTPUT / 'overview.png'), cases=reports)),
        indent=2, allow_nan=False), encoding='utf-8')
    return reports


if __name__ == '__main__':
    main()
