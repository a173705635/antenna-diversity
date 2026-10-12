"""Latest single-emitter receiver: shared four-channel IQ, independent 2/3/4-antenna runs.

Research assumptions remain unchanged: normal startup, known nominal attitude,
ephemeris and coarse timing aid. This is not certified continuous navigation.
"""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import asdict
import hashlib
import json
import numpy as np

from core.spoofing.single_source import frontend, navigate
from core.spoofing.single_source_scene import Case, generate
from core.provenance import source_hashes, freeze_source
from core.serialization import serializable
from experiments.paths import SPOOFING_RESULTS as RESULTS, reset_output

OUTPUT = RESULTS/'single_source'
INPUT = None
CASE = None
SEED = 113001
DURATION_S = 2.
SATELLITES = 24
CN0_DB_HZ = 45.
ATTACKED = True
ANTENNAS = [3, 4]
SAVE_IQ = False
FREEZE_SOURCE = False


def save(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False), encoding='utf-8')


def evaluate(record, subset, case):
    observations = frontend(record, subset)
    navigation = navigate(observations)
    times = observations['available_time']
    eligible = times >= case['attack_start']
    truth = np.zeros_like(navigation['position'])
    truth[:, 0] = case['velocity_east'] * times
    errors = np.linalg.norm((navigation['position'] - truth)[:, :2], axis=1)[eligible]
    available = np.isfinite(errors)
    runs = np.diff(np.flatnonzero(np.r_[True, available, True])) - 1
    init = navigation['initialization_time']
    score = dict(assessment_status='evaluated' if len(errors) else 'not_evaluated_no_post_attack_observations',
                 initialization_status='initialized' if init is not None else 'not_initialized_in_record',
                 opportunities=len(errors), solved=int(available.sum()),
                 good20=int((errors <= 20).sum()), error30=int((errors >= 30).sum()),
                 error50=int((errors >= 50).sum()), errors_m=errors,
                 initialization_time=init, attack_start_s=case['attack_start'],
                 initialized_before_attack=init is not None and init < case['attack_start'],
                 initialization_gap_s=None if init is None else case['attack_start'] - init,
                 longest_unavailable_s=float(runs.max() * .01) if len(errors) else None)
    return observations, navigation, score


def main():
    if INPUT is None:
        parameters = (json.loads(CASE.read_text(encoding='utf-8')) if CASE is not None else
                      dict(duration=DURATION_S, stars=SATELLITES, cn0=CN0_DB_HZ,
                           attack_start=.3, phase_deg=10., amplitude_db=1., velocity_east=2.))
        case = Case(**parameters)
        if case.duration <= .17:
            raise ValueError('Single-source evaluation needs more than 170 ms for its 150 ms start and 20 ms processing window.')
    output = reset_output(OUTPUT)
    if FREEZE_SOURCE:
        freeze_source(output)

    if INPUT is not None:
        meta = json.loads((INPUT/'metadata.json').read_text(encoding='utf-8'))
        with np.load(INPUT/'observations.npz') as data:
            record = dict(data)
        record['samples'] = np.load(INPUT/'samples.npy', mmap_mode='r')
        # The manifest binds actual input bytes, not just the old metadata's claims.
        hashes = {}
        for name in ('samples.npy', 'observations.npz', 'metadata.json'):
            with (INPUT/name).open('rb') as stream:
                hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        identity = dict(input=str(INPUT.resolve()), input_sha256=hashes)
    else:
        samples, nominal, codes, times, satellites, radii = generate(case, SEED, ATTACKED)
        record = dict(samples=samples, nominal=nominal, codes=np.array(codes), times=times,
                      satellites=satellites, coarse=np.round((radii + 75000) / 10000) * 10000)
        meta = dict(case=asdict(case), seed=SEED, attacked=ATTACKED)
        identity = dict(generated=True, **meta)
        if SAVE_IQ:
            inp = output/'input'
            inp.mkdir()
            np.save(inp/'samples.npy', samples)
            np.savez(inp/'observations.npz', **{k: v for k, v in record.items() if k != 'samples'})
            save(inp/'metadata.json', meta)
    save(output/'manifest.json', identity | dict(scope=__doc__, source_sha256=source_hashes(),
         case=meta['case'], antennas=ANTENNAS))
    summaries = []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    for antennas in ANTENNAS:
        z, nav, score = evaluate(record, list(range(antennas)), meta['case'])
        folder = output/f'{antennas}_antennas'
        folder.mkdir()
        np.savez_compressed(folder/'tracking.npz', **z)
        np.savez_compressed(folder/'navigation.npz', **{k: v for k, v in nav.items() if k != 'initialization_time'})
        save(folder/'score.json', score)
        summaries.append(dict(antennas=antennas, **score))
        times = z['available_time']
        axes[0].plot(times[times >= meta['case']['attack_start']], score['errors_m'], label=f'{antennas} antennas')
        axes[1].plot(times, nav['selected'].sum(axis=1), label=f'{antennas} antennas')
        print(antennas, {k: score[k] for k in ('opportunities', 'solved', 'good20', 'error30', 'error50')}, flush=True)
    save(output/'summary.json', summaries)
    axes[0].set(ylabel='Horizontal error (m)')
    axes[1].set(xlabel='Receiver time (s)', ylabel='Selected satellites')
    for axis in axes:
        axis.grid(alpha=.3)
        axis.legend()
    figure.savefig(output/'navigation.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    main()
