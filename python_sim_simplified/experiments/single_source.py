"""Latest single-emitter receiver: shared four-channel IQ, independent 2/3/4-antenna runs.

Research assumptions remain unchanged: normal startup, known nominal attitude,
ephemeris and coarse timing aid. This is not certified continuous navigation.
"""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import asdict
import hashlib
import json
import numpy as np

from core.single_source import frontend, navigate
from core.single_source_scene import Case, generate
from core.provenance import source_hashes, freeze_source
from core.serialization import serializable


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
    score = dict(opportunities=len(errors), solved=int(available.sum()),
                 good20=int((errors <= 20).sum()), error30=int((errors >= 30).sum()),
                 error50=int((errors >= 50).sum()), errors_m=errors,
                 initialization_time=init, attack_start_s=case['attack_start'],
                 initialized_before_attack=init is not None and init < case['attack_start'],
                 initialization_gap_s=None if init is None else case['attack_start'] - init,
                 longest_unavailable_s=float(runs.max() * .01))
    return observations, navigation, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--input', type=Path, help='Existing samples.npy, observations.npz and metadata.json folder')
    source.add_argument('--case', type=Path, help='JSON fields for core.single_source_scene.Case')
    parser.add_argument('--seed', type=int, default=113001)
    parser.add_argument('--duration', type=float, default=2.)
    parser.add_argument('--stars', type=int, default=24)
    parser.add_argument('--cn0', type=float, default=45.)
    parser.add_argument('--no-attack', action='store_true')
    parser.add_argument('--antennas', nargs='+', type=int, choices=(2, 3, 4), default=[3, 4])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--save-iq', action='store_true', help='Also save the generated input for exact replay')
    parser.add_argument('--freeze-source', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.freeze_source:
        freeze_source(args.output)

    if args.input is not None:
        meta = json.loads((args.input/'metadata.json').read_text(encoding='utf-8'))
        with np.load(args.input/'observations.npz') as data:
            record = dict(data)
        record['samples'] = np.load(args.input/'samples.npy', mmap_mode='r')
        # The manifest binds actual input bytes, not just the old metadata's claims.
        hashes = {}
        for name in ('samples.npy', 'observations.npz', 'metadata.json'):
            with (args.input/name).open('rb') as stream:
                hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        identity = dict(input=str(args.input.resolve()), input_sha256=hashes)
    else:
        parameters = (json.loads(args.case.read_text(encoding='utf-8')) if args.case is not None else
                      dict(duration=args.duration, stars=args.stars, cn0=args.cn0,
                           attack_start=.3, phase_deg=10., amplitude_db=1., velocity_east=2.))
        case = Case(**parameters)
        samples, nominal, codes, times, satellites, radii = generate(case, args.seed, not args.no_attack)
        record = dict(samples=samples, nominal=nominal, codes=np.array(codes), times=times,
                      satellites=satellites, coarse=np.round((radii + 75000) / 10000) * 10000)
        meta = dict(case=asdict(case), seed=args.seed, attacked=not args.no_attack)
        identity = dict(generated=True, **meta)
        if args.save_iq:
            inp = args.output/'input'
            inp.mkdir()
            np.save(inp/'samples.npy', samples)
            np.savez(inp/'observations.npz', **{k: v for k, v in record.items() if k != 'samples'})
            save(inp/'metadata.json', meta)
    save(args.output/'manifest.json', identity | dict(scope=__doc__, source_sha256=source_hashes(),
         case=meta['case'], antennas=args.antennas))
    summaries = []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    for antennas in args.antennas:
        z, nav, score = evaluate(record, list(range(antennas)), meta['case'])
        folder = args.output/f'{antennas}_antennas'
        folder.mkdir()
        np.savez_compressed(folder/'tracking.npz', **z)
        np.savez_compressed(folder/'navigation.npz', **{k: v for k, v in nav.items() if k != 'initialization_time'})
        save(folder/'score.json', score)
        summaries.append(dict(antennas=antennas, **score))
        times = z['available_time']
        axes[0].plot(times[times >= meta['case']['attack_start']], score['errors_m'], label=f'{antennas} antennas')
        axes[1].plot(times, nav['selected'].sum(axis=1), label=f'{antennas} antennas')
        print(antennas, {k: score[k] for k in ('opportunities', 'solved', 'good20', 'error30', 'error50')}, flush=True)
    save(args.output/'summary.json', summaries)
    axes[0].set(ylabel='Horizontal error (m)')
    axes[1].set(xlabel='Receiver time (s)', ylabel='Selected satellites')
    for axis in axes:
        axis.grid(alpha=.3)
        axis.legend()
    figure.savefig(args.output/'navigation.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    main()
