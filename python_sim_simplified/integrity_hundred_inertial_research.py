"""Run one actual source revision and paired experiment, iterations 081--100.

Each command mutates the receiver source, executes old/new against byte-identical
observations, saves both sources and all epoch traces. Seeds are replications,
never counted as algorithm iterations. Run in increasing order exactly once.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from integrity_hundred_inertial_model import NAMES, make_case, score

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'integrity_hundred_inertial.py'
RESULTS = ROOT/'results'/'research-hundred'

# Each replacement changes a model term or an inference operation, not a tuning threshold.
EDITS = {
81: ('Local inertial innovation admission', 'step',
     'A fixed local innovation gate can reject a 200 m jump but may follow smooth common drift.',
     'accepted = available',
     'accepted = available and np.linalg.norm(g-p) <= 20.'),
82: ('Independent inertial backbone', 'creep',
     'Keeping the propagated anchor independent prevents a passing GNSS fix from moving the next admission center.',
     '            p = g.copy()\n            output = p.copy()',
     '            output = g.copy()'),
83: ('Bounded acceleration reachable set', 'normal',
     'An acceleration support bound exposes quadratic inertial error growth.',
     'a_bound = np.zeros(2)', 'a_bound = o[\'accel_bound\'].copy()'),
84: ('Initial velocity support propagation', 'velocity',
     'Initial velocity uncertainty must contribute linearly to position support.',
     'rv = np.zeros(2)', 'rv = o[\'initial_velocity_radius\'].copy()'),
85: ('Gravity leakage support', 'tilt',
     'Small attitude errors produce gravity leakage that accelerometer bias bounds do not cover.',
     "a_bound = o['accel_bound'].copy()",
     "a_bound = o['accel_bound'].copy() + 9.81*o['tilt_bound']"),
86: ('Gyro driven attitude growth', 'gyro',
     'Uncorrected gyro error creates time-growing gravity leakage and cubic position uncertainty.',
     "9.81*o['tilt_bound']", "9.81*(o['tilt_bound']+t[k]*o['gyro_bound'])"),
87: ('Preserve anchor across a GNSS outage', 'outage',
     'A reacquired coherent fix must not erase the pre-outage independent anchor.',
     "        if available and not previous_available:\n            p = g.copy()\n            rp = o['initial_position_radius'].copy()",
     '        # GNSS arc changes carry no absolute trust.'),
88: ('Unobservable absolute initialization', 'unknown',
     'Unknown absolute position requires an unbounded translation set, even with coherent GNSS.',
     "rp = o['initial_position_radius'].copy()",
     "rp = o['initial_position_radius'].copy() if o['initial_position_trusted'] else np.full(2, np.inf)"),
89: ('Output centered protection level', 'tilt',
     'The radius must cover the emitted GNSS point relative to the independent set; exceeding 20 m protects output.',
     '        bound = np.linalg.norm(rp)\n        is_normal = True',
     '        bound = np.linalg.norm(rp+np.abs(output-p))\n        is_normal = bool(bound <= 20. and feasible)'),
90: ('Independent position certificate intersection', 'auth',
     'An explicit independent absolute observation can reduce the reachable set without trusting RF agreement.',
     '        # INDEPENDENT_POSITION',
     "        if k in o['absolute_certificates']:\n            z, r = o['absolute_certificates'][k]\n            lo = np.maximum(p-rp, z-r); hi = np.minimum(p+rp, z+r)\n            feasible = feasible and bool(np.all(lo <= hi))\n            if feasible:\n                p = (lo+hi)/2; rp = (hi-lo)/2"),
91: ('Independent velocity certificate', 'zero_velocity',
     'External velocity or zero-velocity evidence bounds subsequent drift without assuming stillness from GNSS.',
     '        # INDEPENDENT_VELOCITY',
     "        if k in o['velocity_certificates']:\n            v, rv = (x.copy() for x in o['velocity_certificates'][k])"),
92: ('Explicit stationary bias calibration', 'bias_calibration',
     'An independent bounded bias estimate can reduce acceleration uncertainty while retaining sample noise.',
     '        # ACCEL_CORRECTIONS',
     "        if o['bias_certificate'] is not None:\n            bias, r = o['bias_certificate']\n            a -= bias\n            a_bound += r+.002-o['accel_bound']\n        # ACCEL_CORRECTIONS"),
93: ('Initial certificate time transport', 'timing',
     'Delayed position certificates require velocity transport plus delay and velocity uncertainty.',
     '    # INITIAL_ALIGNMENT',
     "    p += v*o['time_delay']\n    rp += rv*abs(o['time_delay']) + (np.abs(v)+rv)*o['time_delay_radius']\n    # INITIAL_ALIGNMENT"),
94: ('Antenna/body lever arm frame', 'lever',
     'Position comparisons must share an explicit frame and retain lever uncertainty.',
     "        g = o['gnss'][k]", "        g = o['gnss'][k]-o['lever_arm']"),
95: ('Certified vehicle nonholonomic velocity', 'mounted',
     'A lateral velocity constraint helps a mounted straight vehicle but must not be applied to a handheld phone.',
     '        # NONHOLONOMIC',
     "        if o['nonholonomic_certificate']:\n            v[1] = 0.; rv[1] = .01"),
96: ('Independent attitude calibration', 'calibrated',
     'A certified tilt estimate removes gravity leakage and replaces its support by calibration uncertainty.',
     '        # ACCEL_CORRECTIONS',
     "        if o['attitude_certificate'] is not None:\n            tilt, r = o['attitude_certificate']\n            a -= 9.81*tilt\n            a_bound += 9.81*(r-o['tilt_bound'])\n        # ACCEL_CORRECTIONS"),
97: ('Union of ambiguous absolute hypotheses', 'ambiguous',
     'A 200 m ambiguity cannot be collapsed by choosing the GNSS-matching initial hypothesis.',
     '    # HYPOTHESES',
     "    if o['hypotheses']:\n        runs = [receive(dict(o, initial_position=p, hypotheses=[])) for p in o['hypotheses']]\n        pos = np.array([r['position'] for r in runs])\n        rad = np.array([r['bound'] for r in runs])\n        viable = np.ones_like(rad, dtype=bool)\n        lo = np.min(np.where(viable[:, :, None], pos-rad[:, :, None], np.inf), axis=0)\n        hi = np.max(np.where(viable[:, :, None], pos+rad[:, :, None], -np.inf), axis=0)\n        p = (lo+hi)/2; radius = np.linalg.norm((hi-lo)/2, axis=1)\n        return dict(position=p, bound=radius, normal=radius <= 20.,\n                    accepted=np.any([r['accepted'] for r in runs], axis=0),\n                    feasible=np.any(viable, axis=0))\n    # HYPOTHESES"),
98: ('Independent hypothesis elimination', 'resolved',
     'Only incompatibility with an independent certificate can remove a location hypothesis.',
     'viable = np.ones_like(rad, dtype=bool)',
     "viable = np.array([r['feasible'] for r in runs])"),
99: ('Bias drift support', 'drift',
     'A fixed calibrated bias assumption cannot cover bias drift; its integral must enter the support.',
     "t[k]*o['gyro_bound'])", "t[k]*o['gyro_bound']) + t[k]*o['bias_drift_bound']"),
100: ('Minimax center for protected output', 'normal',
      'Emitting the reachable-set center minimizes the worst-case box distance; GNSS agreement remains diagnostic only.',
      '        if accepted:\n            output = g.copy()',
      '        # Keep the minimax independent center; accepted is a diagnostic only.'),
}


def load_receiver(source):
    namespace = {}
    exec(compile(source, str(SOURCE), 'exec'), namespace)
    return namespace['receive']


def aggregate(metrics):
    return dict(normal_output_rate=float(np.mean([x['normal_output_rate'] for x in metrics.values()])),
                all_epochs_le20_cases=sum(x['all_epochs_le20'] for x in metrics.values()),
                normal_error_ge30=sum(x['normal_error_ge30'] for x in metrics.values()),
                normal_error_ge50=sum(x['normal_error_ge50'] for x in metrics.values()),
                bound_violations=sum(x['bound_violations'] for x in metrics.values()))


def run(iteration):
    title, target, hypothesis, old, new = EDITS[iteration]
    folder = RESULTS/f'iteration{iteration:03d}'
    folder.mkdir()
    before = SOURCE.read_text(encoding='utf-8')
    assert before.count(old) == 1, (iteration, old, before.count(old))
    proposal = dict(iteration=iteration, title=title, hypothesis=hypothesis, target=target,
                    same_input_seed=810, scope='declared bounded 2-D observation domain, not RF',
                    independent_inputs='Explicit p/v initialization and optional certificates; unavailable inputs remain unavailable.',
                    change=dict(old=old, new=new))
    (folder/'proposal.json').write_text(json.dumps(proposal, indent=2), encoding='utf-8')
    after = before.replace(old, new)
    if iteration == 94:
        after = after.replace('rp+np.abs(output-p)', "rp+o['lever_radius']+np.abs(output-p)")
        after = after.replace('        # OUTPUT_FRAME', "        output = output+o['lever_arm']")
    SOURCE.write_text(after, encoding='utf-8')
    (folder/'source_before.py').write_text(before, encoding='utf-8')
    (folder/'source_snapshot.py').write_text(after, encoding='utf-8')
    (folder/'model_snapshot.py').write_text((ROOT/'integrity_hundred_inertial_model.py').read_text(encoding='utf-8'), encoding='utf-8')
    old_receiver = load_receiver(before); new_receiver = load_receiver(after)
    metrics = {'before': {}, 'after': {}}
    traces = {}
    for name in NAMES:
        observations, truth = make_case(name, 810)
        for label, receiver in [('before', old_receiver), ('after', new_receiver)]:
            result = receiver(observations)
            metrics[label][name] = score(result, truth)
            for key in ['position', 'bound', 'normal', 'accepted', 'feasible']:
                traces[f'{name}_{label}_{key}'] = result[key]
        traces[f'{name}_truth'] = truth
    np.savez_compressed(folder/'traces.npz', **traces)
    summary = {label: aggregate(rows) for label, rows in metrics.items()}
    results = dict(iteration=iteration, metrics=metrics, aggregate=summary,
                   source_sha256=hashlib.sha256(after.encode()).hexdigest())
    (folder/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    a = metrics['before'][target]; b = metrics['after'][target]
    conclusion = dict(iteration=iteration, target=target, before=a, after=b,
                      observed_delta={key: summary['after'][key]-summary['before'][key] for key in summary['after']},
                      next_step='Pending researcher inspection of paired metrics; see decision below.')
    (folder/'conclusion.json').write_text(json.dumps(conclusion, indent=2), encoding='utf-8')
    print(json.dumps(dict(iteration=iteration, target=target, aggregate=summary,
                         target_before=a, target_after=b), indent=2))


def replay(iteration):
    folder = RESULTS/f'iteration{iteration:03d}'
    assert (folder/'model_snapshot.py').read_text(encoding='utf-8') == (ROOT/'integrity_hundred_inertial_model.py').read_text(encoding='utf-8')
    recorded = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    for label, filename in [('before', 'source_before.py'), ('after', 'source_snapshot.py')]:
        receiver = load_receiver((folder/filename).read_text(encoding='utf-8'))
        for name in NAMES:
            o, truth = make_case(name, 810)
            assert score(receiver(o), truth) == recorded['metrics'][label][name], (iteration, label, name)
    print(f'iteration{iteration:03d}: recorded same-input metrics reproduced')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('iteration', type=int)
    parser.add_argument('--after-review')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if args.after_review:
        previous = RESULTS/f'iteration{args.iteration-1:03d}'/'conclusion.json'
        note = json.loads(previous.read_text(encoding='utf-8'))
        note['next_step'] = args.after_review
        previous.write_text(json.dumps(note, indent=2), encoding='utf-8')
    if args.replay:
        replay(args.iteration)
    else:
        run(args.iteration)
