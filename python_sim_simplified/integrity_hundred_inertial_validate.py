"""Frozen new-seed and model-mismatch checks. These are not research rounds."""
import hashlib
import json
from pathlib import Path
import numpy as np
from integrity_hundred_inertial import receive
from integrity_hundred_inertial_model import NAMES, make_case, score
from integrity_hundred_inertial_research import aggregate

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'results'/'research-hundred'


def main():
    digest = hashlib.sha256((ROOT/'integrity_hundred_inertial.py').read_bytes()).hexdigest()
    rows = {}; mismatch = {}; traces = {}
    for seed in [12081, 14083, 16087, 18089]:
        for name in NAMES:
            o, truth = make_case(name, seed)
            result = receive(o)
            rows[f'{name}_{seed}'] = score(result, truth)
            if seed == 12081:
                for key, value in result.items():
                    traces[f'{name}_{key}'] = value
        o, truth = make_case('normal', seed)
        o['gnss'][:] = np.nan
        rows[f'no_gnss_{seed}'] = score(receive(o), truth)
        o, truth = make_case('unknown', seed)
        o['gnss'][:] = np.nan
        rows[f'unknown_no_gnss_translated_{seed}'] = score(receive(o), truth+np.array([200., 0.]))
    # These two cases deliberately violate the declared physical certificates.
    o, truth = make_case('normal', 20011)
    o['acceleration'] += np.array([.2, 0.])
    mismatch['undeclared_acceleration_error'] = score(receive(o), truth)
    o, truth = make_case('handheld', 20011)
    o['nonholonomic_certificate'] = True
    mismatch['false_vehicle_certificate_for_handheld'] = score(receive(o), truth)
    result = dict(source_sha256=digest, seed_use='validation only, after iteration100 freeze',
                  metrics=rows, aggregate=aggregate(rows), out_of_domain=mismatch,
                  validated_domain='2-D piecewise constant acceleration, externally bounded errors; no RF simulation',
                  normal_label='bound <= 20 m; not satellite authentication',
                  continuous_positioning_success='Every epoch must be normal and <=20 m; missing/protected epochs fail.')
    (DEST/'integrity_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(DEST/'integrity_validation_traces.npz', **traces)
    assert hashlib.sha256((ROOT/'integrity_hundred_inertial.py').read_bytes()).hexdigest() == digest
    print(json.dumps(dict(aggregate=result['aggregate'], out_of_domain=mismatch), indent=2))


if __name__ == '__main__':
    main()
