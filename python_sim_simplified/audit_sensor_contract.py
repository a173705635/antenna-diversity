"""Read-only legacy replay audit; output is a new report, not archive mutation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sensor_contract import adapt_legacy_rf
from spatial_spoofing import observations_from_records


def audit(archive):
    names = ('TL', 'TR', 'BL', 'BR')
    wavelength = 299792458/1575.42e6
    rows, hashes = [], {}
    for reference in sorted(archive.glob('rf_*/*_TL.json')):
        stem = reference.name[:-len('_TL.json')]
        records = {}
        for channel in names:
            path = reference.with_name(f'{stem}_{channel}.json')
            raw = path.read_bytes()
            hashes[str(path.relative_to(archive))] = hashlib.sha256(raw).hexdigest()
            records[channel] = json.loads(raw)
        for count in (2, 3, 4):
            ids = names[:count]
            result = adapt_legacy_rf({c: records[c] for c in ids}, ids,
                                    run_id=stem, wavelength_m=wavelength)
            t, x, v = observations_from_records([records[c] for c in ids], wavelength)
            np.testing.assert_array_equal(result['sample_time_s'], t)
            np.testing.assert_array_equal(result['carrier_m'], x)
            np.testing.assert_array_equal(result['valid'], v)
            rows.append(dict(record=stem, channels=list(ids), shape=list(x.shape),
                             valid_count=int(v.sum()), invalid_count=int((~v).sum()),
                             evidence_gaps=result['evidence_gaps']))
    for path, digest in hashes.items():
        if hashlib.sha256((archive/path).read_bytes()).hexdigest() != digest:
            raise AssertionError(f'input modified: {path}')
    return dict(scope='legacy replay only; not hardware synchronization or authenticity',
                source_record_count=len(hashes), layout_batches=len(rows),
                observation_count=sum(np.prod(r['shape']).item() for r in rows),
                inputs_unchanged=True, source_sha256=hashes, results=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Exclusive creation also prevents accidentally replacing an archive file.
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'results')}))
