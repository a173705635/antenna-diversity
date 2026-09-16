"""Independent frozen holdout and archive replay; no holdout tuning."""
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from research_hundred import load
from research_warm_array import ROOT, SUBSETS, generate, score, save
from validate_research_hundred import source_closure


def freeze():
    save(ROOT/'freeze.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
         selected=25, seeds=[112001,112013,112037],
         kinds=['normal','attack','weak','near','absent','drift'], stress_epochs=64,
         sources=source_closure(['warm_array.py','research_warm_array.py','validate_warm_array.py']),
         domain='independent coherent sampled-code snapshots; not continuous RF tracking',
         initialization='three observed PVT solutions and horizontal spread <=20m; no authentication claim'))


def run():
    freeze_record = json.loads((ROOT/'freeze.json').read_text(encoding='utf-8'))
    for name, digest in freeze_record['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    models = {'baseline': load(ROOT/'baseline.py'), 'candidate': load(ROOT/'iteration025/warm_array.py')}
    rows = []
    for seed in freeze_record['seeds']:
        for kind in freeze_record['kinds']:
            for epochs in (8, freeze_record['stress_epochs']):
                record, truth = generate(seed, kind, epochs)
                for subset, channels in SUBSETS.items():
                    scores = {name: score(model.process(record, channels), truth) for name, model in models.items()}
                    rows.append(dict(seed=seed, kind=kind, epochs=epochs, subset=subset, scores=scores,
                                     input_sha256=hashlib.sha256(record['samples'].tobytes()).hexdigest()))
    save(ROOT/'holdout.json', rows)
    summary = {}
    for epochs in (8, freeze_record['stress_epochs']):
        for subset in SUBSETS:
            selected = [r for r in rows if r['epochs']==epochs and r['subset']==subset]
            summary[f'{epochs}_{subset}'] = {}
            for version in models:
                scores = [r['scores'][version] for r in selected]
                totals = {key: sum(r[key] for r in scores) for key in ('opportunities','solved','good20','error30','error50')}
                errors = [e for r in scores for e in r['errors_m'] if e is not None]
                totals.update(initialization_failed=sum(not r['initialized'] for r in scores),
                              conditional_p95_m=float(np.percentile(errors, 95)),
                              unavailable=totals['opportunities']-totals['solved'])
                summary[f'{epochs}_{subset}'][version] = totals
    save(ROOT/'summary.json', summary)
    print(json.dumps(summary), flush=True)


def replay():
    rows = []; semantic = set()
    for number in range(1,26):
        folder = ROOT/f'iteration{number:03d}'
        proposal = json.loads((folder/'proposal.json').read_text(encoding='utf-8'))
        result = json.loads((folder/'results.json').read_text(encoding='utf-8'))
        source = folder/'warm_array.py'
        tree = ast.dump(ast.parse(source.read_text(encoding='utf-8')), include_attributes=False)
        assert tree not in semantic
        semantic.add(tree)
        parent = ROOT/'baseline.py' if proposal['parent']==0 else ROOT/f"iteration{proposal['parent']:03d}/warm_array.py"
        models = dict(before=load(parent), after=load(source))
        for row in result['rows']:
            record, truth = generate(111001, row['kind'])
            assert hashlib.sha256(record['samples'].tobytes()).hexdigest()==row['input_sha256']
            for version, module in models.items():
                actual = module.process(record, SUBSETS[row['subset']])
                assert score(actual, truth)==row[version], (number,row['kind'],row['subset'],version)
                with np.load(folder/f"{row['kind']}_{row['subset']}.npz") as z:
                    np.testing.assert_array_equal(actual['position'], z[version])
        rows.append(dict(number=number, passed=True, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    save(ROOT/'replay.json', dict(rounds=rows, unique_executable_ast=len(semantic), passed=True))
    print('25 paired source versions replayed exactly', flush=True)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
