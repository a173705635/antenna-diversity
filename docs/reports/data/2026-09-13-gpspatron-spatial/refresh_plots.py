"""Re-render existing RF results; no signal generation or changed decisions."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]/'python_sim_simplified'))
import audit_spatial_diversity_rf as rf

thresholds = json.loads((HERE/'model/thresholds.json').read_text(encoding='utf-8'))['thresholds']
for part in ('rf_a', 'rf_b'):
    directory = HERE/part
    rows = json.loads((directory/'summary.json').read_text(encoding='utf-8'))['rows']
    traces = {}
    for r in rows:
        key = (r['case'], r['seed'])
        if key not in traces:
            obs = np.load(directory/f'{key[0]}_{key[1]}_observations.npz')
            traces[key] = tuple(obs[name] for name in ('time_s', 'carrier_m', 'valid', 'tl_error_m'))
    rf.plot_rf(directory, rows, traces, thresholds)
    index = directory/'sha256.json'
    # Only the rendered PNG changes. All original numeric hashes remain fixed.
    hashes = json.loads(index.read_text(encoding='utf-8'))
    hashes['rf_results.png'] = hashlib.sha256((directory/'rf_results.png').read_bytes()).hexdigest()
    index.write_text(json.dumps(hashes, indent=2), encoding='utf-8')
source = Path(rf.__file__)
(HERE/'plot_code').mkdir(exist_ok=True)
(HERE/'plot_code'/source.name).write_bytes(source.read_bytes())
(HERE/'plot_refresh.json').write_text(json.dumps({
    'change': 'empty/unknown panel uses receiver time range and explicit unknown annotation',
    'numeric_results_changed': False, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'source_snapshot': 'plot_code/'+source.name,
    'figures_visually_checked': ['model/detection_comparison.png', 'model/boundaries.png',
                                 'rf_a/rf_results.png', 'rf_b/rf_results.png']}, indent=2), encoding='utf-8')
print('RF plots refreshed; numeric result hashes preserved')
