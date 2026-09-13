"""Compare archived external and in-house receiver outputs without branch adjustment."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
receiver=root/'receiver'
old=root.parent/'2026-09-13-sdcp-hardware-bias'
provenance=json.loads((root/'upstream_provenance.json').read_text())
for name,expected in provenance['files_sha256'].items():
    assert hashlib.sha256((root/'upstream/GNSSSimulator'/name).read_bytes()).hexdigest()==expected,name
external=json.loads((receiver/'summary.json').read_text())['rows']
internal={(r['seed'],r['case']):r for r in json.loads((old/'summary.json').read_text())['rows']}
rows=[]
for r in external:
    key=(r['seed'],r['case']);baseline=internal[key]
    current=json.loads((receiver/f'{key[0]}_{key[1]}.json').read_text())
    reference=json.loads((old/f'{key[0]}_{key[1]}.json').read_text())
    maxima={}
    for metric in ('epoch_start_samples','carrier_nco_phase_rad','carrier_nco_frequency_hz','pll_error_rad','phase_lock_metric'):
        maxima[metric]=max(float(np.max(abs(np.array(a[metric])-np.array(b[metric]))))
                           for a,b in zip(current['channels'],reference['channels']))
    rows.append({'seed':key[0],'case':key[1],'max_tracking_array_difference':maxima,
        'residual_mean_difference_mm':r['geometry_residual']['mean_mm']-baseline['geometry_residual']['mean_mm'],
        'residual_std_difference_mm':r['geometry_residual']['std_mm']-baseline['geometry_residual']['std_mm']})
log=(root/'upstream_tests.log').read_text(encoding='utf-8',errors='replace')
tests=re.findall(r'^([^\n|]+)\|\s+(\d+)\s+(\d+)\s+[\d.]+s\s*$',log,re.M)
assert len(tests)==12 and all(a==b for _,a,b in tests)
result={'upstream_files_verified':len(provenance['files_sha256']),
        'upstream_test_passes':sum(int(a) for _,a,_ in tests),'rows':rows}
(root/'cross_source_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with np.load(receiver/'traces.npz') as traces,np.load(old/'traces.npz') as original:
    fig,axs=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    for ax,seed in zip(axs,(9900,9902)):
        for label,color in [('zero','tab:blue'),('combined','tab:orange')]:
            name=f'{seed}_{label}_residual_m'
            ax.plot(traces['time_s'],original[name]*1000,color=color,lw=2,alpha=.5,label=f'{label}: in-house source')
            ax.plot(traces['time_s'],traces[name]*1000,color=color,lw=.7,ls='--',label=f'{label}: external source')
        ax.set(title=f'Seed pair {seed}/{seed+1}; raw accumulated phase, no modulo',
               xlabel='Time (s)',ylabel='SDCP - geometry (mm)')
        ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=2)
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
print(json.dumps({'upstream_files':result['upstream_files_verified'],
    'upstream_test_passes':result['upstream_test_passes'],'receiver_rows':len(rows),
    'max_nco_phase_difference_rad':max(r['max_tracking_array_difference']['carrier_nco_phase_rad'] for r in rows)}))
