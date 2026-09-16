"""Reproduce frozen model counterexamples; validation seeds are not research rounds."""
import numpy as np
from cold_next import calibrate,complex_calibrate
from presets import mate70_phase_centers
from signal_recovery import sky,steering
from cold_next_research import ROOT,save


def main():
    a=steering(sky(48020,8),np.array(list(mate70_phase_centers().values())),0.)
    rows=[];parents=[]
    for seed in range(48020,48024):
        rng=np.random.default_rng(seed)
        z=(rng.normal(size=(4,8,100))+1j*rng.normal(size=(4,8,100)))/np.sqrt(2)
        _,info=complex_calibrate(z,a);rows.append(dict(seed=seed,diagnostics=info))
        for revision in (125,134,136,137,138,139,140):
            _,info=calibrate(z,a,revision);parents.append(dict(seed=seed,revision=revision,diagnostics=info))
    common=dict(layer='prompt-domain IID complex noise; no RF receiver/PVT',after_freeze=True)
    save(ROOT/'cold_noise_control.json',dict(**common,rows=rows))
    save(ROOT/'cold_noise_parent_control.json',dict(**common,rows=parents))
    print('150 false support',sum(r['diagnostics']['usable'] for r in rows),'/',len(rows))


if __name__=='__main__':main()
