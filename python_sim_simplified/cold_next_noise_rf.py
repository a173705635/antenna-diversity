"""A separate measured-RF absence control; not inferred from prompt-model tests."""
from unittest.mock import patch
import numpy as np
import cold_next_rf as receiver
from cold_next_research import ROOT,save
from navigation_hundred import candidate
from audit_recovery_two_hundred import score


def main():
    original=receiver.generate;seed=62120;case=receiver.Case(duration=.4,cn0=35.,attack_start=0.)
    def only_noise(*args):
        x,*rest=original(*args)
        rng=np.random.default_rng(seed)
        x=np.sqrt(receiver.FS/10**(case.cn0/10)/2)*(rng.normal(size=x.shape)+1j*rng.normal(size=x.shape))
        return (x,*rest)
    folder=ROOT/'cold_noise_rf';folder.mkdir(exist_ok=True);rows=[]
    for revision in (125,150):
        with patch.object(receiver,'generate',only_noise):
            data,metadata=receiver.experiment(case,seed,False,['raw','calibrated'],extended=True,revision=revision)
        nav=candidate(data)
        if not data['joint_usable']:nav[:]=np.nan
        np.savez_compressed(folder/f'{revision}.npz',**data)
        rows.append(dict(revision=revision,startup_prns=data['startup_prns'].tolist(),
            finite_raw_ranges=int(np.isfinite(data['raw_ranges']).sum()),
            finite_calibrated_ranges=int(np.isfinite(data['calibrated_ranges']).sum()),
            navigation=score(nav,data['truth_position']),metadata=metadata))
    save(folder/'results.json',dict(seed=seed,domain='sampled complex Gaussian RF only, all 8 PRNs absent',rows=rows))
    print([(r['revision'],r['startup_prns'],r['navigation']['solved']) for r in rows])


if __name__=='__main__':main()
