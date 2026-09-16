"""Replay one paired seed from every development round with the final source."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_recovery_fifty import Case, run_case, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;checked=[]
    for number in range(1,41):
        directory=out/f'round{number:02d}'
        frozen=json.loads((directory/'frozen.json').read_text())
        seed=frozen['seeds'][0];case=Case(**frozen['case'])
        for attacked in (False,True):
            _,trace,_=run_case(case,seed,attacked)
            original=np.load(directory/f'{seed}_{int(attacked)}.npz')
            for key in original.files:
                np.testing.assert_allclose(trace[key],original[key],rtol=1e-9,atol=1e-7,equal_nan=True)
            checked.append(dict(round=number,seed=seed,attacked=attacked))
        print(f'replayed {number}/40',flush=True)
    save(out/'replay_verification.json',dict(checked=checked,rtol=1e-9,atol=1e-7,
        result='Final source reproduces every saved array for one normal/attack seed per development round.'))


if __name__=='__main__':main()
