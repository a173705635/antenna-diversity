"""Controlled Jones/array counterexamples. No RF, field data or detector rates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from antenna import jones_vector
from config import WavePolarization
from diversity import combine_lcmv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rh = jones_vector(WavePolarization(kind='rhcp'))
    lh = jones_vector(WavePolarization(kind='lhcp'))
    # Explicit illustrative 20 dB cross-polar leakage, not measured antenna data.
    ports = np.column_stack([rh+.1*lh,lh+.1*rh])/np.sqrt(1.01)
    cases = {'los_rhcp':rh, 'nlos_lhcp':lh, 'nlos_rhcp_counterexample':rh,
             'los_plus_inphase_rhcp_echo':1.5*rh,
             'los_plus_quadrature_rhcp_echo':(1+.5j)*rh}
    pol = {}
    for name,signal in cases.items():
        v = ports.conj().T@signal
        pol[name] = {'rhcp_lhcp_power_ratio_db':float(20*np.log10(abs(v[0]/v[1]))),
                     'rhcp_phase_rad':float(np.angle(v[0]))}
    # Synthetic post-correlation samples with known steering: an oracle upper bound.
    los = np.ones(2,dtype=complex)
    echo = np.array([1.,-1.],dtype=complex)
    x = np.tile((los+.5j*echo)[:,None],(1,100))
    spatial = {}
    for name,s in [('distinct_direction',echo),('same_direction',los)]:
        samples = x if name=='distinct_direction' else np.tile((1+.5j)*los[:,None],(1,100))
        _,info = combine_lcmv(samples,los,[s],1e-3,.99)
        w = info['weights']
        spatial[name] = {'los_response_abs':float(abs(np.vdot(w,los))),
                         'echo_response_abs':float(abs(np.vdot(w,s))),
                         'rejected_constraints':info['rejected_constraint_indices']}
    result = {'scope':__doc__,'polarization':pol,'spatial_oracle':spatial,
              'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [Path(__file__),Path(__file__).with_name('antenna.py'),Path(__file__).with_name('diversity.py')]}}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
