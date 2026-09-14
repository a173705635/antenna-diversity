"""Known-target projection identifiability audit; complex snapshots, not RF."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_spoofing_spatial import response
from diversity import combine_lcmv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(9720)
    a, b = response(10,25), response(100,10)
    # q^H a = 0. In two channels this complement is only one-dimensional.
    q = np.array([-np.conj(a[1]), 1])/np.sqrt(2)
    assert abs(np.vdot(q,a)) < 1e-12
    noise_power = .0025
    rows = []
    for rho in [0., .25, .5, .9, 1.]:
        values = []
        for trial in range(300):
            u = (rng.normal(size=4000)+1j*rng.normal(size=4000))/np.sqrt(2)
            independent = (rng.normal(size=4000)+1j*rng.normal(size=4000))/np.sqrt(2)
            v = rho*np.exp(.7j)*u + np.sqrt(1-rho*rho)*independent
            x = a[:,None]*u+b[:,None]*v
            x += np.sqrt(noise_power/2)*(rng.normal(size=x.shape)+1j*rng.normal(size=x.shape))
            z = q.conj()@x
            # Exact white-noise variance supplied as an optimistic calibration.
            estimate = x@z.conj()/len(z)-noise_power*q
            estimate /= np.linalg.norm(estimate)
            _, info = combine_lcmv(x,a,[estimate],.001,.99)
            w = info['weights']
            values.append([int(not info['rejected_constraint_indices']),
                abs(np.vdot(estimate,b))/np.linalg.norm(b),
                -20*np.log10(max(abs(np.vdot(w,b)),1e-12)),
                10*np.log10(np.vdot(w,w).real),
                float(np.mean(abs(w.conj()@x-u)**2))])
        vals = np.array(values)
        np.savez_compressed(out/f'rho_{rho:.2f}.npz', values=vals)
        rows.append({'rho':rho,'trials':300,'accepted':int(vals[:,0].sum()),
                     'response_similarity_median':float(np.median(vals[:,1])),
                     'source_suppression_db_p05_median':np.quantile(vals[:,2],[.05,.5]).tolist(),
                     'noise_gain_db_median':float(np.median(vals[:,3])),
                     'reconstruction_mse_median':float(np.median(vals[:,4]))})
    # Exact observational ambiguity: identical received samples admit another source.
    b_alt = response(250,20)
    mixing = a + np.exp(.7j)*b
    coefficients = np.linalg.solve(np.column_stack([a,b_alt]),mixing)
    residual = float(np.linalg.norm(mixing-np.column_stack([a,b_alt])@coefficients))
    assert residual < 1e-12
    # A projector cannot reveal source angle in its one-dimensional output subspace.
    projector = np.eye(2)-np.outer(a,a.conj())/np.vdot(a,a)
    projected = np.column_stack([projector@b,projector@b_alt])
    singular = np.linalg.svd(projected,compute_uv=False)
    assert singular[1] < 1e-12
    result={'scope':__doc__,'seed':9720,'snapshots_per_trial':4000,
        'target':[10,25],'source':[100,10],'source_to_target_power_db':0,
        'known_white_noise_power':noise_power,'rows':rows,
        'columns':['accepted','response_similarity','suppression_db','noise_gain_db','reconstruction_mse'],
        'coherent_ambiguity':{'alternative_source':[250,20],
            'alternative_target_coefficient':[float(coefficients[0].real),float(coefficients[0].imag)],
            'alternative_source_coefficient':[float(coefficients[1].real),float(coefficients[1].imag)],
            'identical_observation_residual':residual,'projected_singular_values':singular.tolist()},
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('audit_spoofing_spatial.py'),Path(__file__).with_name('diversity.py')]}}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
