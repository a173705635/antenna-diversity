"""Replay fixed PRN 8 reference, including compromised-reference counterexamples."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_spatial_diversity import W, NAMES, LAYOUTS, save_json
from presets import mate70_phase_centers
from run_gps_spp import receiver_reference, satellite_setup
from spatial_direction import select_common_prns
from spatial_reference_direction import estimate_with_reference
from audit_spatial_direction import direction_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    p=np.array([mate70_phase_centers()[name] for name in NAMES])
    rx,rot=receiver_reference()
    reference=rot@(satellite_setup()[7]['position_m']-rx)
    reference/=np.linalg.norm(reference)
    az,el=np.deg2rad([100,10])
    truth=np.array([np.cos(el)*np.sin(az),np.cos(el)*np.cos(az),np.sin(el)])
    rows=[]
    for batch,cases in [('rf_a',['all_spoof','partial_3']),('rf_b',['partial_4','mixture_0','mixture_6'])]:
        for case in cases:
            for seed in [610000,620000]:
                with np.load(args.archive/batch/f'{case}_{seed}_observations.npz') as a:
                    x,valid=a['carrier_m'],a['valid']
                for layout in ['3','4']:
                    indices=LAYOUTS[layout]
                    for k in range(4):
                        block=x[k*10:(k+1)*10,indices]
                        if not valid[k*10:(k+1)*10,indices].all():
                            rows.append(dict(case=case,seed=seed,layout=layout,window=k,status='invalid_observation'))
                            continue
                        selected=select_common_prns(block,W)
                        fit=estimate_with_reference(block,p[indices],W,reference_index=7,
                              target_indices=selected,reference_direction=reference)
                        bias=np.array([0,.012,-.018,.025])[indices]
                        shifted=estimate_with_reference(block+bias[None,:,None],p[indices],W,reference_index=7,
                              target_indices=selected,reference_direction=reference)
                        # Bias-invariant candidate sets can be ordered differently at equal costs.
                        if fit['candidates']:
                            a=np.array([c['unit_vector'] for c in fit['candidates']])
                            b=np.array([c['unit_vector'] for c in shifted['candidates']])
                            assert len(a)==len(b)
                            assert np.max(np.min(np.linalg.norm(a[:,None]-b[None],axis=-1),axis=1))<1e-4
                        rows.append(dict(case=case,seed=seed,layout=layout,window=k,status=fit['status'],
                            selected_prns=[i+1 for i in selected],result=fit,
                            reference_actually_genuine=case.startswith('partial_'),
                            reference_policy='fixed PRN 8; assumed, not independently authenticated',
                            candidate_set_min_error_deg=direction_error(fit,truth),
                            added_common_bias_did_not_change_candidates=True))
    save_json(args.output,rows)
    print('Reference replay completed:',len(rows),'windows')


if __name__=='__main__':
    main()
