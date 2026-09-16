from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Dividing all observations by a weak reference channel amplifies its noise; a reference-free projective fit avoids that error propagation.',
 change='Compare reference-channel ratio equations with orthogonal projective equations for a full matrix under weak first-channel reception.',
 input_model='Authentic-only unknown amplitudes, first hardware row attenuated 0.03, output noise 0.01; both methods use same 24 measured responses.')
def experiment(seed=41060):
    d=scene(seed);h=d['h'].copy();h[0]*=.03
    y=(h@d['a'])*d['amplitudes']+.01*noise(d['rng'],d['clean'].shape)
    rows=[]
    for a,z in zip(d['a'].T,y.T):
        for i in range(1,4):
            row=np.zeros((4,4),complex);row[i]=a;row[0]=-z[i]/z[0]*a;rows.append(row.ravel())
    _,_,vh=np.linalg.svd(rows,full_matrices=False);reference=vh[-1].conj().reshape(4,4)
    global_fit,_=homogeneous_fit(d['a'],y)
    return dict(reference_ratio_error=projective_error(reference@d['test_a'],h@d['test_a']),
      reference_free_error=projective_error(global_fit@d['test_a'],h@d['test_a']))
