from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Projecting a measured attacker direction does not identify a full matrix: v q-transpose remains invisible even with authentic satellites.',
 change='Fit full matrix after projection using a measured source-only training block; compare unconstrained fit and near-identity ridge prior.',
 input_model='Explicit source-only training block gives covariance direction, followed by authentic plus 10x same-source correlations; prior identity is unmeasured.')
def experiment(seed=41060):
    d=scene(seed);train=np.outer(d['v'],noise(d['rng'],(100,)))
    _,u=np.linalg.eigh(train@train.conj().T);v=u[:,-1];p=np.eye(4)-np.outer(v,v.conj())
    y=d['clean']+10*np.outer(d['v'],noise(d['rng'],(24,)))
    basis=np.eye(16).reshape(16,4,4);q=constraints(d['a'],p@y,np.einsum('ab,kbc->kac',p,basis))
    _,s,vh=np.linalg.svd(q,full_matrices=False)
    ridge=np.linalg.solve(q.conj().T@q+.01*np.eye(16),.01*np.eye(4).ravel()).reshape(4,4)
    return dict(projected_nullity=int(sum(s<np.finfo(float).eps*96*s[0])),
      ridge_heldout_error=projective_error(ridge@d['test_a'],d['test_y']),
      hidden_matrix_components=4,prior='identity ridge, synthetic assumption, not hardware measurement')
