from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Low projective residual alone falsely accepts a single-source absent-authentic scene; nullspace dimension exposes nonidentifiability.',
 change='Add explicit rank-one source-only observation model and compute algebraic nullity rather than treating a least singular vector as a calibration.',
 input_model='Same 24 directions; absent observations v times unknown amplitudes; authentic positive control.')
def experiment(seed=41060):
    d=scene(seed);y=np.outer(d['v'],d['amplitudes'])
    h,s=homogeneous_fit(d['a'],y);_,sa=homogeneous_fit(d['a'],d['clean'])
    tol=np.finfo(float).eps*96*s[0]
    return dict(absent_fit_error=projective_error(h@d['a'],y),absent_matrix_rank=int(np.linalg.matrix_rank(h)),
      absent_nullity=int(sum(s<tol)),authentic_nullity=int(sum(sa<tol)),
      decision='unobservable: a rank-one matrix maps every nominal satellite into the attacker direction')
