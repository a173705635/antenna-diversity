from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Ill-conditioned angular calibration amplifies measurement noise; a declared small-pattern prior regularizes only angular coefficients.',
 change='Penalize H1/H2 with Gaussian-prior ridge while leaving the constant matrix unpenalized.',
 input_model='Same angular fixture geometry as 057, noise 0.02; explicit prior angular coefficient RMS 0.1 chosen as synthetic design assumption, not measured hardware evidence.')
def experiment(seed=41060):
    d=scene(seed);u,v=d['uv'];a=np.column_stack((d['a'],d['test_a']))
    h1=.1*noise(d['rng'],(4,4));h2=.1*noise(d['rng'],(4,4))
    truth=d['h']@a+(h1@a)*u+(h2@a)*v
    y=truth[:,:24]+.02*noise(d['rng'],(4,24));design=np.vstack((a,a*u,a*v));x=design[:,:24]
    raw=pilot_fit(x,y)
    penalty=np.diag([0]*4+[1]*8)*(.02/.1)**2
    ridge=np.linalg.solve((x@x.conj().T+penalty).T,(y@x.conj().T).T).T
    return dict(unregularized_error=relative_error(raw@design[:,24:],truth[:,24:]),
       angular_prior_error=relative_error(ridge@design[:,24:],truth[:,24:]),prior_rms=.1,
       limitation='Prior mismatch may bias actual patterns; no measured prior or off-distribution hand/body model.')
