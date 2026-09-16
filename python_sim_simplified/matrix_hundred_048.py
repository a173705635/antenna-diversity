from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Nonuniform acquisition timing and ambient curvature invalidate affine cancellation; a quadratic nuisance model removes curvature leakage.',
 change='Add time-squared nuisance response to joint pilot identification.',
 input_model='16 known nonuniform times per port; ambient v*(4+1.2*t+3*t^2); same noisy observations for affine and quadratic regressions.')
def experiment(seed=41060):
    d=scene(seed);t=np.tile(np.linspace(0,1,16)**1.5,4)
    x=np.repeat(np.eye(4),16,axis=1)*np.tile([1,-1],32)
    y=d['h']@x+np.outer(d['v'],4+1.2*t+3*t*t)+.01*noise(d['rng'],x.shape)
    affine=pilot_fit(np.vstack((x,np.ones(64),t)),y)[:,:4]
    quadratic=pilot_fit(np.vstack((x,np.ones(64),t,t*t)),y)[:,:4]
    return dict(affine_error=relative_error(affine,d['h']),quadratic_error=relative_error(quadratic,d['h']),
      limitation='An attacker coherent with secret pilot commands remains inseparable; low-order drift is assumed, not certified.')
