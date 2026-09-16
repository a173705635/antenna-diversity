from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='A single direction-independent matrix cannot describe direction-dependent antenna patterns; an angular matrix basis can predict heldout fixture directions.',
 change='Extend response to (H0+u H1+v H2) a(u,v), estimate all coefficients from known OTA commands.',
 input_model='24 known complex OTA fixture excitations, 16 heldout directions, angular response terms noise-generated at 0.1, measurement noise 0.001; no online GNSS amplitude truth.')
def experiment(seed=41060):
    d=scene(seed);u,v=d['uv'];a=np.column_stack((d['a'],d['test_a']))
    h1=.1*noise(d['rng'],(4,4));h2=.1*noise(d['rng'],(4,4))
    truth=d['h']@a+(h1@a)*u+(h2@a)*v
    y=truth[:,:24]+.001*noise(d['rng'],(4,24))
    flat=pilot_fit(d['a'],y)
    design=np.vstack((a,a*u,a*v));angular=pilot_fit(design[:,:24],y)
    return dict(flat_heldout_error=relative_error(flat@d['test_a'],truth[:,24:]),
       angular_heldout_error=relative_error(angular@design[:,24:],truth[:,24:]),
       design_rank=int(np.linalg.matrix_rank(design[:,:24])),design_condition=float(np.linalg.cond(design[:,:24])),
       caveat='Hand/body state and unmodeled angular structure not tested; fixture directions/amplitudes are externally commanded measurements.')
