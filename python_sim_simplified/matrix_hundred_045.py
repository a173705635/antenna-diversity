from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Independent commanded port excitation makes the hardware matrix identifiable; a common injected tone does not.',
 change='Introduce explicit synthetic multiport pilot measurement with rank-four commands; compare full fit and diagonal fit on identical measurements; rank-one pilot negative control.',
 input_model='Hypothetical calibrated injection before the full modeled coupling network, known complex pilot amplitudes, 0.01 output noise. No bench evidence this topology exists in a phone.')
def experiment(seed=41060):
    d=scene(seed);x=np.tile(np.eye(4),(1,32));n=.01*noise(d['rng'],x.shape);y=d['h']@x+n
    full=pilot_fit(x,y);diagonal=np.diag(np.diag(full));common=np.ones_like(x)/2
    bad=pilot_fit(common,d['h']@common+n)
    return dict(full_heldout_error=relative_error(full@d['test_a'],d['test_y']),
       diagonal_heldout_error=relative_error(diagonal@d['test_a'],d['test_y']),
       common_pilot_heldout_error=relative_error(bad@d['test_a'],d['test_y']),
       independent_command_rank=int(np.linalg.matrix_rank(x)),common_command_rank=int(np.linalg.matrix_rank(common)))
