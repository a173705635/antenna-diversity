from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='An internal pilot injected after antenna coupling calibrates only the receiver electronics; known over-air fixture measurements can identify the end-to-end matrix.',
 change='Separate antenna coupling C from receiver R and move the simulated injection point; add explicit over-air known-direction, known-amplitude observations.',
 input_model='Internal commands traverse R, sky signals traverse R C; over-air fixture is an unbuilt external calibration facility, not GNSS truth given to receiver.')
def experiment(seed=41060):
    d=scene(seed);r=np.diag(np.diag(d['h']));c=np.eye(4)+.15*noise(d['rng'],(4,4));h=r@c
    x=np.tile(np.eye(4),(1,24));internal=r@x+.003*noise(d['rng'],x.shape)
    ota=h@d['a']+.003*noise(d['rng'],d['a'].shape)
    internal_fit=pilot_fit(x,internal);ota_fit=pilot_fit(d['a'],ota)
    return dict(post_coupling_injection_error=relative_error(internal_fit@d['test_a'],h@d['test_a']),
       over_air_calibration_error=relative_error(ota_fit@d['test_a'],h@d['test_a']),
       input_change='All comparisons receive internal and OTA records; internal policy ignores OTA, candidate uses independently commanded OTA fixture.')
