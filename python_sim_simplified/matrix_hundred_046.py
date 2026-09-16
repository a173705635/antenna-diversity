from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Sign-reversed port pilots cancel unknown stationary ambient source contamination.',
 change='Replace direct pilot fit by differential plus/minus measurements using both samples.',
 input_model='32 paired excitations per port; constant additive ambient vector within each pair; independent noise 0.01; full network injection assumption retained.')
def experiment(seed=41060):
    d=scene(seed);x=np.tile(np.eye(4),(1,32));b=4*np.outer(d['v'],np.ones(128))
    plus=d['h']@x+b+.01*noise(d['rng'],x.shape)
    minus=-d['h']@x+b+.01*noise(d['rng'],x.shape)
    raw=pilot_fit(x,plus);diff=pilot_fit(x,(plus-minus)/2)
    return dict(direct_error=relative_error(raw,d['h']),differential_error=relative_error(diff,d['h']),samples=256)
