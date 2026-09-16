from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='A first-order frequency model reduces pilot noise and predicts unmeasured subbands.',
 change='Fit affine complex matrix versus frequency; compare piecewise-linear interpolation of independently estimated subbands.',
 input_model='Delayed-tap scene; noise 0.08; seven even-index tones used for estimation and six odd-index tones held out.')
def experiment(seed=41060):
    f,h,y=frequency_scene(seed,.08);ft=f[::2];yt=y[::2];q=f[1::2]
    interpolation=(yt[:-1]+yt[1:])/2
    linear=polynomial_response(ft,yt,q,1)
    return dict(interpolation_heldout_error=relative_error(interpolation,h[1::2]),
        linear_heldout_error=relative_error(linear,h[1::2]),train_indices=list(range(0,13,2)),test_indices=list(range(1,13,2)))
