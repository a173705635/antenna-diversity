from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='A frequency-flat full matrix cannot model delayed off-diagonal coupling; independently measured subband matrices can.',
 change='Add a 180 ns delayed coupling tap over +/-1 MHz and estimate a matrix at each of 13 calibration tones.',
 input_model='Explicit known independent port pilots at each frequency; equivalent averaged matrix noise 0.01; no RF front end.')
def experiment(seed=41060):
    f,h,y=frequency_scene(seed);flat=np.broadcast_to(y.mean(axis=0),h.shape)
    return dict(flat_error=relative_error(flat,h),subband_error=relative_error(y,h),tones=13,
        frequency_hz=(f*1e6).tolist(),delayed_tap_s=180e-9)
