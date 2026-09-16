from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Periodic independent pilots track changing receiver phase while a startup matrix becomes stale.',
 change='Introduce time-dependent row phase drift and causal exponentially weighted matrix updates from recurring pilots.',
 input_model='61 one-second epochs, four row phases drift by different known-to-simulator rates; receiver sees pilot matrices with 0.01 noise, no truth rates.')
def experiment(seed=41060):
    d=scene(seed);times=np.arange(61)
    h=np.exp(1j*times[:,None]*np.array([0,.007,-.01,.012]))[:,:,None]*d['h']
    y=h+.01*noise(d['rng'],h.shape);estimated=y[0].copy();static=[];online=[]
    for k in range(61):
        estimated=.5*estimated+.5*y[k]
        static.append(relative_error(y[0],h[k]));online.append(relative_error(estimated,h[k]))
    return dict(static_final_error=static[-1],online_final_error=online[-1],static_max_error=max(static),online_max_error=max(online),
       seconds=times.tolist(),static_error=static,online_error=online)
