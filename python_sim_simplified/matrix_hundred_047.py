from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Differential pilots leak linear ambient drift; jointly fitting a local affine nuisance removes that bias.',
 change='Augment commanded pilot design with constant and timestamp nuisance columns.',
 input_model='Alternating signs for each port over 16 local times, ambient v*(4+1.2*t); known acquisition times; all policies use identical outputs.')
def experiment(seed=41060):
    d=scene(seed);t=np.tile(np.linspace(-1,1,16),4);sign=np.tile([1,-1],32)
    x=np.repeat(np.eye(4),16,axis=1)*sign
    y=d['h']@x+np.outer(d['v'],4+1.2*t)+.01*noise(d['rng'],x.shape)
    differential=pilot_fit(x,y)
    joint=pilot_fit(np.vstack((x,np.ones(64),t)),y)[:,:4]
    return dict(differential_error=relative_error(differential,d['h']),affine_nuisance_error=relative_error(joint,d['h']))
