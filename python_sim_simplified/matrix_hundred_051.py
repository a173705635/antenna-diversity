from scipy.optimize import minimize_scalar
from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='A shared delayed coupling tap describes phase curvature more efficiently than an affine frequency model.',
 change='Variable-projection fit jointly estimates one unknown delay and two complex matrix coefficients from pilot measurements.',
 input_model='Same noisy seven training/six heldout tones as 050; delay constrained to 1 ns through 500 ns, an explicit unmeasured structural prior.')
def fit_delayed(f,y,query):
    def basis(tau,grid):
        return np.column_stack((np.ones(len(grid)),np.exp(-2j*np.pi*grid*tau)))
    def loss(tau):
        b=basis(tau,f);c=np.linalg.lstsq(b,y.reshape(len(f),-1),rcond=None)[0]
        return np.linalg.norm(b@c-y.reshape(len(f),-1))**2
    fitted=minimize_scalar(loss,bounds=(.001,.5),method='bounded')
    c=np.linalg.lstsq(basis(fitted.x,f),y.reshape(len(f),-1),rcond=None)[0]
    return (basis(fitted.x,query)@c).reshape(len(query),4,4),fitted.x
def experiment(seed=41060):
    f,h,y=frequency_scene(seed,.08);q=f[1::2]
    linear=polynomial_response(f[::2],y[::2],q,1)
    delayed,tau=fit_delayed(f[::2],y[::2],q)
    return dict(linear_error=relative_error(linear,h[1::2]),delayed_error=relative_error(delayed,h[1::2]),estimated_delay_ns=float(tau*1000))
