"""Prompt-domain calibration research. Inputs contain observations and nominal steering only."""
import numpy as np


def cross_calibration(z,a):
    cross=np.sum(z*z[0].conj()[None,:,:]*(a[0]/a)[:,:,None],axis=(1,2))
    return np.exp(1j*np.angle(cross))


def projected_calibration(z,a,v,weights):
    # P diag(g) a is not diag(g) P a: solve the actual projected model.
    m=len(z);p=np.eye(m)-np.outer(v,v.conj());constraints=[]
    for s in range(a.shape[1]):
        y=p@z[:,s];_,vectors=np.linalg.eigh(y@y.conj().T);b=vectors[:,-1]
        constraints.append(np.sqrt(weights[s])*(np.eye(m)-np.outer(b,b.conj()))@p@np.diag(a[:,s]))
    matrix=np.vstack(constraints)
    _,singular,vh=np.linalg.svd(matrix,full_matrices=False)
    g=vh[-1].conj();g/=g[0]
    return g,singular


def calibrate(z,a):
    # Iteration 14: alternate source and hardware estimates in the two-component model.
    flat=z.reshape(len(z),-1);_,vectors=np.linalg.eigh(flat@flat.conj().T)
    v=vectors[:,-1];weights=np.ones(a.shape[1])
    for _ in range(6):
        g,singular=projected_calibration(z,a,v,weights);remaining=np.empty_like(z);errors=[]
        for s in range(a.shape[1]):
            desired=g*a[:,s];basis=np.column_stack((desired,v))
            coefficients=np.linalg.lstsq(basis,z[:,s],rcond=None)[0]
            remaining[:,s]=z[:,s]-desired[:,None]*coefficients[0]
            errors.append(np.mean(abs(z[:,s]-basis@coefficients)**2))
        # Iteration 15: a PRN violating the two-component model cannot dominate calibration.
        errors=np.array(errors);weights=np.median(errors)/(np.median(errors)+errors)
        r=(remaining*np.sqrt(weights)[None,:,None]).reshape(len(z),-1)
        _,vectors=np.linalg.eigh(r@r.conj().T);v=vectors[:,-1]
    # Iteration 18: a unique null direction is required to identify relative gain.
    usable=len(z)>=3 and singular[-2]>.1 and singular[-1]<.1*singular[-2]
    return g,dict(usable=bool(usable),singular=singular.tolist())
