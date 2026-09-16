from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Reciprocal complex-symmetric hardware may constrain the unobserved matrix row/column ambiguity.',
 change='Replace full matrix basis by complex-symmetric reciprocal coupling basis and test projected identifiability.',
 input_model='Favorable synthetic reciprocal hardware; measured source-only covariance training; no diagonal receiver gains.')
def experiment(seed=41060):
    d=scene(seed);h=(d['h']+d['h'].T)/2
    train=np.outer(d['v'],noise(d['rng'],(100,)));_,u=np.linalg.eigh(train@train.conj().T)
    p=np.eye(4)-np.outer(u[:,-1],u[:,-1].conj());y=p@h@d['a']*d['amplitudes']
    basis=[]
    for i in range(4):
        for j in range(i,4):
            b=np.zeros((4,4),complex);b[i,j]=1;b[j,i]=1;basis.append(b)
    q=constraints(d['a'],y,np.einsum('ab,kbc->kac',p,basis));s=np.linalg.svd(q,compute_uv=False)
    return dict(reciprocal_parameters=10,nullity=int(sum(s<np.finfo(float).eps*96*s[0])),singular_values=s.tolist(),
       invisible_symmetric_component='v v-transpose',calibration_identified=False)
