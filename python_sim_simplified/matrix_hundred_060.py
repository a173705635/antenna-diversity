from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='A calibrated two-component model can fit absent-authentic data well; comparing a source-only nested model tests whether satellite-specific signal information is present.',
 change='Add source-only versus source-plus-constant-authentic-coefficient likelihood comparison, including parameter penalty; retain absent control explicitly.',
 input_model='2/3/4 fixed antennas, 24 PRNs and 32 coherent correlations per PRN; measured pilot matrix noise 0.001; ideal source-only covariance block; complex observation noise 0.02.')
def experiment(seed=41060):
    rows=[]
    for m in (2,3,4):
        d=scene(seed,m);h=d['h']+.001*noise(d['rng'],(m,m))
        block=np.outer(d['v'],noise(d['rng'],(100,)));_,u=np.linalg.eigh(block@block.conj().T)
        v=u[:,-1];p=np.eye(m)-np.outer(v,v.conj());t=32;sigma=.02
        attack=10*d['v'][:,None,None]*noise(d['rng'],(24,t))[None,:,:]
        n=sigma*noise(d['rng'],attack.shape)
        for genuine in (False,True):
            y=attack+n+genuine*.2*(d['h']@d['a'])[:,:,None]
            evidence=[];residual=[]
            for j in range(24):
                z=p@y[:,j];a=p@h@d['a'][:,j]
                alpha=np.vdot(a,z.mean(axis=1))/np.vdot(a,a)
                r0=np.linalg.norm(z)**2;r1=np.linalg.norm(z-a[:,None]*alpha)**2
                evidence.append(float(2*(r0-r1)/sigma**2-2*np.log(m*t)))
                residual.append(float(r1/(m*t*sigma**2)))
            rows.append(dict(antennas=m,genuine=genuine,mean_two_component_normalized_residual=float(np.mean(residual)),
                total_nested_bic_support=float(sum(evidence)),positive_prn_support=int(sum(x>0 for x in evidence))))
    return dict(cases=rows,absent_navigation='No authentic range information; no trusted sustained navigation output.',
       caveat='Evidence tests modeled spatial information, not cryptographic authenticity or position accuracy; model selection is not an integrity guarantee.')
