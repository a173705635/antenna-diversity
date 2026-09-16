from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='Measured noise covariance improves null-constrained weights after full calibration only when spatial degrees of freedom remain.',
 change='Replace Euclidean projection with covariance-weighted constrained least variance using independent noise-only observations; compare physical 2, 3 and 4 antenna arrays.',
 input_model='Explicit pilot matrix noise 0.001, source-training observations, 4000 independent noise-only samples with colored covariance; fixed small phone aperture.')
def experiment(seed=41060):
    rows=[]
    for m in (2,3,4):
        d=scene(seed,m);h=d['h']+.001*noise(d['rng'],(m,m))
        train=np.outer(d['v'],noise(d['rng'],(300,)))+.001*noise(d['rng'],(m,300))
        _,u=np.linalg.eigh(train@train.conj().T);v=u[:,-1]
        mixing=np.eye(m,dtype=complex);mixing[0,0]=5;mixing[-1,0]=2j
        quiet=mixing@noise(d['rng'],(m,4000));r=quiet@quiet.conj().T/4000;truth_r=mixing@mixing.conj().T
        euclidean=[];whitened=[];gain=[]
        for a in d['test_a'].T:
            b=np.column_stack((h@a,v));e=np.array([1.,0.])
            w0=b@np.linalg.solve(b.conj().T@b,e)
            rb=np.linalg.solve(r,b);w=rb@np.linalg.solve(b.conj().T@rb,e)
            euclidean.append(float(np.vdot(w0,truth_r@w0).real));whitened.append(float(np.vdot(w,truth_r@w).real))
            gain.append(float(abs(np.vdot(w,d['h']@a)-1)))
        rows.append(dict(antennas=m,mean_euclidean_noise=float(np.mean(euclidean)),mean_whitened_noise=float(np.mean(whitened)),
           noise_ratio=float(np.mean(whitened)/np.mean(euclidean)),max_actual_gain_error=max(gain)))
    return dict(arrays=rows,limitation='Two constraints consume all two-antenna degrees of freedom; calibration cannot create geometric separation.')
