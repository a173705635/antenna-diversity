"""Single-source receiver-observable counterexamples, not RF synthesis.
All antenna labels receive byte-identical observations; no array evidence exists.
"""
import numpy as np
KINDS=("normal","code_drag","joint_drag","joint_step","clock_drag","code_fault","slip","dropout","weak","ionosphere","turn","reacquire_attack","slow_drag","phase_fault","colored","authentic_absent")
def generate(seed,kind):
    rng=np.random.default_rng(seed);n=240;t=np.arange(n)*.5
    az=np.linspace(0,2*np.pi,10,endpoint=False)+rng.normal(0,.1,10);el=rng.uniform(.3,1.3,10)
    directions=np.column_stack((np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)))
    h=np.column_stack((-directions,np.ones(10)))
    truth=np.column_stack((t*1.2,.3*t,np.zeros(n)));attack=60
    if kind=="turn":truth[100:,1]+=.01*(t[100:]-t[100])**2
    clock=200+ .02*t;state=np.column_stack((truth,clock));ranges=state@h.T
    code=ranges+rng.normal(0,3,(n,10));phase=ranges+rng.normal(0,.015,(n,10))+rng.integers(-100,100,10)*.1903
    lock=np.ones((n,10),bool);cn=np.full((n,10),45.)
    drag=np.maximum(t-t[attack],0);offset=np.outer(drag,h[:,0])
    if kind=="code_drag":code+=1.5*offset
    if kind in ("joint_drag","reacquire_attack","authentic_absent"):code+=1.5*offset;phase+=1.5*offset
    if kind=="slow_drag":code+=.3*offset;phase+=.3*offset
    if kind=="joint_step":code[attack:]+=80*h[:,0];phase[attack:]+=80*h[:,0]
    if kind=="clock_drag":code[attack:]+=drag[attack:,None]*10;phase[attack:]+=drag[attack:,None]*10
    if kind=="code_fault":code[attack:,0]+=70
    if kind=="slip":phase[100:,0]+=12
    if kind in ("dropout","reacquire_attack","authentic_absent"):
        lock[80:100]=False;phase[100:]+=rng.integers(-50,50,10)*.1903
    if kind=="authentic_absent":lock[140:]=False
    if kind=="weak":cn[attack:]=27;code[attack:]+=rng.normal(0,14,(n-attack,10));phase[attack:]+=rng.normal(0,.15,(n-attack,10))
    if kind=="ionosphere":code+=.1*offset;phase-=.1*offset
    if kind=="phase_fault":phase[attack:,0]+=.8*drag[attack:]
    if kind=="colored":
        noise=np.zeros((n,10))
        for k in range(1,n):noise[k]=.95*noise[k-1]+rng.normal(0,1,10)
        code+=noise
    return dict(design=h,code=code,phase=phase,lock=lock,cn0=cn,time=t),truth,attack
