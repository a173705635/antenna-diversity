"""W6 conditional small-error budgets. Gaussian coverage is not an integrity bound."""
from math import factorial

import numpy as np
from scipy.stats import chi2


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def static_ins_budget(times_s, initial_mean, initial_covariance, imu_noise_intensity, gravity=9.80665):
    """State order p,v,theta,accel bias,gyro bias; each is a 3-vector.

Level static linearization; constant bias and continuous accel/gyro white noise.
Cross-axis and cross-sensor correlations in both covariances are retained.
"""
    a=np.zeros((15,15));a[:3,3:6]=np.eye(3)
    a[3:6,6:9]=-skew([0.,0.,gravity]);a[3:6,9:12]=np.eye(3);a[6:9,12:15]=np.eye(3)
    g=np.zeros((15,6));g[3:6,:3]=np.eye(3);g[6:9,3:]=np.eye(3)
    q=g@imu_noise_intensity@g.T
    powers=[np.eye(15),a,a@a,a@a@a];rows=[]
    for t in times_s:
        transition=sum(p*t**i/factorial(i) for i,p in enumerate(powers))
        process=sum(pi@q@pj.T*t**(i+j+1)/(factorial(i)*factorial(j)*(i+j+1))
                    for i,pi in enumerate(powers) for j,pj in enumerate(powers))
        mean=transition@initial_mean;cov=transition@initial_covariance@transition.T+process
        radius=np.linalg.norm(mean[:2])+np.sqrt(chi2.ppf(.95,2)*np.linalg.eigvalsh(cov[:2,:2])[-1])
        rows.append(dict(time_s=float(t),mean=mean,covariance=cov,horizontal_radius95_m=float(radius)))
    return rows


def spatial_difference_budget(mean, covariance):
    """Input mean is channel/PRN; covariance uses channel-major flattened order."""
    channels,prns=mean.shape
    antenna=np.column_stack((-np.ones(channels-1),np.eye(channels-1)))
    satellite=np.column_stack((-np.ones(prns-1),np.eye(prns-1)))
    sd=np.kron(antenna,np.eye(prns));dd=np.kron(antenna,satellite)
    return dict(sd_mean=sd@mean.ravel(),sd_covariance=sd@covariance@sd.T,
                dd_mean=dd@mean.ravel(),dd_covariance=dd@covariance@dd.T)
