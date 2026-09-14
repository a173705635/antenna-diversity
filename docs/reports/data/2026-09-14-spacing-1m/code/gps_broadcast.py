"""GPS broadcast Kepler orbit and satellite-clock model (IS-GPS-200)."""
import numpy as np

MU=3.986005e14
OMEGA_E=7.2921151467e-5


def week_difference(seconds):
    return (seconds+302400.)%604800.-302400.


def satellite_state(eph, tow_s):
    """Transmit-time ECEF metres and clock seconds, including relativity.

    Input time is GPS system time. TGD is kept separate for observable-specific
    code correction. No receiver position or observation truth is consumed.
    """
    a=eph['sqrt_a']**2; e=eph['e']
    if not (2e7<a<3e7 and 0<=e<1):
        raise ValueError('invalid GPS broadcast orbit')
    tk=week_difference(tow_s-eph['toe'])
    m=eph['m0']+(np.sqrt(MU/a**3)+eph['delta_n'])*tk
    anomaly=m
    for _ in range(20):
        step=(anomaly-e*np.sin(anomaly)-m)/(1-e*np.cos(anomaly))
        anomaly-=step
        if abs(step)<1e-13:break
    else: raise ValueError('Kepler equation did not converge')
    phi=np.arctan2(np.sqrt(1-e*e)*np.sin(anomaly),np.cos(anomaly)-e)+eph['omega']
    s,c=np.sin(2*phi),np.cos(2*phi)
    u=phi+eph['cus']*s+eph['cuc']*c
    r=a*(1-e*np.cos(anomaly))+eph['crs']*s+eph['crc']*c
    inc=eph['i0']+eph['idot']*tk+eph['cis']*s+eph['cic']*c
    longitude=eph['omega0']+(eph['omega_dot']-OMEGA_E)*tk-OMEGA_E*eph['toe']
    x,y=r*np.cos(u),r*np.sin(u)
    xyz=np.array([x*np.cos(longitude)-y*np.cos(inc)*np.sin(longitude),
                  x*np.sin(longitude)+y*np.cos(inc)*np.cos(longitude),y*np.sin(inc)])
    dt=week_difference(tow_s-eph['toc'])
    clock=eph['af0']+eph['af1']*dt+eph['af2']*dt*dt-4.442807633e-10*eph['e']*eph['sqrt_a']*np.sin(anomaly)
    return xyz,float(clock)


def reception_frame(xyz, travel_s):
    """Rotate transmit-time ECEF into reception-time ECEF (Sagnac)."""
    a=OMEGA_E*travel_s;c,s=np.cos(a),np.sin(a)
    return np.array([c*xyz[0]+s*xyz[1],-s*xyz[0]+c*xyz[1],xyz[2]])
