"""Evolving bounded 3-D receiver; iteration snapshots are the audit source."""
import numpy as np

def receive(o):
    t=o['time']; p=o['initial_position'].copy(); v=o['initial_velocity'].copy()
    rp=o['initial_radius'].copy(); rv=o['velocity_radius'].copy()
    bias=np.zeros(3); br=np.full(3,.022)
    certified=True
    if not o['origin_trusted']: rp[:]=np.inf
    bias=o['training'].mean(axis=0)
    br=np.max(np.abs(o['training']-bias),axis=0)+o['noise_bound']
    certified=o['envelope_origin']=='synthetic_declared_domain' and o['noise_model']=='bounded'
    if certified:
        r=o['reference_radius']+o['noise_bound']
        lo=np.max(o['training']-r,axis=0); hi=np.min(o['training']+r,axis=0)
        bias=(lo+hi)/2; br=(hi-lo)/2+o['noise_bound']
    certified=certified and bool(np.all(np.abs(o['holdout']-bias)<=br+o['reference_radius']))
    p+=v*o['initial_delay']
    rp+=rv*abs(o['initial_delay'])+(np.abs(v)+rv)*o['time_radius']
    # INITIAL
    positions=[]; bounds=[]; normals=[]
    acceleration_history=[]; support_history=[]; history_available=[]
    valid=True
    for k in range(len(t)):
        a=o['acceleration'][k]-bias
        ar=br.copy()
        if o['drift_enabled']: ar+=t[k]*o['drift_bound']
        a-=(o['temperature'][k]-20)*o['temperature_coefficient']
        ar+=abs(o['temperature'][k]-20)*o['temperature_radius']
        if t[k]>o['envelope_expiry']: certified=False
        ar+=9.81*o['tilt_bound']
        if o['attitude'] is not None:
            angle,angle_radius=o['attitude']; a-=9.81*np.sin(angle)
            ar+=9.81*(angle_radius-o['tilt_bound'])
        ar+=9.81*t[k]*o['gyro_bound']
        ar+=o['scale_bound']/(1-o['scale_bound'])*(np.abs(a)+ar)
        if o['saturated'][k]: certified=False
        acceleration_history.append(a.copy()); support_history.append(ar.copy())
        history_available.append(certified)
        # ACCEL
        if k:
            p+=v+a/2; v+=a
            rp+=rv+ar/2; rv+=ar
        if k in o['velocities']:
            if k-10 in o['velocities'] and not o['drift_enabled'] and o['attitude'] is None and not np.any(o['tilt_bound']) and not np.any(o['temperature_coefficient']) and o['scale_bound']==0:
                prev,pr=o['velocities'][k-10]; cur,cr=o['velocities'][k]
                estimate=o['acceleration'][k-9:k+1].mean(axis=0)-(cur-prev)/10
                er=(pr+cr)/10+o['noise_bound']
                lo=np.maximum(bias-(br-o['noise_bound']),estimate-er)
                hi=np.minimum(bias+(br-o['noise_bound']),estimate+er)
                bias=(lo+hi)/2; br=(hi-lo)/2+o['noise_bound']
            v,rv=(x.copy() for x in o['velocities'][k])
        if o['stationary_claim'] and o['stationary_origin']=='fixture':
            v[:]=0.; rv[:]=0.
        if k in o['anchors'] and o['anchors'][k]['origin']=='independent_synthetic':
            anchor=o['anchors'][k]; lag=anchor['lag']
            if lag<=k and all(history_available[k-lag+1:k+1]):
                # p_k-p_(k-lag)=lag*v_k-sum((j+.5)*a_(k-lag+1+j)).
                weights=(np.arange(lag)+.5)[:,None]
                history=np.array(acceleration_history[k-lag+1:k+1]).reshape(lag,3)
                support=np.array(support_history[k-lag+1:k+1]).reshape(lag,3)
                z=anchor['z']+v*lag-np.sum(weights*history,axis=0)
                zr=anchor['radius']+rv*lag+np.sum(weights*support,axis=0)
                lo=np.maximum(p-rp,z-zr); hi=np.minimum(p+rp,z+zr)
                valid=valid and bool(np.all(lo<=hi))
                if valid: p=(lo+hi)/2; rp=(hi-lo)/2
        # OBSERVE
        bound=float(np.linalg.norm(rp)) if certified else float('inf')
        output=p.copy()
        output+=o['lever'][k]
        bound=float(np.linalg.norm(rp+o['lever_radius'])) if certified else float('inf')
        if k in o['anchors'] and not certified and o['anchors'][k]['origin']=='independent_synthetic':
            anchor=o['anchors'][k]
            if anchor['lag']==0:
                output=anchor['z']+o['lever'][k]
                bound=float(np.linalg.norm(anchor['radius']+o['lever_radius']))
        # OUTPUT
        positions.append(output); bounds.append(bound); normals.append(bool(valid and bound<=20))
    return dict(position=np.array(positions),bound=np.array(bounds),normal=np.array(normals))
