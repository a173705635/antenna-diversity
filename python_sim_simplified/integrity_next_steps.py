"""Explicit mechanism changes; each is applied and evaluated separately."""
import sys
from integrity_next_research import run

STEPS={
151: ('    # INITIAL','    if not o[\'origin_trusted\']: rp[:]=np.inf\n    # INITIAL',
      'Unknown initial origin is unobservable; propagate an unbounded position set.', 'unknown_origin',
      'Reject finite absolute claims without an independent origin. Translation remains unobservable.'),
152: ('    # INITIAL','    bias=o[\'training\'].mean(axis=0)\n    # INITIAL',
      'Estimate bias from a separate stationary bench record, not runtime truth.', 'bounded',
      'Independent synthetic training improves center accuracy; its validity still depends on the fixture and sensor stability.'),
153: ('    # INITIAL','    br=np.max(np.abs(o[\'training\']-bias),axis=0)+o[\'noise_bound\']\n    # INITIAL',
      'Try empirical calibration residual maxima as an error envelope.', 'empirical_only',
      'Empirical finite-sample maxima improve declared availability but cannot certify unobserved errors; this proposal is not a hard guarantee.'),
154: ('    # INITIAL','    certified=o[\'envelope_origin\']==\'synthetic_declared_domain\' and o[\'noise_model\']==\'bounded\'\n    # INITIAL',
      'Separate empirical or unbounded Gaussian support from hard bounded support.', 'gaussian',
      'Gaussian noise and empirical-only bounds cannot authorize finite hard protection levels; protected output is a continuity failure.'),
155: ('    # INITIAL',"    if certified:\n        r=o['reference_radius']+o['noise_bound']\n        lo=np.max(o['training']-r,axis=0); hi=np.min(o['training']+r,axis=0)\n        bias=(lo+hi)/2; br=(hi-lo)/2+o['noise_bound']\n    # INITIAL",
      'Identify a bias interval by intersecting separately measured bench intervals with declared fixture error.', 'bounded',
      'This interval is conditional on externally supplied fixture and noise support; simulation declaration is not a phone certificate.'),
156: ('        # ACCEL',"        if o['drift_enabled']: ar+=t[k]*o['drift_bound']\n        # ACCEL",
      'Allow time-correlated bias drift rather than treating it as independent white noise.', 'drift',
      'Drift support increases cubically in position and prevents optimistic long-duration claims, at the cost of continuity.'),
157: ('    # INITIAL',"    certified=certified and bool(np.all(np.abs(o['holdout']-bias)<=br+o['reference_radius']))\n    # INITIAL",
      'Use a held-out bench sequence as a falsification check for calibration validity.', 'holdout_fault',
      'A holdout contradiction invalidates the declared calibration. Passing holdout cannot prove a hard bound.'),
158: ('        # ACCEL',"        a-=(o['temperature'][k]-20)*o['temperature_coefficient']\n        ar+=abs(o['temperature'][k]-20)*o['temperature_radius']\n        # ACCEL",
      'Compensate measured temperature with an externally calibrated coefficient interval.', 'temperature',
      'Temperature compensation reduces drift only under the declared coefficient interval; no coefficient is learned from truth online.'),
159: ('        # OBSERVE',"        if k in o['velocities']: v,rv=(x.copy() for x in o['velocities'][k])\n        # OBSERVE",
      'Use independent bounded velocity observations to limit integrated drift.', 'velocity',
      'Velocity observations constrain drift but do not establish absolute position; their synthetic provenance is explicit.'),
160: ('        # OBSERVE',"        if o['stationary_claim'] and o['stationary_origin']=='fixture':\n            v[:]=0.; rv[:]=0.\n        # OBSERVE",
      'Permit zero-velocity updates only from an independent stationary fixture, not quiet accelerometers.', 'stationary',
      'Verified stationarity helps continuity. Constant motion has the same zero acceleration and cannot justify zero velocity.'),
161: ('    # INITIAL',"    p+=v*o['initial_delay']\n    # INITIAL",
      'Transport an old initial position measurement to the integration epoch.', 'initial_delay',
      'Known timestamp correction removes a systematic position offset; this model assumes constant velocity during the initial delay.'),
162: ('    # INITIAL',"    rp+=rv*abs(o['initial_delay'])+(np.abs(v)+rv)*o['time_radius']\n    # INITIAL",
      'Propagate initial timestamp uncertainty and velocity support into position support.', 'jitter',
      'Timestamp uncertainty belongs in the reachable set and can reduce availability even when the center improves.'),
163: ('        # ACCEL',"        if t[k]>o['envelope_expiry']: certified=False\n        # ACCEL",
      'Stop using a calibration after its declared validity interval expires.', 'expired',
      'An expired sensor envelope cannot be silently extended to a 600-second claim.'),
164: ('        # OBSERVE',"        if k in o['anchors']:\n            anchor=o['anchors'][k]; lag=anchor['lag']\n            z=anchor['z']+v*lag\n            zr=anchor['radius']+rv*lag+(np.abs(a)+ar)*lag*lag/2\n            lo=np.maximum(p-rp,z-zr); hi=np.minimum(p+rp,z+zr)\n            p=(lo+hi)/2; rp=np.abs(hi-lo)/2\n        # OBSERVE",
      'Transport delayed independent position measurements with velocity and acceleration support.', 'delayed_anchor',
      'A delayed anchor is not a current position. Transport uncertainty is retained; acceleration variation is tested next by final stress cases.'),
165: ('        if k in o[\'velocities\']: v,rv=(x.copy() for x in o[\'velocities\'][k])',"        if k in o['velocities']:\n            if k-10 in o['velocities'] and not o['drift_enabled'] and o['attitude'] is None and not np.any(o['tilt_bound']) and not np.any(o['temperature_coefficient']) and o['scale_bound']==0:\n                prev,pr=o['velocities'][k-10]; cur,cr=o['velocities'][k]\n                estimate=o['acceleration'][k-9:k+1].mean(axis=0)-(cur-prev)/10\n                er=(pr+cr)/10+o['noise_bound']\n                lo=np.maximum(bias-(br-o['noise_bound']),estimate-er)\n                hi=np.minimum(bias+(br-o['noise_bound']),estimate+er)\n                bias=(lo+hi)/2; br=(hi-lo)/2+o['noise_bound']\n            v,rv=(x.copy() for x in o['velocities'][k])",
      'Intersect constant-bias support with independent velocity increments using time-correlated noise support.', 'bias_identify',
      'No square-root-of-N hard-bound reduction is used. Velocity increments provide bounded extra information but do not anchor translation.'),
166: ('        # ACCEL',"        ar+=9.81*o['tilt_bound']\n        # ACCEL",
      'Account for three-axis gravity leakage using the sine Lipschitz bound.', 'tilt3d',
      'All three position components are scored. This is a componentwise attitude leakage model, not full SO(3) inertial mechanization.'),
167: ('        # ACCEL',"        if o['attitude'] is not None:\n            angle,angle_radius=o['attitude']; a-=9.81*np.sin(angle)\n            ar+=9.81*(angle_radius-o['tilt_bound'])\n        # ACCEL",
      'Use an independent attitude observation and its interval to remove nominal gravity leakage.', 'attitude',
      'Independent attitude improves the center, but residual gravity uncertainty still causes quadratic position growth.'),
168: ('        # ACCEL',"        ar+=9.81*t[k]*o['gyro_bound']\n        # ACCEL",
      'Propagate time-correlated gyro error into attitude and gravity support.', 'gyro',
      'A fixed attitude calibration cannot stop later gyro drift; long-time continuity remains limited.'),
169: ('        # OUTPUT',"        output+=o['lever'][k]\n        # OUTPUT",
      'Transform the body-center position to the time-varying antenna phase center.', 'lever',
      'The measured navigation-frame rotating lever is applied at each epoch; angular-rate-induced sensor acceleration is outside this declared model.'),
170: ('        # OUTPUT',"        bound=float(np.linalg.norm(rp+o['lever_radius'])) if certified else float('inf')\n        # OUTPUT",
      'Include uncertain antenna/body lever geometry in the output-centered error set.', 'lever_uncertain',
      'Known lever correction and its uncertainty are different inputs; only the latter belongs in protection support.'),
171: ('        # ACCEL',"        ar+=o['scale_bound']/(1-o['scale_bound'])*(np.abs(a)+ar)\n        # ACCEL",
      'Bound multiplicative scale error from measured acceleration rather than true acceleration.', 'scale',
      'The inverse multiplicative support covers the declared scale interval, including interaction with bias and attitude residuals.'),
172: ('        # ACCEL',"        if o['saturated'][k]: certified=False\n        # ACCEL",
      'Invalidate integrated inertial support after a flagged clipped sample with unknown missing acceleration.', 'clipping',
      'Clipping loses an unbounded impulse in this model. Disabling a claim is not recovering the missed motion.'),
173: ('            p=(lo+hi)/2; rp=np.abs(hi-lo)/2',"            valid=valid and bool(np.all(lo<=hi))\n            if valid: p=(lo+hi)/2; rp=(hi-lo)/2",
      'Treat contradictory anchor intersections as an integrity inconsistency, not an absolute-valued radius.', 'conflicting_anchor',
      'An empty set cannot support a normal output. This policy stays invalid after a contradiction, and does not identify which instrument failed.'),
174: ('        # OUTPUT',"        if k in o['anchors'] and not certified:\n            anchor=o['anchors'][k]\n            if anchor['lag']==0:\n                output=anchor['z']+o['lever'][k]\n                bound=float(np.linalg.norm(anchor['radius']+o['lever_radius']))\n        # OUTPUT",
      'Allow a fresh absolute observation to provide a single-epoch position when inertial support is unavailable.', 'anchor_recovery',
      'Fresh independent position gives isolated valid epochs, not continuous recovery of unknown velocity and acceleration.'),
175: ("        if k in o['anchors']:","        if k in o['anchors'] and o['anchors'][k]['origin']=='independent_synthetic':",
      'Reject position certificates whose provenance depends on the attacked GNSS stream.', 'circular_trust',
      'GNSS agreement cannot create independent absolute information. Circular anchors are excluded from both intersection and direct recovery.')
}

if __name__=='__main__':
    n=int(sys.argv[1]); old,new,hypothesis,target,conclusion=STEPS[n]
    if n==154:
        # Protection reporting is part of distinguishing hard support from statistics.
        from integrity_next_research import SOURCE
        old=SOURCE.read_text(encoding='utf-8')
        new=old.replace('    # INITIAL',new,1).replace('bound=float(np.linalg.norm(rp))',"bound=float(np.linalg.norm(rp)) if certified else float('inf')")
    if n==175:
        from integrity_next_research import SOURCE
        old=SOURCE.read_text(encoding='utf-8')
        new=old.replace("if k in o['anchors']:","if k in o['anchors'] and o['anchors'][k]['origin']=='independent_synthetic':").replace("if k in o['anchors'] and not certified:","if k in o['anchors'] and not certified and o['anchors'][k]['origin']=='independent_synthetic':")
    run(n,old,new,hypothesis,target,conclusion)
