"""W4-W6 software audit. Default inputs are SYNTHETIC, not measured hardware."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bench_calibration import (rf_features,fit_rf,evaluate_rf,fit_imu,correct_imu,
                               time_offset_profile,long_run_covariance)
from error_budget import static_ins_budget,spatial_difference_budget
from inertial import measurements,propagate,trajectory
from run_gps_spp import serializable

G=np.array([0.,0.,-9.80665]);W=299792458/1575.42e6


def synthetic_bench():
    rng=np.random.default_rng(440560)
    phase_coef=np.array([[.2,-.4,.1],[.003,.001,-.002],[.02,-.01,.015],
                         [-.006,.012,-.018],[.0002,-.0001,.0003],[.12,-.08,.04],[-.1,.2,.1]])
    gain_coef=phase_coef/3
    def rf(temperatures,boots):
        conditions=np.array(list(product(temperatures,[-6.,0.,6.],boots)))
        temp,gain,boot=conditions.T;frequency=1575.42e6+np.linspace(-2e6,2e6,9)
        x=rf_features(temp,gain,boot,frequency,boots=np.array([0,1,2]))
        phase=np.concatenate((np.zeros((*x.shape[:2],1)),x@phase_coef),axis=2)
        amplitude=np.concatenate((np.zeros((*x.shape[:2],1)),x@gain_coef),axis=2)
        fixture=10**(np.array([0.,.6,-.3,.2])/20)*np.exp(-2j*np.pi*frequency[:,None]*np.array([0.,2e-9,-1e-9,.5e-9]))
        source=np.exp(1j*rng.uniform(-np.pi,np.pi,(*x.shape[:2],1)))
        transfer=(source*fixture*10**((amplitude+rng.normal(0,.02,amplitude.shape))/20)
                  *np.exp(1j*(phase+rng.normal(0,.004,phase.shape))))
        return dict(temperature_c=temp,gain_db=gain,boot_id=boot.astype(int),frequency_hz=frequency,
                    transfer=transfer,fixture_transfer=fixture,phase_anchor_rad=np.zeros((len(temp),3)))
    matrix=np.eye(6)+rng.normal(0,.001,(6,6));bias=np.array([.01,-.015,.008,.0001,-.00015,.00008])
    temp_coefficient=np.array([.0004,-.0003,.0002,1e-5,-8e-6,5e-6])
    def imu(n):
        direction=rng.normal(size=(n,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
        reference=np.column_stack((9.80665*direction,rng.uniform(-.5,.5,(n,3))))
        temp=rng.uniform(5,45,n)
        measured=(reference@matrix.T+bias+(temp-25)[:,None]*temp_coefficient
                  +rng.normal(size=(n,6))*np.array([.008]*3+[8e-5]*3))
        return dict(reference=reference,measured=measured,temperature_c=temp)
    train=imu(600);test=imu(300)
    stationary=(np.tile([0.,0.,9.80665,0.,0.,0.],(6000,1))@matrix.T+bias
                +rng.normal(size=(6000,6))*np.array([.008]*3+[8e-5]*3))
    rt=np.arange(-20,2021)*.01;t=np.arange(2001)*.01
    def rate(t):return np.column_stack((.3*np.sin(.7*t),.2*np.sin(1.1*t+.2),.4*np.cos(.4*t)))
    sync=dict(time_s=t,measured=rate(t-.017)+bias[3:]+rng.normal(0,1e-5,(len(t),3)),
              reference_time_s=rt,reference=rate(rt),offsets_s=np.arange(41)*.001)
    return dict(rf_train=rf([5.,25.,45.],[0,1,2]),rf_holdout=rf([15.,35.],[0,1,2]),
                rf_outside=rf([65.],[7]),imu_train=train,imu_holdout=test,
                imu_stationary=dict(measured=stationary,temperature_c=np.full(6000,25.),dt_s=np.array(.01),
                                    reference=np.array([0.,0.,9.80665,0.,0.,0.])),
                imu_sync=sync),dict(source_kind='synthetic',seed=440560,device_id=None,
                                   fixture_status='synthetic exact transfer; actual uncertainty unmeasured',
                                   expected_time_offset_s=.017)


def analyze_bench(data):
    rf_model=fit_rf(data['rf_train']);rf_test=evaluate_rf(rf_model,data['rf_holdout'])
    outside=evaluate_rf(rf_model,data['rf_outside'])
    imu_model=fit_imu(**data['imu_train'])
    test=data['imu_holdout'];imu_residual=correct_imu(imu_model,test['measured'],test['temperature_c'])-test['reference']
    stationary=data['imu_stationary']
    corrected=correct_imu(imu_model,stationary['measured'],stationary['temperature_c'])
    intensity=long_run_covariance(corrected,float(stationary['dt_s']),50)
    sync=time_offset_profile(**data['imu_sync'])
    return dict(rf_model=rf_model,imu_model=imu_model,imu_noise_intensity=intensity,
                imu_stationary_mean_residual=corrected.mean(axis=0)-stationary['reference'],
                imu_operating_feature=np.r_[1.,stationary['temperature_c'].mean()-25.,stationary['reference']],
                rf_phase_holdout_rms_rad=np.sqrt(np.mean(rf_test['phase_residual_rad']**2,axis=(0,1))),
                rf_gain_holdout_rms_db=np.sqrt(np.mean(rf_test['gain_residual_db']**2,axis=(0,1))),
                rf_holdout_valid=bool(rf_test['calibration_valid'].all()),
                rf_outside_valid_count=int(outside['calibration_valid'].sum()),
                imu_holdout_mean=imu_residual.mean(axis=0),imu_holdout_rms=np.sqrt(np.mean(imu_residual**2,axis=0)),
                sync_profile=sync)


def motion_audit():
    edges=np.arange(3001)*.02;zero=np.zeros(3);lever=np.array([.07,.03,.01]);rows=[];curves={}
    covariance=np.diag(np.array([.02]*3+[2e-4]*3)**2);covariance[0,1]=covariance[1,0]=.00016
    for motion in ('static','straight','turn'):
        initial=trajectory(np.array([0.,-.01]),motion);truth=trajectory(edges,motion)
        for case in ('ideal','accel_bias','gyro_bias','initial_velocity','initial_tilt','time_offset','lever_ignored','axis_swap','correlated_noise','combined'):
            bias=np.zeros(6);velocity=initial['velocity_mps'][0].copy();rotation=initial['rotation'][0].as_matrix()
            noise=np.zeros((6,6));offset=0.;axis=np.eye(6)
            if case in ('accel_bias','combined'):bias[0]=.02
            if case in ('gyro_bias','combined'):bias[3]=np.deg2rad(.01)
            if case in ('initial_velocity','combined'):velocity[0]+=.2
            if case in ('initial_tilt','combined'):rotation=Rotation.from_euler('y',.2,degrees=True).as_matrix()@rotation
            if case in ('time_offset','combined'):offset=.02
            if case in ('correlated_noise','combined'):noise=covariance
            if case=='axis_swap':axis[[0,1]]=axis[[1,0]]
            obs=measurements(edges,motion,lever_arm_m=lever,gravity_mps2=G,time_offset_s=offset,
                             bias=bias,axis_matrix=axis,noise_covariance=noise,correlation=.8,seed=441)
            previous=trajectory(np.array([-.01-offset]),motion)['rate_rad_s'][0]+bias[3:]
            result=propagate(edges,obs['force_mps2'],obs['rate_rad_s'],position_m=initial['position_m'][0],
                             velocity_mps=velocity,body_to_nav=rotation,accel_bias_mps2=zero,gyro_bias_rad_s=zero,
                             gravity_mps2=G,lever_arm_m=zero if case=='lever_ignored' else lever,
                             previous_rate_rad_s=previous)
            horizontal=np.linalg.norm((result['position_m']-truth['position_m'])[:,:2],axis=1)
            curves[motion+'_'+case]=horizontal
            rows.append(dict(motion=motion,case=case,error_10s_m=float(horizontal[500]),
                             error_30s_m=float(horizontal[1500]),error_60s_m=float(horizontal[3000]),
                             maximum_first_10s_m=float(horizontal[:501].max()),
                             maximum_first_30s_m=float(horizontal[:1501].max()),
                             maximum_first_60s_m=float(horizontal.max())))
    return rows,dict(time_s=edges,**curves)


def budget_audit(bench):
    # Explicit assumed initial/fixture uncertainty, NOT inferred from missing hardware.
    sigma=np.array([3.,3.,5.,.2,.2,.3,*np.deg2rad([.2,.2,1.]),.002,.002,.002,*np.deg2rad([.005]*3)])
    initial_cov=np.diag(sigma**2)
    imu=bench['imu_model'];inverse=np.linalg.inv(imu['matrix'])
    feature=bench['imu_operating_feature']
    bias_cov=inverse@((feature@imu['feature_covariance']@feature)*imu['residual_covariance'])@inverse.T
    initial_cov[9:15,9:15]+=bias_cov
    mean=np.zeros(15);mean[9:15]=bench['imu_stationary_mean_residual']
    times=np.arange(601)*.1
    rows=static_ins_budget(times,mean,initial_cov,bench['imu_noise_intensity'])
    rng=np.random.default_rng(4600);checks=[]
    for index in (100,300,600):
        row=rows[index]
        samples=rng.multivariate_normal(row['mean'][:2],row['covariance'][:2,:2],20000)
        checks.append(dict(time_s=row['time_s'],horizontal_radius95_m=row['horizontal_radius95_m'],
                           gaussian_snapshot_coverage=float((np.linalg.norm(samples,axis=1)<=row['horizontal_radius95_m']).mean())))
    channel_cov=np.zeros((4,4));channel_cov[1:,1:]=bench['rf_model']['residual_covariance']*(W/(2*np.pi))**2
    common_cov=np.kron(channel_cov,np.ones((8,8)))
    carrier_mean=np.array([0.,.03,-.02,.01])[:,None]*np.ones((1,8))
    common=spatial_difference_budget(carrier_mean,common_cov)
    prn_dependent=spatial_difference_budget(carrier_mean,common_cov+np.eye(32)*.003**2)
    return dict(scope='conditional level/static Gaussian sensitivity, NOT a trusted upper bound',
                assumed_initial_sigma=sigma,initial_mean=mean,initial_covariance=initial_cov,
                snapshots=checks,time_s=times,radius95_m=[r['horizontal_radius95_m'] for r in rows],
                first_over_20m_s=next((r['time_s'] for r in rows if r['horizontal_radius95_m']>20),None),
                first_over_50m_s=next((r['time_s'] for r in rows if r['horizontal_radius95_m']>50),None),
                common_hardware=common,with_assumed_3mm_prn_error=prn_dependent,
                release_status='not_evaluated_without_hardware_and_dynamic_validation',
                omitted=['Earth rotation and curvature','nonlinear attitude/large errors','IMU scale drift outside bench domain',
                         'RF multipath/body effects and coherence faults','RF/IMU shared-clock cross covariance',
                         'non-Gaussian tails','initial-state authenticity'])


def detection_audit():
    from audit_spatial_diversity import model_observations,wilson
    from spatial_spoofing import spatial_scores,calibrate_threshold
    n=2000;layouts={2:[0,1],3:[0,1,2],4:[0,1,2,3]};rows=[]
    calibration=model_observations(4610,n,'los')
    thresholds={k:calibrate_threshold(spatial_scores(calibration[:,:,ids],W)['cluster3']) for k,ids in layouts.items()}
    for case in ('los','all_spoof','partial_3'):
        base=model_observations(4620,n,case)
        for error in ('none','channel_common_30mm','prn_independent_20mm'):
            rng=np.random.default_rng(4630)
            perturb={'none':np.zeros((n,1,4,8)),
                     'channel_common_30mm':rng.normal(0,.03,(n,1,4,1)),
                     'prn_independent_20mm':rng.normal(0,.02,(n,1,4,8))}[error]
            for count,ids in layouts.items():
                scores=spatial_scores((base+perturb)[:,:,ids],W)['cluster3'];hits=int((scores<thresholds[count]).sum())
                rows.append(dict(case=case,error=error,channels=count,hits=hits,events=n,
                                 fraction=hits/n,wilson95=wilson(hits,n),threshold=thresholds[count]))
    return dict(scope='assumed error structure sensitivity; not measured device detection',rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bench-dir',type=Path,help='directory with explicit manifest.json and the seven documented NPZ inputs')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    def save(name,data):
        (out/name).write_text(json.dumps(serializable(data),indent=2,allow_nan=False),encoding='utf-8')
    if args.bench_dir is None:
        data,manifest=synthetic_bench()
    else:
        manifest=json.loads((args.bench_dir/'manifest.json').read_text(encoding='utf-8'))
        data={key:dict(np.load(args.bench_dir/(key+'.npz'))) for key in
              ('rf_train','rf_holdout','rf_outside','imu_train','imu_holdout','imu_stationary','imu_sync')}
    (out/'bench_input').mkdir()
    for key,value in data.items():np.savez_compressed(out/'bench_input'/(key+'.npz'),**value)
    save('bench_input/manifest.json',manifest)
    save('manifest.json',dict(bench_provenance=manifest,hardware_test_performed=False,
                             source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                            [Path(__file__),*(Path(__file__).with_name(n) for n in ('inertial.py','bench_calibration.py','error_budget.py'))]}))
    bench=analyze_bench(data);save('bench_result.json',bench);print('W5 numeric workflow done',flush=True)
    motion,curves=motion_audit();save('imu_summary.json',motion);np.savez_compressed(out/'imu_curves.npz',**curves)
    print('W4 motion propagation done',flush=True)
    budget=budget_audit(bench);save('budget.json',budget)
    detector=detection_audit();save('detection_sensitivity.json',detector)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for case in ('ideal','accel_bias','gyro_bias','initial_tilt','combined'):
        axes[0].plot(curves['time_s'],curves['static_'+case],label=case)
    axes[0].set(title='Synthetic static inertial drift',xlabel='Time (s)',ylabel='Horizontal error (m)')
    axes[0].legend();axes[0].grid()
    axes[1].plot(budget['time_s'],budget['radius95_m'],label='Conditional Gaussian radius')
    axes[1].axhline(20,color='orange',linestyle='--');axes[1].axhline(50,color='red',linestyle='--')
    axes[1].set(title='Assumed initial/fixture uncertainties',xlabel='Time (s)',ylabel='Radius (m)');axes[1].grid()
    fig.savefig(out/'imu_budget.png',dpi=150);plt.close(fig)
    save('sha256.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})
    print(json.dumps(dict(imu_cases=len(motion),bench_source=manifest['source_kind'],snapshots=budget['snapshots'],
                          first_over_20m_s=budget['first_over_20m_s'],first_over_50m_s=budget['first_over_50m_s']),indent=2))


if __name__=='__main__':main()
