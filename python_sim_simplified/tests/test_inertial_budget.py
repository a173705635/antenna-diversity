import numpy as np
from scipy.spatial.transform import Rotation

from inertial import propagate, measurements, trajectory
from error_budget import static_ins_budget, spatial_difference_budget

G=np.array([0.,0.,-9.80665])


def integrate(edges, force, rate, *, velocity=np.zeros(3), rotation=np.eye(3), lever=np.zeros(3), previous=np.zeros(3)):
    return propagate(edges,force,rate,position_m=np.zeros(3),velocity_mps=velocity,
                     body_to_nav=rotation,accel_bias_mps2=np.zeros(3),gyro_bias_rad_s=np.zeros(3),
                     gravity_mps2=G,lever_arm_m=lever,previous_rate_rad_s=previous)


def test_static_accel_initial_velocity_and_tilt_have_expected_growth():
    edges=np.linspace(0,10,1001);n=len(edges)-1
    force=np.tile(-G,(n,1));rate=np.zeros_like(force)
    bias=integrate(edges,force+[.02,0,0],rate)
    np.testing.assert_allclose(bias['position_m'][-1],[1.,0,0],atol=1e-10)
    speed=integrate(edges,force,rate,velocity=np.array([.2,0,0]))
    np.testing.assert_allclose(speed['position_m'][-1],[2.,0,0],atol=1e-10)
    tilt=integrate(edges,force,rate,rotation=Rotation.from_euler('y',.01).as_matrix())
    np.testing.assert_allclose(tilt['position_m'][-1,0],9.80665*np.sin(.01)*50,atol=1e-10)


def test_gyro_bias_cubic_limit_matches_nonlinear_integrator():
    edges=np.linspace(0,10,2001);n=len(edges)-1;b=1e-4
    run=integrate(edges,np.tile(-G,(n,1)),np.tile([b,0.,0.],(n,1)),previous=np.array([b,0.,0.]))
    np.testing.assert_allclose(run['position_m'][-1,1],-9.80665*b*10**3/6,rtol=1e-6)


def test_analytic_turn_with_lever_converges_without_truth_in_integrator():
    errors=[]
    for dt in (.02,.01):
        edges=np.arange(round(10/dt)+1)*dt;lever=np.array([.07,.03,.01])
        obs=measurements(edges,'turn',lever_arm_m=lever,gravity_mps2=G,time_offset_s=0.,
                         bias=np.zeros(6),axis_matrix=np.eye(6),noise_covariance=np.zeros((6,6)),correlation=0.,seed=1)
        initial=trajectory(np.array([0.,-dt/2]),'turn')
        run=integrate(edges,obs['force_mps2'],obs['rate_rad_s'],velocity=initial['velocity_mps'][0],
                      lever=lever,previous=initial['rate_rad_s'][1])
        errors.append(np.linalg.norm(run['position_m'][-1]-trajectory(np.array([10.]),'turn')['position_m'][0]))
    assert errors[1]<errors[0] and errors[1]<.002


def test_linear_budget_matches_closed_form_and_retains_noise_correlations():
    mean=np.zeros(15);mean[9]=.02;mean[13]=1e-4
    noise=np.zeros((6,6));noise[0,0]=.001;noise[1,1]=.001;noise[0,1]=noise[1,0]=.0004
    row=static_ins_budget([10.],mean,np.zeros((15,15)),noise)[0]
    np.testing.assert_allclose(row['mean'][0],.02*100/2+9.80665*1e-4*1000/6)
    np.testing.assert_allclose(row['covariance'][0,1],.0004*1000/3)


def test_spatial_common_modes_cancel_but_shared_reference_covariance_remains():
    mean=np.array([0.,.03,-.02,.01])[:,None]*np.ones((1,8))
    result=spatial_difference_budget(mean,np.eye(32)*1e-6)
    np.testing.assert_allclose(result['dd_mean'],0.,atol=1e-15)
    np.testing.assert_allclose(np.diag(result['sd_covariance']),2e-6)
    np.testing.assert_allclose(np.diag(result['dd_covariance']),4e-6)
    np.testing.assert_allclose(result['sd_covariance'][0,8],1e-6)
    np.testing.assert_allclose(result['dd_covariance'][0,1],2e-6)
