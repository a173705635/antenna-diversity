"""Boundaries for the optional, geometry checked history extension."""
import numpy as np

from core.gps_simulation import receiver_reference
from core.recovery_geometry import sky
from core.mechanism_smoothing import smooth_history
from core.mechanism_phase import fit_carrier_displacement


def record(times, count=6):
    reference,rotation=receiver_reference()
    directions=sky(8102,count)
    satellites=reference+np.tile((directions*21e6)@rotation,(len(times),1,1))
    shape=(len(times),count)
    return dict(time=np.array(times),satellites=satellites,
                ranges=np.zeros(shape),carrier=np.zeros(shape),arc=np.zeros(shape,int))


def test_long_gap_requires_joint_carrier_motion_and_preserves_default():
    z=record([0.,.09]);eligible=np.ones((2,6),bool)
    reference,rotation=receiver_reference()
    directions=(z['satellites'][1]-reference)@rotation.T/21e6
    # Two metres east motion and three metres receiver clock change.
    change=-(directions@np.array([2.,0.,0.]))+3.
    z['carrier'][1]=change;z['ranges'][1]=change
    default,default_age=smooth_history(z,eligible)
    extended,extended_age=smooth_history(z,eligible,history_gap_s=.105)
    assert np.all(default_age[1]==1)
    assert np.all(extended_age[1]==2)
    np.testing.assert_allclose(extended[1],change,atol=1e-8)


def test_single_satellite_25m_carrier_jump_rejects_long_gap():
    times=[k*.01 for k in range(20)]+[.28]
    z=record(times);eligible=np.ones((len(times),6),bool)
    z['carrier'][-1,0]=25.
    smoothed,ages=smooth_history(z,eligible,history_gap_s=.105)
    assert ages[-2,0]==20
    assert np.all(ages[-1]==1)
    assert smoothed[-1,0]==0.
    # Blind retention at capped age 20 would output .95 * 25 = 23.75 m.


def test_code_carrier_50m_limit_is_unchanged():
    z=record([0.,.09]);eligible=np.ones((2,6),bool)
    reference,rotation=receiver_reference()
    directions=(z['satellites'][1]-reference)@rotation.T/21e6
    z['carrier'][1]=-(directions@np.array([60.,0.,0.]))
    z['ranges'][1]=z['carrier'][1]
    z['ranges'][1,0]+=50.1
    _,ages=smooth_history(z,eligible,history_gap_s=.105)
    assert ages[1,0]==1
    assert np.all(ages[1,1:]==2)


def test_each_satellite_uses_its_own_last_accepted_epoch():
    z=record([0.,.01,.1]);eligible=np.zeros((3,6),bool)
    eligible[0,:5]=True;eligible[1,5]=True;eligible[2]=True
    _,ages=smooth_history(z,eligible,history_gap_s=.105)
    np.testing.assert_array_equal(ages[2],[2,2,2,2,2,1])


def test_recently_accepted_satellites_can_witness_older_target_gap():
    z=record([0.,.08,.1]);eligible=np.ones((3,6),bool)
    eligible[1,0]=False
    _,ages=smooth_history(z,eligible,history_gap_s=.105)
    assert ages[2,0]==2
    assert np.all(ages[2,1:]==3)


def test_arc_count_and_gap_boundary():
    z=record([0.,.105]);eligible=np.ones((2,6),bool)
    _,ages=smooth_history(z,eligible,history_gap_s=.105)
    assert np.all(ages[1]==2)
    z['time'][1]=.1051
    _,ages=smooth_history(z,eligible,history_gap_s=.105)
    assert np.all(ages[1]==1)
    z['time'][1]=.09;z['arc'][1,0]=1
    _,ages=smooth_history(z,eligible,history_gap_s=.105)
    assert ages[1,0]==1 and np.all(ages[1,1:]==2)
    eligible[1,1:]=False
    _,ages=smooth_history(z,eligible,history_gap_s=.105)
    assert ages[1,0]==1


def test_phase_fit_rejects_condition_residual_and_horizontal_uncertainty():
    reference,rotation=receiver_reference()
    directions=sky(8102,6)
    def satellites(unit):
        return reference+(unit*21e6)@rotation
    broad=satellites(directions)
    outlier=np.zeros(6);outlier[0]=.5
    def gate_values(unit,change):
        h=np.column_stack((-unit,np.ones(len(unit))))
        fitted=np.linalg.lstsq(h,change,rcond=None)[0]
        residual=h@fitted-change
        variance=max(.02**2,float(residual@residual)/(len(unit)-4))
        horizontal_std=np.sqrt(np.linalg.eigvalsh(variance*np.linalg.inv(h.T@h)[:2,:2])[-1])
        return np.linalg.cond(h),max(abs(residual)),horizontal_std

    condition,residual,std=gate_values(directions,outlier)
    assert condition<1000 and residual>.1 and std<.5
    assert fit_carrier_displacement(broad,broad,outlier) is None
    az=np.arange(6)*np.pi/3
    el=np.pi/4+np.radians(np.array([0,1,-1,2,-2,.5]))*1e-4
    narrow_elevation=np.column_stack((np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)))
    condition,residual,std=gate_values(narrow_elevation,np.zeros(6))
    assert condition>1000 and residual<=.1 and std<.5
    points=satellites(narrow_elevation)
    assert fit_carrier_displacement(points,points,np.zeros(6)) is None

    weak_horizontal=np.array([1.,0.,0.])+.4*directions
    weak_horizontal/=np.linalg.norm(weak_horizontal,axis=1)[:,None]
    condition,residual,std=gate_values(weak_horizontal,np.zeros(6))
    assert condition<1000 and residual<=.1 and std>.5
    points=satellites(weak_horizontal)
    assert fit_carrier_displacement(points,points,np.zeros(6)) is None


def test_coherent_false_displacement_is_an_unresolved_boundary():
    z=record([0.,.09]);eligible=np.ones((2,6),bool)
    reference,rotation=receiver_reference()
    directions=(z['satellites'][1]-reference)@rotation.T/21e6
    forged=-(directions@np.array([25.,0.,0.]))
    z['carrier'][1]=forged;z['ranges'][1]=forged
    _,ages=smooth_history(z,eligible,history_gap_s=.105)
    assert np.all(ages[1]==2)
