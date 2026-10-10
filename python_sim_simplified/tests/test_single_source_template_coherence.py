"""Template-rate and five-millisecond trigger factors without local IQ."""
import numpy as np

from core.single_source import residual_trigger
import experiments.single_source_template_coherence as diagnosis
from experiments.single_source_template_coherence import (
    aided_template,group_five_ms,projected_power_retention,trigger_fit,
    phase_masks,truth_code_phases,wrapped_code_error,
)


def test_five_ms_groups_same_prn_complex_prompts_without_bit_knowledge():
    prompts=np.zeros((3,2,20),complex)
    prompts[:,0,:5]=np.array([1+1j,2-1j,3j])[:,None]
    prompts[:,0,5:10]=-prompts[:,0,:5]
    prompts[:,1,:5]=np.array([2.,1j,-1.])[:,None]
    prompts[:,1,3:5]=-prompts[:,1,:2]
    grouped=group_five_ms(prompts)
    assert grouped.shape==(3,8)
    np.testing.assert_allclose(grouped[:,0],np.sqrt(5)*prompts[:,0,0],atol=1e-14)
    np.testing.assert_allclose(grouped[:,1],-np.sqrt(5)*prompts[:,0,0],atol=1e-14)
    np.testing.assert_allclose(grouped[:,4],prompts[:,1,0]/np.sqrt(5),atol=1e-14)


def test_aided_template_changes_only_rate_from_startup_carrier():
    template=(.0995,123.25,1023100.,-1200.)
    aided=aided_template(template)
    assert aided[0]==template[0] and aided[1]==template[1]
    assert aided[3]==template[3]
    assert aided[2]==1.023e6*(1-1200./1575.42e6)
    assert wrapped_code_error(1024.,1.)==0.
    assert wrapped_code_error(.25,1022.75)==.5


def test_trigger_fit_uses_actual_group_column_count_and_current_projector():
    rng=np.random.default_rng(37)
    prompts=rng.normal(size=(3,2,20))+1j*rng.normal(size=(3,2,20))
    desired=np.exp(1j*rng.normal(size=(3,2)))
    one=prompts.reshape(3,-1)
    five=group_five_ms(prompts)
    for data in (one,five):
        repeated=np.repeat(desired,data.shape[1]//2,axis=1)
        fitted,raw,would,gain=trigger_fit(data,repeated,desired,noise=.8)
        reference,expected=residual_trigger(data,repeated,.8)
        np.testing.assert_allclose(fitted,reference,rtol=0,atol=1e-13)
        assert raw==expected
        initial=repeated*np.sum(repeated.conj()*data,axis=0)/3
        projected=fitted@repeated
        amplitude=np.sum(projected.conj()*data,axis=0)/(
            np.sum(repeated.conj()*projected,axis=0).real+.03)
        residual=fitted@(data-repeated*amplitude)
        reduction=np.sum(abs(data-initial)**2)-np.sum(abs(residual)**2)
        np.testing.assert_allclose(raw,reduction/(.8*(data.shape[1]+2)),
                                   rtol=0,atol=1e-12)
        np.testing.assert_allclose(would,reference if raw>=3 else np.eye(3))
        np.testing.assert_allclose(gain,
            np.einsum('mi,mn,ni->i',desired.conj(),would,desired).real/3)
    assert one.shape[1]==40 and five.shape[1]==8


def test_truth_only_projection_retention_distinguishes_fake_and_genuine():
    fake=np.array([1.,1.,0j])/np.sqrt(2)
    genuine=np.column_stack((fake,np.array([1.,-1.,0j])/np.sqrt(2)))
    projector=np.eye(3)-np.outer(fake,fake.conj())
    assert abs(projected_power_retention(projector,fake))<1e-14
    np.testing.assert_allclose(projected_power_retention(projector,genuine),
                               [0.,1.],atol=1e-14)
    np.testing.assert_allclose(projected_power_retention(np.eye(3),genuine),
                               [1.,1.])


def test_truth_phase_audit_distinguishes_offset_only_after_geometry():
    case=dict(stars=2,doppler_scale=0.,doppler_rate=0.,velocity_east=0.,
              fake_east=0.,fake_rate=0.,attack_start=.3)
    genuine,fake=truth_code_phases(case,2026100301,1.)
    np.testing.assert_array_equal(genuine,fake)
    case['fake_east']=100.
    genuine2,fake2=truth_code_phases(case,2026100301,1.)
    np.testing.assert_array_equal(genuine,genuine2)
    assert np.any(abs(wrapped_code_error(fake2,genuine2))>.01)


def test_first_five_blocks_unmeasured_and_future_block_cannot_change_past(monkeypatch):
    rng=np.random.default_rng(17)
    width=diagnosis.WIDTH
    samples=rng.normal(size=(3,7*width))+1j*rng.normal(size=(3,7*width))
    nominal=np.array([[1.,1.],[1.,-1.],[1.,1j]])
    def startup(_record,_tracking):
        return (np.array([0,1]),np.array([[0.,0.,1.023e6,0.],
                                         [0.,0.,1.023e6,0.]]),
                np.ones(3,complex),np.zeros(3))
    monkeypatch.setattr(diagnosis,'startup_templates',startup)
    monkeypatch.setattr(diagnosis,'local_signal',
                        lambda _code,_template,time:np.ones(len(time),complex))
    record=dict(samples=samples,nominal=nominal,codes=np.arange(2))
    tracking=dict(time=np.array([.09]),ratio=np.array([0.]))
    full=diagnosis.fit_saved_iq(record,tracking)
    prefix=diagnosis.fit_saved_iq({**record,'samples':samples[:,:6*width]},tracking)
    assert np.isnan(full['raw'][:,:5]).all()
    assert np.isfinite(full['raw'][:,5]).all()
    np.testing.assert_array_equal(full['raw'][:,5],prefix['raw'][:,5])
    np.testing.assert_array_equal(full['would_projector'][:,5],
                                  prefix['would_projector'][:,5])
    aided_tracking=dict(time=np.array([.11]),ratio=np.array([full['raw'][3,5]]))
    diagnosis.fit_saved_iq(record,aided_tracking,reference_variant=3)
    with np.testing.assert_raises(AssertionError):
        diagnosis.fit_saved_iq(record,{**aided_tracking,'ratio':
            aided_tracking['ratio']+1.},reference_variant=3)


def test_phase_masks_count_actual_evaluable_blocks_without_startup_hindsight():
    starts=np.arange(400)*.02
    masks=phase_masks(starts,.3)
    assert [int(masks[key].sum()) for key in ('clean','post','steady')]==[
        10,385,375]
    assert not any(mask[:5].any() for mask in masks.values())
