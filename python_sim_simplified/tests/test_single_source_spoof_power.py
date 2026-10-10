"""Fixed spoof-power pair and observability limits, independent of local RF."""
from copy import deepcopy
import json

import numpy as np
import pytest

from core import single_source_scene
from core.recovery_geometry import FS
from core.serialization import serializable
from experiments.single_source_history_study import generate_new_record
from experiments.single_source_spoof_power import (
    cached_positions,compare_power,observation_quality,
)


@pytest.mark.parametrize('scenario',('weak','genuine_absent'))
def test_only_spoof_power_changes_attacked_samples_not_rng_or_clean_prefix(
        monkeypatch,scenario):
    plan=dict(seed=2026100301,duration_s=.25,satellites=2,
              startup_cn0_db_hz=45.,post_0p3s_cn0_db_hz=30.,
              attack_start_s=.15,spoof_db=0.,angle_deg=5.,fake_east_m=100.)
    original=np.random.default_rng
    generators=[]
    def tracked(seed):
        generator=original(seed)
        generators.append(generator)
        return generator
    monkeypatch.setattr(single_source_scene.np.random,'default_rng',tracked)
    zero,zero_identity=generate_new_record(plan,scenario)
    zero_state=[deepcopy(item.bit_generator.state) for item in generators]
    generators.clear()
    plan['spoof_db']=10.
    ten,ten_identity=generate_new_record(plan,scenario)
    ten_state=[deepcopy(item.bit_generator.state) for item in generators]
    assert zero_state==ten_state and len(zero_state)==3
    start=round(plan['attack_start_s']*FS)
    np.testing.assert_array_equal(zero['samples'][:,:start],ten['samples'][:,:start])
    assert np.any(zero['samples'][:,start:]!=ten['samples'][:,start:])
    for field in ('nominal','codes','times','satellites','coarse'):
        np.testing.assert_array_equal(zero[field],ten[field])
    assert {key:value for key,value in zero_identity.items()
            if key not in ('case','rf_sha256')}=={
                key:value for key,value in ten_identity.items()
                if key not in ('case','rf_sha256')}
    assert {key:value for key,value in zero_identity['case'].items()
            if key!='spoof_db'}=={
                key:value for key,value in ten_identity['case'].items()
                if key!='spoof_db'}


def test_absent_steady_without_finite_observations_is_not_authentication():
    tracking=dict(time=np.array([.27,.49,.59]),
                  available_time=np.array([.29,.51,.61]),
                  ranges=np.array([[1.,1.,1.,1.],[np.nan]*4,[np.nan]*4]),
                  lock=np.array([[1.]*4,[np.nan]*4,[np.nan]*4]),
                  cn0=np.array([[30.]*4,[np.nan]*4,[np.nan]*4]),
                  gain=np.array([[1.]*4,[0.]*4,[0.]*4]),
                  support=np.array([[4.]*4,[np.nan]*4,[np.nan]*4]))
    navigation=dict(selected=np.array([[True]*4,[False]*4,[False]*4]),
                    position=np.array([[0.,0.,0.],[np.nan]*3,[np.nan]*3]),
                    diagnostics=dict(phase_ok=np.array([False,False,False])))
    quality=observation_quality(tracking,navigation,.3)
    assert quality['steady_opportunities']==2
    assert quality['steady_finite_ge4']==0
    assert quality['steady_quality_ge4']==0
    assert quality['steady_support_ge4']==0
    assert quality['steady_selected_ge4']==0
    assert quality['steady_published']==0
    assert quality['steady_published_phase_ok']==0


def test_cached_path_first_positions_compare_and_summary_json():
    times=np.array([.25,.35,.45])
    tracking=dict(time=times,available_time=times+.02)
    baseline=np.array([[0.,0.,0.],[np.nan]*3,[.9,0.,0.]])
    candidate=np.array([[0.,0.,0.],[.7,0.,0.],[.8,0.,0.]])
    saved={f'baseline_{mode}_position':baseline for mode in ('original','short')}
    saved.update({f'candidate_{mode}_position':candidate
                  for mode in ('original','short')})
    older=cached_positions(saved,'baseline')
    fresh={mode:dict(position=saved[f'candidate_{mode}_position'])
           for mode in ('original','short')}
    compared=compare_power(fresh,older,tracking,
                           dict(velocity_east=2.,attack_start=.3))
    assert compared['short']['common_published']==1
    assert compared['short']['candidate_only_published']==1
    document=dict(cases=dict(weak=dict(zero_ten_db_same_mode=compared,
                                       paths=dict(history_only_blocks=np.array([388])))))
    encoded=json.dumps(serializable(document),allow_nan=False)
    assert json.loads(encoded)['cases']['weak']['paths']['history_only_blocks']==[388]


def test_finite_steady_observation_without_publication_is_distinct_from_admission():
    tracking=dict(available_time=np.array([.51,.61]),
                  ranges=np.ones((2,6)),lock=np.ones((2,6)),
                  cn0=np.full((2,6),30.),gain=np.ones((2,6)),
                  support=np.full((2,6),4.))
    navigation=dict(selected=np.array([[True]*6,[False]*6]),
                    position=np.array([[np.nan]*3,[1.,0.,0.]]),
                    diagnostics=dict(phase_ok=np.array([False,True])))
    quality=observation_quality(tracking,navigation,.3)
    assert quality['steady_finite_ge6']==2
    assert quality['steady_quality_ge6']==2
    assert quality['steady_support_ge6']==2
    assert quality['steady_selected_ge6']==1
    assert quality['steady_published']==1
    assert quality['steady_published_phase_ok']==1
