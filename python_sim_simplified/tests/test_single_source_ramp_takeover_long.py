"""Long takeover scoring uses observed signal time and actual publication runs."""
import numpy as np
import pytest

import experiments.spoofing.single_source_ramp_takeover_long as study
from experiments.spoofing.single_source_ramp_takeover_long import long_tail_scores


def test_long_tail_masks_and_error_runs_use_observation_time():
    observation=np.array([6.96,6.97,10.29,10.30,16.96,16.97,16.98])
    tracking=dict(time=observation,available_time=observation+.02)
    position=np.column_stack((2*observation+np.array([0,20,31,32,51,52,np.nan]),
                              np.zeros(len(observation)),np.zeros(len(observation))))
    navigation=dict(position=position)
    case=dict(attack_start=.3,velocity_east=2.,fake_east=0.,fake_rate=3.)
    scores=long_tail_scores(tracking,navigation,case)
    assert [scores[f'fake_offset_ge_{limit}']['opportunities']
            for limit in (20,30,50)]==[6,4,2]
    assert [scores[f'fake_offset_ge_{limit}']['published']
            for limit in (20,30,50)]==[5,3,1]
    assert scores['observed_attack']['longest_error_run_epochs']['ge20']==5
    assert scores['observed_attack']['longest_error_run_epochs']['ge30']==4
    assert scores['observed_attack']['longest_error_run_epochs']['ge50']==2
    # A missing publication breaks a run even if adjacent published errors persist.
    navigation['position'][3]=np.nan
    broken=long_tail_scores(tracking,navigation,case)
    assert broken['observed_attack']['longest_error_run_epochs']['ge20']==2


def test_new_20s_pair_requires_exact_clean_iq_and_geometry(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'PLAN',{**study.PLAN,'attack_start_s':.003})
    onset=round(.003*study.FS)
    record=dict(samples=np.zeros((2,onset+5),complex),
                nominal=np.ones((2,1),complex),codes=np.ones((1,4)),
                times=np.array([.15]),satellites=np.zeros((1,1,3)),
                coarse=np.array([2.1e7]))
    identity=dict(seed=1,case=dict(attack_start=.003))
    (tmp_path/'weak').mkdir()
    (tmp_path/'control').mkdir()
    _,prefix,geometry=study.save_new_input(tmp_path/'weak',record,identity,
                                            None,None)
    pair={**record,'samples':record['samples'].copy()}
    pair['samples'][:,onset:]=2.
    study.save_new_input(tmp_path/'control',pair,identity,prefix,geometry)
    pair['samples'][0,onset-1]=1.
    with pytest.raises(AssertionError):
        study.save_new_input(tmp_path/'invalid',pair,identity,prefix,geometry)
