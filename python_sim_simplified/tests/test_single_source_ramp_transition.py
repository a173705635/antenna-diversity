import json

import numpy as np
import pytest

from experiments import single_source_ramp_transition as transition
from experiments.single_source_ramp_transition import (
    TRANSITION_PLAN,interpolation_alpha,verify_interpolated_samples,
    verify_power_metadata,trigger_counts,
)
from experiments.single_source_history_study import generate_new_record


def test_fixed_power_interpolation_checks_every_sample():
    rng=np.random.default_rng(42)
    true=rng.normal(size=(3,407))+1j*rng.normal(size=(3,407))
    forged=rng.normal(size=(3,407))+1j*rng.normal(size=(3,407))
    low=true+10**(-5/20)*forged
    high=true+forged
    middle=true+10**(-4/20)*forged
    alpha=interpolation_alpha()
    assert alpha==(10**(-4/20)-10**(-5/20))/(1-10**(-5/20))
    assert verify_interpolated_samples(middle,low,high,73)<1e-12
    changed=middle.copy()
    changed[2,393]+=.001
    with pytest.raises(AssertionError):
        verify_interpolated_samples(changed,low,high,73)


def test_actual_generator_changes_only_spoof_power():
    plan={**TRANSITION_PLAN,'duration_s':.08,'attack_start_s':.03,
          'satellites':4}
    old_low,low_identity=generate_new_record({**plan,'spoof_db':-5.},'weak',
                                               fake_rate=plan['fake_rate_mps'])
    middle,middle_identity=generate_new_record(plan,'weak',
                                                fake_rate=plan['fake_rate_mps'])
    old_high,high_identity=generate_new_record({**plan,'spoof_db':0.},'weak',
                                                fake_rate=plan['fake_rate_mps'])
    onset=round(.03*2.046e6)
    verify_power_metadata(middle,middle_identity,old_low,low_identity,onset)
    verify_power_metadata(middle,middle_identity,old_high,high_identity,onset)
    assert verify_interpolated_samples(middle['samples'],old_low['samples'],
                                       old_high['samples'],791)<1e-11


def test_trigger_counts_keep_startup_and_attack_denominators_distinct():
    raw=np.array([np.nan]*5+[4.]*10+[2.]*985)
    result=trigger_counts(raw,.3)
    assert result['post_startup']==dict(triggered=10,opportunities=995)
    assert result['post_attack']==dict(triggered=0,opportunities=985)


def test_main_writes_plan_before_any_rf_generation(tmp_path,monkeypatch):
    class GeneratorReached(Exception):
        pass

    monkeypatch.setattr(transition,'OUTPUT',tmp_path/'transition')
    monkeypatch.setattr(transition,'preflight',lambda:({},{}))
    monkeypatch.setattr(transition,'source_snapshot',lambda *args:None)
    monkeypatch.setattr(transition,'digest',lambda *args:'test-sha')
    monkeypatch.setattr(transition,'generate_new_record',
                        lambda *args,**kwargs:(_ for _ in ()).throw(GeneratorReached))
    with pytest.raises(GeneratorReached):
        transition.main()
    plan=json.loads((transition.OUTPUT/'plan.json').read_text(encoding='utf-8'))
    assert plan['spoof_db']==-4.
    assert plan['receiver_pll']==[.15,1.]
    assert plan['beam_coherent_ms']==5
