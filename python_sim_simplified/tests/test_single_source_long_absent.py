"""A new absent IQ must be paired before one saved-observation replay."""
import hashlib
import json
import zipfile

import numpy as np
import pytest

from core.single_source import navigate
from experiments import single_source_long_absent as study
from experiments.single_source_long_absent import replay_three, verify_pair
from test_single_source_phase_state import fixed_code_geometry, observations
from test_single_source_phase_state_replay import saved_short


def small_pair():
    case=dict(duration=20.,fake_rate=3.,genuine=True,angle=5.)
    weak_metadata=dict(seed=7,scenario='weak',attacked=True,
                       rf_sha256='weak',case=case)
    control_metadata=dict(seed=7,scenario='no_attack_control',attacked=False,
                          rf_sha256='control',case=case)
    absent_metadata=dict(seed=7,scenario='genuine_absent',attacked=True,
                         rf_sha256='absent',case={**case,'genuine':False})
    weak=dict(samples=np.ones((2,10),complex),
              geometry=np.arange(3),times=np.array([.1,.2]))
    control={**weak,'samples':weak['samples'].copy()}
    absent={**weak,'samples':weak['samples'].copy()}
    absent['samples'][:,5:]=2.
    return absent,absent_metadata,((weak,weak_metadata),(control,control_metadata))


def test_only_genuine_flag_may_differ_from_paired_weak_model():
    absent,metadata,references=small_pair()
    verify_pair(absent,metadata,references,onset_sample=5)
    absent['geometry']=np.array([0,1,4])
    with pytest.raises(AssertionError):
        verify_pair(absent,metadata,references,onset_sample=5)
    absent,metadata,references=small_pair()
    metadata['case']['angle']=6.
    with pytest.raises(AssertionError):
        verify_pair(absent,metadata,references,onset_sample=5)


def test_attack_sample_may_change_but_one_sample_before_must_match_both_controls():
    absent,metadata,references=small_pair()
    absent['samples'][:,4]=3.
    with pytest.raises(AssertionError):
        verify_pair(absent,metadata,references,onset_sample=5)


def test_control_case_must_match_weak_case():
    absent,metadata,references=small_pair()
    references[1][1]['case']={**references[1][1]['case'],'fake_rate':4.}
    with pytest.raises(AssertionError):
        verify_pair(absent,metadata,references,onset_sample=5)


def test_preflight_rejects_snapshot_with_wrong_source_bytes(tmp_path,monkeypatch):
    stage37=tmp_path/'stage37'
    stage37.mkdir()
    (stage37/'manifest.json').write_text('{}',encoding='utf-8')
    (stage37/'overview.png').write_bytes(b'figure')
    with zipfile.ZipFile(stage37/'source_snapshot.zip','w') as archive:
        archive.writestr('core/check.py',b'wrong source')
    expected=hashlib.sha256(b'expected source').hexdigest()
    summary=dict(manifest_sha256=study.digest(stage37/'manifest.json'),
                 source_snapshot_sha256=study.digest(stage37/'source_snapshot.zip'),
                 figure_sha256=study.digest(stage37/'overview.png'),
                 source_sha256={'core/check.py':expected})
    (stage37/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    history={'long':{'cases':{'weak':{'input_sha256':{}},
                               'no_attack_control':{'input_sha256':{}}}}}
    monkeypatch.setattr(study,'STAGE37',stage37)
    monkeypatch.setattr(study,'preflight_stage37',lambda:(history,{},summary['source_sha256']))
    with pytest.raises(AssertionError):
        study.preflight()


def test_three_nav_states_check_saved_baseline_and_gate_independently(monkeypatch):
    fixed_code_geometry(monkeypatch)
    z=observations()
    old=navigate(z,diagnostics=True,code_blend=0.,phase_leave_one_out=True)
    mask=np.zeros(len(z['time']),bool)
    baseline,open_state,gated=replay_three(z,saved_short(old),mask)
    np.testing.assert_array_equal(baseline['position'],gated['position'])
    assert np.isnan(baseline['position'][27]).all()
    assert np.isfinite(open_state['position'][27]).all()
    saved=saved_short(old)
    saved['short_diagnostic_reason'][3]='wrong'
    with pytest.raises(AssertionError):
        replay_three(z,saved,mask)
