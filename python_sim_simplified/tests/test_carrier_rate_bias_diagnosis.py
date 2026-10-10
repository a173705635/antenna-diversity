import numpy as np
import pytest

from tools import carrier_rate_bias_diagnosis as diagnosis


def test_final_mask_uses_actual_lag_and_exclusion():
    z = dict(carrier=np.ones((3, 6)), arc=np.ones((3, 6), int))
    valid = np.ones((3, 6), bool)
    valid[1, 0] = False
    z['arc'][0, 1] = 2
    details = dict(lag_epochs=2, excluded_prn=3,
                   candidate_common_count=4,
                   full_common_count=np.array([0, 5]))
    mask = diagnosis.final_bridge_mask(z, valid, 2, details)
    np.testing.assert_array_equal(mask, [True, False, False,
                                         True, True, True])
    assert valid[2].all()
    details.update(lag_epochs=1, excluded_prn=0,
                   full_common_count=np.array([5, 4]))
    lag_one = diagnosis.final_bridge_mask(z, valid, 2, details)
    assert not lag_one[0]
    assert lag_one[1]


def test_training_requires_connected_prn_support_rank():
    def step(stars, values):
        mask = np.zeros(4, bool)
        mask[stars] = True
        return dict(mask=mask, b=np.array(values, float))

    disconnected = [step([0, 1], [3., 1.]),
                    step([2, 3], [4., 2.])]
    report, beta = diagnosis.fit_fixed_bias(disconnected, 4)
    assert report['status'] == 'unidentifiable'
    assert report['design_rank'] == 2
    assert report['required_rank'] == 3
    assert beta is None

    connected = [step([0, 1], [3., 1.]),
                 step([1, 2], [4., 2.])]
    report, beta = diagnosis.fit_fixed_bias(connected, 4)
    assert report['status'] == 'identified'
    assert report['design_rank'] == report['required_rank'] == 2
    assert beta[:3].sum() == pytest.approx(0., abs=1e-12)
    assert np.isnan(beta[3])
    assert report['beta_mps'][3] is None


def test_validation_keeps_new_prn_unknown_and_lag_two_parent_once():
    h = np.array([[1., 0., 0., 1.], [0., 1., 0., 1.],
                  [0., 0., 1., 1.], [-1., 0., 0., 1.],
                  [0., -1., 0., 1.]])
    beta = np.array([.1, -.2, .15, -.05, .0, np.nan])
    assert beta[:5].sum() == pytest.approx(0.)
    mask = np.array([True]*5+[False])
    dt = .02
    observed = dt*(np.linalg.pinv(h)@beta[:5])[:2]
    parent = np.zeros(3)
    first_prediction = parent + [observed[0], observed[1], 0.]
    first = dict(k=2, parent=0, lag=2, mask=mask, h=h, dt=dt,
                 b=beta[:5], observed_en=observed,
                 parent_position=parent, prediction=first_prediction)
    second = dict(first, k=4, parent=2,
                  parent_position=first_prediction,
                  prediction=first_prediction+[observed[0], observed[1], 0.])
    new_mask = np.array([True, True, True, True, False, True])
    unknown = dict(first, k=5, parent=4, mask=new_mask)
    time = np.array([9.99, 10., 10.01, 10.02, 10.03, 10.04])
    report, detail = diagnosis.evaluate_bridges(
        [first, second, unknown], beta, time)
    assert report['validation_bridges'] == 3
    assert report['supported_bridges'] == 2
    assert report['new_prn_bridges'] == 1
    assert report['lag2_bridges'] == 2
    assert report['reanchored_segments'] == 1
    assert report['max_k_en_ones'] < 1e-10
    np.testing.assert_allclose(detail['cumulative_observed_en_m'][-1], 2*observed)
    np.testing.assert_allclose(detail['predicted_en_m'], [observed, observed])
    np.testing.assert_allclose(np.nansum(detail['per_prn_en_m'], axis=1),
                               detail['predicted_en_m'])
    np.testing.assert_allclose(detail['bridge_sse_predicted_m2'], 0., atol=1e-20)
    assert detail['reanchor_reason'].tolist() == [
        'missing_parent_bridge', 'continued']
    np.testing.assert_array_equal(detail['new_prn_k'], [5])


def test_no_late_bridge_is_no_validation_opportunity():
    report, detail = diagnosis.evaluate_bridges([], np.zeros(6), np.array([0., 20.]))
    assert report == dict(status='no_validation_bridge', validation_bridges=0)
    assert detail['k'].size == 0


@pytest.mark.local_evidence
def test_formal_continuity_source_matches_recorded_sha():
    import json
    from pathlib import Path

    formal_path = (diagnosis.study.OUTPUT
                   /'single_path_carrier_continuity_summary.json')
    assert diagnosis.digest(formal_path) == diagnosis.CONTINUITY_SUMMARY_SHA256
    formal = json.loads(formal_path.read_text(encoding='utf-8'))
    source = Path(diagnosis.__file__).resolve().parents[1]
    assert diagnosis.digest(source/'tools/single_path_carrier_continuity.py') == (
        formal['source_sha256'])


@pytest.mark.local_evidence
def test_saved_path_records_actual_final_bridge_and_formal_replay():
    import json

    frozen = json.loads((diagnosis.study.OUTPUT/'summary.json').read_text())
    row = frozen['cases']['2026100404']['weak']
    folder = diagnosis.study.OUTPUT/'2026100404'/'weak'/'gate3'
    array_path = diagnosis.study.OUTPUT/'single_path_carrier_continuity_2026100404_weak.npz'
    result, detail = diagnosis.analyze_path(
        folder, row['paths']['gate3']['output_sha256'],
        (array_path, diagnosis.digest(array_path)), 2.)
    assert result['fit']['training_bridges'] > 0
    assert result['baseline']['validation_bridges'] > 0
    assert result['candidate']['validation_bridges'] > 0
    assert detail['baseline']['mask'].shape[1] == 24
    json.dumps(result, allow_nan=False)
