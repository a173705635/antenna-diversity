"""Legacy numerical replay and truth separation, not defensive input tests."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from sensor_contract import adapt_legacy_rf


@pytest.fixture(scope='module')
def legacy_records():
    root = Path(__file__).resolve().parents[2]
    data = root/'docs/reports/data/2026-09-13-gpspatron-spatial/rf_a'
    return {c: json.loads((data/f'los_610000_{c}.json').read_text())
            for c in ('TL', 'TR', 'BL', 'BR')}



def test_legacy_replay_preserves_observations_without_inventing_evidence(legacy_records):
    from spatial_spoofing import observations_from_records
    channels = ('TL', 'TR', 'BL', 'BR')
    lam = 299792458/1575.42e6
    before = deepcopy(legacy_records)
    result = adapt_legacy_rf(legacy_records, channels, run_id='legacy-los', wavelength_m=lam)
    t, carrier, valid = observations_from_records(list(legacy_records.values()), lam)
    np.testing.assert_array_equal(result['sample_time_s'], t)
    np.testing.assert_array_equal(result['carrier_m'], carrier)
    np.testing.assert_array_equal(result['valid'], valid)
    assert result['available_time_s'] is None
    assert result['clock_id'] is None
    assert result['calibration_id'] is None and result['arc_ids'] is None
    assert result['evidence_gaps'] and result['authenticity_verified'] is False
    assert legacy_records == before



def test_adapter_does_not_read_truth_or_pvt(legacy_records):
    clean = deepcopy(legacy_records)
    for record in clean.values():
        for field in ('position_ecef_m', 'enu_error_m', 'summary', 'clock_bias_m', 'residual_m'):
            record.pop(field, None)
    kwargs = dict(channel_ids=('TL', 'TR', 'BL', 'BR'), run_id='legacy', wavelength_m=.190293672798)
    a, b = adapt_legacy_rf(legacy_records, **kwargs), adapt_legacy_rf(clean, **kwargs)
    np.testing.assert_array_equal(a['carrier_m'], b['carrier_m'])


def test_acquisition_failure_retains_missing_carrier_and_quality_state(legacy_records):
    records = deepcopy(legacy_records)
    records['TR']['satellite_tracking'][0] = {'prn': 1, 'acquired': False}
    result = adapt_legacy_rf(records, tuple(records), run_id='no-acquisition',
                             wavelength_m=299792458/1575.42e6)
    assert np.isnan(result['carrier_m'][:, 1, 0]).all()
    assert not result['valid'][:, 1, 0].any()
    assert result['valid'][:, 0, 0].any()
