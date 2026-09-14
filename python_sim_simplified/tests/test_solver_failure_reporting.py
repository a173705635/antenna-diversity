"""Research failures must preserve cause; programming errors must escape."""
import json
from pathlib import Path

import numpy as np
import pytest

from positioning import PositioningFailure, solve_position


@pytest.fixture(scope='module')
def pair():
    root = Path(__file__).resolve().parents[2]
    data = root/'docs/reports/data/2026-09-13-gpspatron-spatial/rf_a'
    return [json.loads((data/f'los_610000_{channel}.json').read_text())
            for channel in ('TL', 'TR')]


def test_unexpected_solver_error_is_not_relabelled_as_unsolved_epoch(monkeypatch, pair):
    import audit_antenna_selection as audit

    def solver(*args, **kwargs):
        raise ValueError('unexpected solver regression')

    monkeypatch.setattr(audit, 'solve_position', solver)
    with pytest.raises(ValueError, match='unexpected solver regression'):
        audit.evaluate(pair, 'phase_lock_metric')


def test_defined_geometry_failure_keeps_each_epoch_reason(monkeypatch, pair):
    import audit_antenna_selection as audit

    def solver(*args, **kwargs):
        return solve_position(np.tile([2e7, 0., 1e7], (5, 1)), np.full(5, 2.1e7))

    monkeypatch.setattr(audit, 'solve_position', solver)
    result = audit.evaluate(pair, 'phase_lock_metric')
    for mode in result.values():
        assert mode['solved'] == 0
        assert len(mode['failures']) == mode['epochs'] == 40
        assert mode['failures'][0] == {'index': 0, 'reason': 'insufficient satellite geometry'}
        assert mode['failures'][-1]['index'] == 39


def test_geometry_failure_remains_distinct_from_array_shape_error():
    with pytest.raises(PositioningFailure):
        solve_position(np.tile([2e7, 0., 1e7], (5, 1)), np.full(5, 2.1e7))
    with pytest.raises(ValueError) as error:
        solve_position(np.zeros((5, 2)), np.ones(5))
    assert not isinstance(error.value, PositioningFailure)
