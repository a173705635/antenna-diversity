import itertools

import numpy as np
import pytest

from antenna_diversity.array.geometry import (
    active_corner_combinations,
    mate70_phase_centers,
    maximum_baseline_m,
    select_phase_centers,
)


def test_active_corner_combinations_returns_all_eleven_subsets():
    combinations = active_corner_combinations()

    assert len(combinations) == 11
    assert len(set(combinations)) == 11
    assert sum(len(item) == 2 for item in combinations) == 6
    assert sum(len(item) == 3 for item in combinations) == 4
    assert sum(len(item) == 4 for item in combinations) == 1


def test_select_phase_centers_preserves_requested_order():
    centers = mate70_phase_centers()
    selected = select_phase_centers(centers, ("BR", "TL", "TR"))

    np.testing.assert_allclose(
        selected,
        [centers["BR"], centers["TL"], centers["TR"]],
    )


def test_diagonal_maximum_baseline_matches_phone_outline():
    positions = select_phase_centers(mate70_phase_centers(), ("TL", "BR"))

    assert maximum_baseline_m(positions) == pytest.approx(
        np.hypot(0.1609, 0.0759)
    )
