"""Numerical equivalence to the frozen receiver, isolated in subprocesses."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.local_evidence

from core.single_source import frontend, navigate
from core.single_source_scene import Case, generate

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "archive" / "pre_refactor_2026-09-16"


def frozen_run(source, *arguments):
    subprocess.run(
        [sys.executable, "-c", source, *map(str, arguments)],
        cwd=FROZEN, check=True, capture_output=True, text=True,
    )


def assert_same_arrays(actual, expected):
    assert set(actual) == set(expected)
    for key, value in actual.items():
        np.testing.assert_array_equal(value, expected[key], err_msg=key)


@pytest.mark.parametrize("fixture", [
    "00adaf87ed6c58c597ea747119d8e33764f98cc629bc875b7463fbe964bb3db4",
    "ff4a4c81ec7a83a8208f54f2e4808c170e2596cdcdd9d3237d63fbf55a9c1e39",
])
def test_navigation_matches_frozen_observations(tmp_path, fixture):
    observation = ROOT / "results" / "mechanism-four" / "frontends" / fixture / "tracking.npz"
    expected_file = tmp_path / "navigation.npz"
    frozen_run("""
import sys
import numpy as np
from single_source_receiver_history import navigate
with np.load(sys.argv[1]) as record:
    result = navigate(record)
result['initialization_time'] = np.nan if result['initialization_time'] is None else result['initialization_time']
np.savez(sys.argv[2], **result)
""", observation, expected_file)
    with np.load(observation) as record:
        actual = navigate(record)
    actual["initialization_time"] = np.nan if actual["initialization_time"] is None else actual["initialization_time"]
    with np.load(expected_file) as expected:
        assert_same_arrays(actual, expected)


def test_generated_rf_and_frontend_match_frozen_receiver(tmp_path):
    expected_file = tmp_path / "receiver.npz"
    frozen_run("""
import sys
import numpy as np
from continuous_improvement import Case, generate
from single_source_receiver_history import frontend
case = Case(duration=.24, phase_deg=4., amplitude_db=.5, velocity_east=2., fake_rate=3.)
samples, nominal, codes, times, satellites, radii = generate(case, 8123, True)
record = dict(samples=samples, nominal=nominal, codes=codes, times=times,
              satellites=satellites, coarse=np.round((radii+75000)/10000)*10000)
result = frontend(record, [0,1,2,3])
np.savez(sys.argv[1], samples=samples, nominal=nominal, codes=codes, times=times,
         satellites=satellites, radii=radii, **{'front_'+k:v for k,v in result.items()})
""", expected_file)
    case = Case(duration=.24, phase_deg=4., amplitude_db=.5, velocity_east=2., fake_rate=3.)
    samples, nominal, codes, times, satellites, radii = generate(case, 8123, True)
    record = dict(samples=samples, nominal=nominal, codes=codes, times=times,
                  satellites=satellites, coarse=np.round((radii+75000)/10000)*10000)
    result = frontend(record, [0, 1, 2, 3])
    assert len(result["startup_prns"]) >= 4
    assert np.isfinite(result["ranges"]).any()
    actual = dict(samples=samples, nominal=nominal, codes=codes, times=times,
                  satellites=satellites, radii=radii,
                  **{"front_"+key:value for key,value in result.items()})
    with np.load(expected_file) as expected:
        assert_same_arrays(actual, expected)
