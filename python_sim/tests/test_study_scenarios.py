import copy
import json

import numpy as np
import pytest

from antenna_diversity.array.geometry import active_corner_combinations
from antenna_diversity.config import default_config
from antenna_diversity.studies.config import resolve_study_config
from antenna_diversity.studies.models import strict_snapshot
from antenna_diversity.studies.scenarios import (
    build_multi_jammer_cases,
    derive_child_seed,
    parse_combination_spec,
    sample_combined_errors,
    stable_case_id,
)


def test_combination_parser_supports_one_list_and_all():
    assert parse_combination_spec("TL+BR") == (("TL", "BR"),)
    assert parse_combination_spec("TL+BR,TL+TR+BR") == (
        ("TL", "BR"), ("TL", "TR", "BR"),
    )
    assert parse_combination_spec("all") == active_corner_combinations()


@pytest.mark.parametrize("value", ["TL", "TL+TL", "TL+XX", "TL+TR+BL+BR+TL"])
def test_combination_parser_rejects_invalid_subarrays(value):
    with pytest.raises(ValueError, match="combination"):
        parse_combination_spec(value)


def test_combined_error_samples_are_repeatable_and_zero_mean_across_channels():
    resolved = resolve_study_config(preset="quick")
    first = sample_combined_errors(resolved, "medium", ("TL", "TR", "BR"), 4)
    second = sample_combined_errors(resolved, "medium", ("TL", "TR", "BR"), 4)

    assert first == second
    for sample in first:
        assert sum(sample.amplitude_errors_db.values()) == pytest.approx(0.0)
        assert sum(sample.phase_errors_deg.values()) == pytest.approx(0.0)
    json.dumps([sample.to_dict() for sample in first], allow_nan=False)


def test_scenario_builders_do_not_mutate_base_configuration():
    cfg = default_config()
    before = strict_snapshot(copy.deepcopy(cfg))
    resolved = resolve_study_config(preset="quick")

    build_multi_jammer_cases(resolved, cfg, ("TL", "BR"), "mvdr")

    assert strict_snapshot(cfg) == before


def test_combination_parser_canonicalizes_and_rejects_duplicate_combinations():
    assert parse_combination_spec((("BR", "TL"), ("BL", "TR"))) == (
        ("TL", "BR"), ("TR", "BL"),
    )
    with pytest.raises(ValueError, match="combination"):
        parse_combination_spec("BR+TL,TL+BR")


def test_combination_parser_rejects_non_sequence_specifications():
    with pytest.raises(ValueError, match="combination"):
        parse_combination_spec(42)  # type: ignore[arg-type]


def test_stable_case_id_ignores_parameter_dictionary_insertion_order():
    first = stable_case_id(
        "multi_jammer", ("BR", "TL"), "mvdr", {"alpha": 1, "beta": {"x": 2}}, 7,
    )
    second = stable_case_id(
        "multi_jammer", ("TL", "BR"), "mvdr", {"beta": {"x": 2}, "alpha": 1}, 7,
    )

    assert first == second
    assert first.startswith("multi-jammer-")
    assert len(first.rsplit("-", 1)[1]) == 12


def test_child_seeds_are_deterministic_and_change_with_master_seed():
    assert derive_child_seed(42, "combined-error:medium:TL+BR", 0) == derive_child_seed(
        42, "combined-error:medium:TL+BR", 0,
    )
    assert derive_child_seed(42, "combined-error:medium:TL+BR", 0) != derive_child_seed(
        43, "combined-error:medium:TL+BR", 0,
    )


def test_combined_error_sampling_redraws_only_invalid_elevation(monkeypatch):
    class StubRng:
        def __init__(self):
            self.calls = []
            self.values = iter((
                1.0, -2.0, 3.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                0.7, 0.8, 0.9, 1.0,
            ))

        def normal(self, loc, scale, size=None):
            self.calls.append((loc, scale, size))
            if size is not None:
                return np.array([next(self.values) for _ in range(size)], dtype=float)
            return next(self.values)

    rng = StubRng()
    monkeypatch.setattr("antenna_diversity.studies.scenarios.np.random.default_rng", lambda seed: rng)

    resolved = resolve_study_config(preset="quick")
    sample = sample_combined_errors(resolved, "light", ("TL", "BR"), 1)[0]

    assert sample.azimuth_error_deg == 1.0
    assert sample.elevation_error_deg == 3.0
    assert [call[2] for call in rng.calls[:3]] == [None, None, None]


def test_multi_jammer_cases_preserve_exact_template_values_and_json_snapshots():
    resolved = resolve_study_config(preset="quick")
    cases = build_multi_jammer_cases(resolved, default_config(), ("TL", "BR"), "mvdr")

    assert [case.parameters["template"] for case in cases] == [
        "two_jammer_separated", "two_jammer_close", "three_jammer_stress",
    ]
    assert cases[0].parameters["interferers"] == [
        {"direction": {"azimuth_deg": 120, "elevation_deg": 10}, "enable": True,
         "power_dbm": -90.0, "offset_hz": 250000.0, "initial_phase_rad": 0.8,
         "start_ms": 0.0, "stop_ms": None},
        {"direction": {"azimuth_deg": 250, "elevation_deg": 25}, "enable": True,
         "power_dbm": -90.0, "offset_hz": -180000.0, "initial_phase_rad": 1.1,
         "start_ms": 0.0, "stop_ms": None},
    ]
    cases[0].parameters["interferers"][0]["power_dbm"] = 0.0
    assert resolved.study.multi_jammer_templates[0].interferers[0].power_dbm == -90.0
    json.dumps([case.parameters for case in cases], allow_nan=False)
