import pytest

from antenna_diversity.config import Direction
from antenna_diversity.studies.ranking import (
    direction_separation_deg,
    select_representative_cases,
    summarize_and_rank,
)
from tests.study_helpers import resolved_for_test


def _direction_row(combination, sinr, *, azimuth=30.0, elevation=30.0, index=0, gain=1.0, baseline=0.178):
    return {
        "case_id": f"direction-{'-'.join(combination)}-{index}",
        "active_antennas": list(combination), "mode": "mvdr",
        "azimuth_deg": azimuth, "elevation_deg": elevation,
        "output_sinr_db": float(sinr), "white_noise_gain": float(gain),
        "maximum_baseline_m": float(baseline),
        "elevation_index": index, "azimuth_index": index,
    }


def test_direction_separation_uses_three_dimensional_unit_vectors():
    assert direction_separation_deg(Direction(0, 90), Direction(180, 90)) == pytest.approx(0.0)
    assert direction_separation_deg(Direction(0, 0), Direction(90, 0)) == pytest.approx(90.0)


def test_exclusion_boundary_is_in_resolvable_region():
    resolved = resolved_for_test(
        combinations=(("TL", "BR"),), spatial_modes=("mvdr",),
        ranking_mode="mvdr", exclusion_angle_deg=15.0,
    )
    rows = [
        _direction_row(("TL", "BR"), index, elevation=60.0 - separation, index=index)
        for index, separation in enumerate([14.999, 15.0, 30.0])
    ]

    result = summarize_and_rank(rows, resolved)

    assert result.summaries[0]["resolvable_count"] == 2


def test_ranking_uses_exact_stable_key_order():
    resolved = resolved_for_test(
        combinations=(("TL", "BR"), ("TL", "TR"), ("BL", "BR"), ("TR", "BR")),
        spatial_modes=("mvdr",), ranking_mode="mvdr",
    )
    rows = [
        _direction_row(("TL", "BR"), 10, gain=1, baseline=1),
        _direction_row(("TL", "TR"), 10, gain=2, baseline=0),
        _direction_row(("BL", "BR"), 10, gain=2, baseline=3),
        _direction_row(("TR", "BR"), 11, gain=0, baseline=0),
    ]

    result = summarize_and_rank(rows, resolved)

    assert result.ranked_combinations == (
        ("TR", "BR"), ("BL", "BR"), ("TL", "TR"), ("TL", "BR"),
    )


def test_p10_direction_ties_choose_elevation_then_azimuth_index():
    resolved = resolved_for_test(combinations=(("TL", "BR"),), spatial_modes=("mvdr",), ranking_mode="mvdr")
    rows = [_direction_row(("TL", "BR"), 0.0, index=index) for index in range(2)]
    rows[0]["elevation_index"], rows[0]["azimuth_index"] = 1, 0
    rows[1]["elevation_index"], rows[1]["azimuth_index"] = 0, 4

    result = summarize_and_rank(rows, resolved)

    assert result.summaries[0]["representative_p10_direction"]["case_id"] == rows[1]["case_id"]


def test_selection_deduplicates_best_typical_worst_and_keeps_manual_cases():
    resolved = resolved_for_test(
        combinations=(("TL", "BR"),), spatial_modes=("mvdr",), ranking_mode="mvdr",
        automatic_roles=("best", "typical", "worst"),
        manual_cases=({"case_id": "manual", "stage": "full_chain", "active_antennas": ["TL", "BR"],
                       "mode": "mvdr", "seed": 8, "parameters": {"interferers": []}},),
    )
    direction_rows = [_direction_row(("TL", "BR"), 5.0)]
    ranking = summarize_and_rank(direction_rows, resolved)

    selected = select_representative_cases(ranking, direction_rows, (), (), resolved)

    assert len(selected.cases) == 2
    assert selected.cases[0].selection_reasons == ("best", "typical", "worst")
    assert selected.cases[1].case_id == "manual"


def test_selection_uses_lower_middle_rank_for_even_counts():
    combinations = (("TL", "BR"), ("TL", "TR"), ("BL", "BR"), ("TR", "BR"))
    resolved = resolved_for_test(combinations=combinations, spatial_modes=("mvdr",), ranking_mode="mvdr", automatic_roles=("typical",))
    direction_rows = [_direction_row(combination, index + 1) for index, combination in enumerate(combinations)]
    ranking = summarize_and_rank(direction_rows, resolved)

    selected = select_representative_cases(ranking, direction_rows, (), (), resolved)

    assert selected.cases[0].active_antennas == ranking.ranked_combinations[1]


def test_knee_uses_steepest_decline_fallback_and_reports_missing_modes():
    resolved = resolved_for_test(combinations=(("TL", "BR"),), automatic_roles=("knee",))
    direction_rows = [_direction_row(("TL", "BR"), 5.0)]
    ranking = summarize_and_rank(direction_rows, resolved)
    rows = [
        {"stage": "single_factor", "active_antennas": ["TL", "BR"], "factor": "phase", "corner": "TL", "axis": None,
         "value": value, "signed_value": value, "absolute_normalized_severity": severity,
         "mvdr_minus_egc_sinr_db": advantage, "mvdr_output_sinr_db": 3.0,
         "egc_output_sinr_db": 3.0 - advantage}
        for value, severity, advantage in ((0.0, 0.0, 6.0), (1.0, 0.5, 5.0), (2.0, 1.0, 1.0))
    ]

    selected = select_representative_cases(ranking, direction_rows, rows, (), resolved)

    assert selected.cases[0].parameters["single_factor"]["value"] == 2.0
    absent = select_representative_cases(ranking, direction_rows, (), (), resolved)
    assert absent.unselected_roles["knee"] == "no usable EGC/MVDR single-factor curves"


def test_knee_ignores_combined_rows_and_nonfinite_single_factor_values():
    resolved = resolved_for_test(combinations=(("TL", "BR"),), automatic_roles=("knee",))
    direction_rows = [_direction_row(("TL", "BR"), 5.0)]
    ranking = summarize_and_rank(direction_rows, resolved)
    rows = [
        {"stage": "single_factor", "active_antennas": ["TL", "BR"], "factor": "phase", "corner": "TL", "axis": None,
         "value": value, "signed_value": value, "absolute_normalized_severity": severity,
         "mvdr_minus_egc_sinr_db": advantage, "mvdr_output_sinr_db": 3.0, "egc_output_sinr_db": 3.0 - advantage}
        for value, severity, advantage in ((0.0, 0.0, 3.0), (1.0, 1.0, -1.0))
    ]
    rows.extend((
        {"stage": "combined_error", "active_antennas": ["TL", "BR"], "mvdr_output_sinr_db": 2.0,
         "egc_output_sinr_db": 1.0, "signed_value": None, "absolute_normalized_severity": None},
        {"stage": "single_factor", "active_antennas": ["TL", "BR"], "factor": "phase", "corner": "BR", "axis": None,
         "value": 1.0, "signed_value": 1.0, "absolute_normalized_severity": float("nan"),
         "mvdr_minus_egc_sinr_db": -5.0, "mvdr_output_sinr_db": 3.0, "egc_output_sinr_db": 8.0},
    ))

    selected = select_representative_cases(ranking, direction_rows, rows, (), resolved)

    assert selected.cases[0].parameters["single_factor"]["corner"] == "TL"


def test_knee_slope_ties_choose_lower_higher_endpoint_severity_before_factor_order():
    resolved = resolved_for_test(combinations=(("TL", "BR"),), automatic_roles=("knee",))
    direction_rows = [_direction_row(("TL", "BR"), 5.0)]
    ranking = summarize_and_rank(direction_rows, resolved)
    rows = [
        {"stage": "single_factor", "case_id": case_id, "active_antennas": ["TL", "BR"], "factor": factor,
         "corner": "TL" if factor == "phase" else None, "axis": None, "value": value, "signed_value": value,
         "absolute_normalized_severity": severity, "mvdr_minus_egc_sinr_db": advantage,
         "mvdr_output_sinr_db": 5.0, "egc_output_sinr_db": 5.0 - advantage}
        for factor, case_id, value, severity, advantage in (
            ("phase", "phase-zero", 0.0, 0.0, 3.0),
            ("phase", "phase-half", 1.0, 0.5, 2.0),
            ("direction_azimuth", "az-zero", 0.0, 0.0, 4.0),
            ("direction_azimuth", "az-one", 1.0, 1.0, 2.0),
        )
    ]

    selected = select_representative_cases(ranking, direction_rows, rows, (), resolved)

    assert selected.cases[0].parameters["single_factor"]["case_id"] == "phase-half"


def test_knee_slope_fallback_does_not_treat_a_smaller_slope_as_a_tie():
    resolved = resolved_for_test(combinations=(("TL", "BR"),), automatic_roles=("knee",))
    direction_rows = [_direction_row(("TL", "BR"), 5.0)]
    ranking = summarize_and_rank(direction_rows, resolved)
    rows = [
        {"stage": "single_factor", "case_id": case_id, "active_antennas": ["TL", "BR"], "factor": factor,
         "corner": "TL" if factor == "phase" else None, "axis": None, "value": value, "signed_value": value,
         "absolute_normalized_severity": severity, "mvdr_minus_egc_sinr_db": advantage,
         "mvdr_output_sinr_db": 5.0, "egc_output_sinr_db": 5.0 - advantage}
        for factor, case_id, value, severity, advantage in (
            ("phase", "phase-zero", 0.0, 0.0, 3.0),
            ("phase", "phase-half", 1.0, 0.5, 2.50000000000005),
            ("direction_azimuth", "az-zero", 0.0, 0.0, 3.0),
            ("direction_azimuth", "az-one", 1.0, 1.0, 2.0),
        )
    ]

    selected = select_representative_cases(ranking, direction_rows, rows, (), resolved)

    assert selected.cases[0].parameters["single_factor"]["case_id"] == "az-one"


def test_knee_ignores_single_factor_rows_with_invalid_apply_schema():
    resolved = resolved_for_test(combinations=(("TL", "BR"),), automatic_roles=("knee",))
    direction_rows = [_direction_row(("TL", "BR"), 5.0)]
    ranking = summarize_and_rank(direction_rows, resolved)
    rows = [
        {"stage": "single_factor", "active_antennas": ["TL", "BR"], "factor": "position", "corner": "TL", "axis": None,
         "value": value, "signed_value": value, "absolute_normalized_severity": severity,
         "mvdr_minus_egc_sinr_db": advantage, "mvdr_output_sinr_db": 3.0, "egc_output_sinr_db": 3.0 - advantage}
        for value, severity, advantage in ((0.0, 0.0, 3.0), (1.0, 1.0, -1.0))
    ]

    selected = select_representative_cases(ranking, direction_rows, rows, (), resolved)

    assert selected.cases == ()
    assert selected.unselected_roles["knee"] == "no usable EGC/MVDR single-factor curves"


def test_selection_rejects_manual_case_with_unsupported_stage_or_missing_reconstruction_inputs():
    resolved = resolved_for_test(
        combinations=(("TL", "BR"),), automatic_roles=(),
        manual_cases=({"case_id": "banana", "stage": "banana", "active_antennas": ["TL", "BR"],
                       "mode": "mvdr", "seed": None, "parameters": {"any": "json"}},),
    )
    direction_rows = [_direction_row(("TL", "BR"), 5.0)]
    ranking = summarize_and_rank(direction_rows, resolved)

    with pytest.raises(ValueError, match="unsupported selected case stage"):
        select_representative_cases(ranking, direction_rows, (), (), resolved)

    missing_parameters = resolved_for_test(
        combinations=(("TL", "BR"),), automatic_roles=(),
        manual_cases=({"case_id": "incomplete", "stage": "full_chain", "active_antennas": ["TL", "BR"],
                       "mode": "mvdr", "seed": None, "parameters": {}},),
    )
    with pytest.raises(ValueError, match="lacks reconstruction parameter: interferers"):
        select_representative_cases(ranking, direction_rows, (), (), missing_parameters)


@pytest.mark.parametrize("factor,value,corner,axis", [
    ("unknown", 1.0, None, None),
    ("loading", True, None, None),
    ("phase", [], "TL", None),
    ("direction_azimuth", 1.0, "TL", None),
    ("amplitude", 1.0, None, None),
    ("position", 1.0, "TL", "q"),
])
def test_selection_rejects_manual_single_factor_without_a_legal_apply_schema(factor, value, corner, axis):
    resolved = resolved_for_test(
        combinations=(("TL", "BR"),), automatic_roles=(),
        manual_cases=({"case_id": "bad-single", "stage": "single_factor", "active_antennas": ["TL", "BR"],
                       "mode": "mvdr", "seed": None,
                       "parameters": {"single_factor": {"factor": factor, "value": value, "corner": corner, "axis": axis}}},),
    )
    direction_rows = [_direction_row(("TL", "BR"), 5.0)]
    ranking = summarize_and_rank(direction_rows, resolved)

    with pytest.raises(ValueError, match="single_factor"):
        select_representative_cases(ranking, direction_rows, (), (), resolved)


def test_selection_uses_saved_rows_without_rerunning_analytical_code(monkeypatch):
    import antenna_diversity.studies.analytical as analytical

    monkeypatch.setattr(analytical, "run_baseline", lambda *_: pytest.fail("ranking must not rerun baseline"))
    resolved = resolved_for_test(combinations=(("TL", "BR"),), automatic_roles=("best",))
    rows = [_direction_row(("TL", "BR"), 5.0)]
    ranking = summarize_and_rank(rows, resolved)

    selected = select_representative_cases(ranking, rows, (), (), resolved)

    assert selected.cases[0].selection_reasons == ("best",)


def test_selection_keeps_exact_multi_jammer_and_median_combined_parameters():
    resolved = resolved_for_test(
        preset="extended", combinations=(("TL", "BR"),),
        automatic_roles=("two_jammer_separated", "combined_light"),
    )
    direction_rows = [_direction_row(("TL", "BR"), 5.0)]
    ranking = summarize_and_rank(direction_rows, resolved)
    multi = [{
        "case_id": "multi-case", "active_antennas": ["TL", "BR"], "mode": "mvdr",
        "template": "two_jammer_separated", "child_seed": 99,
        "interferers": [{"direction": {"azimuth_deg": 120.0, "elevation_deg": 10.0}, "enable": True,
                          "power_dbm": -90.0, "offset_hz": 250e3, "initial_phase_rad": 0.8,
                          "start_ms": 0.0, "stop_ms": None}],
    }]
    combined = [
        {"case_id": f"combined-{index}", "active_antennas": ["TL", "BR"], "level": "light",
         "sample_index": index, "child_seed": 10 + index, "mvdr_output_sinr_db": sinr,
         "realization": {"level": "light", "sample_index": index, "child_seed": 10 + index,
                         "azimuth_error_deg": 0.0, "elevation_error_deg": 0.0,
                         "position_errors_m": {"TL": [0.0, 0.0, 0.0], "BR": [0.0, 0.0, 0.0]},
                         "amplitude_errors_db": {"TL": 0.0, "BR": 0.0},
                         "phase_errors_deg": {"TL": 0.0, "BR": 0.0}}}
        for index, sinr in enumerate((1.0, 3.0, 5.0))
    ]

    selected = select_representative_cases(ranking, direction_rows, combined, multi, resolved)

    assert selected.cases[0].parameters["interferers"] == multi[0]["interferers"]
    assert selected.cases[1].parameters["realization"] == combined[1]["realization"]
    assert selected.cases[1].parameters["sample_index"] == 1
