"""Deterministic summaries and representative full-chain case selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import numpy as np

from antenna_diversity.array.geometry import CORNER_ORDER
from antenna_diversity.array.manifold import direction_unit_vector
from antenna_diversity.config import Direction

from .models import RankingResult, ResolvedStudyConfig, SelectionResult, StudyCase, strict_snapshot
from .scenarios import stable_case_id


_CORNER_INDEX = {corner: index for index, corner in enumerate(CORNER_ORDER)}
_FACTOR_INDEX = {
    "direction_azimuth": 0, "direction_elevation": 1, "position": 2,
    "amplitude": 3, "phase": 4, "loading": 5,
}


def direction_separation_deg(first: Direction, second: Direction) -> float:
    """Return the great-circle separation in the phone-frame convention."""

    dot = float(np.dot(direction_unit_vector(first), direction_unit_vector(second)))
    return float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))


def summarize_and_rank(
    direction_rows: Iterable[Mapping[str, object]], resolved: ResolvedStudyConfig,
) -> RankingResult:
    """Summarize all and resolvable directions, then rank geometries stably."""

    target = resolved.simulation.signal.true_direction
    if target is None:
        raise ValueError("ranking requires a true target direction")
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in direction_rows:
        if row.get("mode") == resolved.study.ranking_mode:
            grouped[tuple(str(name) for name in row["active_antennas"])] .append(row)
    if not grouped:
        raise ValueError("no ranking-mode direction rows")

    summaries = []
    for combination in resolved.study.combinations:
        rows = grouped.get(combination, [])
        if not rows:
            raise ValueError(f"no ranking-mode direction rows for {'+'.join(combination)}")
        decorated = [
            (row, direction_separation_deg(target, Direction(float(row["azimuth_deg"]), float(row["elevation_deg"]))))
            for row in rows
        ]
        resolvable = [
            item for item in decorated
            if item[1] >= resolved.study.exclusion_angle_deg
            or np.isclose(item[1], resolved.study.exclusion_angle_deg, rtol=0.0, atol=1e-12)
        ]
        if not resolvable:
            raise ValueError(f"empty resolvable region for {'+'.join(combination)}")
        all_values = np.asarray([float(row["output_sinr_db"]) for row, _ in decorated], dtype=float)
        values = np.asarray([float(row["output_sinr_db"]) for row, _ in resolvable], dtype=float)
        p10 = float(np.percentile(values, 10, method="linear"))
        chosen = min(
            (row for row, _ in resolvable),
            key=lambda row: (
                abs(float(row["output_sinr_db"]) - p10),
                int(row["elevation_index"]), int(row["azimuth_index"]),
            ),
        )
        summary = {
            "active_antennas": list(combination), "combination_name": "+".join(combination),
            "mode": resolved.study.ranking_mode,
            "all_count": len(decorated), "resolvable_count": len(resolvable),
            "p10_all_db": float(np.percentile(all_values, 10, method="linear")),
            "median_all_db": float(np.median(all_values)),
            "minimum_all_db": float(np.min(all_values)), "maximum_all_db": float(np.max(all_values)),
            "p10_resolvable_db": p10, "median_resolvable_db": float(np.median(values)),
            "minimum_resolvable_db": float(np.min(values)), "maximum_resolvable_db": float(np.max(values)),
            "representative_p10_direction": strict_snapshot(dict(chosen)),
            "white_noise_gain": float(chosen["white_noise_gain"]),
            "maximum_baseline_m": float(chosen["maximum_baseline_m"]),
        }
        summaries.append(summary)
    summaries.sort(key=lambda summary: (
        -float(summary["p10_resolvable_db"]), -float(summary["median_resolvable_db"]),
        -float(summary["white_noise_gain"]), -float(summary["maximum_baseline_m"]),
        str(summary["combination_name"]),
    ))
    return RankingResult(
        tuple(strict_snapshot(summary) for summary in summaries),  # type: ignore[arg-type]
        tuple(tuple(summary["active_antennas"]) for summary in summaries),  # type: ignore[arg-type]
    )


def select_representative_cases(
    ranking: RankingResult,
    direction_rows: Iterable[Mapping[str, object]],
    robustness_rows: Iterable[Mapping[str, object]],
    multi_rows: Iterable[Mapping[str, object]],
    resolved: ResolvedStudyConfig,
) -> SelectionResult:
    """Select only from saved normalized records; never invoke study execution."""

    if not ranking.ranked_combinations:
        raise ValueError("cannot select cases from an empty ranking")
    directions = tuple(direction_rows)
    robust = tuple(robustness_rows)
    multi = tuple(multi_rows)
    summaries = {tuple(summary["active_antennas"]): summary for summary in ranking.summaries}
    cases: list[StudyCase] = []
    unselected: dict[str, str] = {}
    best = ranking.ranked_combinations[0]

    def add(case: StudyCase) -> None:
        _validate_case(case, resolved)
        for index, existing in enumerate(cases):
            if existing.case_id == case.case_id:
                reasons = existing.selection_reasons + tuple(
                    reason for reason in case.selection_reasons if reason not in existing.selection_reasons
                )
                cases[index] = StudyCase(
                    existing.case_id, existing.stage, existing.active_antennas, existing.mode,
                    existing.seed, existing.parameters, reasons,
                )
                return
        cases.append(case)

    automatic = set(resolved.study.automatic_roles)
    for role, combination in (
        ("best", best),
        ("typical", ranking.ranked_combinations[(len(ranking.ranked_combinations) - 1) // 2]),
        ("worst", ranking.ranked_combinations[-1]),
    ):
        if role in automatic:
            add(_direction_case(role, combination, summaries[combination], resolved))
    if "knee" in automatic:
        knee = _knee_row(best, robust)
        if knee is None:
            unselected["knee"] = "no usable EGC/MVDR single-factor curves"
        else:
            add(_robustness_case("knee", knee, resolved))
    for template in resolved.study.multi_jammer_templates:
        if template.name not in automatic:
            continue
        rows = [row for row in multi if tuple(row.get("active_antennas", ())) == best and row.get("mode") == "mvdr" and row.get("template") == template.name]
        if not rows:
            unselected[template.name] = "no saved MVDR multi-jammer row for best combination"
            continue
        row = rows[0]
        parameters = {"template": row["template"], "interferers": strict_snapshot(row["interferers"])}
        add(StudyCase(str(row["case_id"]), "multi_jammer", best, "mvdr", _optional_int(row.get("child_seed")), parameters, (template.name,)))
    for level in resolved.study.error_levels:
        role = f"combined_{level.name}"
        if role not in automatic:
            continue
        rows = [row for row in robust if tuple(row.get("active_antennas", ())) == best and row.get("level") == level.name and row.get("mode", "mvdr") == "mvdr"]
        if not rows:
            unselected[role] = "no saved MVDR combined-error rows for best combination"
            continue
        values = np.asarray([float(row["mvdr_output_sinr_db"]) for row in rows], dtype=float)
        median = float(np.percentile(values, 50, method="linear"))
        row = min(rows, key=lambda item: (abs(float(item["mvdr_output_sinr_db"]) - median), int(item["sample_index"])))
        parameters = {"level": level.name, "sample_index": row["sample_index"], "child_seed": row["child_seed"], "realization": strict_snapshot(row["realization"])}
        add(StudyCase(str(row["case_id"]), "combined_error", best, "mvdr", _optional_int(row.get("child_seed")), parameters, (role,)))
    for manual in resolved.study.manual_cases:
        add(_manual_case(manual))
    return SelectionResult(tuple(cases), unselected)


def _direction_case(role: str, combination: tuple[str, ...], summary: Mapping[str, object], resolved: ResolvedStudyConfig) -> StudyCase:
    direction = dict(summary["representative_p10_direction"])
    interferer = strict_snapshot(resolved.simulation.interferers[0])
    parameters = {
        "interferers": [{**interferer, "direction": {"azimuth_deg": direction["azimuth_deg"], "elevation_deg": direction["elevation_deg"]}}],
        "direction_row_case_id": direction["case_id"],
    }
    case_id = stable_case_id("representative_direction", combination, resolved.study.ranking_mode, parameters, None)
    return StudyCase(case_id, "representative_direction", combination, resolved.study.ranking_mode, None, parameters, (role,))


def _robustness_case(role: str, row: Mapping[str, object], resolved: ResolvedStudyConfig) -> StudyCase:
    combination = tuple(str(value) for value in row["active_antennas"])
    parameters = {"single_factor": strict_snapshot(dict(row))}
    case_id = stable_case_id("single_factor", combination, "mvdr", parameters, None)
    return StudyCase(case_id, "single_factor", combination, "mvdr", None, parameters, (role,))


def _knee_row(combination: tuple[str, ...], rows: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    matched = [
        row for row in rows
        if row.get("stage") == "single_factor"
        and tuple(row.get("active_antennas", ())) == combination
        and _usable_knee_row(row)
    ]
    groups: dict[tuple[object, object, object], list[Mapping[str, object]]] = defaultdict(list)
    for row in matched:
        groups[(row.get("factor"), row.get("corner"), row.get("axis"))].append(row)
    crossings: list[Mapping[str, object]] = []
    slopes: list[tuple[float, Mapping[str, object]]] = []
    for key, curve in groups.items():
        del key
        for side in (-1, 1):
            ordered = sorted((row for row in curve if _side(row) in (0, side)), key=lambda row: float(row["absolute_normalized_severity"]))
            distinct = _distinct_severity(ordered)
            if len(distinct) < 2:
                continue
            crossing = next((row for row in distinct if float(row["mvdr_minus_egc_sinr_db"]) <= 0.0), None)
            if crossing is not None:
                crossings.append(crossing)
            for previous, current in zip(distinct, distinct[1:]):
                delta = float(current["absolute_normalized_severity"]) - float(previous["absolute_normalized_severity"])
                if delta > 0:
                    decline = float(previous["mvdr_minus_egc_sinr_db"]) - float(current["mvdr_minus_egc_sinr_db"])
                    slopes.append((decline / delta, current))
    if crossings:
        return min(crossings, key=lambda row: (float(row["absolute_normalized_severity"]), *_curve_tie_key(row)))
    if slopes:
        maximum = max(value for value, _ in slopes)
        return min(
            (row for value, row in slopes if value == maximum),
            key=lambda row: (float(row["absolute_normalized_severity"]), *_curve_tie_key(row)),
        )
    return None


def _usable_knee_row(row: Mapping[str, object]) -> bool:
    required = (
        "factor", "signed_value", "absolute_normalized_severity",
        "mvdr_minus_egc_sinr_db", "mvdr_output_sinr_db", "egc_output_sinr_db",
    )
    if any(name not in row for name in required):
        return False
    try:
        return (
            _valid_single_factor_parameters(row, tuple(str(name) for name in row["active_antennas"]))
            and all(_is_finite_real_scalar(row[name]) for name in required[1:])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _side(row: Mapping[str, object]) -> int:
    value = float(row.get("signed_value", row.get("value", 0.0)))
    return 1 if value > 0 else -1 if value < 0 else 0


def _distinct_severity(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    chosen: dict[float, Mapping[str, object]] = {}
    for row in rows:
        severity = float(row["absolute_normalized_severity"])
        chosen.setdefault(severity, row)
    return [chosen[key] for key in sorted(chosen)]


def _curve_tie_key(row: Mapping[str, object]) -> tuple[int, int, int, str]:
    return (
        _FACTOR_INDEX.get(str(row.get("factor")), len(_FACTOR_INDEX)),
        _CORNER_INDEX.get(str(row.get("corner")), len(_CORNER_INDEX)),
        {"x": 0, "y": 1, "z": 2, None: 3}.get(row.get("axis"), 4),
        str(row.get("case_id", "")),
    )


def _manual_case(raw: Mapping[str, object]) -> StudyCase:
    required = {"case_id", "stage", "active_antennas", "mode", "seed", "parameters"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"manual case missing {sorted(missing)[0]}")
    if raw["stage"] in {"representative_direction", "multi_jammer", "full_chain"}:
        parameters = raw["parameters"]
        if isinstance(parameters, Mapping) and "interferers" in parameters:
            validate_interferer_snapshots(parameters["interferers"])
    return StudyCase(str(raw["case_id"]), str(raw["stage"]), tuple(raw["active_antennas"]), str(raw["mode"]), _optional_int(raw["seed"]), strict_snapshot(raw["parameters"]), ("manual",))  # type: ignore[arg-type]


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _validate_case(case: StudyCase, resolved: ResolvedStudyConfig) -> None:
    if case.active_antennas not in resolved.study.combinations:
        raise ValueError(f"selected case has unknown combination: {'+'.join(case.active_antennas)}")
    if case.mode not in {"single", "egc", "mvdr"}:
        raise ValueError(f"selected case has unknown mode: {case.mode}")
    parameters = case.parameters
    if case.stage in {"representative_direction", "multi_jammer", "full_chain"}:
        _require_case_parameters(case, parameters, {"interferers"})
        validate_interferer_snapshots(parameters["interferers"])
    elif case.stage == "single_factor":
        _require_case_parameters(case, parameters, {"single_factor"})
        factor = parameters["single_factor"]
        if not isinstance(factor, Mapping) or not _valid_single_factor_parameters(factor, case.active_antennas):
            raise ValueError("selected case single_factor has invalid reconstruction parameters")
    elif case.stage == "combined_error":
        _require_case_parameters(case, parameters, {"level", "sample_index", "child_seed", "realization"})
        realization = parameters["realization"]
        required = {
            "level", "sample_index", "child_seed", "azimuth_error_deg", "elevation_error_deg",
            "position_errors_m", "amplitude_errors_db", "phase_errors_deg",
        }
        if not isinstance(realization, Mapping) or not required <= set(realization):
            raise ValueError("selected case combined_error lacks reconstruction parameters")
    else:
        raise ValueError(f"unsupported selected case stage: {case.stage}")
    strict_snapshot(case.parameters)


def _require_case_parameters(case: StudyCase, parameters: Mapping[str, object], required: set[str]) -> None:
    missing = required - set(parameters)
    if missing:
        raise ValueError(f"selected case {case.stage} lacks reconstruction parameter: {sorted(missing)[0]}")


def validate_interferer_snapshots(value: object) -> None:
    """Validate the exact strict-JSON schema needed to rebuild CW interferers."""

    if not isinstance(value, (list, tuple)):
        raise ValueError("selected case parameters.interferers must be a sequence")
    required = {
        "direction", "enable", "power_dbm", "offset_hz", "initial_phase_rad",
        "start_ms", "stop_ms",
    }
    direction_fields = {"azimuth_deg", "elevation_deg"}
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("selected case interferer must contain the exact CWInterferer fields")
        direction = raw["direction"]
        if not isinstance(direction, Mapping) or set(direction) != direction_fields:
            raise ValueError("selected case interferer direction must contain exact direction fields")
        if not all(_is_finite_real_scalar(direction[name]) for name in direction_fields):
            raise ValueError("selected case interferer direction must be finite real scalars")
        try:
            Direction(direction["azimuth_deg"], direction["elevation_deg"]).validate()
        except (TypeError, ValueError) as error:
            raise ValueError("selected case interferer direction is invalid") from error
        if type(raw["enable"]) is not bool:
            raise ValueError("selected case interferer enable must be a bool")
        for name in ("power_dbm", "offset_hz", "initial_phase_rad", "start_ms"):
            if not _is_finite_real_scalar(raw[name]):
                raise ValueError(f"selected case interferer {name} must be a finite real scalar")
        start_ms = raw["start_ms"]
        stop_ms = raw["stop_ms"]
        if start_ms < 0:
            raise ValueError("selected case interferer start_ms must be nonnegative")
        if stop_ms is not None:
            if not _is_finite_real_scalar(stop_ms):
                raise ValueError("selected case interferer stop_ms must be None or a finite real scalar")
            if stop_ms < start_ms:
                raise ValueError("selected case interferer stop_ms must not precede start_ms")


def _valid_single_factor_parameters(parameters: Mapping[str, object], combination: tuple[str, ...]) -> bool:
    required = {"factor", "value", "corner", "axis"}
    if not required <= set(parameters):
        return False
    factor = parameters["factor"]
    if not isinstance(factor, str) or factor not in _FACTOR_INDEX:
        return False
    if not _is_finite_real_scalar(parameters["value"]):
        return False
    corner, axis = parameters["corner"], parameters["axis"]
    if factor == "position":
        return corner in combination and axis in {"x", "y", "z"}
    if factor in {"amplitude", "phase"}:
        return corner in combination and axis is None
    return corner is None and axis is None


def _is_finite_real_scalar(value: object) -> bool:
    if isinstance(value, (bool, complex)):
        return False
    candidate = np.asarray(value)
    return (
        candidate.ndim == 0
        and np.issubdtype(candidate.dtype, np.number)
        and not np.issubdtype(candidate.dtype, np.complexfloating)
        and bool(np.isfinite(candidate))
    )
