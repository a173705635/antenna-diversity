import json
from pathlib import Path

import numpy as np
import pytest

from antenna_diversity.studies.config import resolve_study_config
from antenna_diversity.studies.models import ResolvedStudyConfig


def test_quick_preset_has_exact_axes_and_counts():
    resolved = resolve_study_config(preset="quick")

    np.testing.assert_array_equal(resolved.study.azimuths_deg, np.arange(0, 360, 30))
    np.testing.assert_array_equal(resolved.study.elevations_deg, np.arange(0, 91, 15))
    np.testing.assert_allclose(resolved.study.powers_dbm, [-110, -100, -90, -80])
    assert resolved.study.combined_error_samples_per_level == 50
    assert resolved.study.full_chain_run_count == 3
    assert resolved.study.ranking_mode == "mvdr"


def test_cli_overrides_json_which_overrides_preset(tmp_path: Path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({
        "study": {"name": "from-json", "exclusion_angle_deg": 20.0},
        "simulation": {"signal": {"carrier_hz": 1.2e9}},
    }), encoding="utf-8")

    resolved = resolve_study_config(
        path,
        preset="quick",
        cli_overrides={"name": "from-cli", "combinations": "TL+BR"},
    )

    assert resolved.study.name == "from-cli"
    assert resolved.study.exclusion_angle_deg == 20.0
    assert resolved.study.combinations == (("TL", "BR"),)
    assert resolved.simulation.signal.carrier_hz == 1.2e9


def test_unknown_json_key_is_rejected(tmp_path: Path):
    path = tmp_path / "study.json"
    path.write_text('{"study":{"exlusion_angle_deg":15}}', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown.*exlusion_angle_deg"):
        resolve_study_config(path)


def test_resolved_snapshots_do_not_alias_caller_overrides():
    overrides = {"modes": ["egc", "mvdr"]}
    resolved = resolve_study_config(preset="quick", cli_overrides=overrides)
    before = resolved.to_dict()

    overrides["modes"].append("single")

    assert resolved.to_dict() == before


@pytest.mark.parametrize(
    ("preset", "azimuths", "elevations", "powers", "combined", "runs"),
    [
        ("standard", np.arange(0, 360, 10), np.arange(0, 91, 10), np.linspace(-110, -70, 9), 500, 40),
        ("extended", np.arange(0, 360, 5), np.arange(0, 91, 5), np.linspace(-115, -65, 21), 5000, 200),
    ],
)
def test_nonquick_presets_have_documented_grids_and_counts(preset, azimuths, elevations, powers, combined, runs):
    resolved = resolve_study_config(preset=preset)

    np.testing.assert_array_equal(resolved.study.azimuths_deg, azimuths)
    np.testing.assert_array_equal(resolved.study.elevations_deg, elevations)
    np.testing.assert_allclose(resolved.study.powers_dbm, powers)
    assert resolved.study.combined_error_samples_per_level == combined
    assert resolved.study.full_chain_run_count == runs


def test_snapshot_is_strict_json_and_scientific_hash_ignores_output_controls(tmp_path: Path):
    base = resolve_study_config(preset="quick")
    changed = ResolvedStudyConfig.from_parts(
        __import__("dataclasses").replace(base.study, name="another", output_dir=tmp_path),
        __import__("dataclasses").replace(
            base.simulation,
            plot=__import__("dataclasses").replace(base.simulation.plot, enable=False, show=False, output_dir=tmp_path),
        ),
    )

    json.dumps(base.to_dict(), allow_nan=False)
    assert changed.config_sha256 == base.config_sha256


@pytest.mark.parametrize(
    "overrides",
    [
        {"azimuths_deg": [0.0, np.nan]},
        {"powers_dbm": [0.0 + 1.0j]},
        {"loading_factors": [1e-2, 1e-5]},
        {"spatial_modes": ["egc", "music"]},
        {"combinations": "TL+BR,TL+BR"},
    ],
)
def test_invalid_study_overrides_are_rejected(overrides):
    with pytest.raises(ValueError):
        resolve_study_config(preset="quick", cli_overrides=overrides)


def test_invalid_simulation_frequency_is_rejected(tmp_path: Path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"simulation": {"signal": {"carrier_hz": 0.0}}}), encoding="utf-8")

    with pytest.raises(ValueError, match="frequencies"):
        resolve_study_config(path, preset="quick")


@pytest.mark.parametrize("value", [complex(1.0, 2.0), float("nan"), np.array([float("inf")])])
def test_strict_snapshot_rejects_complex_and_nonfinite_values(value):
    from antenna_diversity.studies.models import strict_snapshot

    with pytest.raises(ValueError):
        strict_snapshot(value)


def test_json_modes_shorthand_is_rejected(tmp_path: Path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"study": {"modes": ["egc", "mvdr"]}}), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown.*modes"):
        resolve_study_config(path)


def test_nested_simulation_dataclass_keys_are_validated(tmp_path: Path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"simulation": {"rf": {"stages": [{"name": "LNA", "gain_db": 1.0, "noise_figre_db": 1.0}]}}}), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown.*noise_figre_db"):
        resolve_study_config(path)


@pytest.mark.parametrize(
    "study_update",
    [
        {"error_levels": [{"name": "light", "direction_sigma_deg": 1.0, "position_sigma_m": .001, "amplitude_sigma_db": .25, "phase_sigma_deg": 3.0, "phase_sigme_deg": 4.0}]},
        {"multi_jammer_templates": [{"name": "custom", "interferers": [], "unexpected": True}]},
        {"error_levels": {"name": "not-a-list"}},
        {"multi_jammer_templates": [{"name": "custom", "interferers": "not-a-list"}]},
    ],
)
def test_nested_study_records_reject_unknown_keys_and_invalid_shapes(tmp_path: Path, study_update):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"study": study_update}), encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_study_config(path)


def test_resolved_state_cannot_be_mutated_through_public_arrays_cases_or_simulation():
    resolved = resolve_study_config(
        preset="quick",
        cli_overrides={"manual_cases": [{"label": "fixed", "nested": [1, 2]}]},
    )
    before = resolved.to_dict()

    with pytest.raises(ValueError):
        resolved.study.azimuths_deg[0] = 123.0
    with pytest.raises(TypeError):
        resolved.study.manual_cases[0]["label"] = "changed"
    with pytest.raises(AttributeError):
        resolved.study.manual_cases[0]["nested"].append(3)

    simulation_copy = resolved.simulation
    simulation_copy.signal.carrier_hz = 1.2e9

    assert resolved.to_dict() == before
    assert resolved.simulation.signal.carrier_hz != 1.2e9


@pytest.mark.parametrize("single_antennas", [(), ("TL", "TL"), ("TL", "XX")])
def test_single_antennas_must_be_a_nonempty_unique_corner_subset(single_antennas):
    with pytest.raises(ValueError, match="single_antennas"):
        resolve_study_config(preset="quick", cli_overrides={"single_antennas": single_antennas})
