import json
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from antenna_diversity.array.antenna_config import AntennaStateConfig, PhoneAntennaConfig
from antenna_diversity.array.effects import AttenuationRegion, CouplingPath, ElementOrientation
from antenna_diversity.array.patterns import ParametricPatternParameters
from antenna_diversity.array.polarization import WavePolarization
from antenna_diversity.config import default_config
from antenna_diversity.studies import config as study_config_module
from antenna_diversity.studies.config import resolve_study_config
from antenna_diversity.studies.models import (
    InterfererTemplate,
    ResolvedStudyConfig,
    strict_snapshot,
)


_CORNERS = ("TL", "TR", "BL", "BR")


def _polarization(kind: str = "rhcp", *, tilt_deg: float = 0.0) -> dict[str, object]:
    return {
        "kind": kind,
        "tilt_deg": tilt_deg,
        "az_real": 1.0,
        "az_imag": 0.0,
        "el_real": 0.0,
        "el_imag": 0.0,
    }


def _parametric_state(*, gain_db: float, polarization_kind: str) -> dict[str, object]:
    return {
        "pattern_kind": "parametric",
        "parametric_patterns": {
            corner: {
                "peak_gain_db": gain_db + index,
                "cosine_exponent": 1.5,
                "back_lobe_gain_db": -18.0,
                "nominal_polarization": _polarization(polarization_kind),
                "cross_polar_discrimination_db": 24.0,
                "cross_phase_deg": 12.0,
            }
            for index, corner in enumerate(_CORNERS)
        },
        "measured_pattern_csv": None,
        "measured_pattern_sha256": None,
        "orientations": {
            corner: {
                "yaw_deg": float(index * 5),
                "pitch_deg": float(index),
                "roll_deg": float(-index),
            }
            for index, corner in enumerate(_CORNERS)
        },
        "attenuation_regions": [{
            "elements": ["TL", "BR"],
            "center_azimuth_deg": 110.0,
            "center_elevation_deg": 25.0,
            "radius_deg": 20.0,
            "attenuation_db": 4.0,
            "transition_width_deg": 5.0,
        }],
        "coupling_paths": [{
            "source": "TL",
            "receiver": "BR",
            "magnitude_db": -22.0,
            "phase_deg": 40.0,
        }],
    }


def _parametric_override() -> dict[str, object]:
    return {
        "simulation": {
            "signal": {
                "true_polarization": _polarization("rhcp"),
                "assumed_polarization": _polarization("lhcp"),
            },
            "array": {
                "antenna_model": {
                    "model_label": "strict parametric fixture",
                    "model_source": "parametric",
                    "true_state": _parametric_state(
                        gain_db=1.0, polarization_kind="rhcp"
                    ),
                    "assumed_state": _parametric_state(
                        gain_db=-1.0, polarization_kind="linear"
                    ),
                }
            },
            "interferers": [{
                "direction": {"azimuth_deg": 120.0, "elevation_deg": 10.0},
                "enable": True,
                "power_dbm": -90.0,
                "offset_hz": 250e3,
                "initial_phase_rad": 0.8,
                "start_ms": 0.0,
                "stop_ms": None,
                "polarization": _polarization("linear", tilt_deg=35.0),
            }],
        }
    }


def _isotropic_state() -> dict[str, object]:
    return {
        "pattern_kind": "isotropic",
        "parametric_patterns": {},
        "measured_pattern_csv": None,
        "measured_pattern_sha256": None,
        "orientations": {
            corner: {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
            for corner in _CORNERS
        },
        "attenuation_regions": [],
        "coupling_paths": [],
    }


def _measured_override(
    csv_name: str, *, expected_sha256: str | None = None
) -> dict[str, object]:
    true_state = _isotropic_state()
    true_state["pattern_kind"] = "measured"
    true_state["measured_pattern_csv"] = csv_name
    if expected_sha256 is None:
        true_state.pop("measured_pattern_sha256")
    else:
        true_state["measured_pattern_sha256"] = expected_sha256
    return {
        "simulation": {
            "array": {
                "antenna_model": {
                    "model_label": "measured fixture",
                    "model_source": "measured",
                    "true_state": true_state,
                    "assumed_state": _isotropic_state(),
                }
            }
        }
    }


def _write_pattern_csv(path: Path, *, component: float = 1.0) -> None:
    rows = [
        "element_name,frequency_hz,azimuth_deg,elevation_deg,h_az_real,h_az_imag,h_el_real,h_el_imag",
        *(f"{corner},1575420000,0,0,{component},0,0,0" for corner in _CORNERS),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _unresolved_measured_simulation() -> object:
    simulation = default_config()
    simulation.array.antenna_model = PhoneAntennaConfig(
        "measured failure fixture",
        "measured",
        AntennaStateConfig(
            pattern_kind="measured",
            measured_pattern_csv=Path("patterns.csv"),
        ),
        AntennaStateConfig(),
    )
    return simulation


def test_quick_preset_has_exact_axes_and_counts():
    resolved = resolve_study_config(preset="quick")

    assert resolved.study.spatial_modes == ("egc", "mvdr")
    assert resolved.study.full_chain_modes == (
        "single", "egc", "mvdr", "blind_null",
    )
    np.testing.assert_array_equal(resolved.study.azimuths_deg, np.arange(0, 360, 30))
    np.testing.assert_array_equal(resolved.study.elevations_deg, np.arange(0, 91, 15))
    np.testing.assert_allclose(resolved.study.powers_dbm, [-110, -100, -90, -80])
    assert resolved.study.combined_error_samples_per_level == 50
    assert resolved.study.full_chain_run_count == 3
    assert resolved.study.ranking_mode == "mvdr"


def test_blind_null_is_full_chain_only():
    resolved = resolve_study_config(
        preset="quick",
        cli_overrides={"full_chain_modes": ["single", "blind_null"]},
    )
    assert resolved.study.full_chain_modes == ("single", "blind_null")

    with pytest.raises(ValueError, match="blind_null.*full-chain"):
        resolve_study_config(
            preset="quick",
            cli_overrides={"spatial_modes": ["mvdr", "blind_null"]},
        )


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


def test_json_overrides_tracking_carrier_nco_fields(tmp_path: Path):
    path = tmp_path / "tracking-nco.json"
    path.write_text(
        json.dumps(
            {
                "simulation": {
                    "tracking": {
                        "carrier_nco_mode": "fixed",
                        "carrier_nco_clock_hz": 6_600_000.0,
                        "carrier_nco_word_bits": 43,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_study_config(path, preset="quick")

    assert resolved.simulation.tracking.carrier_nco_mode == "fixed"
    assert resolved.simulation.tracking.carrier_nco_clock_hz == 6.6e6
    assert resolved.simulation.tracking.carrier_nco_word_bits == 43
    resolved.simulation.validate()


def test_json_overrides_channel_delay_maps_and_hashes_them(tmp_path: Path):
    true_delays = {"TL": 0.0, "TR": 10.0, "BL": 2.0, "BR": 4.0}
    estimated_delays = {"TL": 0.0, "TR": 8.0, "BL": 1.0, "BR": 3.0}
    path = tmp_path / "channel-delays.json"
    path.write_text(
        json.dumps(
            {
                "simulation": {
                    "array": {
                        "channel_delay_ns": true_delays,
                        "estimated_channel_delay_ns": estimated_delays,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    baseline = resolve_study_config(preset="quick")
    resolved = resolve_study_config(path, preset="quick")

    assert resolved.simulation.array.channel_delay_ns == true_delays
    assert resolved.simulation.array.estimated_channel_delay_ns == estimated_delays
    assert resolved.config_sha256 != baseline.config_sha256


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


@pytest.mark.parametrize(
    "axis_name",
    (
        "azimuths_deg",
        "elevations_deg",
        "powers_dbm",
        "direction_errors_deg",
        "position_errors_m",
        "amplitude_errors_db",
        "phase_errors_deg",
        "loading_factors",
    ),
)
def test_resolved_scientific_axes_cannot_have_write_access_restored(axis_name):
    resolved = resolve_study_config(preset="quick")
    before = resolved.to_dict()
    digest = resolved.config_sha256
    axis = getattr(resolved.study, axis_name)

    with pytest.raises(ValueError):
        axis.setflags(write=True)
    with pytest.raises(ValueError):
        axis[0] = axis[0] + 1.0

    assert resolved.to_dict() == before
    assert resolved.config_sha256 == digest


def test_from_parts_irreversibly_freezes_detached_caller_axis():
    base = resolve_study_config(preset="quick")
    caller_axis = np.array([10.0, 20.0])
    study = __import__("dataclasses").replace(base.study, azimuths_deg=caller_axis)

    resolved = ResolvedStudyConfig.from_parts(study, base.simulation)
    before = resolved.to_dict()
    digest = resolved.config_sha256
    caller_axis[0] = 99.0

    with pytest.raises(ValueError):
        resolved.study.azimuths_deg.setflags(write=True)
    assert resolved.to_dict() == before
    assert resolved.study.azimuths_deg.tolist() == [10.0, 20.0]
    assert resolved.config_sha256 == digest


@pytest.mark.parametrize("single_antennas", [(), ("TL", "TL"), ("TL", "XX")])
def test_single_antennas_must_be_a_nonempty_unique_corner_subset(single_antennas):
    with pytest.raises(ValueError, match="single_antennas"):
        resolve_study_config(preset="quick", cli_overrides={"single_antennas": single_antennas})


@pytest.mark.parametrize(
    "study_update",
    [
        {"error_levels": [{}]},
        {"multi_jammer_templates": [{"name": "x", "interferers": [{}]}]},
    ],
)
def test_incomplete_nested_study_records_raise_value_error(tmp_path: Path, study_update):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"study": study_update}), encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_study_config(path)


def test_resolved_containers_are_canonical_and_immutable_after_from_parts():
    base = resolve_study_config(preset="quick", cli_overrides={"manual_cases": [{"key": "value"}]})
    study_with_lists = __import__("dataclasses").replace(
        base.study,
        spatial_modes=["egc", "mvdr"],
        full_chain_modes=["single", "egc", "mvdr"],
        combinations=[["TL", "BR"]],
        single_antennas=["TL"],
        automatic_roles=["best"],
    )
    resolved = ResolvedStudyConfig.from_parts(study_with_lists, base.simulation)
    before = resolved.to_dict()
    digest = resolved.config_sha256

    with pytest.raises(TypeError):
        resolved.study.manual_cases[0] |= {"other": "value"}
    with pytest.raises(AttributeError):
        resolved.study.spatial_modes.append("single")

    assert resolved.study.spatial_modes == ("egc", "mvdr")
    assert resolved.study.combinations == (("TL", "BR"),)
    assert resolved.to_dict() == before
    assert resolved.config_sha256 == digest


def test_from_parts_canonicalizes_inner_interferer_template_containers():
    base = resolve_study_config(preset="quick")
    cw = base.study.multi_jammer_templates[0].interferers[0]
    mutable_interferers = [cw]
    raw = __import__("dataclasses").replace(
        base.study,
        multi_jammer_templates=[InterfererTemplate("mutable", mutable_interferers)],
    )
    resolved = ResolvedStudyConfig.from_parts(raw, base.simulation)
    before = resolved.to_dict()
    digest = resolved.config_sha256

    mutable_interferers.append(cw)

    assert isinstance(resolved.study.multi_jammer_templates, tuple)
    assert isinstance(resolved.study.multi_jammer_templates[0].interferers, tuple)
    with pytest.raises(AttributeError):
        resolved.study.multi_jammer_templates[0].interferers.append(cw)
    assert resolved.to_dict() == before
    assert resolved.config_sha256 == digest


def test_resolver_parses_complete_parametric_antenna_and_polarization_records(
    tmp_path: Path,
):
    path = tmp_path / "study.json"
    path.write_text(json.dumps(_parametric_override()), encoding="utf-8")

    resolved = resolve_study_config(path, preset="quick")
    model = resolved.simulation.array.antenna_model

    assert isinstance(model, PhoneAntennaConfig)
    assert model.model_source == "parametric"
    assert isinstance(model.true_state.parametric_patterns["TL"], ParametricPatternParameters)
    assert model.true_state.parametric_patterns["TL"].peak_gain_db == 1.0
    assert model.assumed_state.parametric_patterns["TL"].nominal_polarization.kind == "linear"
    assert isinstance(model.true_state.orientations["TR"], ElementOrientation)
    assert model.true_state.orientations["TR"].yaw_deg == 5.0
    assert model.true_state.attenuation_regions == (
        AttenuationRegion(("TL", "BR"), 110.0, 25.0, 20.0, 4.0, 5.0),
    )
    assert model.true_state.coupling_paths == (
        CouplingPath("TL", "BR", -22.0, 40.0),
    )
    assert resolved.simulation.signal.true_polarization.kind == "rhcp"
    assert resolved.simulation.signal.assumed_polarization.kind == "lhcp"
    assert resolved.simulation.interferers[0].polarization.kind == "linear"


def test_antenna_and_source_polarization_fields_change_scientific_hash(tmp_path: Path):
    first_path = tmp_path / "first.json"
    first_path.write_text(json.dumps(_parametric_override()), encoding="utf-8")
    changed_override = _parametric_override()
    changed_override["simulation"]["array"]["antenna_model"]["assumed_state"][  # type: ignore[index]
        "parametric_patterns"
    ]["TL"]["peak_gain_db"] = 9.0  # type: ignore[index]
    changed_override["simulation"]["signal"]["assumed_polarization"] = (  # type: ignore[index]
        _polarization("linear", tilt_deg=22.0)
    )
    second_path = tmp_path / "second.json"
    second_path.write_text(json.dumps(changed_override), encoding="utf-8")

    first = resolve_study_config(first_path, preset="quick")
    second = resolve_study_config(second_path, preset="quick")

    assert first.config_sha256 != second.config_sha256


def test_measured_pattern_relative_path_and_missing_digest_are_resolved(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    csv_path = config_dir / "patterns.csv"
    _write_pattern_csv(csv_path)
    path = config_dir / "study.json"
    original_json = json.dumps(_measured_override("patterns.csv"))
    path.write_text(original_json, encoding="utf-8")

    resolved = resolve_study_config(path, preset="quick")
    state = resolved.simulation.array.antenna_model.true_state
    expected_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    assert state.measured_pattern_csv == csv_path.resolve()
    assert state.measured_pattern_sha256 == expected_digest
    assert path.read_text(encoding="utf-8") == original_json


def test_measured_pattern_digest_changes_scientific_hash_without_pinned_sha(
    tmp_path: Path,
):
    csv_path = tmp_path / "patterns.csv"
    _write_pattern_csv(csv_path, component=1.0)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(_measured_override(csv_path.name)), encoding="utf-8")

    first = resolve_study_config(path, preset="quick")
    first_snapshot = first.to_dict()
    _write_pattern_csv(csv_path, component=0.5)
    second = resolve_study_config(path, preset="quick")

    assert first.config_sha256 != second.config_sha256
    assert first.to_dict() == first_snapshot


def test_measured_pattern_changed_pinned_digest_is_rejected_without_mutation(
    tmp_path: Path,
):
    csv_path = tmp_path / "patterns.csv"
    _write_pattern_csv(csv_path, component=1.0)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    path = tmp_path / "study.json"
    original_json = json.dumps(_measured_override(csv_path.name, expected_sha256=digest))
    path.write_text(original_json, encoding="utf-8")
    resolve_study_config(path, preset="quick")
    _write_pattern_csv(csv_path, component=0.25)

    with pytest.raises(ValueError, match="SHA-256.*match"):
        resolve_study_config(path, preset="quick")

    assert path.read_text(encoding="utf-8") == original_json


def test_measured_pattern_missing_file_is_a_contextual_value_error(tmp_path: Path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps(_measured_override("missing.csv")), encoding="utf-8")

    with pytest.raises(ValueError, match="measured pattern CSV"):
        resolve_study_config(path, preset="quick")


def test_invalid_utf8_json_is_normalized_to_contextual_value_error(tmp_path: Path):
    path = tmp_path / "study.json"
    raw = b'\xff{"simulation": {}}'
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="JSON configuration.*UTF-8") as caught:
        resolve_study_config(path, preset="quick")

    assert type(caught.value) is ValueError
    assert path.read_bytes() == raw


def test_json_path_resolution_failure_is_contextual_and_preserves_file(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "study.json"
    raw = b'{"simulation": {}}'
    path.write_bytes(raw)
    original_resolve = Path.resolve

    def fail_source_resolve(self, *args, **kwargs):
        if self == path:
            raise OSError("fixture resolve failure")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_source_resolve)

    with pytest.raises(ValueError, match="JSON configuration path") as caught:
        resolve_study_config(path, preset="quick")

    assert type(caught.value) is ValueError
    assert path.read_bytes() == raw


def test_json_read_failure_is_contextual_and_preserves_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "study.json"
    raw = b'{"simulation": {}}'
    path.write_bytes(raw)
    original_read_text = Path.read_text

    def fail_source_read(self, *args, **kwargs):
        if self == path:
            raise OSError("fixture read failure")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_source_read)

    with pytest.raises(ValueError, match="cannot read JSON configuration") as caught:
        resolve_study_config(path, preset="quick")

    assert type(caught.value) is ValueError
    assert path.read_bytes() == raw


def test_measured_csv_path_resolution_failure_is_contextual_and_preserves_caller(
    tmp_path: Path, monkeypatch
):
    simulation = _unresolved_measured_simulation()
    before = strict_snapshot(simulation)
    csv_path = tmp_path / "patterns.csv"
    original_resolve = Path.resolve

    def fail_csv_resolve(self, *args, **kwargs):
        if self == csv_path:
            raise OSError("fixture resolve failure")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_csv_resolve)

    with pytest.raises(
        ValueError,
        match="true_state.*measured pattern CSV path",
    ) as caught:
        study_config_module.resolve_pattern_resources(simulation, tmp_path)

    assert type(caught.value) is ValueError
    assert strict_snapshot(simulation) == before


def test_measured_csv_read_failure_is_contextual_and_preserves_caller(
    tmp_path: Path, monkeypatch
):
    csv_path = tmp_path / "patterns.csv"
    csv_path.write_text("fixture", encoding="utf-8")
    simulation = _unresolved_measured_simulation()
    before = strict_snapshot(simulation)
    original_read_bytes = Path.read_bytes

    def fail_csv_read(self, *args, **kwargs):
        if self == csv_path:
            raise OSError("fixture read failure")
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", fail_csv_read)

    with pytest.raises(ValueError, match="cannot read.*true_state.*measured pattern CSV") as caught:
        study_config_module.resolve_pattern_resources(simulation, tmp_path)

    assert type(caught.value) is ValueError
    assert strict_snapshot(simulation) == before


def test_measured_csv_invalid_utf8_is_contextual_and_preserves_caller(tmp_path: Path):
    csv_path = tmp_path / "patterns.csv"
    csv_path.write_bytes(b"\xff")
    simulation = _unresolved_measured_simulation()
    before = strict_snapshot(simulation)

    with pytest.raises(ValueError, match="true_state.*measured pattern CSV must be UTF-8") as caught:
        study_config_module.resolve_pattern_resources(simulation, tmp_path)

    assert type(caught.value) is ValueError
    assert strict_snapshot(simulation) == before


def test_resolve_pattern_resources_returns_a_detached_resolved_copy(tmp_path: Path):
    assert hasattr(study_config_module, "resolve_pattern_resources")
    csv_path = tmp_path / "patterns.csv"
    _write_pattern_csv(csv_path)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    simulation = default_config()
    simulation.array.antenna_model = PhoneAntennaConfig(
        "measured direct fixture",
        "measured",
        AntennaStateConfig(
            pattern_kind="measured",
            measured_pattern_csv=Path("patterns.csv"),
            measured_pattern_sha256=digest,
        ),
        AntennaStateConfig(),
    )

    resolved = study_config_module.resolve_pattern_resources(simulation, tmp_path)

    assert resolved is not simulation
    assert resolved.array.antenna_model.true_state.measured_pattern_csv == csv_path.resolve()
    assert simulation.array.antenna_model.true_state.measured_pattern_csv == Path(
        "patterns.csv"
    )


def test_old_json_without_antenna_or_polarization_fields_stays_ideal(tmp_path: Path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({
        "simulation": {
            "signal": {"carrier_hz": 1_575_420_000.0},
            "interferers": [{
                "direction": {"azimuth_deg": 120.0, "elevation_deg": 10.0},
                "enable": True,
                "power_dbm": -90.0,
                "offset_hz": 250e3,
                "initial_phase_rad": 0.8,
                "start_ms": 0.0,
                "stop_ms": None,
            }],
        }
    }), encoding="utf-8")

    resolved = resolve_study_config(path, preset="quick")

    assert resolved.simulation.array.antenna_model is None
    assert resolved.simulation.signal.true_polarization.kind == "rhcp"
    assert resolved.simulation.interferers[0].polarization.kind == "rhcp"


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda data: data["simulation"]["array"]["antenna_model"]["true_state"][  # type: ignore[index]
                "parametric_patterns"
            ]["TL"].pop("cross_phase_deg"),  # type: ignore[index]
            id="incomplete-parametric-pattern",
        ),
        pytest.param(
            lambda data: data["simulation"]["array"]["antenna_model"]["true_state"].update(  # type: ignore[index]
                {"orientations": []}
            ),
            id="orientation-list-instead-of-map",
        ),
        pytest.param(
            lambda data: data["simulation"]["array"]["antenna_model"]["true_state"][  # type: ignore[index]
                "attenuation_regions"
            ][0].update({"elements": "TL"}),  # type: ignore[index]
            id="region-elements-string",
        ),
        pytest.param(
            lambda data: data["simulation"]["array"]["antenna_model"]["true_state"][  # type: ignore[index]
                "coupling_paths"
            ][0].pop("phase_deg"),  # type: ignore[index]
            id="incomplete-coupling-path",
        ),
        pytest.param(
            lambda data: data["simulation"]["signal"].update(  # type: ignore[index]
                {"true_polarization": _polarization("invalid")}
            ),
            id="invalid-source-polarization",
        ),
        pytest.param(
            lambda data: data["simulation"]["signal"]["true_polarization"].update(  # type: ignore[index]
                {"tilt_deg": True}
            ),
            id="boolean-source-polarization-number",
        ),
        pytest.param(
            lambda data: data["simulation"]["interferers"][0].update(  # type: ignore[index]
                {"power_dbm": True}
            ),
            id="boolean-interferer-number",
        ),
        pytest.param(
            lambda data: data["simulation"]["array"]["antenna_model"]["true_state"][  # type: ignore[index]
                "parametric_patterns"
            ]["TL"].update({"peak_gain_db": float("nan")}),  # type: ignore[index]
            id="nonfinite-pattern-number",
        ),
    ],
)
def test_nested_antenna_records_fail_strictly_without_mutating_json(
    tmp_path: Path, mutate
):
    data = _parametric_override()
    mutate(data)
    path = tmp_path / "study.json"
    original_json = json.dumps(data)
    path.write_text(original_json, encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_study_config(path, preset="quick")

    assert path.read_text(encoding="utf-8") == original_json


def test_from_parts_detaches_complete_antenna_scientific_snapshot():
    simulation = default_config()
    patterns = {
        corner: ParametricPatternParameters(peak_gain_db=float(index))
        for index, corner in enumerate(_CORNERS)
    }
    orientations = {
        corner: ElementOrientation(yaw_deg=float(index))
        for index, corner in enumerate(_CORNERS)
    }
    state = AntennaStateConfig(
        pattern_kind="parametric",
        parametric_patterns=patterns,
        measured_pattern_csv=Path("unused.csv"),
        orientations=orientations,
        attenuation_regions=(
            AttenuationRegion(("TL",), 0.0, 20.0, 10.0, 3.0),
        ),
        coupling_paths=(CouplingPath("TL", "BR", -20.0, 15.0),),
    )
    model = PhoneAntennaConfig("detached fixture", "parametric", state, AntennaStateConfig())
    source_polarization = WavePolarization("linear", tilt_deg=30.0)
    simulation.array.antenna_model = model
    simulation.signal.true_polarization = source_polarization
    base = resolve_study_config(preset="quick")

    resolved = ResolvedStudyConfig.from_parts(base.study, simulation)
    before = resolved.to_dict()
    digest = resolved.config_sha256
    patterns["TL"] = ParametricPatternParameters(peak_gain_db=99.0)
    orientations["TL"] = ElementOrientation(yaw_deg=99.0)
    state.attenuation_regions += (
        AttenuationRegion(("BR",), 180.0, 30.0, 5.0, 8.0),
    )
    state.coupling_paths += (CouplingPath("BR", "TL", -10.0, 90.0),)
    state.measured_pattern_csv = Path("changed.csv")
    object.__setattr__(source_polarization, "tilt_deg", 80.0)

    assert resolved.to_dict() == before
    assert resolved.config_sha256 == digest
