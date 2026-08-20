"""Serializable true and assumed phone-antenna model configuration."""

from dataclasses import dataclass, field
from pathlib import Path
import re

from .effects import AttenuationRegion, CouplingPath, ElementOrientation
from .patterns import ParametricPatternParameters


_CORNERS = ("TL", "TR", "BL", "BR")
_CORNER_SET = frozenset(_CORNERS)
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


def zero_orientations() -> dict[str, ElementOrientation]:
    return {corner: ElementOrientation() for corner in _CORNERS}


@dataclass
class AntennaStateConfig:
    pattern_kind: str = "isotropic"
    parametric_patterns: dict[str, ParametricPatternParameters] = field(
        default_factory=dict
    )
    measured_pattern_csv: Path | None = None
    measured_pattern_sha256: str | None = None
    orientations: dict[str, ElementOrientation] = field(
        default_factory=zero_orientations
    )
    attenuation_regions: tuple[AttenuationRegion, ...] = ()
    coupling_paths: tuple[CouplingPath, ...] = ()

    def validate(self) -> None:
        if not isinstance(self.pattern_kind, str) or self.pattern_kind not in {
            "isotropic",
            "parametric",
            "measured",
        }:
            raise ValueError(
                "antenna state pattern_kind must be isotropic, parametric, or measured"
            )
        if not isinstance(self.orientations, dict) or set(self.orientations) != _CORNER_SET:
            raise ValueError("antenna state orientations must map exactly TL/TR/BL/BR")
        if any(
            not isinstance(orientation, ElementOrientation)
            for orientation in self.orientations.values()
        ):
            raise ValueError("antenna state orientations must be ElementOrientation records")
        if not isinstance(self.parametric_patterns, dict):
            raise ValueError("parametric_patterns must be a corner mapping")
        if any(
            not isinstance(parameters, ParametricPatternParameters)
            for parameters in self.parametric_patterns.values()
        ):
            raise ValueError(
                "parametric_patterns values must be ParametricPatternParameters"
            )
        if (
            self.pattern_kind == "parametric" or self.parametric_patterns
        ) and set(self.parametric_patterns) != _CORNER_SET:
            raise ValueError(
                "parametric pattern_kind requires parameter records for TL/TR/BL/BR"
            )

        if self.measured_pattern_sha256 is not None and (
            not isinstance(self.measured_pattern_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.measured_pattern_sha256) is None
        ):
            raise ValueError("measured pattern SHA-256 must contain 64 hexadecimal digits")
        if self.pattern_kind == "measured":
            if not isinstance(self.measured_pattern_csv, Path):
                raise ValueError("measured pattern_kind requires a measured pattern CSV path")
            if not self.measured_pattern_csv.is_file():
                raise ValueError("measured pattern CSV path must exist and name a file")
            if self.measured_pattern_sha256 is None:
                raise ValueError("measured pattern_kind requires a measured pattern SHA-256")

        if not isinstance(self.attenuation_regions, tuple) or any(
            not isinstance(region, AttenuationRegion)
            for region in self.attenuation_regions
        ):
            raise ValueError("attenuation_regions must be AttenuationRegion records")
        for region in self.attenuation_regions:
            region.validate()

        if not isinstance(self.coupling_paths, tuple) or any(
            not isinstance(path, CouplingPath) for path in self.coupling_paths
        ):
            raise ValueError("coupling_paths must be CouplingPath records")
        seen_paths: set[tuple[str, str]] = set()
        for path in self.coupling_paths:
            path.validate()
            key = (path.source, path.receiver)
            if key in seen_paths:
                raise ValueError("antenna state contains a duplicate coupling path")
            seen_paths.add(key)


@dataclass
class PhoneAntennaConfig:
    model_label: str
    model_source: str
    true_state: AntennaStateConfig
    assumed_state: AntennaStateConfig

    def validate(self) -> None:
        if not isinstance(self.model_label, str) or not self.model_label:
            raise ValueError("antenna model_label must be a nonempty string")
        if not isinstance(self.model_source, str) or self.model_source not in {
            "ideal",
            "parametric",
            "measured",
        }:
            raise ValueError("antenna model_source must be ideal, parametric, or measured")
        if not isinstance(self.true_state, AntennaStateConfig) or not isinstance(
            self.assumed_state, AntennaStateConfig
        ):
            raise ValueError("antenna true_state and assumed_state must be state records")
        self.true_state.validate()
        self.assumed_state.validate()
        expected_true_kind = {
            "ideal": "isotropic",
            "parametric": "parametric",
            "measured": "measured",
        }[self.model_source]
        if self.true_state.pattern_kind != expected_true_kind:
            raise ValueError("model_source conflicts with true_state pattern_kind")


def resolved_phone_antenna_config(array_cfg: object) -> PhoneAntennaConfig:
    """Resolve a structured model or the legacy ideal-isotropic selector."""
    try:
        pattern_model = array_cfg.pattern_model
        antenna_model = array_cfg.antenna_model
    except AttributeError as error:
        raise ValueError("array_cfg must be a phone array configuration") from error
    if pattern_model != "isotropic":
        if antenna_model is not None:
            raise ValueError("pattern_model conflicts with antenna_model")
        raise ValueError("legacy pattern_model must be isotropic")
    if antenna_model is not None:
        if not isinstance(antenna_model, PhoneAntennaConfig):
            raise ValueError("antenna_model must be a PhoneAntennaConfig")
        antenna_model.validate()
        return antenna_model
    model = PhoneAntennaConfig(
        model_label="legacy isotropic",
        model_source="ideal",
        true_state=AntennaStateConfig(),
        assumed_state=AntennaStateConfig(),
    )
    model.validate()
    return model
