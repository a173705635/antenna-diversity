"""独立源复响应：在理想空间响应之后、真实通道时延之前作用一次。"""

from dataclasses import dataclass, field

import numpy as np

from core.config import CORNER_NAMES
from core.gnss import generate_scenario
from core.models import AntennaDiagnostics, ScenarioTruth


RESPONSE_CONVENTION = (
    'Source-level voltage gain and phase multiply the ideal geometric manifold '
    'once, before true hardware delay. They exclude geometry, hardware channel '
    'amplitude/phase, and delay; assumed coefficients change steering only. '
    'Estimated delay supplies narrowband phase compensation, not code-envelope alignment.'
)


@dataclass
class SourceResponse:
    gain_db: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in CORNER_NAMES}
    )
    phase_deg: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in CORNER_NAMES}
    )

    def coefficients(self, active_antennas: tuple[str, ...]) -> np.ndarray:
        """按激活天线顺序返回无量纲复电压系数。"""
        return np.asarray([
            10.0 ** (self.gain_db[name] / 20.0)
            * np.exp(1j * np.deg2rad(self.phase_deg[name]))
            for name in active_antennas
        ])


@dataclass
class DirectResponses:
    desired_true: SourceResponse
    desired_assumed: SourceResponse
    interferers_true: tuple[SourceResponse, ...]


def _scale_diagnostics(stages: AntennaDiagnostics, coefficients: np.ndarray) -> None:
    """仅更新源相关电压级；硬件通道响应不受源系数影响。"""
    for key in ('pattern_polarization_voltage', 'open_circuit_response',
                'coupled_response', 'effective_manifold'):
        stages[key] = stages[key] * coefficients


def generate_direct_scenario(cfg, responses: DirectResponses) -> tuple[np.ndarray, ScenarioTruth]:
    """在理想无互耦场景的目标/干扰分量分别注入非几何源响应。

    源系数乘于理想阵元电压，不能直接套用于已有互耦的通用天线场景。
    """
    _, truth = generate_scenario(cfg)
    active_antennas = cfg.phone.active_antennas
    desired_true = responses.desired_true.coefficients(active_antennas)
    desired_assumed = responses.desired_assumed.coefficients(active_antennas)
    interferers = [response.coefficients(active_antennas)
                   for response in responses.interferers_true]

    truth['desired_component'] = truth['desired_component'] * desired_true[:, None]
    truth['desired_steering_true'] = truth['desired_steering_true'] * desired_true
    truth['desired_steering_assumed'] = truth['desired_steering_assumed'] * desired_assumed
    truth['desired_open_circuit_response_true'] = (
        truth['desired_open_circuit_response_true'] * desired_true
    )
    truth['desired_open_circuit_response_assumed'] = (
        truth['desired_open_circuit_response_assumed'] * desired_assumed
    )
    diagnostics = truth['antenna_model_diagnostics']
    _scale_diagnostics(diagnostics['desired_true'], desired_true)
    _scale_diagnostics(diagnostics['desired_assumed'], desired_assumed)
    for index, coefficients in enumerate(interferers):
        truth['interferer_components'][index] = (
            truth['interferer_components'][index] * coefficients[:, None]
        )
        truth['interferer_steering_effective'][index] = (
            truth['interferer_steering_effective'][index] * coefficients
        )
        _scale_diagnostics(diagnostics['interferers'][index], coefficients)
    truth['source_response_convention'] = RESPONSE_CONVENTION
    truth['source_response_coefficients'] = {
        'desired_true': desired_true,
        'desired_assumed': desired_assumed,
        'interferers_true': interferers,
    }
    scene = truth['desired_component'].copy()
    for component in truth['interferer_components']:
        scene += component
    return scene, truth


def direct_response_row_fields(truth: ScenarioTruth, weights: np.ndarray) -> dict:
    """事后计算源/天线响应诊断，不包含后续 DFE 相位处理，也不参与求权。"""
    true_target_response = truth['desired_steering_true']
    assumed_target_response = truth['desired_steering_assumed']
    return {
        'mean_input_target_antenna_gain_db': float(
            10.0 * np.log10(np.vdot(true_target_response, true_target_response).real /
                            len(true_target_response))
        ),
        'target_gain': float(abs(np.vdot(weights, true_target_response))),
        'assumed_target_gain': float(abs(np.vdot(weights, assumed_target_response))),
        'true_assumed_correlation': float(abs(np.vdot(
            true_target_response, assumed_target_response)) /
            np.sqrt(np.vdot(true_target_response, true_target_response).real *
                    np.vdot(assumed_target_response, assumed_target_response).real)),
    }
