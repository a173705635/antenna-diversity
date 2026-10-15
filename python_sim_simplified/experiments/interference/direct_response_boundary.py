"""Independent source-level direct-response boundary on ideal phone geometry.

Each per-corner voltage coefficient multiplies the ideal spatial manifold, not
the geometry, channel hardware response, or channel-delay carrier phase.
"""

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from core.config import AcquisitionConfig, CORNER_NAMES
from core.gnss import generate_scenario
from core.models import AntennaDiagnostics, ScenarioTruth
from core.presets import default_config
from core.provenance import source_hashes
from core.serialization import serializable
from experiments.interference.antenna_ablation import run_case
from experiments.paths import INTERFERENCE_RESULTS_DIR, reset_output


OUTPUT_DIR = INTERFERENCE_RESULTS_DIR / 'direct_response_boundary'
ANTENNAS = CORNER_NAMES
SEEDS = (42,)
DURATION_MS = 100
PHASE_ERRORS_DEG = (0, 20, 40, 60, 90, 120, 180)
PERTURBED_ANTENNA = ANTENNAS[1]
SHOW_PLOTS_FLAG = True
EVALUATION_EDGE_TRIM_MS = 10

# Edit each source's complete four-corner voltage gain and phase independently.
DESIRED_GAIN_DB = {name: 0.0 for name in CORNER_NAMES}
DESIRED_PHASE_DEG = {name: 0.0 for name in CORNER_NAMES}
CW_GAIN_DB = {name: 0.0 for name in CORNER_NAMES}
CW_PHASE_DEG = {name: 0.0 for name in CORNER_NAMES}
WIDEBAND_GAIN_DB = {name: 0.0 for name in CORNER_NAMES}
WIDEBAND_PHASE_DEG = {name: 0.0 for name in CORNER_NAMES}
# Hardware channel delays are separate from source coefficients. Estimated
# delays affect only assumed narrowband steering, not input code alignment.
TRUE_CHANNEL_DELAY_NS = {name: 0.0 for name in CORNER_NAMES}
ESTIMATED_CHANNEL_DELAY_NS = {name: 0.0 for name in CORNER_NAMES}

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

    def coefficients(self, antennas: tuple[str, ...]) -> np.ndarray:
        return np.asarray([
            10.0 ** (self.gain_db[name] / 20.0)
            * np.exp(1j * np.deg2rad(self.phase_deg[name]))
            for name in antennas
        ])


@dataclass
class DirectResponses:
    desired_true: SourceResponse
    desired_assumed: SourceResponse
    interferers_true: tuple[SourceResponse, ...]


def _scale_diagnostics(stages: AntennaDiagnostics, coefficients: np.ndarray) -> None:
    """Update only source-dependent voltage stages (ideal, uncoupled base)."""
    for key in ('pattern_polarization_voltage', 'open_circuit_response',
                'coupled_response', 'effective_manifold'):
        stages[key] = stages[key] * coefficients


def generate_direct_scenario(cfg, responses: DirectResponses) -> tuple[np.ndarray, ScenarioTruth]:
    """Apply independent non-geometric source voltages to an ideal scene.

    The caller supplies the ideal antenna configuration.
    Interferer coefficients follow the enabled-interferer order; this experiment
    uses both default CW and wideband sources for the fixed budget audit.
    """
    _, truth = generate_scenario(cfg)
    antennas = cfg.phone.active_antennas
    desired_true = responses.desired_true.coefficients(antennas)
    desired_assumed = responses.desired_assumed.coefficients(antennas)
    interferers = [response.coefficients(antennas)
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


def build_config(seed, duration_ms, antennas):
    cfg = default_config()
    cfg.random_seed = seed
    cfg.signal.duration_ms = cfg.tracking.max_epochs = duration_ms
    cfg.phone.active_antennas = antennas
    cfg.phone.channel_delay_ns = TRUE_CHANNEL_DELAY_NS.copy()
    cfg.phone.estimated_channel_delay_ns = ESTIMATED_CHANNEL_DELAY_NS.copy()
    return cfg


def build_responses(phase_error_deg, *, perturbed_antenna) -> tuple[DirectResponses, DirectResponses]:
    target_phase = DESIRED_PHASE_DEG.copy()
    target_phase[perturbed_antenna] += phase_error_deg
    target_true = SourceResponse(DESIRED_GAIN_DB.copy(), target_phase)
    interferers = (SourceResponse(CW_GAIN_DB.copy(), CW_PHASE_DEG.copy()),
                   SourceResponse(WIDEBAND_GAIN_DB.copy(), WIDEBAND_PHASE_DEG.copy()))
    known = DirectResponses(target_true,
                            SourceResponse(DESIRED_GAIN_DB.copy(), target_phase.copy()),
                            interferers)
    mismatch = DirectResponses(
        target_true,
        SourceResponse(DESIRED_GAIN_DB.copy(), DESIRED_PHASE_DEG.copy()),
        interferers,
    )
    return known, mismatch


def run_sweep(seeds=SEEDS, duration_ms=DURATION_MS, phases=PHASE_ERRORS_DEG,
              antennas=ANTENNAS, evaluation_edge_trim_ms=EVALUATION_EDGE_TRIM_MS):
    """Run each paired case through the existing audited RF/DFE receiver."""
    rows = []
    for seed in seeds:
        cfg = build_config(seed, duration_ms, antennas)
        for phase in phases:
            known, mismatch = build_responses(
                phase, perturbed_antenna=PERTURBED_ANTENNA
            )
            for name, response in (('known_response', known),
                                   ('unmodeled_phase', mismatch)):
                scene = generate_direct_scenario(cfg, response)
                case_rows = run_case(cfg, name, 'direct_response', evaluation_edge_trim_ms,
                                     scene=scene)
                for row in case_rows:
                    row['phase_error_deg'] = phase
                    row['antennas'] = ','.join(antennas)
                rows.extend(case_rows)
                print(seed, phase, name, 'complete', flush=True)
    return rows


def plot_summary(rows, output_dir, phases=PHASE_ERRORS_DEG):
    import matplotlib.pyplot as plt

    modes = tuple(dict.fromkeys(row['mode'] for row in rows))
    case_names = ('known_response', 'unmodeled_phase')
    colors = {'single': '#555555', 'egc': '#1f77b4',
              'mvdr': '#d95f02', 'blind_null': '#2ca02c'}
    def series(case, mode, key):
        values = []
        for phase in phases:
            members = [row[key] for row in rows
                       if row['case'] == case and row['mode'] == mode
                       and row['phase_error_deg'] == phase and row[key] is not None]
            values.append(float(np.mean(members)) if members else np.nan)
        return values

    figures = (
        ('acquisition_metric', (('acquisition_metric', 'Acquisition peak ratio'),)),
        ('target_and_interference', (
            ('target_output_gain_db', 'Target gain vs SINGLE (dB)'),
            ('cw_suppression_db', 'CW suppression vs SINGLE (dB)'),
            ('wideband_suppression_db', 'Wideband suppression vs SINGLE (dB)'),
        )),
        ('cn0_oracle_receiver', (
            ('thermal_oracle_prompt_cn0_db_hz', 'Thermal-only oracle C/N0 (dB-Hz)'),
            ('tracking_cn0_late_median_db_hz', 'Receiver late C/N0 (dB-Hz)'),
        )),
    )
    figure_paths = []
    for filename, metrics in figures:
        fig, axes = plt.subplots(len(metrics), len(case_names),
                                 figsize=(12, 3.8 * len(metrics)),
                                 squeeze=False, constrained_layout=True)
        for index, (key, ylabel) in enumerate(metrics):
            for case_index, case in enumerate(case_names):
                axis = axes[index, case_index]
                for mode in modes:
                    axis.plot(phases, series(case, mode, key),
                              marker='o', color=colors[mode], label=mode)
                if key == 'acquisition_metric':
                    axis.axhline(AcquisitionConfig().threshold, color='black',
                                 linestyle=':', label='acquisition threshold')
                axis.set(title=case.replace('_', ' '),
                         xlabel='Injected target phase (deg)', ylabel=ylabel)
                axis.grid(alpha=0.3)
                axis.legend(fontsize=8)
        figure_path = output_dir / f'{filename}.png'
        fig.savefig(figure_path, dpi=150)
        figure_paths.append(figure_path)
        if not SHOW_PLOTS_FLAG:
            plt.close(fig)
    if SHOW_PLOTS_FLAG and plt.get_backend().lower() != 'agg':
        plt.show()
    return figure_paths


def main():
    output_dir = reset_output(OUTPUT_DIR)
    manifest = {
        'seeds': SEEDS, 'duration_ms': DURATION_MS, 'phases_deg': PHASE_ERRORS_DEG,
        'antennas': ANTENNAS, 'perturbed_antenna': PERTURBED_ANTENNA,
        'evaluation_edge_trim_ms': EVALUATION_EDGE_TRIM_MS,
        'receiver_config_by_seed': {
            str(seed): asdict(build_config(seed, DURATION_MS, ANTENNAS))
            for seed in SEEDS
        },
        'true_channel_delay_ns': TRUE_CHANNEL_DELAY_NS,
        'estimated_channel_delay_ns': ESTIMATED_CHANNEL_DELAY_NS,
        'source_responses': {
            'desired_gain_db': DESIRED_GAIN_DB, 'desired_phase_deg': DESIRED_PHASE_DEG,
            'cw_gain_db': CW_GAIN_DB, 'cw_phase_deg': CW_PHASE_DEG,
            'wideband_gain_db': WIDEBAND_GAIN_DB,
            'wideband_phase_deg': WIDEBAND_PHASE_DEG,
        },
        'case_responses_by_phase': {
            str(phase): {
                'known_response': asdict(build_responses(
                    phase, perturbed_antenna=PERTURBED_ANTENNA
                )[0]),
                'unmodeled_phase': asdict(build_responses(
                    phase, perturbed_antenna=PERTURBED_ANTENNA
                )[1]),
            }
            for phase in PHASE_ERRORS_DEG
        },
        'response_convention': RESPONSE_CONVENTION,
        'metric_definitions': {
            'acquisition_metric': 'Actual receiver peak ratio; dotted line is configured acquisition threshold.',
            'target_output_gain_db': 'Post-hoc linear target power gain relative to same-case SINGLE.',
            'cw_suppression_db': 'Post-hoc CW power reduction relative to same-case SINGLE.',
            'wideband_suppression_db': 'Post-hoc wideband power reduction relative to same-case SINGLE.',
            'thermal_oracle_prompt_cn0_db_hz': 'Post-hoc true-code/Doppler 1 ms projection of target versus thermal noise only; not receiver-observable.',
            'tracking_cn0_late_median_db_hz': 'Median finite receiver C/N0 over late half of tracking; absent when unavailable.',
        },
        'source_sha256': source_hashes(),
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(serializable(manifest), indent=2, allow_nan=False, default=str),
        encoding='utf-8',
    )
    rows = run_sweep(SEEDS, DURATION_MS, PHASE_ERRORS_DEG,
                     ANTENNAS, EVALUATION_EDGE_TRIM_MS)
    summary = {'rows': rows}
    (output_dir / 'summary.json').write_text(
        json.dumps(serializable(summary), indent=2, allow_nan=False), encoding='utf-8'
    )
    with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot_summary(rows, output_dir, phases=PHASE_ERRORS_DEG)
    return summary


if __name__ == '__main__':
    main()
