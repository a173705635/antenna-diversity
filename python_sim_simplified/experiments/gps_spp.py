"""Single-antenna, eight-PRN assisted GPS L1 positioning experiment."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
from dataclasses import asdict
import numpy as np
from core.provenance import source_hashes, freeze_source
from core.serialization import serializable
from core.gps_simulation import receiver_reference, satellite_setup, build_scene, run_record
from experiments.paths import RESULTS, reset_output

OUTPUT = RESULTS/'gps_spp'
DURATION_MS = 1000
SEEDS = [3000, 3001, 3002]
FREEZE_SOURCE = False

def main():
    if DURATION_MS < 210:
        raise ValueError('GPS positioning is not evaluated: need at least 210 ms to include the first 200 ms observation and end margin.')
    output = reset_output(OUTPUT.name)
    if FREEZE_SOURCE:
        freeze_source(output)
    hashes = source_hashes()
    satellites = satellite_setup()
    print('Generating eight GPS PRNs before shared RF/DFE...', flush=True)
    scene, cfg = build_scene(satellites, DURATION_MS)
    from dataclasses import asdict
    metadata = {'scope': 'single antenna assisted synthetic GPS L1 SPP; no multipath/interference',
                'satellites': satellites, 'duration_ms': DURATION_MS, 'seeds': SEEDS,
                'satellite_frame': 'synthetic reception ECEF, already rotation-corrected by definition',
                'trajectory': 'linear radial motion, not broadcast ephemeris/orbital propagation',
                'time_aid': 'per-PRN pseudorange rounded to 10 km at t=0; only selects 1 ms ambiguity',
                'clock_bias_truth_m': 75000., 'receiver_truth_ecef_m': receiver_reference()[0],
                'satellite_clock_and_atmosphere': 'zero in source and solver',
                'quality_screen': 'fixed >=200 ms; lock metric >=0.8 and abs PLL discriminator <pi/2',
                'signal_config_last_prn': asdict(cfg.signal), 'rf_config': asdict(cfg.rf),
                'dfe_config': asdict(cfg.dfe), 'tracking_config': asdict(cfg.tracking),
                'source_sha256': hashes}
    (output/'manifest.json').write_text(json.dumps(serializable(metadata), indent=2, allow_nan=False), encoding='utf-8')
    all_records = []
    for seed in SEEDS:
        result = run_record(seed, satellites, scene, cfg)
        (output/f'seed_{seed}.json').write_text(json.dumps(serializable(result), indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(result['summary']), flush=True)
        all_records.append(result)
    (output/'summary.json').write_text(json.dumps([r['summary'] for r in all_records], indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    for record in all_records:
        label = f"seed {record['summary']['seed']}"
        axes[0].plot(record['time_s'], np.linalg.norm(record['enu_error_m'], axis=1), label=label)
        axes[1].plot(record['time_s'], record['clock_bias_m']-75000., label=label)
    axes[0].set(ylabel='3D position error (m)', title='Single antenna: eight GPS L1 PRNs, assisted synthetic SPP')
    axes[1].set(xlabel='Receiver time (s)', ylabel='Clock range error (m)')
    for axis in axes:
        axis.grid(alpha=.3)
        axis.legend()
    fig.savefig(output/'position_error.png', dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
