"""Deterministic observation-model identifiability audit; NOT RF or PVT evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sdcp import geometry_free_sdcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    t = np.arange(1000)/1000
    a = np.zeros((len(t), 1))
    g = (.0759*np.sin(2*np.pi*t))[:, None]
    l1, l5 = 299792458/1575.42e6, 299792458/1176.45e6
    cases = {'changing_geometry': (g, g),
             'static_l1_multipath_8mm': (g+.008, g),
             'hardware_l1_bias_8mm': (g+.008, g),
             'common_frequency_error_20mm': (g+.020, g+.020),
             'one_cycle_each_frequency': (g+l1, g+l5),
             'l1_cycle_slip_at_500ms': (g+(t>=.5)[:,None]*l1, g)}
    outputs = {name: geometry_free_sdcp(t,a,b1,t,a,b5) for name,(b1,b5) in cases.items()}
    result = {'scope': __doc__, 'epochs': len(t), 'baseline_amplitude_m': .0759,
              'wavelengths_m': [l1,l5], 'cases': {}}
    for name, v in outputs.items():
        result['cases'][name] = {'mean_m': float(v.mean()), 'std_m': float(v.std()),
                                'max_abs_m': float(abs(v).max())}
    result['static_multipath_vs_hardware_max_difference_m'] = float(abs(
        outputs['static_l1_multipath_8mm']-outputs['hardware_l1_bias_8mm']).max())
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), Path(__file__).with_name('sdcp.py')]}
    path = args.output/'summary.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
