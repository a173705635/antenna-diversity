"""Compare decoded fields and orbit/clocks with unmodified upstream RTKLIB C."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3] / 'python_sim_simplified'))
from gps_broadcast import satellite_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', action='store_true', help='Requires GCC; rebuild and execute the external C reference')
    args = parser.parse_args()
    source = ROOT / 'source/rtklib'
    manifest = json.loads((source / 'source.json').read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((source / Path(entry['path']).name).read_bytes()).hexdigest() == entry['sha256']
    navigation = json.loads((ROOT / 'run/position.json').read_text())['navigation']
    if args.rebuild:
        build = ROOT.parents[3] / 'python_sim_simplified/results/rtklib_rebuild'
        build.mkdir(exist_ok=True)
        exe = build / 'compare.exe'
        command = ['gcc', '-O2', '-flto', '-ffunction-sections', '-fdata-sections', '-I', str(source),
                   str(ROOT / 'rtklib_compare.c'),
                   *[str(source / f) for f in ('rtkcmn.c', 'rcvraw.c', 'ephemeris.c')],
                   '-Wl,--gc-sections', '-lwinmm', '-lws2_32', '-o', str(exe)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        lines = []
        for prn, nav in navigation.items():
            frames = {f['subframe_id']: f['payload_words'] for f in nav['subframes']}
            for t in (590465., 590480., 590499.):
                lines.append(' '.join(map(str, [prn, t, *sum([frames[i] for i in (1, 2, 3)], [])])))
        result = subprocess.run([str(exe)], input='\n'.join(lines)+'\n', capture_output=True, text=True, check=True)
        vectors = [json.loads(s) for s in result.stdout.splitlines()]
        (ROOT / 'rtklib_vectors.json').write_text(json.dumps(vectors, indent=2), encoding='utf-8')
        (ROOT / 'rtklib_build.json').write_text(json.dumps({'command': command,
            'compiler': subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0],
            'source_commit': manifest['commit'], 'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest()}, indent=2))
    vectors = json.loads((ROOT / 'rtklib_vectors.json').read_text())
    differences = []; clocks = []; checks = 0
    for v in vectors:
        eph = navigation[str(v['prn'])]['ephemeris']
        for name, value in v['ephemeris'].items():
            # Upstream SC2RAD=3.1415926535898 has fewer digits than numpy.pi.
            np.testing.assert_allclose(eph[name], value, rtol=5e-15, atol=1e-18)
            checks += 1
        xyz, clock = satellite_state(eph, v['tow_s'])
        differences.append(float(np.linalg.norm(xyz-v['position_m'])))
        clocks.append(float(abs(clock-v['clock_s'])))
    assert max(differences) < .001
    assert max(clocks) < 1e-12
    summary = {'source_commit': manifest['commit'], 'vectors': len(vectors),
        'satellites': len(set(v['prn'] for v in vectors)), 'ephemeris_field_checks': checks,
        'max_orbit_difference_m': max(differences), 'max_clock_difference_s': max(clocks),
        'scope': 'Same parity-validated subframe payloads; independent C field decoding and orbit/clock propagation. Not an independent ephemeris truth source or full-receiver comparison.'}
    (ROOT / 'rtklib_comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
