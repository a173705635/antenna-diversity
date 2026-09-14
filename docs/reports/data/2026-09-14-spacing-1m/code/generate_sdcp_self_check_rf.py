"""Fresh 3-second dual RF records; unchanged earlier LOS calibration only."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from audit_sdcp_positioning import receiver, satellite_setup, serializable, WAVELENGTH
from audit_calibrated_sdcp import residuals
from sdcp import calibrated_sdcp


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=root/'docs/reports/data/2026-09-12-sdcp-fresh-rf')
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    old = root/'docs/reports/data/2026-09-11-sdcp-long'
    def save(name, value):
        (out/name).write_text(json.dumps(serializable(value),indent=2,allow_nan=False),encoding='utf-8')
    satellites = satellite_setup()
    save('manifest.json', {'satellites':satellites,'duration_ms':3000,
         'evaluation_seeds':[6400,6402],'calibration_seeds':[6200,6202,6204],
         'scope':__doc__})
    for seed in [6200,6202,6204]:
        name = f'residual_los_{seed}.json'
        shutil.copyfile(old/name,out/name)
    for seed in [6400,6402]:
        records = {}
        for channel,kind in [(0,'los'),(1,'los'),(1,'strong'),(1,'weak'),
                             (1,'static'),(0,'common'),(1,'common')]:
            print('NEW RF',seed,channel,kind,flush=True)
            d = receiver(seed,channel,kind,satellites,3000)
            records[channel,kind] = d
            save(f'{kind}_{seed}_{channel}.json',d)
        for kind in ['los','strong','weak','static','common']:
            pair = [records[0,'common' if kind=='common' else 'los'],records[1,kind]]
            t,s,g,_ = residuals(pair,satellites)
            save(f'residual_{kind}_{seed}.json',{'time_s':t,
                       'residual_m':calibrated_sdcp(s,g,WAVELENGTH,0)})
    save('sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')})


if __name__ == '__main__': main()
