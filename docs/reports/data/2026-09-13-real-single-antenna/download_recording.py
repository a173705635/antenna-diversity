"""Download the exact first 10 s used in this audit (200 MB, no RF synthesis)."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL='https://sdr.ion.org/HackRF/HackRF_Bands-L1.int8'
EXPECTED='220b37c6e57255483fb7858e7fab10255d7db0459d5134531ea660e9da87ca78'
SIZE=200000000


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        with args.output.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=EXPECTED: raise ValueError('existing file differs; refusing to overwrite')
        print('Existing recording verified');return
    args.output.parent.mkdir(parents=True,exist_ok=True)
    partial=args.output.with_suffix(args.output.suffix+'.part')
    req=urllib.request.Request(URL,headers={'Range':f'bytes=0-{SIZE-1}'})
    with urllib.request.urlopen(req,timeout=60) as response,partial.open('xb') as f:
        if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes 0-{SIZE-1}/'):
            raise ValueError('server did not honor the requested byte range')
        remaining=SIZE; digest=hashlib.sha256()
        while remaining:
            block=response.read(min(remaining,1024*1024))
            if not block: raise ValueError('truncated response')
            f.write(block);digest.update(block);remaining-=len(block)
    if digest.hexdigest()!=EXPECTED: raise ValueError('download hash mismatch; partial retained')
    partial.rename(args.output)
    print('Verified',args.output)


if __name__=='__main__':main()
