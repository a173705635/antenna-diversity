from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/reports/integration_work'
OUT.mkdir(exist_ok=True)
files=list(ROOT.glob('*.docx'))+list((ROOT/'验证方案').glob('*.docx'))+list((ROOT/'docs/reports').glob('*.docx'))
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
inventory=[]
for i,p in enumerate(sorted(files)):
    if p.name.startswith('~$'): continue
    with ZipFile(p) as z:
        root=etree.fromstring(z.read('word/document.xml'))
        lines=[]
        for el in root.xpath('//w:body/*',namespaces=ns):
            text=''.join(el.xpath('.//w:t/text()|.//m:t/text()',namespaces=ns))
            if text: lines.append(text)
        (OUT/f'source_{i:02}.txt').write_text('\n'.join(lines),encoding='utf-8')
        inventory.append({'id':i,'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'chars':sum(map(len,lines)),'tables':len(root.xpath('//w:tbl',namespaces=ns)),
            'math':len(root.xpath('//m:oMath',namespaces=ns)),'extract':f'source_{i:02}.txt'})
(OUT/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(inventory,ensure_ascii=False,indent=2))
