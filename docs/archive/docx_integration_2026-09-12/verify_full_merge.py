from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from collections import Counter
import json,hashlib
import pypdfium2 as pdfium
P=Path(__file__).parent; R=P.parents[2]
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
def extract(blob):
    import io
    with ZipFile(io.BytesIO(blob)) as z:
        r=E.fromstring(z.read('word/document.xml'))
        text=''.join(r.xpath('//w:body//w:t/text() | //w:body//m:t/text()',namespaces=ns))
        counts={x:len(r.findall('.//'+x,ns)) for x in ['w:tbl','m:oMath','w:drawing','w:pict']}
        media=Counter(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/'))
        return text,counts,media
out=R/'GNSS天线分集全量整合报告.docx'
text,counts,media=extract(out.read_bytes()); report=[]; totals=Counter()
with ZipFile(P/'root_docx_before_full_merge.zip') as z:
    for item in json.loads((P/'full_merge_manifest.json').read_text(encoding='utf-8')):
        raw=z.read(item['name']); assert hashlib.sha256(raw).hexdigest()==item['sha256']
        t,c,m=extract(raw); totals.update(c)
        # Word may split runs, but concatenated text is unchanged.
        ok=t in text
        report.append({'name':item['name'],'full_text_contiguous':ok,'counts':c,'media_preserved':set(m)<=set(media)})
assert all(x['full_text_contiguous'] and x['media_preserved'] for x in report),report
assert all(counts[k]>=v for k,v in totals.items()),(counts,totals)
(P/'full_merge_verification.json').write_text(json.dumps({'sources':report,'expected_counts':dict(totals),'actual_counts':counts},ensure_ascii=False,indent=2),encoding='utf-8')
pdf=pdfium.PdfDocument(str(P/'full_merge.pdf')); d=P/'full_render';d.mkdir(exist_ok=True)
for i,page in enumerate(pdf): page.render(scale=1).to_pil().save(d/f'page-{i+1}.png')
print('All source texts, object counts and media verified; pages',len(pdf))
