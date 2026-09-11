from pathlib import Path
import pypdfium2 as pdfium
import json
from zipfile import ZipFile
from lxml import etree
P=Path(__file__).parent
pdf=pdfium.PdfDocument(str(P/'report.pdf'))
out=P/'render';out.mkdir(exist_ok=True)
pages=[]
for i,page in enumerate(pdf):
    page.render(scale=1.5).to_pil().save(out/f'page-{i+1}.png')
    text=page.get_textpage().get_text_range()
    pages.append({'page':i+1,'text':text})
(P/'pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
for p in pages:print(p['page'],len(p['text']),p['text'][:65].replace('\r\n',' / '))
with ZipFile(P.parents[2]/'GNSS天线分集可行性与验证综合报告.docx') as z:
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    root=etree.fromstring(z.read('word/document.xml'))
    fields=root.xpath('//w:instrText/text()',namespaces=ns)
    assert any('TOC ' in x for x in fields)
    assert len(root.xpath('//m:oMath',namespaces=ns))==12
    assert not any('Error!' in p['text'] or '错误！' in p['text'] for p in pages)
print('OMML and field presence verified')
