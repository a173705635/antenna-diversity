from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,hashlib
P=Path(__file__).parent; R=P.parents[2]
out=R/'GNSS天线分集全量整合报告.docx'
files=sorted(f for f in R.glob('*.docx') if f!=out and not f.name.startswith('~$'))
archive=P/'root_docx_before_full_merge.zip'
assert not archive.exists(), 'Existing archive: stop to avoid overwrite'
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for f in files:z.write(f,f.name)
manifest=[{'name':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]
(P/'full_merge_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
script=r'''$ErrorActionPreference='Stop'
$root='F:\antenna_diversity'
$work=Join-Path $root 'docs\reports\integration_work'
$items=Get-Content -Raw -Encoding UTF8 (Join-Path $work 'full_merge_manifest.json') | ConvertFrom-Json
$word=New-Object -ComObject Word.Application
$word.Visible=$false
$word.DisplayAlerts=0
try {
 $doc=$word.Documents.Open((Join-Path $root 'GNSS天线分集可行性与验证综合报告.docx'),$false,$true)
 $doc.SaveAs2((Join-Path $root 'GNSS天线分集全量整合报告.docx'),16)
 foreach($item in $items) {
  if($item.name -eq 'GNSS天线分集可行性与验证综合报告.docx'){continue}
  $end=$doc.Content.End-1
  $range=$doc.Range($end,$end)
  $range.InsertBreak(7)
  $end=$doc.Content.End-1
  $range=$doc.Range($end,$end)
  $range.InsertAfter(('原文完整收录：'+$item.name+"`r"+'以下保留来源全文及其原始表述；历史状态、理论示例和待验证内容以综合报告的来源说明为准。'+"`r"))
  $end=$doc.Content.End-1
  $range=$doc.Range($end,$end)
  $range.InsertFile((Join-Path $root $item.name))
  Write-Output ('Inserted '+$item.name)
 }
 $doc.Repaginate()
 $doc.Save()
 $doc.ExportAsFixedFormat((Join-Path $work 'full_merge.pdf'),17)
 Write-Output ('Pages: '+$doc.ComputeStatistics(2))
 $doc.Close(0)
} finally {$word.Quit()}
'''
(P/'full_merge_word.ps1').write_text(script,encoding='utf-8-sig')
print('Archived',len(files),'source documents')
