$ErrorActionPreference = 'Stop'
$reportPath = 'F:\antenna_diversity\GNSS天线分集可行性与验证综合报告.docx'
$pdfPath = 'F:\antenna_diversity\docs\reports\integration_work\report.pdf'
$wordApp = New-Object -ComObject Word.Application
Write-Output 'Word created'
$wordApp.Visible = $false
$wordApp.DisplayAlerts = 0
try {
    $reportDoc = $wordApp.Documents.Open($reportPath, $false, $false)
    Write-Output 'Report opened'
    $reportDoc.Fields.Update() | Out-Null
    $reportDoc.Repaginate()
    foreach ($toc in $reportDoc.TablesOfContents) { $toc.Update() }
    $reportDoc.Fields.Update() | Out-Null
    $reportDoc.Repaginate()
    foreach ($toc in $reportDoc.TablesOfContents) { $toc.Update() }
    $reportDoc.Save()
    $reportDoc.ExportAsFixedFormat($pdfPath, 17)
    Write-Output ('Pages: ' + $reportDoc.ComputeStatistics(2))
    $reportDoc.Close(0)
} finally {
    $wordApp.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApp) | Out-Null
}
