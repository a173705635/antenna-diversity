$ErrorActionPreference = 'Stop'
$reportPath = 'F:\antenna_diversity\GNSS天线分集全量整合报告.docx'
$pdfPath = 'F:\antenna_diversity\docs\reports\integration_work\full_merge.pdf'
$wordApp = New-Object -ComObject Word.Application
Write-Output 'Word created'
$wordApp.Visible = $false
$wordApp.DisplayAlerts = 0
try {
    $reportDoc = $wordApp.Documents.Open($reportPath, $false, $false)
    Write-Output 'Report opened'

    $reportDoc.Repaginate()


    $reportDoc.Repaginate()

    $reportDoc.Save()
    $reportDoc.ExportAsFixedFormat($pdfPath, 17)
    Write-Output ('Pages: ' + $reportDoc.ComputeStatistics(2))
    $reportDoc.Close(0)
} finally {
    $wordApp.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApp) | Out-Null
}
