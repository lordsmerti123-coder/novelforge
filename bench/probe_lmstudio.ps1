# Probe: can the LM Studio local server be started, and does port 1234 come up.
# ASCII only: Windows PowerShell reads .ps1 as ANSI and mangles Cyrillic.

$exe = Join-Path $env:LOCALAPPDATA 'Programs\LM Studio\LM Studio.exe'
$lms = Join-Path $env:USERPROFILE '.lmstudio\bin\lms.exe'
$log = Join-Path $env:APPDATA 'LM Studio\logs\main.log'

Write-Output "=== launching LM Studio ==="
$before = if (Test-Path $log) { (Get-Item $log).Length } else { 0 }
$p = Start-Process -FilePath $exe -PassThru
Write-Output ("  pid={0}" -f $p.Id)

for ($i = 1; $i -le 12; $i++) {
    Start-Sleep -Seconds 5
    $procs = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like 'LM Studio*' })
    $port = Get-NetTCPConnection -State Listen -LocalPort 1234 -ErrorAction SilentlyContinue
    Write-Output ("  {0,3}s: processes={1} port1234={2}" -f ($i * 5), $procs.Count, $(if ($port) { 'LISTENING' } else { 'no' }))
    if ($procs.Count -gt 0 -and $port) { break }
}

Write-Output ""
Write-Output "=== lms server start ==="
$out = & $lms server start 2>&1 | Select-Object -First 6
$out | ForEach-Object { "  $_" }

Write-Output ""
Write-Output "=== new lines in main.log ==="
if (Test-Path $log) {
    $fs = [System.IO.File]::Open($log, 'Open', 'Read', 'ReadWrite')
    $fs.Seek($before, 'Begin') | Out-Null
    $reader = New-Object System.IO.StreamReader($fs)
    $new = $reader.ReadToEnd()
    $reader.Close()
    $fs.Close()
    if ($new.Trim()) {
        ($new -split "`n" | Select-Object -Last 12) | ForEach-Object { "  " + $_.Trim() }
    } else {
        Write-Output "  (no new lines)"
    }
}
