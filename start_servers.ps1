<#
.SYNOPSIS
    Starts the GraphRAG API and UI dev servers.
.DESCRIPTION
    start_servers.ps1 - Start the GraphRAG API + UI dev servers on Windows
    Usage: .\start_servers.ps1 [-Stop]
#>
param(
    [switch]$Stop
)

$Root = $PSScriptRoot
$ApiPidFile = "$env:TEMP\graphrag_api.pid"
$UiPidFile = "$env:TEMP\graphrag_ui.pid"
$ApiLog = "$env:TEMP\graphrag_api.log"
$UiLog = "$env:TEMP\graphrag_ui.log"
$PythonPath = "$Root\.venv\Scripts\python.exe"

if (-not (Test-Path $PythonPath)) {
    $PythonPath = "python"
}

if ($Stop) {
    Write-Host "Stopping servers…"
    if (Test-Path $ApiPidFile) {
        $pidApi = Get-Content $ApiPidFile
        Stop-Process -Id $pidApi -Force -ErrorAction SilentlyContinue
        Write-Host "  ✓ API server stopped"
        Remove-Item $ApiPidFile -ErrorAction SilentlyContinue
    }
    if (Test-Path $UiPidFile) {
        $pidUi = Get-Content $UiPidFile
        Stop-Process -Id $pidUi -Force -ErrorAction SilentlyContinue
        Write-Host "  ✓ UI dev server stopped"
        Remove-Item $UiPidFile -ErrorAction SilentlyContinue
    }
    # Also attempt to stop by name if pids were lost
    Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "api_server.py" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "vite" } | Stop-Process -Force -ErrorAction SilentlyContinue
    exit 0
}

Write-Host "=== GraphRAG Servers ==="

# ── API server ──
# Kill any existing processes
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "api_server.py" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500

$apiProc = Start-Process -FilePath $PythonPath -ArgumentList "-u", "$Root\api_server.py" -WorkingDirectory $Root -NoNewWindow -PassThru -RedirectStandardOutput $ApiLog -RedirectStandardError $ApiLog
$apiProc.Id | Set-Content $ApiPidFile

$ApiUp = $false
for ($i = 1; $i -le 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:5001/api/health" -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            Write-Host "  ✓ API server running  → http://localhost:5001  (PID $($apiProc.Id))"
            $ApiUp = $true
            break
        }
    } catch {
        # ignore
    }
    if ($i -eq 15) {
        Write-Host "  ✗ API server failed to start. Log:"
        Get-Content $ApiLog -Tail 20
        exit 1
    }
}

# ── Vite UI ──
Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "vite" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500

$uiProc = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev", "--", "--host", "0.0.0.0" -WorkingDirectory "$Root\ui" -NoNewWindow -PassThru -RedirectStandardOutput $UiLog -RedirectStandardError $UiLog
$uiProc.Id | Set-Content $UiPidFile

$UiUp = $false
for ($i = 1; $i -le 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:5173" -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            Write-Host "  ✓ UI dev server running → http://localhost:5173  (PID $($uiProc.Id))"
            $UiUp = $true
            break
        }
    } catch {
        # ignore
    }
    if ($i -eq 15) {
        Write-Host "  ✗ UI dev server failed to start. Log:"
        Get-Content $UiLog -Tail 20
        exit 1
    }
}

Write-Host ""
Write-Host "  Open → http://localhost:5173"
Write-Host "  Stop → .\start_servers.ps1 -Stop"
Write-Host "  Logs → $ApiLog  |  $UiLog"
