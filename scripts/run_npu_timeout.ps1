# Run an IRON Python script with a wall-clock limit (for experiments that
# may hang the NPU). A hung XRT wait holds the Python GIL, so an in-process
# watchdog cannot fire; this kills the Python process from outside instead.
# Usage: .\scripts\run_npu_timeout.ps1 -Script experiments\x.py [-ScriptArgs "--a 1"] [-Seconds 300]
param([Parameter(Mandatory)][string]$Script, [string]$ScriptArgs = '', [int]$Seconds = 300)
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$log = [System.IO.Path]::GetTempFileName()
$argList = @('-NoProfile', '-File', (Join-Path $PSScriptRoot 'iron_python.ps1'), $Script)
if ($ScriptArgs) { $argList += $ScriptArgs.Split(' ') }
$p = Start-Process powershell -ArgumentList $argList -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $log -PassThru -WindowStyle Hidden
if (-not $p.WaitForExit($Seconds * 1000)) {
    # The model process is the large python.exe (weights are several hundred MB).
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.WorkingSetSize -gt 100MB } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    "TIMEOUT after $Seconds s"
}
Get-Content $log | Select-String -NotMatch "vswhere|operable program|^\s*$"
