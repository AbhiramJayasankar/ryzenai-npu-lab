# Run experiment 013 NPU cases one at a time, each in a fresh PowerShell
# process with a wall-clock limit. Stops at the first sign of a hang or a
# device reset (no retries). Usage, from the repo root:
#   powershell -NoProfile -File experiments\013_run_npu.ps1 [-Cases a,b,c] [-Seconds 300]
param([string[]]$Cases = @(), [int]$Seconds = 300)
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo
$log = Join-Path $repo 'cache\013\npu_run.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
$Cases = @($Cases | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
if ($Cases.Count -eq 0) {
    $Cases = & powershell -NoProfile -File scripts\iron_python.ps1 experiments\013_npu_bench.py --list |
        Where-Object { $_ -match '^[a-z0-9_]+$' }
}
foreach ($c in $Cases) {
    $t = Get-Date
    $out = & powershell -NoProfile -File scripts\run_npu_timeout.ps1 -Script experiments\013_npu_bench.py -ScriptArgs $c -Seconds $Seconds | Out-String
    $secs = [int]((Get-Date) - $t).TotalSeconds
    Add-Content $log "=== $c ($secs s, $(Get-Date -Format s))`n$out"
    $line = ($out -split "`n" | Where-Object { $_ -match '^\{"case"' } | Select-Object -Last 1)
    if ($line) { Write-Output $line.Trim() } else { Write-Output "=== $c produced no result line:`n$out" }
    if ($out -match 'TIMEOUT|ERT_CMD_STATE_TIMEOUT|ERT_CMD_STATE_ABORT|0xc01e0009|Failed to create context|hw_context') {
        Write-Output "STOP: possible hang or device reset after case $c. No more NPU runs until checked."
        exit 3
    }
}
