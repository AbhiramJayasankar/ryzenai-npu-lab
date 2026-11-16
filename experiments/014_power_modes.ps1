# Experiment 014, lead 1: run the unchanged, known-stable NPU programs in the
# NPU power mode that is currently set, one per fresh PowerShell process with
# a wall-clock limit. Stops at the first sign of a hang or device reset (no
# retries). The power mode itself is set separately with
#   & C:\Windows\System32\AMD\xrt-smi.exe configure --pmode <mode>
# Usage, from the repo root:
#   powershell -NoProfile -File experiments\014_power_modes.ps1 -Label default [-Jobs a,b]
param([Parameter(Mandatory)][string]$Label, [string[]]$Jobs = @())
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo
$dir = Join-Path $repo 'cache\014'
New-Item -ItemType Directory -Force $dir | Out-Null
$log = Join-Path $dir "power_$Label.log"
$smi = 'C:\Windows\System32\AMD\xrt-smi.exe'

# name, script, arguments, timeout in seconds. All of these ran cleanly in
# experiments 010, 012 and 013 with exactly these programs.
$all = @(
    @('health',    'experiments\010_x8_validate.py',     '--check',                          900),
    @('decode',    'experiments\010_x8_validate.py',     '--repeats 3',                      900),
    @('bw4',       'experiments\010_bandwidth_probe.py', '--streams 4 --repeats 20',         600),
    @('bw8',       'experiments\010_bandwidth_probe.py', '--streams 8 --repeats 20',         600),
    @('dma64m',    'experiments\013_npu_bench.py',       'memcpy_dma_64m',                   300),
    @('core1_16m', 'experiments\013_npu_bench.py',       'memcpy_core1_16m',                 300),
    @('gemm_bf16', 'experiments\013_npu_bench.py',       'gemm_bf16f32_wa_1024',             300),
    @('gemm_i8',   'experiments\013_npu_bench.py',       'gemm_i8_wa_1024',                  300),
    @('call',      'experiments\013_call_overhead.py',   'direct',                           300),
    @('vbench',    'experiments\012_vbench.py',          '',                                 300),
    @('encoder',   'experiments\012_encoder_test.py',    '1 24 --clip 0 --repeats 3',       1800),
    @('context',   'experiments\010_context_bench.py',   '',                                1800)
)
$Jobs = @($Jobs | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$mode = (& $smi examine -r platform --batch | Select-String 'Power Mode').ToString().Trim()
$os = Get-CimInstance Win32_OperatingSystem
$bat = Get-CimInstance Win32_Battery
Add-Content $log ("### {0} label={1} {2} free_ram_gb={3:N2} battery_status={4} charge={5}" -f `
    (Get-Date -Format s), $Label, $mode, ($os.FreePhysicalMemory / 1MB), $bat.BatteryStatus, $bat.EstimatedChargeRemaining)
Write-Output "mode: $mode"
foreach ($j in $all) {
    if ($Jobs.Count -and -not ($Jobs -contains $j[0])) { continue }
    $t = Get-Date
    $wrap = @('-NoProfile', '-File', 'scripts\run_npu_timeout.ps1', '-Script', $j[1], '-Seconds', $j[3])
    if ($j[2]) { $wrap += @('-ScriptArgs', $j[2]) }
    $out = & powershell @wrap | Out-String
    $secs = [int]((Get-Date) - $t).TotalSeconds
    Add-Content $log "=== $($j[0]) [$($j[1]) $($j[2])] ($secs s, $(Get-Date -Format s))`n$out"
    Write-Output "=== $($j[0]) ($secs s)"
    Write-Output (($out -split "`n" | Select-Object -Last 25) -join "`n")
    if ($out -match 'TIMEOUT|ERT_CMD_STATE_TIMEOUT|ERT_CMD_STATE_ABORT|0xc01e0009|Failed to create context|Traceback|exited with code') {
        Write-Output "STOP: error or possible hang after $($j[0]). No more NPU runs until checked."
        exit 3
    }
}
