param(
    [Parameter(Mandatory = $true)][string]$Label,
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string[]]$ExecutableArguments,
    [int]$IdleSeconds = 20,
    [int]$WarmupSeconds = 15,
    [int]$MinimumActiveSamples = 15
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$cacheDir = Join-Path $repoRoot 'cache'
New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null

function Get-BatteryReading {
    $battery = @(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus) |
        Where-Object { $_.RemainingCapacity -gt 0 } |
        Select-Object -First 1
    if (-not $battery) { throw 'No battery telemetry is available.' }
    if ($battery.PowerOnline -or -not $battery.Discharging) {
        throw 'Unplug AC power before a battery power run.'
    }
    return [pscustomobject]@{
        Timestamp = Get-Date
        Watts = [double]$battery.DischargeRate / 1000.0
        RemainingMWh = [double]$battery.RemainingCapacity
    }
}

function Get-Median([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) { throw 'No valid battery power samples.' }
    $middle = [int][math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 1) { return [double]$sorted[$middle] }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
}

$startBattery = Get-BatteryReading
$idle = @()
for ($second = 0; $second -lt $IdleSeconds; $second++) {
    $reading = Get-BatteryReading
    if ($reading.Watts -gt 0) { $idle += $reading }
    Start-Sleep -Seconds 1
}
if ($idle.Count -lt 5) { throw 'Battery telemetry did not provide enough idle samples.' }

$safeLabel = $Label -replace '[^a-zA-Z0-9_-]', '_'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdoutPath = Join-Path $cacheDir "battery-$safeLabel-$stamp.stdout.json"
$stderrPath = Join-Path $cacheDir "battery-$safeLabel-$stamp.stderr.txt"
$process = Start-Process -FilePath $Executable -ArgumentList $ExecutableArguments `
    -WorkingDirectory $repoRoot -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -WindowStyle Hidden -PassThru
$processStart = Get-Date
$active = @()
while ($true) {
    $process.Refresh()
    if ($process.HasExited) { break }
    $reading = Get-BatteryReading
    $elapsed = ($reading.Timestamp - $processStart).TotalSeconds
    if ($elapsed -ge $WarmupSeconds -and $reading.Watts -gt 0) {
        $active += $reading
    }
    Start-Sleep -Seconds 1
}
$processEnd = Get-Date
if ($process.ExitCode -ne 0) {
    throw "Benchmark exited with code $($process.ExitCode). See $stderrPath and $stdoutPath."
}
if ($active.Count -lt $MinimumActiveSamples) {
    throw "Only $($active.Count) active power samples; increase benchmark repeats."
}
$idleWatts = Get-Median @($idle | ForEach-Object { $_.Watts })
$activeWatts = Get-Median @($active | ForEach-Object { $_.Watts })
$result = [ordered]@{
    label = $Label
    power_source = 'battery'
    idle_median_watts = $idleWatts
    active_median_watts = $activeWatts
    idle_adjusted_watts = $activeWatts - $idleWatts
    positive_idle_adjusted_power = $activeWatts -gt $idleWatts
    idle_samples = $idle.Count
    active_samples = $active.Count
    process_wall_seconds = ($processEnd - $processStart).TotalSeconds
    battery_start_mwh = $startBattery.RemainingMWh
    battery_end_mwh = (Get-BatteryReading).RemainingMWh
    stdout = $stdoutPath
    stderr = $stderrPath
}
$result | ConvertTo-Json -Depth 4
