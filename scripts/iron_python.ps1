# Run a Python script in the IRON environment with Visual Studio tools loaded.
# Usage: & .\scripts\iron_python.ps1 experiments\script.py [args...]
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot 'cache\iron\mlir-aie\ironenv\Scripts\python.exe'
$devShell = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1'
Push-Location $repoRoot
try {
    & $devShell -Arch amd64 | Out-Null
    . .\cache\iron\mlir-aie\iron_env.ps1
    $env:PYTHONIOENCODING = 'utf-8'
    $ErrorActionPreference = 'Continue'
    & $python @args 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
