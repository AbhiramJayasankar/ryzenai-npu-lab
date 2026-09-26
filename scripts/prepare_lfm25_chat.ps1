param([ValidateRange(2, 96)][int]$Through = 96)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot 'cache\iron\mlir-aie\ironenv\Scripts\python.exe'
$devShell = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1'
if (-not (Test-Path -LiteralPath $python)) { throw "IRON Python is missing: $python" }
if (-not (Test-Path -LiteralPath $devShell)) { throw "Visual Studio Developer PowerShell is missing: $devShell" }

Push-Location $repoRoot
try {
    & $devShell -Arch amd64 | Out-Null
    . .\cache\iron\mlir-aie\iron_env.ps1
    & $python experiments\007_prepare_chat.py --through $Through
    if ($LASTEXITCODE -ne 0) { throw "Attention preparation exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
