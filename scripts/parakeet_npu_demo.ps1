# Live speech-to-text with the Parakeet encoder on the NPU (experiment 012 demo).
# Usage: powershell -NoProfile -File scripts\parakeet_npu_demo.ps1 [--ptt] [--cpu] [--device N] [file.wav ...]
# Unlike iron_python.ps1, output is not piped, so the live level meter works.
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot 'cache\iron\mlir-aie\ironenv\Scripts\python.exe'
$devShell = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1'
Push-Location $repoRoot
try {
    & $devShell -Arch amd64 | Out-Null  # only needed if the NPU program must be (re)compiled
    . .\cache\iron\mlir-aie\iron_env.ps1
    $env:PYTHONIOENCODING = 'utf-8'
    & $python experiments\012_live_demo.py @args
} finally {
    Pop-Location
}
