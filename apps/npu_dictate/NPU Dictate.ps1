# Starts NPU Dictate with the NPU (IRON/XRT) environment of this repository.
# Shortcuts made by the app run: powershell -WindowStyle Hidden -File "NPU Dictate.ps1"
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$devShell = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1'
# The VS tools are only used if the NPU program has to be compiled (first start after code changes).
if (Test-Path $devShell) { & $devShell -Arch amd64 -SkipAutomaticLocation | Out-Null }
. (Join-Path $repo 'cache\iron\mlir-aie\iron_env.ps1')
$env:PYTHONIOENCODING = 'utf-8'
Start-Process -FilePath (Join-Path $repo 'cache\iron\mlir-aie\ironenv\Scripts\pythonw.exe') `
    -ArgumentList "`"$(Join-Path $PSScriptRoot 'npu_dictate.py')`"" -WorkingDirectory $repo
