# Starts NPU Dictate from the repository (development). The packaged app is
# "NPU Dictate.exe" (build\build.ps1) and needs none of this.
# The engine needs only pyxrt (from the XRT SDK) and the exported bundle in
# cache\npu_dictate_bundle (apps\npu_dictate\export_bundle.py).
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $repo 'cache\iron\mlir-aie\iron_env.ps1')
$env:PYTHONIOENCODING = 'utf-8'
$argv = @("`"$(Join-Path $PSScriptRoot 'npu_dictate.py')`"") + $args
Start-Process -FilePath (Join-Path $repo 'cache\iron\mlir-aie\ironenv\Scripts\pythonw.exe') `
    -ArgumentList $argv -WorkingDirectory $repo
