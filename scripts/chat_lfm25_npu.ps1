param(
    [Alias('Prompt')][string]$Message,
    [int]$MaxNewTokens = 16,
    [ValidateSet('fixed64', 'variable', 'chunked')][string]$CacheMode = 'fixed64',
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot 'cache\iron\mlir-aie\ironenv\Scripts\python.exe'
$model = Join-Path $repoRoot 'cache\lfm25-230m\model.safetensors'
$devShell = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1'

if (-not (Test-Path -LiteralPath $python)) {
    throw "IRON Python is missing: $python. See experiments/006_handoff.md."
}
if (-not (Test-Path -LiteralPath $model)) {
    throw "LFM2.5 checkpoint is missing: $model. See experiments/006_handoff.md."
}
if (-not (Test-Path -LiteralPath $devShell)) {
    throw "Visual Studio Developer PowerShell is missing: $devShell."
}

Push-Location $repoRoot
try {
    & $devShell -Arch amd64 | Out-Null
    . .\cache\iron\mlir-aie\iron_env.ps1
    $arguments = @('experiments\007_lfm25_npu_chat.py', '--max-new-tokens', "$MaxNewTokens", '--cache-mode', $CacheMode)
    if ($PSBoundParameters.ContainsKey('Message')) {
        $arguments += @('--prompt', $Message)
    }
    if ($Json) {
        $arguments += '--json'
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "NPU chat exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
