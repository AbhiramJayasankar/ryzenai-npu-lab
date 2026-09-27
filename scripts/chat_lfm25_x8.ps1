# Interactive LFM2.5-230M chat on the Phoenix NPU (experiment 010 engine).
# Usage: .\scripts\chat_lfm25_x8.ps1 [-Message "text"] [-MaxNewTokens 256]
param(
    [Alias('Prompt')][string]$Message,
    [int]$MaxNewTokens = 256
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot 'cache\iron\mlir-aie\ironenv\Scripts\python.exe'
$devShell = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1'
# Build the arguments first: the venv activation script sets its own $Prompt.
$arguments = @('experiments\010_lfm25_npu_chat.py', '--max-new-tokens', "$MaxNewTokens")
if ($PSBoundParameters.ContainsKey('Message')) { $arguments += @('--prompt', $Message) }
Push-Location $repoRoot
try {
    & $devShell -Arch amd64 | Out-Null
    . .\cache\iron\mlir-aie\iron_env.ps1
    $env:PYTHONIOENCODING = 'utf-8'
    # Not piped, so the "You:" prompt and streamed tokens appear immediately.
    & $python @arguments
} finally {
    Pop-Location
}
