param(
    [string]$AmdCheckout = (Join-Path (Split-Path -Parent $PSScriptRoot) '..\ryzenai\RyzenAI-SW')
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $AmdCheckout 'tutorial\yolov8'
$sourceModel = Join-Path $sourceRoot 'yolov8_cpp\implement\DetectionModel_int.onnx'
$sourceLabels = Join-Path $sourceRoot 'yolov8_python\coco.names'
$destination = Join-Path $repoRoot 'models\yolov8'
$expectedHash = '1F65C211A5F147E7B95D33D2B68334854E01C5E125B4111FE3A99248827C4EA7'

if (-not (Test-Path -LiteralPath $sourceModel) -or -not (Test-Path -LiteralPath $sourceLabels)) {
    throw "AMD YOLOv8 files were not found in $sourceRoot. Pass -AmdCheckout with the RyzenAI-SW checkout path."
}

$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceModel).Hash
if ($actualHash -ne $expectedHash) {
    throw "The source model hash differs from the tested model ($actualHash). Review it before using this experiment."
}

New-Item -ItemType Directory -Force -Path $destination | Out-Null
Copy-Item -LiteralPath $sourceModel -Destination (Join-Path $destination 'DetectionModel_int.onnx')
Copy-Item -LiteralPath $sourceLabels -Destination (Join-Path $destination 'coco.names')
Write-Output "YOLOv8 model and labels copied to $destination"
