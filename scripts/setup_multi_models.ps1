param(
    [string]$ArchivePath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'cache\resource_multi_model_demo.zip')
)

# Source: https://www.xilinx.com/bin/public/openDownload?filename=resource_multi_model_demo.zip
$expectedHash = 'A84DF575902C9F29B7A04E1A5B7B40D0E85393114912932D0CB59777476104CB'
$entries = @(
    'resource/mobilenetv2_1.4_int.onnx',
    'resource/nano-YOLOX_int.onnx',
    'resource/pointpainting-nus-FPN_int.onnx',
    'resource/resnet50_pt.onnx',
    'resource/RetinaFace_int.onnx',
    'resource/seg_512_288.avi'
)

if (-not (Test-Path -LiteralPath $ArchivePath)) {
    throw "Download AMD's resource_multi_model_demo.zip to $ArchivePath first."
}
$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArchivePath).Hash
if ($actualHash -ne $expectedHash) {
    throw "Archive SHA-256 differs from the tested package ($actualHash). Review the download before extraction."
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$repoRoot = Split-Path -Parent $PSScriptRoot
$destination = (New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot 'models\amd_multi_model')).FullName
$archive = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $ArchivePath).Path)
try {
    foreach ($name in $entries) {
        $entry = $archive.GetEntry($name)
        if ($null -eq $entry) {
            throw "Missing archive entry: $name"
        }
        $target = Join-Path $destination (Split-Path $name -Leaf)
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
        Write-Output "Extracted $($entry.Name)"
    }
} finally {
    $archive.Dispose()
}
