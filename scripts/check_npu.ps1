$npu = Get-PnpDevice -PresentOnly | Where-Object {
    $_.FriendlyName -eq 'NPU Compute Accelerator Device'
}

if (-not $npu) {
    Write-Output 'NPU device: not found'
    exit 1
}

$driver = Get-CimInstance Win32_PnPSignedDriver | Where-Object {
    $_.DeviceName -eq $npu.FriendlyName
} | Select-Object -First 1

Write-Output "NPU device: $($npu.FriendlyName)"
Write-Output "Status: $($npu.Status)"
Write-Output "Device ID: $(($npu.InstanceId -split '\\')[1])"
Write-Output "Driver version: $($driver.DriverVersion)"

if ($npu.Status -ne 'OK') {
    exit 1
}
