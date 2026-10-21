# Builds the portable app: dist\NPU Dictate\ (folder) and dist\NPU-Dictate-<version>-win64.zip.
# Needs: uv, the XRT SDK's pyxrt.pyd (cache\iron\sdk), and the exported bundle
# (cache\npu_dictate_bundle, from export_bundle.py; only manifest.json goes into the app).
$ErrorActionPreference = 'Stop'
# A clean PATH: PyInstaller copies DLL dependencies from PATH (an old msvcp140.dll
# from another program's folder crashed the app).
$uvDir = Split-Path (Get-Command uv).Source
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0;$uvDir"
$app = Split-Path $PSScriptRoot -Parent
$repo = (Resolve-Path (Join-Path $app '..\..')).Path
$venv = Join-Path $repo 'cache\build-venv'
$py = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $py)) { uv venv --python 3.13.5 $venv }
uv pip install -q --python $py -r (Join-Path $app 'requirements.txt') pyinstaller==6.22.3
$env:NPU_DICTATE_PYXRT = Join-Path $repo 'cache\iron\sdk\xrt_sdk\xrt\python\pyxrt.pyd'
& $py -c "import sys; sys.path.insert(0, r'$app'); import npu_dictate as d; d.icon_image('#30a14e', 256).save(r'$PSScriptRoot\icon.ico', sizes=[(16,16),(32,32),(48,48),(256,256)])"
$version = (& $py -c "import sys; sys.path.insert(0, r'$app'); import npu_dictate as d; print(d.VERSION)").Trim()
$dist = Join-Path $repo 'dist'
& $py -m PyInstaller --noconfirm --clean --distpath $dist --workpath (Join-Path $repo 'cache\pyinstaller') (Join-Path $PSScriptRoot 'npu_dictate.spec')
if ($LASTEXITCODE) { throw "PyInstaller failed ($LASTEXITCODE)" }
Copy-Item (Join-Path $app 'RELEASE_README.txt') (Join-Path $dist 'NPU Dictate\README.txt')
$zip = Join-Path $dist "NPU-Dictate-$version-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path (Join-Path $dist 'NPU Dictate') -DestinationPath $zip
"{0}: {1:N0} MB" -f $zip, ((Get-Item $zip).Length / 1MB)
