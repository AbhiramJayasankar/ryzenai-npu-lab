# PyInstaller spec for the portable NPU Dictate folder (run through build.ps1).
import os
from pathlib import Path

APP = Path(SPECPATH).parent
REPO = APP.parents[1]
PYXRT = os.environ["NPU_DICTATE_PYXRT"]
BUNDLE = REPO / "cache" / "npu_dictate_bundle"

a = Analysis(
    [str(APP / "npu_dictate.py")],
    pathex=[str(APP), str(REPO / "experiments")],
    binaries=[(PYXRT, ".")],
    datas=[(str(BUNDLE / "manifest.json"), ".")],
    hiddenimports=["engine", "npu_runtime", "parakeet_pipeline", "pystray._win32"],
    excludes=["aie", "matplotlib", "scipy", "pandas", "IPython", "torch", "pytest", "unittest",
              "pydoc", "pk_engine", "pk_model", "npu_direct"],
    noarchive=False,
)
# The MSVC runtime must be the system's current one: an old msvcp140.dll found on
# PATH (Oracle XE ships 14.28) crashes onnxruntime's import.
SYSTEM32 = Path(os.environ["SystemRoot"]) / "System32"
RUNTIME = ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll")
a.binaries = [b for b in a.binaries if Path(b[0]).name.lower() not in RUNTIME]
a.binaries += [(name, str(SYSTEM32 / name), "BINARY") for name in RUNTIME if (SYSTEM32 / name).exists()]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="NPU Dictate", console=False,
          icon=str(APP / "build" / "icon.ico"), version=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="NPU Dictate")
