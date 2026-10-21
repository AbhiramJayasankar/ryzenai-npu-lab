NPU Dictate
===========

Push-to-talk dictation for Windows. Speech recognition (NVIDIA Parakeet TDT
0.6B v2) runs its 24 encoder layers on the AMD Ryzen AI NPU, so the GPU stays
free and the CPU mostly idle.

Requirements
  * A Ryzen 7040 / 8040 laptop (Phoenix or Hawk Point NPU, "NPU1").
    Strix Point and newer NPUs need a different NPU program and will not work.
  * The AMD NPU driver (Device Manager > Neural processors). Tested with
    32.0.203.376.
  * About 1.6 GB of free RAM when the model loads.

Set up once
  1. Unzip this folder anywhere, for example to C:\Apps\NPU Dictate.
  2. Run "NPU Dictate.exe". The first start downloads the model
     (1.2 GB, from this project's GitHub release "model-v1") into
     %LOCALAPPDATA%\NPU Dictate\model-v1 and checks every file's SHA-256.
  3. "Start with Windows" is on by default. Turn it off in the window if you
     don't want that.

Use
  Press Ctrl+Shift+Space, speak, press it again: the text is inserted into the
  focused text box. The shortcut, hold-to-talk mode, microphone and how the
  text is inserted can be changed in the window (open it from the tray icon).

Files
  Settings and logs: %APPDATA%\NPU Dictate (settings.json, app.log, engine.log).
  No audio or transcript is ever written to disk.

Uninstall
  Turn off "Start with Windows", quit from the tray icon, then delete this
  folder and %LOCALAPPDATA%\NPU Dictate.

Source, licenses and how it works:
  https://github.com/AbhiramJayasankar/ryzenai-npu-lab/tree/main/apps/npu_dictate
Model: Parakeet TDT 0.6B v2 by NVIDIA, CC-BY-4.0. App code: MIT.
