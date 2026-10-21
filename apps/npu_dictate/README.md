# NPU Dictate

Push-to-talk dictation for Windows in the style of Handy: press a shortcut,
speak, press it again, and the text goes into the focused text box. Speech
recognition is Parakeet TDT 0.6B v2 with its 24 encoder layers on the Ryzen AI
NPU (the 1-block program of [experiment 012](../../experiments/012_all_npu_encoder.md));
mel features, the subsampling prefix and the TDT decoder run on the CPU.

## Install

1. Download `NPU-Dictate-<version>-win64.zip` from the
   [releases](https://github.com/AbhiramJayasankar/ryzenai-npu-lab/releases) and unzip it anywhere.
2. Run `NPU Dictate.exe`. The first start downloads the model (1.2 GB, release
   `model-v1`) into `%LOCALAPPDATA%\NPU Dictate\model-v1`, checking every
   file's SHA-256, and turns on "Start with Windows".

Requirements: a Ryzen 7040/8040 laptop (Phoenix or Hawk Point NPU), the AMD
NPU driver (tested with 32.0.203.376), about 1.2 GB of RAM for the model.
Nothing else: no Python, IRON, Visual Studio or compiler. `NPU Dictate.exe --check`
reports whether the driver, NPU, model files and microphone are usable (it
opens the NPU to read its name and runs nothing on it).

## Settings

| Setting | Options |
| --- | --- |
| Shortcut | any key with Ctrl/Alt/Shift/Win, or an F-key alone (default Ctrl+Shift+Space) |
| Mode | toggle (press, speak, press) or hold to talk (releasing any key of the shortcut stops) |
| Microphone | system default or a specific input (MME) |
| Insert text | paste (your clipboard is restored unless you copied something in the meantime; if it holds an image, files or rich text the text is typed instead), type as keystrokes, or copy only |
| Free memory | unload the model after N seconds idle (0 = keep it loaded, the default; otherwise 5 s to 1 day). Unloading ends the engine process and frees its ~1.4 GB; the next shortcut press loads it again (~3 s) while you already speak, and the text arrives once it is ready. Each reload creates a new NPU context. |
| Other | trailing space, start/stop sounds, recording indicator, start minimized, start with Windows |

If the focused window changed before the text was ready, or Shift/Alt/Win is
still held, the text is copied to the clipboard instead of inserted.
Settings live in `%APPDATA%\NPU Dictate\settings.json`. No audio or
transcript is written to disk; the history list is kept in memory only.

## How it works

* `npu_dictate.py`: Tk window, tray icon (pystray), overlay, recording,
  output, start with Windows (HKCU Run key). The shortcut is a Win32
  `RegisterHotKey` (no keyboard hook).
* `engine.py`: the recognizer in its own process, so NPU calls never stall
  the shortcut, audio capture or UI. The model stays loaded unless "Free
  memory" is set, and the app starts a new engine process only after the
  previous one has exited.
* `npu_runtime.py`: the encoder on the NPU with pyxrt only, from the exported
  bundle (one xclbin, two instruction streams, ctrl/position/weight buffers):
  the same buffers and submissions as `experiments/pk_model.PkEncoder(nblk=1)`.
* `export_bundle.py`: writes that bundle from the IRON compile cache, with a
  manifest of sizes and SHA-256. `model_store.py` finds or downloads it.
* `chunker.py`: the NPU program holds 10.24 s, so a dictation is cut into
  pieces of at most 10 s at pauses (or at the quietest point) and each piece
  is transcribed while you keep speaking. Each piece gets 0.3 s of trailing
  silence: without it Parakeet tends to invent an ending for a sentence cut
  mid-word.
* `win32.py`: hotkey, clipboard (owned by a message-only window), checked
  `SendInput`, a click-through overlay that never takes focus, DPI awareness.

Measured on the laptop (Ryzen 9 8945HS): 180 to 230 ms of NPU time per piece
of up to 10 s; the text is inserted about 0.25 to 0.35 s after the second
press. The engine process uses ~1.4 GB of RAM (the BF16 weights).

## Failure handling

| What happens | What the app does |
| --- | --- |
| The model cannot be loaded (little free memory, missing files, the engine died before touching the NPU) | waits and tries again after 10 s, 30 s, 1 min, 2 min, then every 5 min, and at once when you press the shortcut |
| The model files are missing or damaged | downloads them again |
| An NPU error, or no answer within 30 s of NPU work (sleep time is not counted) | stops dictation for good and says so in a Windows notification: an NPU hang can crash Windows on this driver, so nothing retries. Restart Windows after a hang, then the app |
| A CPU-side error in one piece | that piece is lost, dictation continues |
| The app quits or crashes | the engine process notices and exits (one still loading exits before its first NPU submission); it is never killed in the middle of an NPU submission. A restarted app waits for it, and stops with "restart Windows" if it is still running after 30 s |

Logs: `%APPDATA%\NPU Dictate\app.log` (starts, failures), `engine.log`
(timings), `fault.log` (native crash stacks).

## Build from source

```
powershell -NoProfile -File "apps\npu_dictate\build\build.ps1"
```

needs `uv`, the XRT SDK's `pyxrt.pyd` (`cache\iron\sdk`, see
[docs/setup.md](../../docs/setup.md)) and the bundle from
`python apps\npu_dictate\export_bundle.py` (run in the IRON environment).
Output: `dist\NPU Dictate\` and `dist\NPU-Dictate-<version>-win64.zip`.
For development, `"NPU Dictate.ps1"` runs the app from the repository.

Test hooks (environment variables): `NPU_DICTATE_FAKE_ENGINE=<load s>[,<s per piece>]`
replaces the NPU engine; `NPU_DICTATE_FAKE_MIC=file.wav` plays a 16 kHz WAV as
the microphone; `NPU_DICTATE_TEST_EVENTS=4:hotkey,12:hotkey` posts events;
`NPU_DICTATE_TEST_OUTPUT=1` logs text instead of inserting it;
`NPU_DICTATE_TEST_START_FAIL=1`, `NPU_DICTATE_TEST_DIE_ONCE=<file>` and
`NPU_DICTATE_TEST_QUIT_AFTER=<s>` simulate failures and quit;
`NPU_DICTATE_BUNDLE=<dir>` uses another bundle folder.
