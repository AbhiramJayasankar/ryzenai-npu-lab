# NPU Dictate

Dictation in the style of Handy: press a shortcut, speak, press it again, and
the text is inserted into the focused text box. Speech recognition is
Parakeet TDT 0.6B v2 with its 24 encoder layers on the Ryzen AI NPU (the
1-block program of [experiment 012](../../experiments/012_all_npu_encoder.md));
mel features, the subsampling prefix and the TDT decoder run on the CPU.

## Start

```
powershell -NoProfile -File "apps\npu_dictate\NPU Dictate.ps1"
```

or use "Add Start menu / desktop shortcuts" in the app once. The window
closes to the tray; quit from the tray icon's menu. Startup takes ~6 s (the
compiled NPU program is cached; after changes to the engine code the first
start compiles it for ~2 min).

## Settings

| Setting | Options |
| --- | --- |
| Shortcut | any key with Ctrl/Alt/Shift/Win, or an F-key alone (default Ctrl+Shift+Space) |
| Mode | toggle (press, speak, press) or hold to talk |
| Microphone | system default or a specific input |
| Insert text | paste (the previous clipboard text is restored; if the clipboard holds an image or files the text is typed instead), type as keystrokes, or copy only |
| Free memory | unload the model after N seconds idle (0 = keep it loaded). Unloading ends the engine process and frees its ~1.4 GB; the next shortcut press reloads it (~4 s) while you already speak, and the text arrives once it is ready |
| Other | trailing space, start/stop sounds, recording indicator, start minimized, start with Windows |

Settings live in `%APPDATA%\NPU Dictate\settings.json`. No audio or
transcript is written to disk; `engine.log` holds timings only. The history
list is kept in memory while the app runs.

## How it works

* `npu_dictate.py`: Tk window, tray icon (pystray), overlay, recording and
  output. The shortcut is a Win32 `RegisterHotKey` (no keyboard hook).
* `engine.py`: runs in its own process, so NPU calls never stall the
  shortcut, audio capture or UI. Any NPU error stops the engine (no retries);
  if a piece gets no answer within 30 s the app stops dictation and says so.
* `chunker.py`: the NPU program holds 10.24 s, so a dictation is cut into
  pieces of at most 10 s at pauses (or at the quietest point) and each piece
  is transcribed while you keep speaking. Only the last piece is left when you
  stop. Each piece gets 0.3 s of trailing silence: without it Parakeet tends
  to invent an ending for a sentence cut mid-word (checked with the CPU FP32
  model too).
* `win32.py`: hotkey, clipboard, `SendInput`, a click-through overlay that
  never takes focus, DPI awareness.

Measured on the laptop (Ryzen 9 8945HS): 180 to 230 ms of NPU time per piece
of up to 10 s; the text is inserted about 0.25 to 0.35 s after the second
press. The engine process uses ~1.4 GB of RAM (the BF16 weights).

Test hook: `NPU_DICTATE_FAKE_MIC=file.wav` plays a 16 kHz WAV as the
microphone.
