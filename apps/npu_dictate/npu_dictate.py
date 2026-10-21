"""NPU Dictate: press a shortcut, speak, press it again, and the text is typed
into whatever text box has focus. Speech recognition is Parakeet TDT 0.6B v2
with its encoder on the Ryzen AI NPU (see engine.py).

Start with "NPU Dictate.ps1" (sets up the NPU environment). Settings are kept in
%APPDATA%\\NPU Dictate\\settings.json; no audio or text is written to disk.
"""

import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import wave
import winsound
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chunker  # noqa: E402
import engine  # noqa: E402
import model_store  # noqa: E402
import win32  # noqa: E402

APP = "NPU Dictate"
VERSION = "1.0.0"
APPDIR = Path(os.environ.get("APPDATA", Path.home())) / APP
FROZEN = getattr(sys, "frozen", False)
LAUNCHER = Path(__file__).resolve().parent / "NPU Dictate.ps1"
DEFAULTS = {"shortcut": "ctrl+shift+space", "mode": "toggle", "microphone": "", "output": "paste",
            "trailing_space": True, "sounds": True, "overlay": True, "start_minimized": False,
            "autostart_set": False}
MODES = {"toggle": "Press to start, press again to insert", "hold": "Hold to talk, release to insert"}
OUTPUTS = {"paste": "Paste (your clipboard is restored)", "type": "Type it as keystrokes",
           "clipboard": "Only copy to the clipboard"}
STATES = {"loading": "Loading", "download": "Downloading", "waiting": "Waiting", "ready": "Ready",
          "recording": "Recording", "error": "Stopped"}
COLORS = {"loading": "#8e8e93", "download": "#8e8e93", "waiting": "#d9a400", "ready": "#30a14e",
          "recording": "#e5484d", "error": "#e5484d"}
WATCHDOG_S = 30.0  # NPU work (a piece, or the warm-up) must finish this soon
START_TIMEOUT_S = 60.0  # the engine must reach the NPU this soon after it is spawned
RETRY_S = (10, 30, 60, 120, 300)  # waits between attempts to load the model
S = 1.0  # display scale (DPI / 96), set when the window is created


def px(v):
    return int(round(v * S))


# -- settings, sounds, icon ------------------------------------------------------
def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def load_settings():
    """Saved settings over the defaults; a value of the wrong type or an unknown
    choice falls back to its default (a hand-edited file may have a BOM)."""
    try:
        saved = json.loads((APPDIR / "settings.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        if not isinstance(e, FileNotFoundError):
            log(f"settings.json unreadable, using defaults: {e}")
        saved = {}
    out = dict(DEFAULTS)
    for k, v in (saved.items() if isinstance(saved, dict) else ()):
        if k in DEFAULTS and type(v) is type(DEFAULTS[k]):
            out[k] = v
    try:
        win32.parse(out["shortcut"])
    except (KeyError, AttributeError):
        out["shortcut"] = DEFAULTS["shortcut"]
    if out["mode"] not in MODES:
        out["mode"] = DEFAULTS["mode"]
    if out["output"] not in OUTPUTS:
        out["output"] = DEFAULTS["output"]
    return out


def save_settings(s):
    """Written to a temporary file first, so a crash never leaves a half-written file."""
    APPDIR.mkdir(parents=True, exist_ok=True)
    tmp = APPDIR / "settings.json.tmp"
    tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
    os.replace(tmp, APPDIR / "settings.json")


def make_sounds():
    sounds = {}
    for name, freqs in (("start", (660, 880)), ("stop", (880, 660)), ("error", (330, 247))):
        t = np.arange(int(0.07 * chunker.SR)) / chunker.SR
        env = np.minimum(1, np.minimum(t, t[::-1]) / 0.01)
        tone = np.concatenate([np.sin(2 * np.pi * f * t) * env for f in freqs])
        path = APPDIR / f"{name}.wav"
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(chunker.SR)
            f.writeframes((0.25 * 32767 * tone).astype("<i2").tobytes())
        sounds[name] = str(path)
    return sounds


def icon_image(color, size=64):
    from PIL import Image, ImageDraw

    s = size / 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((2 * s, 2 * s, 62 * s, 62 * s), fill=color)
    d.rounded_rectangle((25 * s, 12 * s, 39 * s, 38 * s), radius=7 * s, fill="white")
    d.arc((18 * s, 22 * s, 46 * s, 46 * s), 0, 180, fill="white", width=max(1, round(3 * s)))
    d.line((32 * s, 46 * s, 32 * s, 52 * s), fill="white", width=max(1, round(3 * s)))
    d.line((25 * s, 52 * s, 39 * s, 52 * s), fill="white", width=max(1, round(3 * s)))
    return img


def _mme():
    import sounddevice as sd

    return next((i for i, h in enumerate(sd.query_hostapis()) if h["name"] == "MME"), None)


def input_devices():
    """Microphones of the MME host API (it resamples to 16 kHz for us), by name."""
    import sounddevice as sd

    mme = _mme()
    return [d["name"] for d in sd.query_devices()
            if d["max_input_channels"] > 0 and d["hostapi"] == mme and "Sound Mapper" not in d["name"]]


def device_index(name):
    """The MME device with this name, else (also for "") the MME default input:
    other host APIs may not open a 16 kHz stream."""
    import sounddevice as sd

    mme = _mme()
    if mme is None:
        return None
    for i, d in enumerate(sd.query_devices()):
        if name and d["name"] == name and d["hostapi"] == mme and d["max_input_channels"] > 0:
            return i
    default = sd.query_hostapis(mme)["default_input_device"]
    return default if default >= 0 else None


def launch_command(background=False):
    """(program, arguments) that start this app: the exe when packaged, else the PowerShell launcher."""
    if FROZEN:
        return sys.executable, "--background" if background else ""
    return "powershell.exe", (f'-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{LAUNCHER}"'
                              + (" --background" if background else ""))


def make_shortcut(path, icon):
    target, args = launch_command()
    q = lambda v: "'" + str(v).replace("'", "''") + "'"  # noqa: E731  (PowerShell single-quoted string)
    ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut({q(path)});"
          f"$s.TargetPath={q(target)};$s.Arguments={q(args)};"
          f"$s.WorkingDirectory={q(Path(target).parent if FROZEN else LAUNCHER.parent)};$s.IconLocation={q(icon)};"
          f"$s.Description='Dictation with Parakeet on the Ryzen AI NPU';$s.WindowStyle=7;$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


# -- start with Windows (HKCU Run key; no admin rights needed) -------------------
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
APPROVED_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"
OLD_STARTUP_LNK = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup" / f"{APP}.lnk"


def autostart_registered():
    """True if a Run value for the app exists (it may point to an older copy)."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            return bool(winreg.QueryValueEx(k, APP)[0])
    except OSError:
        return False


def autostart_enabled():
    """True if Windows will start the app at sign-in: the Run value points at this
    copy of the app and Task Manager's Startup tab has not disabled it."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            value = winreg.QueryValueEx(k, APP)[0]
    except OSError:
        return False
    target, _ = launch_command()
    if target.lower() not in value.lower():
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, APPROVED_KEY) as k:
            flag = winreg.QueryValueEx(k, APP)[0]
        return not (isinstance(flag, bytes) and flag[:1] in (b"\x01", b"\x03"))  # odd = disabled
    except OSError:
        return True


def set_autostart(on):
    import winreg

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
        if on:
            target, args = launch_command(background=True)
            winreg.SetValueEx(k, APP, 0, winreg.REG_SZ, f'"{target}" {args}'.strip())
        else:
            try:
                winreg.DeleteValue(k, APP)
            except FileNotFoundError:
                pass
    try:  # clear a "disabled" mark from Task Manager, which would win over the Run value
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, APPROVED_KEY, 0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, APP)
    except OSError:
        pass
    if OLD_STARTUP_LNK.exists():  # the Startup-folder shortcut of older versions
        OLD_STARTUP_LNK.unlink()


# -- engine process proxy --------------------------------------------------------
class Engine:
    """One engine process. At most one is alive at a time: the app spawns a new
    one only after the previous process has exited (two processes holding NPU
    buffers at once fail to load, and the NPU driver is fragile)."""

    def __init__(self, post, gen):
        self.gen, self.closing = gen, False
        self.npu_used = False  # set once the engine reports its first NPU submission
        self.t0 = time.perf_counter()
        ctx = mp.get_context("spawn")
        self.inq, self.outq = ctx.Queue(), ctx.Queue()
        self.stop_event = ctx.Event()  # set: the engine starts no further piece
        # Audio put into the queue of an engine that died stays unread; without this
        # the app froze at exit waiting for it (seen 2026-10).
        self.inq.cancel_join_thread()
        self.proc = ctx.Process(target=engine.serve, daemon=True, name="npu-engine",
                                args=(self.inq, self.outq, str(APPDIR / "engine.log"),
                                      str(model_store.bundle_dir()), self.stop_event))
        self.proc.start()
        log(f"engine {gen}: spawned pid {self.proc.pid}")
        threading.Thread(target=self._read, args=(post,), daemon=True, name=f"engine-{gen}-read").start()

    def _read(self, post):
        while True:
            try:
                msg = self.outq.get(timeout=1.0)
            except queue.Empty:
                msg = None
            except (EOFError, OSError) as e:  # the queue broke: only the process state is left to go by
                log(f"engine {self.gen}: queue error {e!r}")
                msg = None
                time.sleep(0.5)
            if msg is not None:
                if msg[0] == "npu":
                    self.npu_used = True
                post(("engine", self.gen, msg))
            elif not self.proc.is_alive():
                log(f"engine {self.gen}: pid {self.proc.pid} exited with code {self.proc.exitcode}"
                    f"{' (closed by the app)' if self.closing else ''}")
                if not self.closing:
                    post(("engine", self.gen, ("died", self.proc.exitcode, self.npu_used)))
                return

    def alive(self):
        return self.proc.is_alive()

    def send(self, session, index, audio):
        self.inq.put(("audio", session, index, audio))

    def stop(self):
        """No further piece starts (the one in progress finishes)."""
        self.stop_event.set()

    def close(self, wait_s=30.0):
        """Ends the engine process (blocking; run it off the UI thread). It is never
        killed: the engine exits after the piece in progress, and one stuck in an NPU
        call is left to the driver's own timeout (killing it mid-submission could
        leave the driver in a worse state). It also exits once the app is gone."""
        self.closing = True
        self.stop()
        if self.proc.is_alive():
            try:
                self.inq.put(("quit",))
            except (OSError, ValueError):
                pass
            self.proc.join(wait_s)
        if self.proc.is_alive():
            log(f"engine {self.gen}: pid {self.proc.pid} did not exit in {wait_s:.0f} s (stuck NPU call?); "
                "leaving it to the driver")


# -- audio capture ---------------------------------------------------------------
class Recorder:
    """Microphone -> Chunker -> engine, on its own thread."""

    def __init__(self, app):
        self.app = app
        self.q = queue.Queue()
        self.stream = None
        self.session = None
        self.closer = None  # thread closing the previous stream
        self.level = 0.0
        self.seconds = 0.0
        threading.Thread(target=self._loop, daemon=True, name="recorder").start()

    def start(self, session, device):
        if self.closer is not None:
            self.closer.join(3.0)  # one stream at a time: the previous one must be closed first
        self.session = session
        self.q.put(("start", session))
        fake = os.environ.get("NPU_DICTATE_FAKE_MIC")  # test hook: a 16 kHz WAV played as the microphone

        def cb(data, frames, t, status):  # bound to this recording
            self.q.put(("audio", session, data[:, 0].copy()))

        try:
            if fake:
                self.stream = _FakeStream(fake, cb)
                return
            import sounddevice as sd

            self.stream = sd.InputStream(samplerate=chunker.SR, channels=1, dtype="float32",
                                         blocksize=chunker.BLOCK, device=device, callback=cb)
            self.stream.start()
        except Exception:
            s, self.stream = self.stream, None
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
            self.q.put(("abort", session))
            raise

    def stop(self):
        """Stops capture on a helper thread (closing an MME stream can take a few
        hundred ms); the recorder thread then flushes the last piece."""
        s, self.stream, session = self.stream, None, self.session

        def close():
            try:
                if s is not None:
                    s.stop()
                    s.close()
            except Exception as e:
                log(f"closing the microphone failed: {e}")
            self.q.put(("stop", session))

        self.closer = threading.Thread(target=close, daemon=True, name="mic-close")
        self.closer.start()

    def _loop(self):
        ch, session, index = None, None, 0

        def emit(pieces):
            nonlocal index
            for p in pieces:
                self.app.send_piece(session, index, p)
                index += 1

        while True:
            kind, s, *arg = self.q.get()
            try:
                if kind == "start":
                    if ch is not None:  # the previous recording has not been flushed yet: end it first
                        emit(ch.finish())
                        self.app.post(("ended", session, index, self.seconds))
                    ch, session, index = chunker.Chunker(), s, 0
                    self.level = self.seconds = 0.0
                elif ch is None or s != session:
                    continue  # audio or stop of a recording that has already ended
                elif kind == "abort":
                    ch = None
                elif kind == "audio":
                    emit(ch.push(arg[0]))
                    self.level, self.seconds = ch.level, ch.total / chunker.SR
                elif kind == "stop":
                    emit(ch.finish())
                    self.app.post(("ended", session, index, self.seconds))
                    ch = None
            except Exception:
                import traceback

                log("recorder error:\n" + traceback.format_exc())
                if ch is not None:
                    self.app.post(("recorder_failed", session))
                ch = None


class _FakeStream:
    """Feeds a WAV file in real time through the capture callback (for testing)."""

    def __init__(self, path, cb):
        with wave.open(path, "rb") as f:
            audio = np.frombuffer(f.readframes(f.getnframes()), "<i2").astype(np.float32) / 32768
        self.running = True

        def run():
            t0 = time.perf_counter()
            for i in range(0, len(audio), chunker.BLOCK):
                if not self.running:
                    return
                cb(audio[i:i + chunker.BLOCK, None], chunker.BLOCK, None, None)
                time.sleep(max(0.0, t0 + (i + chunker.BLOCK) / chunker.SR - time.perf_counter()))
            while self.running:  # silence after the file
                cb(np.zeros((chunker.BLOCK, 1), np.float32), chunker.BLOCK, None, None)
                time.sleep(chunker.BLOCK / chunker.SR)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def close(self):
        pass


# -- recording indicator ---------------------------------------------------------
class Overlay:
    BW, BH = 250, 44  # size in 96-DPI units
    KEY = "#ff00fe"  # transparent color

    def __init__(self, app):
        self.app = app
        self.W, self.H = px(self.BW), px(self.BH)
        self.win = tk.Toplevel(app.root)
        self.win.title(f"{APP} indicator")
        self.win.overrideredirect(True)
        self.win.geometry(f"{self.W}x{self.H}+-4000+-4000")  # off screen until styled
        self.win.configure(bg=self.KEY)
        self.win.attributes("-transparentcolor", self.KEY)
        self.c = tk.Canvas(self.win, width=self.W, height=self.H, bg=self.KEY, highlightthickness=0)
        self.c.pack()
        self.win.update_idletasks()
        self.hwnd = win32.toplevel_hwnd(self.win)
        win32.make_overlay(self.hwnd)
        win32.hide(self.hwnd)
        self.state, self.text, self.t0, self.levels = None, "", 0.0, [0.0] * 14
        self.visible = False
        self.hide_at = None
        self._tick()

    def show(self, state, text="", hold_s=None):
        self.state, self.text, self.t0 = state, text, time.perf_counter()
        self.hide_at = self.t0 + hold_s if hold_s else None
        if not self.app.settings["overlay"] and state != "error":
            return self.hide()
        if not self.visible:
            left, _, right, bottom = win32.work_area()
            win32.show_noactivate(self.hwnd, (left + right - self.W) // 2, bottom - self.H - px(28))
            self.visible = True
        self._draw()

    def hide(self):
        if self.visible:
            win32.hide(self.hwnd)
            self.visible = False
        self.state = None

    def _tick(self):
        if self.visible:
            if self.hide_at and time.perf_counter() >= self.hide_at:
                self.hide()
            else:
                if self.state == "recording":
                    self.levels = self.levels[1:] + [self.app.recorder.level]
                self._draw()
        self.app.root.after(50, self._tick)

    def _pill(self, fill):
        w, h, r = self.BW, self.BH, self.BH // 2
        c = self.c
        c.create_oval(0, 0, 2 * r, h - 1, fill=fill, outline=fill)
        c.create_oval(w - 2 * r - 1, 0, w - 1, h - 1, fill=fill, outline=fill)
        c.create_rectangle(r, 0, w - r, h - 1, fill=fill, outline=fill)

    def _draw(self):
        c, h = self.c, self.BH
        c.delete("all")
        self._pill("#1c1c1f")
        t = time.perf_counter() - self.t0
        font = ("Segoe UI", 10)
        if self.state == "recording":
            rad = 5 + 1.5 * abs(np.sin(3 * t))
            c.create_oval(22 - rad, h / 2 - rad, 22 + rad, h / 2 + rad, fill="#ff453a", outline="")
            secs = int(self.app.recorder.seconds)
            c.create_text(38, h / 2, text=f"Listening  {secs // 60}:{secs % 60:02d}", fill="white",
                          font=font, anchor="w")
            for i, lv in enumerate(self.levels):
                bh = 3 + min(22, 22 * (lv / 0.06) ** 0.6)
                x = 150 + i * 6
                c.create_rectangle(x, h / 2 - bh / 2, x + 3, h / 2 + bh / 2, fill="#ff9f9a", outline="")
        else:
            color = {"busy": "#ffd60a", "done": "#30d158", "info": "#8e8e93", "error": "#ff453a"}[self.state]
            if self.state == "busy":
                for i in range(3):
                    a = 0.5 + 0.5 * np.sin(6 * t - i)
                    c.create_oval(14 + i * 8, h / 2 - 3 * a, 20 + i * 8, h / 2 + 3 * a + 1, fill=color, outline="")
                label = self.text or "Transcribing on the NPU…"
            else:
                c.create_oval(17, h / 2 - 5, 27, h / 2 + 5, fill=color, outline="")
                label = self.text
            c.create_text(44, h / 2, text=label, fill="white", font=font, anchor="w", width=px(self.BW - 60))
        c.scale("all", 0, 0, S, S)


# -- the app ---------------------------------------------------------------------
class App:
    def __init__(self, background=False):
        APPDIR.mkdir(parents=True, exist_ok=True)
        self.settings = load_settings()
        self.ui_q = queue.Queue()
        self.post = self.ui_q.put
        self.engine_state, self.engine_msg = "loading", "Starting the NPU engine…"
        self.engine, self.engine_gen = None, 0
        self.closing = []  # engines being closed; a new one starts only once these have exited
        self.want_engine = False
        self.retry_at, self.retries, self.died_retries = None, 0, 0
        self.busy = None  # (engine gen, start time): NPU work in progress, for the watchdog
        self.downloading = None  # (done, total) bytes while the model downloads
        self.quitting = False
        self.send_lock = threading.Lock()
        self.backlog = []  # pieces recorded while no engine process exists yet
        self.dropped_upto = 0  # pieces of sessions up to this one are dropped (their dictation failed)
        self.recording = False
        self.rec_mode = self.settings["mode"]  # mode of the current recording
        self.session = 0
        self.sessions = {}
        self.capturing = False
        self.history = []
        self.sounds = make_sounds()
        self.icon_path = APPDIR / "icon.ico"
        icon_image("#30a14e", 256).save(self.icon_path, sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
        if FROZEN:
            try:
                if not self.settings["autostart_set"]:  # first start of the packaged app: on by default
                    set_autostart(True)
                    self.settings["autostart_set"] = True
                    save_settings(self.settings)
                elif autostart_registered() and not autostart_enabled():
                    set_autostart(True)  # the folder was moved or a new version unzipped elsewhere
                    log(f"autostart now points to {sys.executable}")
            except OSError as e:
                log(f"could not set up autostart: {e}")

        self.root = tk.Tk()
        global S
        S = self.root.winfo_fpixels("1i") / 96
        self.root.title(APP)
        self.root.iconbitmap(default=str(self.icon_path))
        self.root.geometry(f"{px(560)}x{px(640)}")
        self.root.minsize(px(480), px(500))
        self.root.protocol("WM_DELETE_WINDOW", self.hide_window)
        self.root.report_callback_exception = lambda *exc: sys.excepthook(*exc)
        self._build_ui()

        self.recorder = Recorder(self)
        self.overlay = Overlay(self)
        self.out_q = queue.Queue()
        self.out_busy = False
        threading.Thread(target=self._output_loop, daemon=True, name="output").start()
        self.hotkey = win32.Hotkey(lambda: self.post(("hotkey",)))
        self._start_tray()
        if not self._register_shortcut(self.settings["shortcut"]):
            self._notify(f"{win32.label(self.settings['shortcut'])} is used by another app. "
                         "Open NPU Dictate from the tray and choose another shortcut.")
        self._start_model()
        self._refresh()
        if (background or self.settings["start_minimized"]) and self.tray is not None:
            self.root.withdraw()
        self.root.after(20, self._poll)
        if os.environ.get("NPU_DICTATE_TEST_QUIT_AFTER"):  # test hook
            self.root.after(int(float(os.environ["NPU_DICTATE_TEST_QUIT_AFTER"]) * 1000), self.quit)
        for item in filter(None, os.environ.get("NPU_DICTATE_TEST_EVENTS", "").split(",")):  # test hook: "s:event"
            at, name = item.split(":")
            self.root.after(int(float(at) * 1000), lambda n=name: self.post((n,)))

    # ---- UI -----------------------------------------------------------------
    def _build_ui(self):
        root = self.root
        style = ttk.Style()
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 17))
        style.configure("Muted.TLabel", foreground="#6e6e73")
        style.configure("Head.TLabel", font=("Segoe UI Semibold", 11))
        style.configure("Status.TLabel", font=("Segoe UI Semibold", 10))
        style.configure("Treeview", rowheight=px(22))
        outer = ttk.Frame(root, padding=(px(20), px(16), px(20), px(14)))
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")
        ttk.Label(top, text=APP, style="Title.TLabel").pack(side="left")
        self.status_dot = tk.Canvas(top, width=px(12), height=px(12), highlightthickness=0, bg=root.cget("bg"))
        self.status_lbl = ttk.Label(top, text="", style="Status.TLabel")
        self.status_lbl.pack(side="right")
        self.status_dot.pack(side="right", padx=(0, px(6)))
        self.hint = ttk.Label(outer, text="", style="Muted.TLabel", wraplength=px(500), justify="left")
        self.hint.pack(fill="x", pady=(px(4), px(4)))
        self.progress = ttk.Progressbar(outer, maximum=1000)  # shown while the model downloads
        self.progress_shown = False
        self.spacer = ttk.Frame(outer, height=px(8))
        self.spacer.pack(fill="x")

        box = ttk.LabelFrame(outer, text=" Settings ", padding=(px(14), px(8), px(14), px(10)))
        box.pack(fill="x")
        box.columnconfigure(1, weight=1)
        r = 0
        ttk.Label(box, text="Shortcut").grid(row=r, column=0, sticky="w", pady=px(4))
        sc = ttk.Frame(box)
        sc.grid(row=r, column=1, sticky="w", padx=(px(12), px(0)))
        self.shortcut_lbl = ttk.Label(sc, text="", font=("Segoe UI Semibold", 10), width=22)
        self.shortcut_lbl.pack(side="left")
        self.shortcut_btn = ttk.Button(sc, text="Change…", command=self._capture_shortcut)
        self.shortcut_btn.pack(side="left", padx=(px(8), px(0)))
        r += 1
        ttk.Label(box, text="Mode").grid(row=r, column=0, sticky="nw", pady=px(4))
        self.mode_var = tk.StringVar(value=self.settings["mode"])
        mf = ttk.Frame(box)
        mf.grid(row=r, column=1, sticky="w", padx=(px(12), px(0)), pady=px(2))
        for k, v in MODES.items():
            ttk.Radiobutton(mf, text=v, value=k, variable=self.mode_var, command=self._changed).pack(anchor="w")
        r += 1
        ttk.Label(box, text="Microphone").grid(row=r, column=0, sticky="w", pady=px(4))
        self.mic_var = tk.StringVar()
        self.mic_box = ttk.Combobox(box, textvariable=self.mic_var, state="readonly", width=40,
                                    postcommand=self._fill_mics)
        self.mic_box.grid(row=r, column=1, sticky="w", padx=(px(12), px(0)), pady=px(4))
        self.mic_box.bind("<<ComboboxSelected>>", lambda e: self._changed())
        self._fill_mics()
        r += 1
        ttk.Label(box, text="Insert text").grid(row=r, column=0, sticky="nw", pady=px(4))
        self.out_var = tk.StringVar(value=self.settings["output"])
        of = ttk.Frame(box)
        of.grid(row=r, column=1, sticky="w", padx=(px(12), px(0)), pady=px(2))
        for k, v in OUTPUTS.items():
            ttk.Radiobutton(of, text=v, value=k, variable=self.out_var, command=self._changed).pack(anchor="w")
        r += 1
        cf = ttk.Frame(box)
        cf.grid(row=r, column=0, columnspan=2, sticky="w", pady=(px(8), px(0)))
        self.checks = {}
        for i, (k, v) in enumerate((("trailing_space", "Add a space after the text"),
                                    ("sounds", "Play start / stop sounds"),
                                    ("overlay", "Show the recording indicator"),
                                    ("start_minimized", "Start minimized to the tray"))):
            var = tk.BooleanVar(value=self.settings[k])
            ttk.Checkbutton(cf, text=v, variable=var, command=self._changed).grid(
                row=i // 2, column=i % 2, sticky="w", padx=(px(0), px(24)), pady=px(1))
            self.checks[k] = var
        self.autostart_var = tk.BooleanVar(value=autostart_enabled())
        ttk.Checkbutton(cf, text="Start with Windows", variable=self.autostart_var,
                        command=self._toggle_autostart).grid(row=2, column=0, sticky="w", pady=px(1))
        ttk.Button(cf, text="Add Start menu / desktop shortcuts", command=self._make_shortcuts).grid(
            row=2, column=1, sticky="w", pady=(px(4), px(1)))

        hist = ttk.Frame(outer)
        hist.pack(fill="both", expand=True, pady=(px(14), px(0)))
        hh = ttk.Frame(hist)
        hh.pack(fill="x")
        ttk.Label(hh, text="History", style="Head.TLabel").pack(side="left")
        ttk.Label(hh, text="double-click to copy · kept only while the app runs", style="Muted.TLabel").pack(
            side="left", padx=(px(8), px(0)))
        tf = ttk.Frame(hist)
        tf.pack(fill="both", expand=True, pady=(px(6), px(0)))
        self.tree = ttk.Treeview(tf, columns=("time", "text"), show="headings", height=8, selectmode="browse")
        self.tree.heading("time", text="Time")
        self.tree.heading("text", text="Text")
        self.tree.column("time", width=px(70), stretch=False)
        self.tree.column("text", width=px(420))
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self._copy_selected)
        self.footer = ttk.Label(outer, text="", style="Muted.TLabel", wraplength=px(520), justify="left")
        self.footer.pack(fill="x", pady=(px(8), px(0)))

    # ---- model and engine lifecycle -------------------------------------------
    def send_piece(self, session, index, audio):
        """Called from the recorder thread."""
        with self.send_lock:
            if session <= self.dropped_upto:
                return
            if self.engine is not None:
                self.engine.send(session, index, audio)
            else:
                self.backlog.append((session, index, audio))

    def _start_model(self):
        """Download the model if it is missing or damaged, else start the engine.
        Never after an NPU failure: that state ends only with a restart of the app."""
        if self.engine_state == "error":
            return
        if model_store.missing():
            return self._start_download()
        self.want_engine = True
        self.engine_state, self.engine_msg = "loading", "Loading the model onto the NPU…"
        self._spawn_if_possible()
        self._refresh()

    def _spawn_if_possible(self):
        self.closing = [e for e in self.closing if e.alive()]
        if not self.want_engine or self.engine is not None or self.engine_state == "error":
            return
        if self.closing:
            self.engine_msg = "Waiting for the previous engine process to exit…"
            return
        try:
            with self.send_lock:
                eng = Engine(self.post, self.engine_gen + 1)
                self.engine_gen += 1
                self.engine = eng
                for piece in self.backlog:
                    eng.send(*piece)
                self.backlog = []
        except Exception as e:
            self.want_engine = False
            return self._engine_start_failed(f"the engine process could not be started ({e})", retry=True)
        self.want_engine = False
        self.engine_msg = "Loading the model onto the NPU…"

    def _close_engine(self, wait=False):
        with self.send_lock:
            eng, self.engine = self.engine, None
            self.engine_gen += 1  # ignore anything the old process still sends
        self.busy = None
        if eng is not None:
            eng.stop()  # no further piece starts, even ones already in its queue
            self.closing.append(eng)
            if wait:
                eng.close()
            else:
                threading.Thread(target=eng.close, daemon=True, name="engine-close").start()

    def _drop_dictations(self):
        """The dictations in progress cannot be transcribed: forget them."""
        if self.recording:
            self.recording = False
            self.recorder.stop()
        lost = bool(self.sessions)
        with self.send_lock:
            self.dropped_upto = self.session
            self.backlog = []
        self.sessions.clear()
        return lost

    def _engine_start_failed(self, why, retry):
        """The engine failed before its first NPU submission, so loading it again is
        safe. Causes that clear by themselves (memory, files) are retried on their own
        after 10 s, 30 s, 1 min, 2 min, then every 5 min; others (driver errors) only
        when you press the shortcut. A press always retries at once."""
        log(f"engine {self.engine_gen}: start failed ({'retrying' if retry else 'press to retry'}): {why}")
        self._close_engine()
        lost = self._drop_dictations()
        self.engine_state = "waiting"
        memory = "memory" in why
        if retry:
            delay = RETRY_S[min(self.retries, len(RETRY_S) - 1)]
            self.retries += 1
            self.retry_at = time.monotonic() + delay
            self.engine_msg = (f"The model could not be loaded: {why}. " + ("Close some programs; " if memory else "")
                               + f"it tries again in {delay} s, or when you press the shortcut.")
        else:
            self.retry_at = None
            self.engine_msg = (f"The model could not be loaded: {why}. Press the shortcut to try again; "
                               "if it keeps failing, restart Windows.")
            self._notify("The NPU model could not be loaded. Open NPU Dictate from the tray for details.")
        if lost:
            self.overlay.show("error", "Not enough free memory for the model" if memory else
                              "The model could not be loaded", hold_s=3.0)
            self.footer.configure(text="That dictation was lost. Details in app.log.")
        self._refresh()

    def _engine_failed(self, msg):
        """An NPU error or timeout after the engine used the NPU: dictation stops for
        good (an NPU hang can crash Windows on this driver, so nothing retries). The
        engine is told to start no further piece, and its later messages are ignored."""
        log(f"engine {self.engine_gen}: failed: {msg}")
        self.engine_state, self.engine_msg = "error", msg
        self.want_engine, self.retry_at = False, None
        self._close_engine()
        self._drop_dictations()
        self.overlay.show("error", "NPU engine stopped", hold_s=3.0)
        self._notify("Dictation stopped after an NPU error. Quit NPU Dictate from the tray and start it again "
                     "(restart Windows first if it says the NPU did not answer).")
        self._refresh()

    def _start_download(self):
        self.engine_state = "download"
        self.downloading = (0, 1)
        self._refresh()

        def run():
            last = 0.0
            try:
                def progress(done, total, name):
                    nonlocal last
                    if time.monotonic() - last > 0.2 or done == total:
                        last = time.monotonic()
                        self.post(("download", done, total))

                model_store.download(progress, lambda: self.quitting)
                self.post(("download_done", None))
            except Exception as e:
                self.post(("download_done", f"{type(e).__name__}: {e}"))

        threading.Thread(target=run, daemon=True, name="download").start()

    # ---- settings ---------------------------------------------------------------
    def _fill_mics(self):
        try:
            names = input_devices()
        except Exception:
            names = []
        self.mic_box["values"] = ["System default"] + names
        self.mic_var.set(self.settings["microphone"] or "System default")

    def _changed(self):
        s = self.settings
        s["mode"], s["output"] = self.mode_var.get(), self.out_var.get()
        mic = self.mic_var.get()
        s["microphone"] = "" if mic == "System default" else mic
        for k, var in self.checks.items():
            s[k] = var.get()
        save_settings(s)
        self._refresh()

    def _refresh(self):
        s = self.settings
        key = win32.label(s["shortcut"])
        self.shortcut_lbl.configure(text=key if not self.capturing else "Press the new shortcut… (Esc cancels)")
        state = "recording" if self.recording else self.engine_state
        self.status_lbl.configure(text=STATES[state])
        self.status_dot.delete("all")
        self.status_dot.create_oval(1, 1, px(11), px(11), fill=COLORS[state], outline="")
        if self.engine_state == "download":
            done, total = self.downloading
            hint = (f"Downloading the speech model, {done / 2**20:.0f} of {total / 2**20:.0f} MB. "
                    "This happens once; the app is ready when it is done.")
        elif self.engine_state == "loading":
            hint = (self.engine_msg + " (a few seconds). You can already dictate: speech is transcribed "
                    "once the model is ready.")
        elif self.engine_state in ("error", "waiting"):
            hint = self.engine_msg
        elif s["mode"] == "toggle":
            hint = f"Press {key}, speak, then press {key} again: the text goes into the focused text box."
        else:
            hint = f"Hold {key} while you speak; release it and the text goes into the focused text box."
        self.hint.configure(text=hint)
        show = self.engine_state == "download"
        if show != self.progress_shown:
            self.progress_shown = show
            if show:
                self.progress.pack(fill="x", before=self.spacer)
            else:
                self.progress.pack_forget()
        if show:
            done, total = self.downloading
            self.progress["value"] = 1000 * done / max(total, 1)
        tray = getattr(self, "tray", None)
        if tray is not None and getattr(self, "_tray_state", None) != state:
            self._tray_state = state
            tray.icon = icon_image(COLORS[state])
            tray.title = f"{APP}: {STATES[state]}"

    def _copy_selected(self, event=None):
        sel = self.tree.selection()
        if sel:
            text = self.history[len(self.history) - 1 - self.tree.index(sel[0])]["text"]
            self.out_q.put((None, text, "clipboard", None))

    # ---- shortcut ------------------------------------------------------------
    def _register_shortcut(self, combo):
        ok = self.hotkey.set(combo)
        if not ok:
            self.footer.configure(text=f"{win32.label(combo)} is taken by another app; choose another shortcut.")
        return ok

    def _capture_shortcut(self):
        if self.capturing:
            return
        self.capturing = True
        self.hotkey.set(None)
        self.root.deiconify()
        self.root.focus_force()
        self.root.bind("<KeyPress>", self._on_capture_key)
        self._refresh()

    def _on_capture_key(self, event):
        if event.keysym == "Escape":
            return self._end_capture(self.settings["shortcut"])
        combo = win32.combo_from_event(event.keycode, event.state)
        if combo is None:
            return  # a modifier on its own: wait for the main key
        if "+" not in combo and not combo.startswith("f"):
            self.footer.configure(text="Use Ctrl, Alt, Shift or Win with the key, or an F-key on its own.")
            return
        if self._register_shortcut(combo):
            self._end_capture(combo, registered=True)
            self.footer.configure(text=f"Shortcut set to {win32.label(combo)}.")

    def _end_capture(self, combo, registered=False):
        self.root.unbind("<KeyPress>")
        self.capturing = False
        if not registered:
            self._register_shortcut(combo)
        self.settings["shortcut"] = combo
        save_settings(self.settings)
        self._refresh()

    # ---- start with Windows, shortcuts on disk -----------------------------------
    def _toggle_autostart(self):
        try:
            set_autostart(self.autostart_var.get())
        except OSError as e:
            self.footer.configure(text=f"Could not change autostart: {e}")
        self.autostart_var.set(autostart_enabled())

    def _make_shortcuts(self):
        try:
            for folder in (Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs",
                           Path(os.environ["USERPROFILE"]) / "Desktop"):
                make_shortcut(folder / f"{APP}.lnk", sys.executable if FROZEN else self.icon_path)
            self.footer.configure(text="Shortcuts added to the Start menu and the desktop.")
        except Exception as e:
            self.footer.configure(text=f"Could not create shortcuts: {e}")

    # ---- tray ----------------------------------------------------------------
    def _start_tray(self):
        self.tray = None
        try:
            import pystray

            menu = pystray.Menu(pystray.MenuItem("Open NPU Dictate", lambda: self.post(("show",)), default=True),
                                pystray.MenuItem("Quit", lambda: self.post(("quit",))))
            self.tray = pystray.Icon("npu_dictate", icon_image(COLORS["loading"]), f"{APP}: Loading", menu)
            self.tray.run_detached()
        except Exception as e:
            self.tray = None
            self.footer.configure(text=f"No tray icon ({e}); closing the window quits.")

    def _notify(self, text):
        """A Windows notification from the tray icon (the window may be hidden)."""
        self.footer.configure(text=text)
        try:
            if self.tray is not None:
                self.tray.notify(text, APP)
        except Exception:
            pass

    def hide_window(self):
        if self.tray is None:
            return self.quit()
        self.root.withdraw()
        if not getattr(self, "_told_tray", False):
            self._told_tray = True
            self._notify("Still running in the tray. Right-click the icon to quit.")

    def quit(self):
        if self.quitting:
            return
        self.quitting = True  # from here on _handle ignores events
        self.hotkey.set(None)
        if self.recording:
            self.recording = False
            self.recorder.stop()
        self.overlay.hide()
        self.root.withdraw()
        end = time.monotonic() + 7.0  # longer than the output thread's own waits (modifiers 5 s, paste 0.6 s)
        while (not self.out_q.empty() or self.out_busy) and time.monotonic() < end:
            time.sleep(0.05)  # let a paste in progress finish and restore the clipboard
        self._close_engine(wait=True)
        for e in self.closing:  # every engine process, including ones still closing
            e.proc.join(30)
        if self.tray is not None:
            self.tray.stop()
        self.root.destroy()

    # ---- dictation -------------------------------------------------------------
    def _sound(self, name):
        if self.settings["sounds"]:
            winsound.PlaySound(self.sounds[name], winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)

    def _on_hotkey(self):
        if self.capturing:
            return
        if self.recording:  # a recording always ends the way it started, whatever the state
            return self._stop() if self.rec_mode == "toggle" else None
        if self.engine_state == "waiting":
            self.retry_at = None
            self._start_model()  # audio is kept for the engine while it loads
        if self.engine_state == "error":
            self._sound("error")
            return self.overlay.show("error", "NPU engine stopped: restart the app", hold_s=2.0)
        if self.engine_state == "download":
            done, total = self.downloading
            return self.overlay.show("info", f"Downloading the model: {100 * done / max(total, 1):.0f}%",
                                     hold_s=2.0)
        if self._start() and self.rec_mode == "hold":
            threading.Thread(target=self._watch_release, args=(self.settings["shortcut"], self.session),
                             daemon=True, name="hold").start()

    def _watch_release(self, combo, session):
        """Hold to talk: the recording ends when any key of the shortcut is released."""
        mods, vk = win32.parse(combo)
        keys = [(vk,)] + [win32.MOD_VKS[m] for m in win32.MOD_ORDER if mods & win32.MODS[m]]
        while all(any(win32.key_down(v) for v in k) for k in keys):
            time.sleep(0.015)
        self.post(("release", session))

    def _start(self):
        self.session += 1
        self.rec_mode = self.settings["mode"]
        self.sessions[self.session] = {"n": None, "parts": {}, "npu": 0.0, "t_stop": None, "seconds": 0.0,
                                       "target": None, "failed": False}
        try:
            self.recorder.start(self.session, device_index(self.settings["microphone"]))
        except Exception as e:
            self._sound("error")
            self.overlay.show("error", "Microphone error", hold_s=2.0)
            self.footer.configure(text=f"Could not open the microphone: {e}")
            del self.sessions[self.session]
            return False
        self.recording = True
        self._sound("start")
        self.overlay.show("recording")
        self._refresh()
        return True

    def _stop(self):
        self.recording = False
        self.recorder.stop()
        self.sessions[self.session]["t_stop"] = time.perf_counter()
        self.sessions[self.session]["target"] = win32.foreground()  # where the text should go
        self._sound("stop")
        self.overlay.show("busy", "Loading the model, then transcribing…" if self.engine_state == "loading" else "")
        self._refresh()

    def _check_done(self, session):
        s = self.sessions.get(session)
        if s is None or s["n"] is None or len(s["parts"]) < s["n"]:
            return
        del self.sessions[session]
        text = " ".join(s["parts"][i] for i in range(s["n"]) if s["parts"][i]).strip()
        if s.get("failed"):
            self.overlay.show("error", "Part of the dictation could not be transcribed", hold_s=3.0)
            self.footer.configure(text="A piece of the dictation failed on the CPU and is missing"
                                       + (" from the text below." if text else ".") + " Details in engine.log.")
        if not text:
            if not self.recording and not s.get("failed"):
                self.overlay.show("info", "No speech heard", hold_s=1.2)
            return
        latency = time.perf_counter() - (s["t_stop"] or time.perf_counter())
        self.out_q.put((session, text + (" " if self.settings["trailing_space"] else ""), self.settings["output"],
                        s["target"]))
        self.history.append({"time": time.strftime("%H:%M"), "text": text})
        del self.history[:-50]
        self.tree.insert("", 0, values=(self.history[-1]["time"], text))
        for extra in self.tree.get_children()[50:]:
            self.tree.delete(extra)
        if not s.get("failed"):
            self.footer.configure(text=f"Last: {s['seconds']:.1f} s recorded, NPU encoder {1000 * s['npu']:.0f} ms, "
                                       f"text ready {1000 * latency:.0f} ms after you stopped.")

    def _output_loop(self):
        """Inserts finished text (all clipboard use happens on this thread)."""
        while True:
            # This thread owns the clipboard after a paste: it must handle window
            # messages (WM_DESTROYCLIPBOARD) or another program's copy would wait on it.
            win32.pump_messages()
            try:
                session, text, mode, target = self.out_q.get(timeout=0.05)
            except queue.Empty:
                continue
            self.out_busy = True
            err, done = None, mode

            def moved():
                return target and win32.foreground() != target

            try:
                if os.environ.get("NPU_DICTATE_TEST_OUTPUT"):  # test hook: log instead of inserting
                    log(f"output {session}: {text!r}")
                    mode = done = "test"
                if mode not in ("clipboard", "test"):
                    if not win32.wait_modifiers_released(5.0):
                        done = "held"  # Ctrl+V with Shift/Alt/Win held would be another command
                    elif moved():  # the user moved on to another window before the text was ready
                        done = "moved"
                    if done in ("held", "moved"):
                        win32.set_clipboard(text)
                if done == "paste":
                    has, old = win32.clipboard_text()
                    if has and old is None:  # an image, files or rich text: type instead of losing them
                        done = "type"
                    else:
                        ours = win32.set_clipboard(text, history=False)
                        try:
                            time.sleep(0.03)
                            if moved():  # checked right before the keystroke
                                done = "moved"
                            else:
                                win32.send_paste()
                                time.sleep(0.6)  # slow apps read the clipboard late
                        finally:
                            if done == "paste":
                                win32.restore_clipboard(ours, old if has else None)  # unless someone copied since
                if done == "type":
                    if moved():
                        win32.set_clipboard(text)
                        done = "moved"
                    else:
                        win32.type_text(text)
                elif done == "clipboard":
                    win32.set_clipboard(text)
            except Exception as e:
                err = str(e)
            self.out_busy = False
            self.post(("output_done", session, done, err))

    # ---- event loop ------------------------------------------------------------
    def _poll(self):
        try:
            self._poll_once()
        except Exception:
            import traceback

            log("event error:\n" + traceback.format_exc())
        if not self.quitting:
            self.root.after(20, self._poll)

    def _poll_once(self):
        while not self.quitting:
            try:
                ev = self.ui_q.get_nowait()
            except queue.Empty:
                break
            self._handle(ev)
        if self.quitting:
            return
        self._spawn_if_possible()
        if self.engine_state == "waiting" and self.retry_at is not None and time.monotonic() >= self.retry_at:
            self.retry_at = None
            self._start_model()
        # Watchdog on NPU work. The clock excludes sleep, so a laptop that sleeps
        # in the middle of a piece does not count as an NPU hang.
        if self.busy is not None and self.busy[0] == self.engine_gen and win32.awake_time() - self.busy[1] > WATCHDOG_S:
            self._engine_failed(f"The NPU did not answer within {WATCHDOG_S:.0f} s (possible NPU hang). Dictation "
                                "is stopped. Restart Windows before using the NPU again, then start the app.")
        eng = self.engine
        if (eng is not None and self.engine_state == "loading" and not eng.npu_used
                and time.perf_counter() - eng.t0 > START_TIMEOUT_S):
            self._engine_start_failed(f"no answer within {START_TIMEOUT_S:.0f} s", retry=False)

    def _handle(self, ev):
        kind = ev[0]
        if self.quitting:
            return
        if kind == "hotkey":
            self._on_hotkey()
        elif kind == "release":
            if self.recording and ev[1] == self.session and self.rec_mode == "hold":
                self._stop()
        elif kind == "ended":
            _, session, n, seconds = ev
            if session in self.sessions:
                self.sessions[session]["n"] = n
                self.sessions[session]["seconds"] = seconds
                self._check_done(session)
        elif kind == "recorder_failed":
            if self.recording and ev[1] == self.session:
                self.recording = False
                self.recorder.stop()
            if self.sessions.pop(ev[1], None) is not None:
                self._sound("error")
                self.overlay.show("error", "Recording failed", hold_s=2.0)
                self.footer.configure(text="The recording failed and was lost. Details in app.log.")
            self._refresh()
        elif kind == "download":
            self.downloading = (ev[1], ev[2])
            self._refresh()
        elif kind == "download_done":
            self.downloading = None
            if self.engine_state != "download":
                pass
            elif ev[1] is None and not model_store.missing():
                log("model downloaded")
                self._start_model()
            elif not self.quitting:
                log(f"model download failed: {ev[1]}")
                self.engine_state = "waiting"
                delay = RETRY_S[min(self.retries, len(RETRY_S) - 1)]
                self.retries += 1
                self.retry_at = time.monotonic() + delay
                self.engine_msg = (f"The model download failed ({ev[1]}). It tries again in {delay} s, "
                                   "or when you press the shortcut.")
                self._refresh()
        elif kind == "engine":
            if ev[1] != self.engine_gen:
                return  # from an engine that was closed
            msg = ev[2]
            if msg[0] == "status":
                self.engine_msg = msg[1]
            elif msg[0] == "npu":  # warm-up: the first NPU submissions
                self.busy = (self.engine_gen, win32.awake_time())
            elif msg[0] == "ready":
                self.engine_state = "ready"
                self.retries = self.died_retries = 0
                self.busy = None
                self.footer.configure(text=f"NPU engine ready in {msg[1]:.1f} s.")
            elif msg[0] == "start_failed":
                if self.engine_state == "loading":
                    self._engine_start_failed(msg[1], retry=msg[2])
            elif msg[0] == "died":
                _, code, npu_used = msg
                if not npu_used and self.engine_state == "loading":
                    # Before the NPU was used: one automatic retry, then only on a press.
                    self.died_retries += 1
                    self._engine_start_failed(f"the engine process exited (code {code})",
                                              retry=self.died_retries <= 1)
                elif self.engine_state != "error":
                    self._engine_failed(f"The engine process exited (code {code}). Quit and start the app again.")
            elif msg[0] == "fatal":
                log(f"engine {self.engine_gen}: fatal: {msg[1]}")
                if self.engine_state != "error":
                    self._engine_failed(msg[1])
            elif msg[0] == "busy":
                self.busy = (self.engine_gen, win32.awake_time())
            elif msg[0] in ("text", "piece_failed"):
                session, index = msg[1], msg[2]
                self.busy = None
                s = self.sessions.get(session)
                if s is not None:
                    if msg[0] == "text":
                        s["parts"][index] = msg[3]
                        s["npu"] += msg[5]
                    else:
                        log(f"piece {session}.{index} failed on the CPU")
                        s["parts"][index], s["failed"] = "", True
                    self._check_done(session)
            self._refresh()
        elif kind == "output_done":
            _, session, done, err = ev
            if session is None:  # copied from the history list
                self.footer.configure(text=f"Could not copy: {err}" if err else "Copied to the clipboard.")
            elif err:
                self.overlay.show("error", "Could not insert the text", hold_s=2.0)
                self.footer.configure(text=f"Insert failed: {err}. The text is in the history.")
            elif done in ("moved", "held"):
                self.overlay.show("info", "Copied: press Ctrl+V to paste", hold_s=2.5)
                self.footer.configure(text="The text was copied to the clipboard instead of inserted: "
                                      + ("the focused window changed before it was ready." if done == "moved"
                                         else "Shift, Alt or Win was still held down."))
            elif not self.recording:
                label = {"paste": "Inserted", "type": "Typed", "clipboard": "Copied to the clipboard",
                         "test": "Test output"}[done]
                self.overlay.show("done", label, hold_s=0.8)
        elif kind == "show":
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        elif kind == "quit":
            self.quit()


def open_log():
    """app.log, with the previous one kept as app.log.1 once it passes 1 MB."""
    path = APPDIR / "app.log"
    try:
        if path.stat().st_size > 1 << 20:
            os.replace(path, APPDIR / "app.log.1")
    except OSError:
        pass
    return open(path, "a", buffering=1, encoding="utf-8")


def log_exceptions():
    """Every uncaught exception goes to app.log (pythonw has no console to show it)."""
    def hook(kind, value, tb):
        import traceback

        log("uncaught exception:\n" + "".join(traceback.format_exception(kind, value, tb)))

    sys.excepthook = hook
    threading.excepthook = lambda a: hook(a.exc_type, a.exc_value, a.exc_traceback)


def self_check():
    """`NPU Dictate.exe --check`: is everything in place to run? Opens the NPU
    device to read its name but runs nothing on it."""
    lines = [f"{APP} {VERSION}", f"app: {sys.executable}"]

    def item(name, fn):
        try:
            lines.append(f"OK    {name}: {fn()}")
        except Exception as e:
            lines.append(f"FAIL  {name}: {type(e).__name__}: {e}")

    def npu():
        import npu_runtime

        xrt = npu_runtime.import_pyxrt()
        return xrt.device(0).get_info(xrt.xrt_info_device.name)

    def bundle():
        import npu_runtime

        b = model_store.bundle_dir()
        problems = model_store.missing(b) or npu_runtime.verify(b)
        if problems:
            raise FileNotFoundError(f"{b}: " + "; ".join(problems[:4]))
        return str(b)

    def cpu_models():
        import parakeet_pipeline as pp

        b = model_store.bundle_dir()
        for f in ("nemo128.onnx", "encoder_prefix.onnx", "decoder_joint-model.onnx"):
            pp.session(b / f, ["CPUExecutionProvider"])
        return "the three ONNX parts load"

    item("NPU", npu)
    item("model files", bundle)
    item("CPU models", cpu_models)
    item("memory", lambda: "{} MB available, {} MB commit available (loading needs {} MB of commit)".format(
        *engine.memory_mb(), engine.NEED_COMMIT_MB))
    item("microphone", lambda: f"MME input device {device_index('')}")
    item("start with Windows", autostart_enabled)
    text = "\n".join(lines)
    APPDIR.mkdir(parents=True, exist_ok=True)
    (APPDIR / "check.txt").write_text(text + "\n", encoding="utf-8")
    if os.environ.get("NPU_DICTATE_TEST_OUTPUT"):  # test hook: no dialog
        return
    root = tk.Tk()
    root.withdraw()
    (messagebox.showinfo if "FAIL" not in text else messagebox.showwarning)(f"{APP} check", text)


def main():
    import faulthandler

    APPDIR.mkdir(parents=True, exist_ok=True)
    main.fault_log = open(APPDIR / "fault.log", "a", encoding="utf-8")  # native crashes (stack of every thread)
    faulthandler.enable(main.fault_log, all_threads=True)
    win32.set_dpi_aware()
    win32.set_app_id("npu.dictate")
    if "--check" in sys.argv:
        return self_check()
    background = "--background" in sys.argv  # started by Windows at sign-in
    if not win32.single_instance("Local\\NPU-Dictate-single-instance"):
        if not background:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(APP, "NPU Dictate is already running (look for its tray icon).")
        return
    APPDIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open_log()
    print(f"\n=== app start {time.strftime('%Y-%m-%d %H:%M:%S')} v{VERSION} pid {os.getpid()}"
          f"{' (at sign-in)' if background else ''}")
    log_exceptions()
    try:
        app = App(background)
    except Exception:
        import traceback

        log("could not start:\n" + traceback.format_exc())
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(APP, f"NPU Dictate could not start:\n\n{traceback.format_exc(limit=2)}\n"
                                  f"Details in {APPDIR / 'app.log'}")
        os._exit(1)
    app.root.mainloop()
    log("app exit")
    os._exit(0)  # never wait on leftover threads or queues at interpreter shutdown


if __name__ == "__main__":
    mp.freeze_support()
    main()
