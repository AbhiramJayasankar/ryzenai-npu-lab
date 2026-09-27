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
import win32  # noqa: E402

APP = "NPU Dictate"
APPDIR = Path(os.environ.get("APPDATA", Path.home())) / APP
LAUNCHER = Path(__file__).resolve().parent / "NPU Dictate.ps1"
DEFAULTS = {"shortcut": "ctrl+shift+space", "mode": "toggle", "microphone": "", "output": "paste",
            "trailing_space": True, "sounds": True, "overlay": True, "start_minimized": False,
            "unload_after_s": 0}
MODES = {"toggle": "Press to start, press again to insert", "hold": "Hold to talk, release to insert"}
OUTPUTS = {"paste": "Paste (your clipboard is restored)", "type": "Type it as keystrokes",
           "clipboard": "Only copy to the clipboard"}
COLORS = {"loading": "#8e8e93", "unloaded": "#8e8e93", "ready": "#30a14e", "recording": "#e5484d", "busy": "#d9a400",
          "error": "#e5484d"}
WATCHDOG_S = 30.0
S = 1.0  # display scale (DPI / 96), set when the window is created


def px(v):
    return int(round(v * S))


# -- settings, sounds, icon ------------------------------------------------------
def load_settings():
    try:
        return {**DEFAULTS, **json.loads((APPDIR / "settings.json").read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_settings(s):
    APPDIR.mkdir(parents=True, exist_ok=True)
    (APPDIR / "settings.json").write_text(json.dumps(s, indent=1), encoding="utf-8")


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


def input_devices():
    """Microphones of the MME host API (it resamples to 16 kHz for us), by name."""
    import sounddevice as sd

    mme = next((i for i, h in enumerate(sd.query_hostapis()) if h["name"] == "MME"), None)
    return [d["name"] for d in sd.query_devices()
            if d["max_input_channels"] > 0 and d["hostapi"] == mme and "Sound Mapper" not in d["name"]]


def device_index(name):
    import sounddevice as sd

    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d["name"] == name and d["max_input_channels"] > 0:
            return i
    return None  # unplugged: system default


def make_shortcut(path, icon):
    ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{path}');"
          f"$s.TargetPath='powershell.exe';"
          f"$s.Arguments='-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"{LAUNCHER}\"';"
          f"$s.WorkingDirectory='{LAUNCHER.parent}';$s.IconLocation='{icon}';"
          f"$s.Description='Dictation with Parakeet on the Ryzen AI NPU';$s.WindowStyle=7;$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


# -- engine process proxy --------------------------------------------------------
class Engine:
    def __init__(self, post, gen):
        self.gen, self.closing = gen, False
        ctx = mp.get_context("spawn")
        self.inq, self.outq = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=engine.serve, args=(self.inq, self.outq, str(APPDIR / "engine.log")),
                                daemon=True, name="npu-engine")
        self.proc.start()
        threading.Thread(target=self._read, args=(post,), daemon=True).start()

    def _read(self, post):
        while True:
            try:
                post(("engine", self.gen, self.outq.get(timeout=1.0)))
            except queue.Empty:
                if not self.proc.is_alive():
                    if not self.closing:
                        post(("engine", self.gen,
                              ("fatal", f"The engine process exited (code {self.proc.exitcode}).")))
                    return

    def send(self, session, index, audio):
        self.inq.put(("audio", session, index, audio))

    def close(self):
        """Ends the engine process; this frees its ~1.4 GB (weights and NPU context)."""
        self.closing = True
        if self.proc.is_alive():
            self.inq.put(("quit",))
            self.proc.join(10)


# -- audio capture ---------------------------------------------------------------
class Recorder:
    """Microphone -> Chunker -> engine, on its own thread."""

    def __init__(self, app):
        self.app = app
        self.q = queue.Queue()
        self.stream = None
        self.level = 0.0
        self.seconds = 0.0
        threading.Thread(target=self._loop, daemon=True, name="recorder").start()

    def start(self, session, device):
        import sounddevice as sd

        self.q.put(("start", session))
        fake = os.environ.get("NPU_DICTATE_FAKE_MIC")  # test hook: a 16 kHz WAV played as the microphone
        if fake:
            self.stream = _FakeStream(fake, self._cb)
            return
        try:
            self.stream = sd.InputStream(samplerate=chunker.SR, channels=1, dtype="float32",
                                         blocksize=chunker.BLOCK, device=device, callback=self._cb)
            self.stream.start()
        except Exception:
            self.stream = None
            self.q.put(("abort",))
            raise

    def _cb(self, data, frames, t, status):
        self.q.put(("audio", data[:, 0].copy()))

    def stop(self):
        s, self.stream = self.stream, None
        if s is not None:
            s.stop()
            s.close()
        self.q.put(("stop",))

    def _loop(self):
        ch, session, index = None, None, 0
        while True:
            kind, *arg = self.q.get()
            if kind == "start":
                ch, session, index = chunker.Chunker(), arg[0], 0
                self.level = self.seconds = 0.0
            elif kind == "abort":
                ch = None
            elif ch is None:
                continue
            elif kind == "audio":
                pieces = ch.push(arg[0])
                self.level, self.seconds = ch.level, ch.total / chunker.SR
            else:  # stop
                pieces = ch.finish()
            if ch is not None and kind in ("audio", "stop"):
                for p in pieces:
                    self.app.post(("sent", session, index))
                    self.app.engine.send(session, index, p)
                    index += 1
                if kind == "stop":
                    self.app.post(("ended", session, index, self.seconds))
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
    def __init__(self):
        APPDIR.mkdir(parents=True, exist_ok=True)
        self.settings = load_settings()
        self.ui_q = queue.Queue()
        self.post = self.ui_q.put
        self.engine_state, self.engine_msg = "loading", "Starting the NPU engine…"
        self.engine, self.engine_gen = None, 0
        self.last_active = time.perf_counter()
        self.recording = False
        self.session = 0
        self.sessions = {}
        self.sent = {}  # (session, index) -> time sent, for the watchdog
        self.capturing = False
        self.history = []
        self.sounds = make_sounds()
        self.icon_path = APPDIR / "icon.ico"
        icon_image("#30a14e", 256).save(self.icon_path, sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])

        self.root = tk.Tk()
        global S
        S = self.root.winfo_fpixels("1i") / 96
        self.root.title(APP)
        self.root.iconbitmap(default=str(self.icon_path))
        self.root.geometry(f"{px(560)}x{px(660)}")
        self.root.minsize(px(480), px(520))
        self.root.protocol("WM_DELETE_WINDOW", self.hide_window)
        self._build_ui()

        self._load_engine()
        self.recorder = Recorder(self)
        self.overlay = Overlay(self)
        self.out_q = queue.Queue()
        threading.Thread(target=self._output_loop, daemon=True, name="output").start()
        self.hotkey = win32.Hotkey(lambda: self.post(("hotkey",)))
        self._register_shortcut(self.settings["shortcut"])
        self._start_tray()
        self._refresh()
        if self.settings["start_minimized"] and self.tray is not None:
            self.root.withdraw()
        self.root.after(20, self._poll)

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
        self.hint.pack(fill="x", pady=(px(4), px(12)))

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
        ttk.Label(box, text="Free memory").grid(row=r, column=0, sticky="w", pady=px(4))
        uf = ttk.Frame(box)
        uf.grid(row=r, column=1, sticky="w", padx=(px(12), px(0)), pady=px(4))
        ttk.Label(uf, text="Unload the model after").pack(side="left")
        self.unload_var = tk.StringVar(value=str(self.settings["unload_after_s"]))
        sp = ttk.Spinbox(uf, from_=0, to=86400, increment=30, width=7, textvariable=self.unload_var,
                         command=self._unload_changed)
        sp.pack(side="left", padx=px(6))
        sp.bind("<FocusOut>", self._unload_changed)
        sp.bind("<Return>", self._unload_changed)
        ttk.Label(uf, text="s idle (0 = keep it loaded)", style="Muted.TLabel").pack(side="left")
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
        self.autostart_var = tk.BooleanVar(value=self._autostart_path().exists())
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

    def _load_engine(self):
        self.engine_gen += 1
        self.engine_state, self.engine_msg = "loading", "Loading the model onto the NPU…"
        self.engine = Engine(self.post, self.engine_gen)
        self._refresh()

    def _unload_engine(self):
        eng, self.engine = self.engine, None
        self.engine_gen += 1  # ignore anything the old process still sends
        self.engine_state = "unloaded"
        threading.Thread(target=eng.close, daemon=True, name="engine-close").start()
        self.footer.configure(text=f"Model unloaded after {self.settings['unload_after_s']} s idle (memory freed); "
                                   "it reloads when you press the shortcut.")
        self._refresh()

    def _unload_changed(self, event=None):
        try:
            v = max(0, min(86400, int(float(self.unload_var.get()))))
        except ValueError:
            v = self.settings["unload_after_s"]
        self.unload_var.set(str(v))
        if v != self.settings["unload_after_s"]:
            self.settings["unload_after_s"] = v
            self.last_active = time.perf_counter()
            save_settings(self.settings)
            self._refresh()

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
        text = {"loading": "Loading", "ready": "Ready", "recording": "Recording", "error": "Error",
                "unloaded": "Unloaded"}[state]
        self.status_lbl.configure(text=text)
        self.status_dot.delete("all")
        self.status_dot.create_oval(1, 1, px(11), px(11), fill=COLORS[state], outline="")
        if self.engine_state == "loading":
            hint = (self.engine_msg + " (a few seconds; the first start ever compiles the NPU program for ~2 min). "
                    "You can already dictate: speech is transcribed once the model is ready.")
        elif self.engine_state == "error":
            hint = self.engine_msg
        elif self.engine_state == "unloaded":
            hint = (f"The model is unloaded to free memory. Press {key} and just start speaking: it reloads in a "
                    "few seconds and your speech is transcribed once it is ready.")
        elif s["mode"] == "toggle":
            hint = f"Press {key}, speak, then press {key} again: the text goes into the focused text box."
        else:
            hint = f"Hold {key} while you speak; release it and the text goes into the focused text box."
        self.hint.configure(text=hint)
        tray = getattr(self, "tray", None)
        if tray is not None and getattr(self, "_tray_state", None) != state:
            self._tray_state = state
            tray.icon = icon_image(COLORS[state])
            tray.title = f"{APP}: {text}"

    def _copy_selected(self, event=None):
        sel = self.tree.selection()
        if sel:
            win32.set_clipboard(self.history[len(self.history) - 1 - self.tree.index(sel[0])]["text"])
            self.footer.configure(text="Copied to the clipboard.")

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

    # ---- shortcuts on disk ---------------------------------------------------
    @staticmethod
    def _autostart_path():
        return Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Startup" / f"{APP}.lnk"

    def _toggle_autostart(self):
        p = self._autostart_path()
        try:
            if self.autostart_var.get():
                make_shortcut(p, self.icon_path)
            elif p.exists():
                p.unlink()
        except Exception as e:
            self.footer.configure(text=f"Could not change autostart: {e}")
        self.autostart_var.set(p.exists())

    def _make_shortcuts(self):
        try:
            for folder in (Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs",
                           Path(os.environ["USERPROFILE"]) / "Desktop"):
                make_shortcut(folder / f"{APP}.lnk", self.icon_path)
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

    def hide_window(self):
        if self.tray is None:
            return self.quit()
        self.root.withdraw()
        if not getattr(self, "_told_tray", False):
            self._told_tray = True
            try:
                self.tray.notify("Still running in the tray. Right-click the icon to quit.", APP)
            except Exception:
                pass

    def quit(self):
        self.hotkey.set(None)
        if self.recording:
            self.recorder.stop()
        self.overlay.hide()
        self.root.withdraw()
        self.root.update()
        if self.engine is not None:
            self.engine.close()
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
        if self.engine_state == "error":
            self._sound("error")
            return self.overlay.show("error", "NPU engine stopped (see app)", hold_s=1.5)
        if self.engine_state == "unloaded" and not self.recording:
            self._load_engine()  # audio is queued for the engine while it loads
        if self.settings["mode"] == "toggle":
            return self._stop() if self.recording else self._start()
        if not self.recording and self._start():
            vk = win32.parse(self.settings["shortcut"])[1]
            threading.Thread(target=self._watch_release, args=(vk, self.session), daemon=True).start()

    def _watch_release(self, vk, session):
        while win32.key_down(vk):
            time.sleep(0.015)
        self.post(("release", session))

    def _start(self):
        self.session += 1
        self.sessions[self.session] = {"n": None, "parts": {}, "npu": 0.0, "t_stop": None, "audio_s": 0.0}
        try:
            self.recorder.start(self.session, device_index(self.settings["microphone"]))
        except Exception as e:
            self._sound("error")
            self.overlay.show("error", "Microphone error", hold_s=2.0)
            self.footer.configure(text=f"Could not open the microphone: {e}")
            del self.sessions[self.session]
            return False
        self.recording = True
        self.last_active = time.perf_counter()
        self._sound("start")
        self.overlay.show("recording")
        self._refresh()
        return True

    def _stop(self):
        self.recording = False
        self.recorder.stop()
        self.sessions[self.session]["t_stop"] = time.perf_counter()
        self.last_active = time.perf_counter()
        self._sound("stop")
        self.overlay.show("busy", "Loading the model, then transcribing…" if self.engine_state == "loading" else "")
        self._refresh()

    def _check_done(self, session):
        s = self.sessions.get(session)
        if s is None or s["n"] is None or len(s["parts"]) < s["n"]:
            return
        del self.sessions[session]
        text = " ".join(s["parts"][i] for i in range(s["n"]) if s["parts"][i]).strip()
        if not text:
            if not self.recording:
                self.overlay.show("info", "No speech heard", hold_s=1.2)
            return
        latency = time.perf_counter() - s["t_stop"]
        self.out_q.put((session, text + (" " if self.settings["trailing_space"] else ""), self.settings["output"]))
        self.history.append({"time": time.strftime("%H:%M"), "text": text})
        del self.history[:-50]
        self.tree.insert("", 0, values=(self.history[-1]["time"], text))
        for extra in self.tree.get_children()[50:]:
            self.tree.delete(extra)
        self.footer.configure(text=f"Last: {s['audio_s']:.1f} s of speech, NPU encoder {1000 * s['npu']:.0f} ms, "
                                   f"text ready {1000 * latency:.0f} ms after you stopped.")

    def _output_loop(self):
        while True:
            session, text, mode = self.out_q.get()
            err = None
            try:
                win32.wait_modifiers_released()
                if mode == "paste":
                    has, old = win32.clipboard_text()
                    if has and old is None:  # an image or files on the clipboard: type instead of losing it
                        win32.type_text(text)
                    else:
                        win32.set_clipboard(text, history=False)
                        time.sleep(0.03)
                        win32.send_paste()
                        time.sleep(0.3)
                        win32.set_clipboard(old, history=False) if has else win32.set_clipboard(None)
                elif mode == "type":
                    win32.type_text(text)
                else:
                    win32.set_clipboard(text)
            except Exception as e:
                err = str(e)
            self.post(("output_done", session, mode, err))

    # ---- event loop ------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                self._handle(self.ui_q.get_nowait())
        except queue.Empty:
            pass
        if self.sent and self.engine_state == "ready" and time.perf_counter() - min(self.sent.values()) > WATCHDOG_S:
            self._engine_failed("The NPU did not answer within 30 s (possible NPU hang). Dictation is stopped; "
                                "quit the app. If Windows reports a display/NPU reset, reboot before using the NPU.")
        u = self.settings["unload_after_s"]
        if (u > 0 and self.engine_state == "ready" and not self.recording and not self.sent and not self.sessions
                and self.out_q.empty() and time.perf_counter() - self.last_active > u):
            self._unload_engine()
        self.root.after(20, self._poll)

    def _engine_failed(self, msg):
        self.engine_state, self.engine_msg = "error", msg
        if self.recording:
            self.recording = False
            self.recorder.stop()
        self.sent.clear()
        self.sessions.clear()
        self.overlay.show("error", "NPU engine stopped", hold_s=3.0)
        self._refresh()

    def _handle(self, ev):
        kind = ev[0]
        if kind == "hotkey":
            self._on_hotkey()
        elif kind == "release":
            if self.recording and ev[1] == self.session and self.settings["mode"] == "hold":
                self._stop()
        elif kind == "sent":
            self.sent[(ev[1], ev[2])] = time.perf_counter()
        elif kind == "ended":
            _, session, n, seconds = ev
            if session in self.sessions:
                self.sessions[session]["n"] = n
                self._check_done(session)
        elif kind == "engine":
            if ev[1] != self.engine_gen:
                return  # from an engine that was unloaded
            msg = ev[2]
            self.last_active = time.perf_counter()
            if msg[0] == "status":
                self.engine_msg = msg[1]
            elif msg[0] == "ready":
                self.engine_state = "ready"
                now = time.perf_counter()
                self.sent = {k: now for k in self.sent}  # queued while loading: the watchdog starts now
                self.footer.configure(text=f"NPU engine ready in {msg[1]:.1f} s.")
            elif msg[0] == "fatal":
                if self.engine_state != "error":
                    self._engine_failed(msg[1])
            elif msg[0] == "text":
                _, session, index, text, audio_s, npu_s, total_s = msg
                self.sent.pop((session, index), None)
                s = self.sessions.get(session)
                if s is not None:
                    s["parts"][index] = text
                    s["npu"] += npu_s
                    s["audio_s"] += audio_s
                    self._check_done(session)
            self._refresh()
        elif kind == "output_done":
            _, session, mode, err = ev
            self.last_active = time.perf_counter()
            if err:
                self.overlay.show("error", "Could not insert the text", hold_s=2.0)
                self.footer.configure(text=f"Insert failed: {err}. The text is in the history.")
            elif not self.recording:
                done = {"paste": "Inserted", "type": "Typed", "clipboard": "Copied to the clipboard"}[mode]
                self.overlay.show("done", done, hold_s=0.8)
        elif kind == "show":
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        elif kind == "quit":
            self.quit()


def main():
    win32.set_dpi_aware()
    win32.set_app_id("npu.dictate")
    if not win32.single_instance("Local\\NPU-Dictate-single-instance"):
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(APP, "NPU Dictate is already running (look for its tray icon).")
        return
    APPDIR.mkdir(parents=True, exist_ok=True)
    log = open(APPDIR / "app.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = log
    print(f"\n=== app start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    app = App()
    app.root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    main()
