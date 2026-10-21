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
from tkinter import font as tkfont, messagebox, ttk

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chunker  # noqa: E402
import engine  # noqa: E402
import model_store  # noqa: E402
import win32  # noqa: E402

APP = "NPU Dictate"
VERSION = "1.1.0"
APPDIR = Path(os.environ.get("APPDATA", Path.home())) / APP
FROZEN = getattr(sys, "frozen", False)
LAUNCHER = Path(__file__).resolve().parent / "NPU Dictate.ps1"
DEFAULTS = {"shortcut": "ctrl+shift+space", "mode": "toggle", "microphone": "", "output": "paste",
            "trailing_space": True, "sounds": True, "overlay": True, "start_minimized": False,
            "autostart_set": False, "unload_after_s": 0}
MODES = {"toggle": "Press to start, press again to insert", "hold": "Hold to talk, release to insert"}
OUTPUTS = {"paste": "Paste (your clipboard is restored)", "type": "Type it as keystrokes",
           "clipboard": "Only copy to the clipboard"}
STATES = {"loading": "Loading", "download": "Downloading", "waiting": "Waiting", "ready": "Ready",
          "unloaded": "Unloaded", "recording": "Recording", "error": "Stopped"}
WATCHDOG_S = 30.0  # NPU work (a piece, or the warm-up) must finish this soon
START_TIMEOUT_S = 60.0  # the engine must reach the NPU this soon after it is spawned (sleep not counted)
RETRY_S = (10, 30, 60, 120, 300)  # waits between attempts to load the model
MIC_REFRESH_MS = 20000  # how often the microphone list is read again while idle (~40 ms each)
S = 1.0  # display scale (DPI / 96), set when the window is created


def px(v):
    return int(round(v * S))


# -- theme -----------------------------------------------------------------------
TOKENS = {
    "light": {"bg": "#F3F4F6", "surface": "#FFFFFF", "ink": "#1B1F24", "muted": "#5E6670", "line": "#D9DDE3",
              "accent": "#2F5BD3", "ready": "#1F8A4C", "recording": "#D93A3A", "waiting": "#B7791F",
              "idle": "#7A828C", "error": "#D93A3A"},
    "dark": {"bg": "#1C1E22", "surface": "#26292E", "ink": "#E8EAED", "muted": "#9AA1AA", "line": "#3A3E45",
             "accent": "#7C9CF2", "ready": "#46C27A", "recording": "#FF5C5C", "waiting": "#E0A84A",
             "idle": "#8D949D", "error": "#FF5C5C"},
}
STATE_TOKEN = {"loading": "idle", "download": "idle", "unloaded": "idle", "waiting": "waiting", "ready": "ready",
               "recording": "recording", "error": "error"}
COLORS = {k: TOKENS["light"][STATE_TOKEN[k]] for k in STATES}  # the tray icon
FONT = {}  # "display" and "text" font families, set by pick_fonts()
_SEMIBOLD = {}  # family -> its installed Semibold family (None if there is none)
_IMAGES = {}  # images of the flat notebook elements: kept alive, recolored on a theme switch


def blend(color, base, amount):
    """`color` laid over `base` at `amount` (0..1), as "#rrggbb"."""
    a, b = ([int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in (color, base))
    return "#" + "".join(f"{round(x * amount + y * (1 - amount)):02x}" for x, y in zip(a, b))


def contrast(a, b):
    """WCAG contrast ratio of two "#rrggbb" colors (1 to 21)."""
    def lum(c):
        v = [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        v = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def on_accent(tok):
    """Text on an accent fill: white where it is readable (light theme), else the window color (dark theme)."""
    return "#FFFFFF" if contrast("#FFFFFF", tok["accent"]) >= 4.5 else tok["bg"]


def pick_fonts(root):
    """The first installed family of each kind (Bahnschrift and Segoe UI come with Windows 10 and 11)."""
    have = {f.lower(): f for f in tkfont.families(root)}
    for kind, names in (("display", ("Bahnschrift", "Segoe UI", "Helvetica")),
                        ("text", ("Segoe UI Variable Text", "Segoe UI", "Helvetica"))):
        FONT[kind] = next((have[n.lower()] for n in names if n.lower() in have), names[-1])
        _SEMIBOLD[FONT[kind]] = have.get(f"{FONT[kind]} semibold".lower())


def font(kind, size, semibold=False):
    """A Tk font tuple, size in points. Semibold is the family's Semibold face if installed, else bold."""
    family = FONT.get(kind, "Helvetica")
    if not semibold:
        return (family, size)
    face = _SEMIBOLD.get(family)
    return (face, size) if face else (family, size, "bold")


def _flat_notebook_images(root, st, tok):
    """Image elements of Flat.TNotebook: the tab (a 2 px accent underline when selected, then a gap to the
    next tab) and the client area (a 1 px line along its top, under the tab row)."""
    gap, under, rule = px(20), px(2), px(1)
    if "Flat.Notebook.tab" not in st.element_names():
        _IMAGES.update(tab=tk.PhotoImage(master=root, width=1 + gap, height=1 + under),
                       tab_on=tk.PhotoImage(master=root, width=1 + gap, height=1 + under),
                       rule=tk.PhotoImage(master=root, width=1, height=rule + 1))
        st.element_create("Flat.Notebook.tab", "image", _IMAGES["tab"], ("selected", _IMAGES["tab_on"]),
                          border=(0, 0, gap, under), sticky="nswe")
        st.element_create("Flat.Notebook.client", "image", _IMAGES["rule"], border=(0, rule, 0, 0), sticky="nswe")
    _IMAGES["tab_on"].blank()
    _IMAGES["tab_on"].put(tok["accent"], to=(0, 1, 1, 1 + under))
    _IMAGES["rule"].blank()
    _IMAGES["rule"].put(tok["line"], to=(0, 0, 1, rule))


def _recolor_lists(root, opts):
    """Combobox lists that already exist (the option database reaches only new ones)."""
    todo = ["."]
    while todo:
        for child in root.tk.splitlist(root.tk.call("winfo", "children", todo.pop())):
            todo.append(str(child))
            if str(child).endswith(".popdown.f.l"):
                root.tk.call(child, "configure", *opts)


def build_style(root, tok):
    """The ttk styles for the theme colors, on clam (it looks the same on every Windows version).
    Safe to call again when the theme changes: existing widgets follow."""
    st = ttk.Style(root)
    st.theme_use("clam")
    bg, surface, ink, muted, line, accent = (tok[k] for k in ("bg", "surface", "ink", "muted", "line", "accent"))
    on = on_accent(tok)
    tk8 = tk.TkVersion < 9  # Tk 9 scales its element sizes itself; in Tk 8.6 they are pixels
    text = font("text", 10)
    focus = [("focus", accent)]

    st.configure(".", background=bg, foreground=ink, bordercolor=line, lightcolor=bg, darkcolor=bg, troughcolor=line,
                 fieldbackground=surface, selectbackground=accent, selectforeground=on, insertcolor=ink,
                 arrowcolor=ink, focuscolor=accent, focusthickness=1, focussolid=1, font=text)
    st.map(".", background=[], foreground=[("disabled", muted)], selectbackground=[], selectforeground=[])

    st.configure("TFrame", background=bg)
    st.configure("Surface.TFrame", background=surface)
    st.configure("Divider.TFrame", background=line, borderwidth=0, relief="flat")
    st.configure("TLabel", background=bg, foreground=ink, font=text)
    st.configure("Surface.TLabel", background=surface, foreground=ink)
    st.configure("Title.TLabel", foreground=ink, font=font("display", 20, True))
    st.configure("Group.TLabel", foreground=muted, font=font("text", 10, True))
    st.configure("Muted.TLabel", foreground=muted, font=font("text", 9))

    st.configure("TCheckbutton", background=bg, foreground=ink, font=text, padding=(0, px(2)),
                 indicatorbackground=surface, indicatorforeground=on, upperbordercolor=muted, lowerbordercolor=muted,
                 indicatormargin=(0, 0, px(8), 0), indicatorsize=px(14))
    st.map("TCheckbutton", background=[("", bg)], foreground=[("disabled", muted)], indicatorforeground=[("", on)],
           indicatorbackground=[("disabled", bg), ("selected", accent), ("", surface)],
           upperbordercolor=[("focus", accent), ("selected", accent)],
           lowerbordercolor=[("focus", accent), ("selected", accent)])

    # Combobox and spinbox: surface field, line border, a 2 px accent border with the keyboard focus.
    for name in ("TCombobox", "TSpinbox"):
        st.configure(name, fieldbackground=surface, background=surface, foreground=ink, bordercolor=line,
                     lightcolor=surface, darkcolor=surface, arrowcolor=ink, insertcolor=ink, insertwidth=px(1),
                     padding=(px(8), px(4)))
        st.map(name, fieldbackground=[("disabled", bg), ("", surface)], foreground=[("disabled", muted), ("", ink)],
               background=[("disabled", bg), ("pressed", line), ("", surface)], arrowcolor=[("disabled", muted)],
               bordercolor=focus, lightcolor=focus, darkcolor=focus)
    st.configure("TCombobox", selectbackground=surface, selectforeground=ink)  # read-only: focus shows on the border
    st.map("TCombobox", selectbackground=[("", surface)], selectforeground=[("", ink)])
    st.configure("TSpinbox", padding=(px(8), px(4), px(4), px(4)))
    if tk8:
        st.configure("TCombobox", arrowsize=px(14))
        st.configure("TSpinbox", arrowsize=px(10))
    st.configure("ComboboxPopdownFrame", background=surface, bordercolor=line, lightcolor=line, darkcolor=line,
                 relief="solid", borderwidth=1)
    lst = (("background", surface), ("foreground", ink), ("selectBackground", accent), ("selectForeground", on),
           ("font", text), ("borderWidth", 0), ("highlightThickness", 0), ("activeStyle", "none"))
    for k, v in lst:
        root.option_add(f"*TCombobox*Listbox.{k}", v)
    _recolor_lists(root, [x for k, v in lst for x in (f"-{k.lower()}", v)])

    # Thin progress bar: accent on a line-colored track (arrowsize is the bar's thickness; Tk 9 scales it).
    st.configure("Horizontal.TProgressbar", background=accent, lightcolor=accent, darkcolor=accent, bordercolor=line,
                 troughcolor=line, arrowsize=px(6) if tk8 else 6)

    # Quiet scrollbar for surface panels: no arrows, a line-colored thumb that darkens on hover. The Page
    # variant has its track on the window background (the Settings tab).
    st.layout("Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
        ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})])
    st.configure("Vertical.TScrollbar", troughcolor=surface, bordercolor=surface, background=line, lightcolor=line,
                 darkcolor=line, gripcount=0, gripsize=0, arrowpadding=0,
                 arrowsize=px(10) if tk8 else 4)  # 10 px wide (Tk 9: 2 * 4 + 1, scaled by Tk)
    thumb = [("pressed", muted), ("active", muted), ("", line)]
    st.map("Vertical.TScrollbar", background=thumb, lightcolor=thumb, darkcolor=thumb)
    st.configure("Page.Vertical.TScrollbar", troughcolor=bg, bordercolor=bg)

    # Flat notebook: text tabs, the selected one in ink over a 2 px accent underline, the others muted; a 1 px
    # line runs under the whole tab row (do not add another divider there). No tab or client border box.
    _flat_notebook_images(root, st, tok)
    st.configure("Flat.TNotebook", background=bg, borderwidth=0, padding=0, tabmargins=0, tabposition="nw")
    st.layout("Flat.TNotebook", [("Flat.Notebook.client", {"sticky": "nswe"})])
    tab_pad = (0, px(4), 0, px(6))
    st.configure("Flat.TNotebook.Tab", background=bg, foreground=muted, font=text, padding=tab_pad)
    st.map("Flat.TNotebook.Tab", foreground=[("selected", ink), ("active", ink), ("", muted)],
           background=[("", bg)], padding=[("", tab_pad)], expand=[("", 0)])
    st.layout("Flat.TNotebook.Tab", [("Flat.Notebook.tab", {"sticky": "nswe", "children": [
        ("Notebook.padding", {"side": "top", "sticky": "nswe", "children": [
            ("Notebook.focus", {"side": "top", "sticky": "nswe", "children": [
                ("Notebook.label", {"side": "top", "sticky": ""})]})]})]})])


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
    if not valid_shortcut(out["shortcut"]):
        out["shortcut"] = DEFAULTS["shortcut"]
    if out["mode"] not in MODES:
        out["mode"] = DEFAULTS["mode"]
    if out["output"] not in OUTPUTS:
        out["output"] = DEFAULTS["output"]
    out["unload_after_s"] = unload_seconds(out["unload_after_s"])
    return out


def valid_shortcut(combo):
    """A known key with Ctrl, Alt, Shift or Win, or F1 to F24 alone: any other key
    alone is taken from every app (a plain F could no longer be typed)."""
    try:
        mods, _ = win32.parse(combo)
    except (KeyError, AttributeError):
        return False
    return bool(mods) or combo.lower() in {f"f{i}" for i in range(1, 25)}


def unload_seconds(v):
    """The idle time before the model is unloaded: 0 (never), else 5 s to one day."""
    return 0 if v <= 0 else max(5, min(86400, v))


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


NOT_CONNECTED = " (not connected, using the default)"  # after a saved microphone that is missing


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


def refresh_devices():
    """Makes PortAudio read the devices again (it keeps the list from its start, so
    a microphone connected later is missing). Only with no stream open or closing."""
    try:
        import sounddevice as sd

        t = time.perf_counter()
        if sd._initialized:  # not after a failed _initialize()
            sd._terminate()
        sd._initialize()
        dt = time.perf_counter() - t
        if dt > 0.5:  # it delays the start of each recording
            log(f"reading the audio devices again took {dt:.2f} s")
    except Exception as e:
        log(f"could not read the audio devices again: {e}")


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
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:  # PowerShell's error, not the command line
        err = r.stderr.decode("oem", "replace").strip()
        log(f"shortcut {path} failed (exit code {r.returncode}):\n{err}")
        raise OSError(err.splitlines()[0] if err else f"PowerShell exit code {r.returncode}")


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


def autostart_points_here():
    """True if the Run value for the app exists and starts this copy of the app."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            value = winreg.QueryValueEx(k, APP)[0]
    except OSError:
        return False
    target, _ = launch_command()
    return target.lower() in value.lower()


def autostart_enabled():
    """True if Windows will start the app at sign-in: the Run value points at this
    copy of the app and Task Manager's Startup tab has not disabled it."""
    import winreg

    if not autostart_points_here():
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, APPROVED_KEY) as k:
            flag = winreg.QueryValueEx(k, APP)[0]
        return not (isinstance(flag, bytes) and flag[:1] in (b"\x01", b"\x03"))  # odd = disabled
    except OSError:
        return True


def set_autostart(on, clear_disabled=True):
    """Write (on) or delete the Run value. Turning it on also clears a "disabled" mark
    from Task Manager or Settings, unless clear_disabled=False (only the path is repaired)."""
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
    if on and clear_disabled:
        try:  # the "disabled" mark would win over the Run value
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
        self.close_gave_up = False  # close() timed out with the process still alive
        self.npu_since = None  # awake time its NPU work in progress began, by its own messages
        self.t0 = win32.awake_time()
        ctx = mp.get_context("spawn")
        self.inq, self.outq = ctx.Queue(), ctx.Queue()
        self.stop_event = ctx.Event()  # set: the engine starts no further piece
        self.npu_started = ctx.Event()  # set by the engine right before its first NPU submission
        # Audio put into the queue of an engine that died stays unread; without this
        # the app froze at exit waiting for it (seen 2026-10).
        self.inq.cancel_join_thread()
        self.proc = ctx.Process(target=engine.serve, daemon=True, name="npu-engine",
                                args=(self.inq, self.outq, str(APPDIR / "engine.log"),
                                      str(model_store.bundle_dir()), self.stop_event, self.npu_started))
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
                if msg[0] in ("npu", "busy"):  # read here also after the app closed it
                    self.npu_since = win32.awake_time()
                elif msg[0] in ("ready", "text", "piece_failed"):
                    self.npu_since = None
                post(("engine", self.gen, msg))
            elif not self.proc.is_alive():
                log(f"engine {self.gen}: pid {self.proc.pid} exited with code {self.proc.exitcode}"
                    f"{' (closed by the app)' if self.closing else ''}")
                if not self.closing:
                    post(("engine", self.gen, ("died", self.proc.exitcode, self.npu_started.is_set())))
                return

    def alive(self):
        return self.proc.is_alive()

    def npu_overdue(self, now):
        """Alive and in NPU work for longer than WATCHDOG_S (an idle engine is never)."""
        since = self.npu_since  # read once: the reader thread may set it to None meanwhile
        return since is not None and now - since > WATCHDOG_S and self.npu_started.is_set() and self.alive()

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
            self.close_gave_up = True
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

    def idle(self, wait_s=0.0):
        """No stream open or closing (one that is closing is waited for up to wait_s)."""
        c = self.closer
        if c is not None:
            c.join(wait_s)
        return self.stream is None and (c is None or not c.is_alive())

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
        d = TOKENS["dark"]  # the indicator is dark in both themes
        self.fill, self.ink, self.border, self.rec = d["surface"], d["ink"], d["line"], d["recording"]
        self.bar = blend(d["recording"], d["surface"], 0.7)
        self.dots = {"busy": d["waiting"], "done": d["ready"], "info": d["idle"], "error": d["error"]}
        # The timer is in the display font after "Listening": both sit on one baseline
        # (font metrics are in pixels, the drawing in 96-DPI units).
        self.f_text, self.f_time = font("text", 10), font("display", 11)
        ft, fm = tkfont.Font(root=self.win, font=self.f_text), tkfont.Font(root=self.win, font=self.f_time)
        a, dt = ft.metrics("ascent") / S, ft.metrics("descent") / S
        self.time_x = 38 + ft.measure("Listening") / S + 8
        self.time_y = self.BH / 2 + (a - dt) / 2 + fm.metrics("descent") / S  # bottom of the timer's line
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
                if self.app.recording:  # e.g. a message about an earlier dictation: back to this one
                    self.show("recording")
                else:
                    self.hide()
            else:
                if self.state == "recording":
                    self.levels = self.levels[1:] + [self.app.recorder.level]
                self._draw()
        self.app.root.after(50, self._tick)

    def _pill(self, fill, border=None):
        """The pill shape, with a 1-unit border when given (a filled pill inside a larger one)."""
        c, w, h = self.c, self.BW, self.BH
        for i, color in ((0, border), (1, fill)) if border else ((0, fill),):
            x0, y0, x1, y1 = i, i, w - 1 - i, h - 1 - i
            r = (y1 - y0) / 2
            c.create_oval(x0, y0, x0 + 2 * r, y1, fill=color, outline=color)
            c.create_oval(x1 - 2 * r, y0, x1, y1, fill=color, outline=color)
            c.create_rectangle(x0 + r, y0, x1 - r, y1, fill=color, outline=color)

    def _draw(self):
        c, h = self.c, self.BH
        c.delete("all")
        self._pill(self.fill, self.border)
        t = time.perf_counter() - self.t0
        if self.state == "recording":
            rad = 5 + 1.5 * abs(np.sin(3 * t))
            c.create_oval(22 - rad, h / 2 - rad, 22 + rad, h / 2 + rad, fill=self.rec, outline="")
            c.create_text(38, h / 2, text="Listening", fill=self.ink, font=self.f_text, anchor="w")
            secs = int(self.app.recorder.seconds)
            c.create_text(self.time_x, self.time_y, text=f"{secs // 60}:{secs % 60:02d}", fill=self.ink,
                          font=self.f_time, anchor="sw")
            for i, lv in enumerate(self.levels):
                bh = 3 + min(22, 22 * (lv / 0.06) ** 0.6)
                x = 150 + i * 6
                c.create_rectangle(x, h / 2 - bh / 2, x + 3, h / 2 + bh / 2, fill=self.bar, outline="")
        else:
            color = self.dots[self.state]
            if self.state == "busy":
                for i in range(3):
                    a = 0.5 + 0.5 * np.sin(6 * t - i)
                    c.create_oval(14 + i * 8, h / 2 - 3 * a, 20 + i * 8, h / 2 + 3 * a + 1, fill=color, outline="")
                label = self.text or "Transcribing on the NPU…"
            else:
                c.create_oval(17, h / 2 - 5, 27, h / 2 + 5, fill=color, outline="")
                label = self.text
            c.create_text(44, h / 2, text=label, fill=self.ink, font=self.f_text, anchor="w",
                          width=px(self.BW - 60))
        c.scale("all", 0, 0, S, S)


# -- UI components ---------------------------------------------------------------
# Each one adds itself to app.themed and recolors in apply_theme(tok).
def text_width(widget, fnt, text):
    """Width of text in a Tk font, in pixels."""
    return int(widget.tk.call("font", "measure", fnt, text))


def line_height(widget, fnt):
    return int(widget.tk.call("font", "metrics", fnt, "-linespace"))


def round_rect(c, x0, y0, x1, y1, r, **kw):
    """A rectangle with rounded corners of radius r on Canvas c (a smoothed polygon)."""
    r = max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = (x0 + r, y0, x0 + r, y0, x1 - r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y0 + r, x1, y1 - r,
           x1, y1 - r, x1, y1, x1 - r, y1, x1 - r, y1, x0 + r, y1, x0 + r, y1, x0, y1, x0, y1 - r,
           x0, y1 - r, x0, y0 + r, x0, y0 + r, x0, y0)
    return c.create_polygon(pts, smooth=True, **kw)


class LinkButton(tk.Label):
    """A text link: accent text, underlined under the mouse. A click, Return or
    Space runs command; keyboard focus shows an accent outline."""

    def __init__(self, parent, app, text, command, bg="bg"):
        self.app, self.command, self.bg_token = app, command, bg
        self.fonts = (font("text", 10), tuple(font("text", 10)) + ("underline",))
        super().__init__(parent, text=text, font=self.fonts[0], cursor="hand2", takefocus=1, bd=0,
                         padx=px(2), pady=px(1), highlightthickness=px(1), anchor="w", justify="left")
        self.bind("<Enter>", lambda e: self.configure(font=self.fonts[1]))
        self.bind("<Leave>", lambda e: self.configure(font=self.fonts[0]))
        self.bind("<Button-1>", lambda e: self.command())
        for seq in ("<Return>", "<KP_Enter>", "<space>"):
            self.bind(seq, self._key)
        app.themed.append(self)
        self.apply_theme(app.tok)

    def _key(self, event):
        # While a new shortcut is captured its keys are not a click. No "break":
        # the capture binding on the window must still see the key.
        if not self.app.capturing:
            self.command()

    def apply_theme(self, tok):
        bg = tok[self.bg_token]
        self.configure(fg=tok["accent"], bg=bg, activeforeground=tok["accent"], activebackground=bg,
                       highlightcolor=tok["accent"], highlightbackground=bg)


class StatusPill(tk.Canvas):
    """The state as a pill: a dot in the state color and the state's name."""

    def __init__(self, parent, app):
        super().__init__(parent, width=1, height=1, bd=0, highlightthickness=0, takefocus=0, bg=app.tok["bg"])
        self.app, self.state, self.shown = app, None, (1, 1)
        app.themed.append(self)

    def set(self, state):
        if state != self.state:
            self.state = state
            self._draw()

    def apply_theme(self, tok):
        self.configure(bg=tok["bg"])
        self._draw()

    def _draw(self):
        self.delete("all")
        if self.state is None:
            return
        tok = self.app.tok
        color, text, f = tok[STATE_TOKEN[self.state]], STATES[self.state], font("text", 10, semibold=True)
        d, pad = px(8), px(10)
        h = max(px(26), line_height(self, f) + px(8))
        w = pad + d + px(6) + text_width(self, f, text) + px(12)
        if self.shown != (w, h):
            self.shown = (w, h)
            self.configure(width=w, height=h)
        round_rect(self, 0, 0, w - 1, h - 1, (h - 1) / 2, fill=blend(color, tok["surface"], 0.14),
                   outline=blend(color, tok["surface"], 0.4))
        y = (h - d) // 2
        self.create_oval(pad, y, pad + d - 1, y + d - 1, fill=color, outline=color)
        self.create_text(pad + d + px(6), h / 2, text=text, anchor="w", fill=tok["ink"], font=f)


# Ways to fit the keys, tried in order until they fit: label size (pt), padding,
# minimum key width, space each side of "+", narrowest Space (0: its text).
KEY_FITS = ((14, 28, 64, 8, 168), (14, 28, 64, 6, 96), (14, 28, 64, 6, 0), (14, 28, 0, 6, 0),
            (14, 16, 0, 4, 0), (12, 14, 0, 4, 0), (11, 12, 0, 3, 0), (10, 10, 0, 2, 0))


class KeycapHero(ttk.Frame):
    """The shortcut drawn as keycaps (pressed while recording), with a "Change
    shortcut" link to their right, or below them when both do not fit."""

    H, EDGE, R, SPACE, GAP = 44, 3, 6, 168, 20  # 96-DPI units
    CAPTURE = "Press the new shortcut (Esc cancels)"

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.canvas = tk.Canvas(self, width=1, height=px(self.H + self.EDGE), bd=0, highlightthickness=0,
                                takefocus=0, bg=app.tok["bg"])
        self.canvas.grid(row=0, column=0, sticky="w")
        self.link = LinkButton(self, app, "Change shortcut", app._capture_shortcut)
        self.columnconfigure(1, weight=1)
        self.parts, self.mode, self.base = None, None, "normal"  # base: the look when not pressed
        self.p = 0.0  # how far the keys are drawn pressed (0 up, 1 down)
        self.step = None  # the pending animation frame
        self.width, self.shown, self.link_right = 0, None, None
        self.bind("<Configure>", self._resized)
        app.themed.append(self)

    def set(self, parts, mode):
        """parts: the key labels; mode: normal, pressed, muted, error or capturing."""
        parts = list(parts)
        if parts == self.parts and mode == self.mode:
            return
        if self.step is not None:
            self.after_cancel(self.step)
            self.step = None
        animate = parts == self.parts and "capturing" not in (mode, self.mode)
        self.parts, self.mode = parts, mode
        if mode not in ("pressed", "capturing"):
            self.base = mode
        target = 1.0 if mode == "pressed" else 0.0
        if animate and self.p != target:  # one frame half way, the last one 90 ms later
            self._draw((self.p + target) / 2)
            self.step = self.after(90, self._settle)
        else:
            self._draw(target)

    def _settle(self):
        self.step = None
        self._draw(1.0 if self.mode == "pressed" else 0.0)

    def apply_theme(self, tok):
        self.canvas.configure(bg=tok["bg"])
        if self.step is not None:
            self.after_cancel(self.step)
            self.step = None
        if self.parts is not None:
            self._draw(1.0 if self.mode == "pressed" else 0.0)

    def _resized(self, event):
        if event.width != self.width:
            self.width = event.width
            if self.parts is not None:
                self._draw(self.p)

    def _fit(self, avail, fits):
        """(key widths, "+" width, key font, "+" font) of the first way in fits that
        is at most avail wide, else of the last one; and whether it fits."""
        for size, pad, low, gap, space in fits:
            f, fp = font("display", size, semibold=True), font("display", size)
            widths = [max(px(low), text_width(self, f, k) + px(pad)) for k in self.parts]
            sep = text_width(self, fp, "+") + 2 * px(gap)
            if "Space" in self.parts:  # Space is wide, narrower when the room runs out
                i = self.parts.index("Space")
                room = avail - sum(widths) - sep * (len(widths) - 1) + widths[i]
                widths[i] = max(widths[i], px(space), min(px(self.SPACE), room))
            if sum(widths) + sep * (len(widths) - 1) <= avail:
                return (widths, sep, f, fp), True
        return (widths, sep, f, fp), False

    def _draw(self, p):
        self.p = p
        c, tok = self.canvas, self.app.tok
        c.delete("all")
        avail = self.width if self.width > 1 else px(480)  # before the first layout: the default window
        beside = avail - px(self.GAP) - self.link.winfo_reqwidth()
        H, E, r = px(self.H), px(self.EDGE), px(self.R)
        if self.mode == "capturing":
            f = font("text", 10)
            w = text_width(self, f, self.CAPTURE) + px(28)
            right = w <= beside
            w = min(w, avail)  # wraps to two lines if it must
            round_rect(c, 0, 0, w - 1, H - 1, r, fill=tok["bg"], outline=tok["accent"], dash=(px(4), px(3)))
            c.create_text(w / 2, H / 2, text=self.CAPTURE, fill=tok["ink"], font=f, width=w - px(16),
                          justify="center")
        else:
            (widths, sep, f, fp), right = self._fit(beside, KEY_FITS[:2])
            if not right:
                (widths, sep, f, fp), _ = self._fit(avail, KEY_FITS)
            off = round(min(px(2), E - px(1)) * p)  # pressed: the face moves down, the edge shrinks
            border = tok["error"] if self.base == "error" else tok["line"]
            ink = tok["muted"] if self.base == "muted" else tok["ink"]
            face = blend(tok["recording"], tok["surface"], 0.18 * p)
            border, ink = blend(tok["recording"], border, p), blend(tok["ink"], ink, p)
            w = 0
            for i, (k, kw) in enumerate(zip(self.parts, widths)):
                if i:
                    c.create_text(w + sep / 2, H / 2, text="+", fill=tok["muted"], font=fp)
                    w += sep
                round_rect(c, w, E, w + kw - 1, H + E - 1, r, fill=tok["line"], outline=tok["line"])
                round_rect(c, w, off, w + kw - 1, off + H - 1, r, fill=face, outline=border)
                c.create_text(w + kw / 2, off + H / 2, text=k, fill=ink, font=f)
                w += kw
        if self.shown != (w, H + E):
            self.shown = (w, H + E)
            c.configure(width=w, height=H + E)
        if right != self.link_right:
            self.link_right = right
            if right:
                self.link.grid(row=0, column=1, columnspan=1, sticky="w", padx=(px(self.GAP), 0), pady=(0, E))
            else:
                self.link.grid(row=1, column=0, columnspan=2, sticky="w", padx=0, pady=(px(8), 0))


class Segmented(tk.Frame):
    """Joined buttons that choose one value of a tk variable. The selected one is
    filled with the accent color; Left and Right move the choice; the border turns
    accent while one of the buttons has keyboard focus."""

    def __init__(self, parent, app, variable, options, command):
        super().__init__(parent, bd=0, padx=px(1), pady=px(1), highlightthickness=0)
        self.app, self.var, self.hover, self.focused = app, variable, None, None
        self.buttons = []
        for i, (value, label) in enumerate(options):
            b = tk.Radiobutton(self, text=label, value=value, variable=variable, command=command,
                               indicatoron=False, bd=0, relief="flat", offrelief="flat", overrelief="flat",
                               padx=px(10), pady=px(4), highlightthickness=px(1), takefocus=1,
                               font=font("text", 10))
            b.pack(side="left", fill="y", padx=(px(1) if i else 0, 0))
            b.bind("<Enter>", lambda e, i=i: self._hover(i), add="+")
            b.bind("<Leave>", lambda e: self._hover(None), add="+")
            b.bind("<FocusIn>", lambda e, i=i: self._focus(i), add="+")
            b.bind("<FocusOut>", lambda e: self._focus(None), add="+")
            b.bind("<Left>", lambda e, i=i: self._step(i - 1))
            b.bind("<Right>", lambda e, i=i: self._step(i + 1))
            self.buttons.append(b)
        variable.trace_add("write", lambda *a: self._paint())
        app.themed.append(self)
        self.apply_theme(app.tok)

    def _hover(self, i):
        self.hover = i
        self._paint()

    def _focus(self, i):
        self.focused = i
        self._paint()

    def _step(self, i):
        if self.app.capturing:  # the arrow may be part of a new shortcut: no "break"
            return None
        if 0 <= i < len(self.buttons):
            self.buttons[i].focus_set()
            self.buttons[i].invoke()
        return "break"

    def apply_theme(self, tok):
        self.tok = tok
        self._paint()

    def _paint(self):
        # Every color is set here (not left to Tk's active and select colors): Tk on
        # Windows shows no hover on radio buttons and swaps selectcolor on a click.
        tok = self.tok
        self.configure(bg=tok["accent"] if self.focused is not None else tok["line"])
        for i, b in enumerate(self.buttons):
            on = self.var.get() == str(b.cget("value"))
            fill = tok["accent"] if on else tok["line"] if i == self.hover else tok["surface"]
            fg = on_accent(tok) if on else tok["ink"]
            b.configure(bg=fill, activebackground=fill, selectcolor=fill, fg=fg, activeforeground=fg,
                        highlightbackground=fill, highlightcolor=tok["surface"] if on else tok["accent"])


class ScrollFrame(ttk.Frame):
    """A vertically scrolling area on the window background: put widgets in .inner.
    The scrollbar shows only when the content is taller than the area; the mouse
    wheel scrolls it over any widget in it, and a widget that gets keyboard focus
    is scrolled into view. on_width(width) is called when the width changes."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.scrolls, self.wheel, self.on_width = False, 0.0, None
        self.canvas = tk.Canvas(self, width=px(380), height=px(160), bd=0, highlightthickness=0,
                                yscrollincrement=px(8), takefocus=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", style="Page.Vertical.TScrollbar",
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.inner = ttk.Frame(self.canvas)
        self.item = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", self._resized)
        self.inner.bind("<Configure>", lambda e: self._update())
        self.tag = f"ScrollFrame{id(self)}"
        app.themed.append(self)
        self.apply_theme(app.tok)

    def apply_theme(self, tok):
        self.canvas.configure(bg=tok["bg"])

    def bind_wheel(self):
        """Call once the area is filled: binds the wheel and focus on every widget in it."""
        todo = [self]
        while todo:
            w = todo.pop()
            todo.extend(w.winfo_children())
            if self.tag not in w.bindtags():
                w.bindtags((self.tag,) + w.bindtags())
        seqs = ["<MouseWheel>"]
        if self.tk.call("tk", "windowingsystem") == "x11":  # on Windows Button-4/5 are the side buttons
            seqs += ["<Button-4>", "<Button-5>"]
        for seq in seqs:
            self.bind_class(self.tag, seq, self._wheel)
        self.bind_class(self.tag, "<FocusIn>", self._focused)

    def _resized(self, e):
        self.canvas.itemconfigure(self.item, width=e.width)
        if self.on_width:
            self.on_width(e.width)
        self._update()

    def _update(self):
        h, view = self.inner.winfo_reqheight(), self.canvas.winfo_height()
        if (h > view) != self.scrolls:
            self.scrolls = h > view
            if self.scrolls:
                self.bar.grid(row=0, column=1, sticky="ns", padx=(px(6), 0))
            else:
                self.bar.grid_remove()
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), max(h, view)))
        if not self.scrolls:
            self.canvas.yview_moveto(0)

    def _wheel(self, e):
        if not self.scrolls:
            return None  # a combobox or spin box under the mouse handles the wheel as before
        self.wheel += 6 * (-1 if e.num == 4 else 1 if e.num == 5 else -e.delta / 120)  # 48 px a notch
        n = int(self.wheel)
        if n:
            self.wheel -= n
            self.canvas.yview_scroll(n, "units")
        return "break"  # the page scrolls, no value under the mouse changes

    def _focused(self, e):
        if not self.scrolls or str(e.widget) != str(self.tk.call("focus")):
            return
        try:
            y = e.widget.winfo_rooty() - self.inner.winfo_rooty()
            h = e.widget.winfo_height()
        except (AttributeError, tk.TclError):
            return
        top, view, total = self.canvas.canvasy(0), self.canvas.winfo_height(), self.inner.winfo_reqheight()
        if y < top:
            self.canvas.yview_moveto(max(0, y - px(8)) / total)
        elif y + h > top + view:
            self.canvas.yview_moveto((y + h + px(8) - view) / total)


class HistoryList(tk.Frame):
    """The History tab: dictations newest first, the time above the text, 1 px
    dividers between them, all drawn on one Canvas (so the text wraps and the
    flash fills the whole entry). A click, or Return on the keyboard row, calls
    on_click(index) with 0 = the newest entry."""

    PADX, PADY, GAP = 12, 10, 2  # 96-DPI units
    WHEEL, ARROW = 48, 20  # pixels per wheel notch and per scrollbar arrow click

    def __init__(self, parent, app, on_click):
        super().__init__(parent, bd=0, highlightthickness=px(1))
        self.app, self.on_click = app, on_click
        self.items, self.key, self.rows = [], None, []
        self.width = self.content = self.total = 0
        self.cursor, self.focused = 0, False  # keyboard row
        self.flashing, self.flash_job = None, None
        self.sb_shown = False
        self.c = c = tk.Canvas(self, bd=0, highlightthickness=0, takefocus=0, yscrollincrement=1,
                               yscrollcommand=self._scrolled)
        self.sb = ttk.Scrollbar(self, orient="vertical", style="Vertical.TScrollbar", command=self._yview)
        c.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        wheel = ["<MouseWheel>"] + (["<Button-4>", "<Button-5>"] if c.tk.call("tk", "windowingsystem") == "x11" else [])
        for w in (self, c, self.sb):
            for seq in wheel:
                w.bind(seq, self._wheel)
        c.bind("<Configure>", self._configured)
        c.bind("<Button-1>", self._clicked)
        c.bind("<Motion>", self._hover)
        c.bind("<Leave>", lambda e: c.configure(cursor=""))
        c.bind("<FocusIn>", lambda e: self._focus(True))
        c.bind("<FocusOut>", lambda e: self._focus(False))
        for seq, fn in (("<Up>", lambda: self._move(self.cursor - 1)), ("<Down>", lambda: self._move(self.cursor + 1)),
                        ("<Home>", lambda: self._move(0)), ("<End>", lambda: self._move(len(self.items) - 1)),
                        ("<Prior>", lambda: c.yview_scroll(-1, "pages")),
                        ("<Next>", lambda: c.yview_scroll(1, "pages")),
                        ("<Return>", self._activate), ("<KP_Enter>", self._activate), ("<space>", self._activate)):
            c.bind(seq, lambda e, fn=fn: self._key(fn))
        app.themed.append(self)
        self.apply_theme(app.tok)

    def set_items(self, items, empty_text):
        """items: [{"time", "text"}, ...] newest first. Does nothing when neither the
        items nor the empty text changed, so it can be called from every _refresh."""
        key = ([(d["time"], d["text"]) for d in items], empty_text)
        if key == self.key:
            return
        self.key = key
        self.items = [{"time": t, "text": x} for t, x in key[0]]
        self._end_flash()
        tok, c = self.app.tok, self.c
        c.delete("all")
        if not self.items:
            c.create_text(0, 0, text=empty_text, anchor="nw", justify="left", fill=tok["muted"],
                          font=font("text", 10), tags=("empty",))
        for i, d in enumerate(self.items):
            c.create_rectangle(0, 0, 0, 0, fill=tok["surface"], outline="", tags=("bg", f"bg{i}"))
            c.create_text(0, 0, text=d["time"], anchor="nw", fill=tok["muted"], font=font("text", 9),
                          tags=("time", f"t{i}"))
            c.create_text(0, 0, text=d["text"], anchor="nw", justify="left", fill=tok["ink"], font=font("text", 10),
                          tags=("body", f"x{i}"))
            if i < len(self.items) - 1:
                c.create_rectangle(0, 0, 0, 0, fill=tok["line"], outline="", tags=("div", f"d{i}"))
        c.create_rectangle(0, 0, 0, 0, outline=tok["accent"], width=px(1), state="hidden", tags=("focus",))
        c.configure(takefocus=1 if self.items else 0)
        self.cursor = 0
        self._layout()
        c.yview_moveto(0)

    def flash(self, index):
        """The entry fills with accent at 18% for 400 ms (it was just copied)."""
        if not 0 <= index < len(self.items):
            return
        self._end_flash()
        self.flashing = index
        self.c.itemconfigure(f"bg{index}", fill=blend(self.app.tok["accent"], self.app.tok["surface"], 0.18))
        self.flash_job = self.after(400, self._end_flash)

    def apply_theme(self, tok):
        self.configure(bg=tok["surface"], highlightcolor=tok["accent"],
                       highlightbackground=tok["accent"] if self.focused else tok["line"])
        c = self.c
        c.configure(bg=tok["surface"])
        for tag, opt, key in (("bg", "fill", "surface"), ("time", "fill", "muted"), ("body", "fill", "ink"),
                              ("div", "fill", "line"), ("empty", "fill", "muted"), ("focus", "outline", "accent")):
            c.itemconfigure(tag, **{opt: tok[key]})
        if self.flashing is not None:
            c.itemconfigure(f"bg{self.flashing}", fill=blend(tok["accent"], tok["surface"], 0.18))

    def _height(self, tag):
        b = self.c.bbox(tag)
        return b[3] - b[1] if b else 0

    def _layout(self):
        """Places every entry for the current width (the text wraps to it)."""
        c = self.c
        w = self.width = max(c.winfo_width(), px(120))
        padx, pady, gap = px(self.PADX), px(self.PADY), px(self.GAP)
        self.rows, y = [], 0
        if not self.items:
            c.itemconfigure("empty", width=w - 2 * padx)
            c.coords("empty", padx, pady + gap)
            y = 2 * (pady + gap) + self._height("empty")
        for i in range(len(self.items)):
            top = y
            c.coords(f"t{i}", padx, top + pady)
            y = top + pady + self._height(f"t{i}") + gap
            c.itemconfigure(f"x{i}", width=w - 2 * padx)
            c.coords(f"x{i}", padx, y)
            y += self._height(f"x{i}") + pady
            c.coords(f"bg{i}", 0, top, w, y)
            self.rows.append((top, y))
            if i < len(self.items) - 1:
                c.coords(f"d{i}", 0, y, w, y + px(1))
                y += px(1)
        self.content = y
        self._region()
        self._draw_focus()

    def _region(self):
        self.total = max(self.content, self.c.winfo_height(), 1)
        self.c.configure(scrollregion=(0, 0, self.width, self.total))

    def _configured(self, event):
        if event.width != self.width:
            self._layout()
        else:
            self._region()

    def _scrolled(self, first, last):
        """The scrollbar shows only while the list is longer than the view."""
        self.sb.set(first, last)
        need = float(first) > 0.0001 or float(last) < 0.9999
        if need != self.sb_shown:
            self.sb_shown = need
            if need:
                self.sb.grid(row=0, column=1, sticky="ns")
            else:
                self.sb.grid_remove()

    def _yview(self, *args):
        if len(args) == 3 and args[2] == "units":  # an arrow: the canvas scrolls in pixels
            self.c.yview_scroll(int(args[1]) * px(self.ARROW), "units")
        else:
            self.c.yview(*args)

    def _wheel(self, event):
        delta = {4: 120, 5: -120}.get(event.num, event.delta)  # Windows: 120 per notch
        self.c.yview_scroll(int(round(-delta / 120 * px(self.WHEEL))), "units")
        return "break"

    def _index_at(self, y):
        y = self.c.canvasy(y)
        for i, (top, bottom) in enumerate(self.rows):
            if top <= y < bottom + px(1):
                return i
        return None

    def _clicked(self, event):
        i = self._index_at(event.y)
        if i is not None:
            self.cursor = i
            self._draw_focus()
            self.on_click(i)

    def _hover(self, event):
        cursor = "hand2" if self._index_at(event.y) is not None else ""
        if self.c.cget("cursor") != cursor:
            self.c.configure(cursor=cursor)

    def _focus(self, on):
        self.focused = on
        self.configure(highlightbackground=self.app.tok["accent" if on else "line"])
        self._draw_focus()

    def _draw_focus(self):
        """A 1 px accent outline on the keyboard row while the list has focus."""
        c = self.c
        if self.focused and self.cursor < len(self.rows):
            top, bottom = self.rows[self.cursor]
            o = px(1)
            c.coords("focus", o, top + o, self.width - o, bottom - o)
            c.itemconfigure("focus", state="normal")
            c.tag_raise("focus")
        else:
            c.itemconfigure("focus", state="hidden")

    def _key(self, fn):
        if self.app.capturing:
            return None  # the shortcut capture on the window gets the key
        fn()
        return "break"

    def _move(self, i):
        if not self.rows:
            return
        self.cursor = max(0, min(len(self.rows) - 1, i))
        self._draw_focus()
        top, bottom = self.rows[self.cursor]
        view, h = self.c.canvasy(0), self.c.winfo_height()
        if top < view:
            self.c.yview_moveto(top / self.total)
        elif bottom > view + h:
            self.c.yview_moveto((bottom - h) / self.total)

    def _activate(self):
        if self.cursor < len(self.items):
            self.on_click(self.cursor)

    def _end_flash(self):
        if self.flash_job is not None:
            self.after_cancel(self.flash_job)
            self.flash_job = None
        if self.flashing is not None:
            self.c.itemconfigure(f"bg{self.flashing}", fill=self.app.tok["surface"])
            self.flashing = None


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
        self.other_engine = None  # (first seen, last checked): an engine of an earlier run holds engine.MUTEX
        self.want_engine = False
        self.retry_at, self.retries, self.died_retries = None, 0, 0
        self.busy = None  # (engine gen, start time): NPU work in progress, for the watchdog
        self.last_active = time.monotonic()  # last recording, engine answer or insert: for unload_after_s
        self.downloading = None  # (done, total) bytes while the model downloads
        self.download_fails = 0  # model downloads in a row that failed with a network error
        self.model_ok = False  # the model files loaded fine in this session
        self.quitting = False
        self.send_lock = threading.Lock()
        self.backlog = []  # pieces recorded while no engine process exists yet
        self.dropped_upto = 0  # pieces of sessions up to this one are dropped (their dictation failed)
        self.recording = False
        self.rec_mode = self.settings["mode"]  # mode of the current recording
        self.session = 0
        self.sessions = {}
        self.capturing = False
        self.shortcut_ok = True  # the last registration of a shortcut worked
        self.history = []
        self.sounds = make_sounds()
        self.icon_path = APPDIR / "icon.ico"
        icon_image(COLORS["ready"], 256).save(self.icon_path, sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
        if FROZEN:
            try:
                if not self.settings["autostart_set"]:  # first start of the packaged app: on by default
                    set_autostart(True)
                    self.settings["autostart_set"] = True
                    save_settings(self.settings)
                elif autostart_registered() and not autostart_points_here():
                    # the folder was moved or a new version unzipped elsewhere; a disable in Windows stays
                    set_autostart(True, clear_disabled=False)
                    log(f"autostart now points to {sys.executable}")
            except OSError as e:
                log(f"could not set up autostart: {e}")

        self.root = tk.Tk()
        self.root.withdraw()  # shown at the end, so a start in the tray never flashes the window
        global S
        S = self.root.winfo_fpixels("1i") / 96
        self.root.title(APP)
        self.root.iconbitmap(default=str(self.icon_path))
        self.root.geometry(f"{px(520)}x{px(600)}")
        self.root.minsize(px(460), px(520))
        self.root.protocol("WM_DELETE_WINDOW", self.hide_window)
        self.root.report_callback_exception = lambda *exc: sys.excepthook(*exc)
        self._init_theme()
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
        if not (background or self.settings["start_minimized"]) or self.tray is None:
            self.root.deiconify()
        self.root.after(20, self._poll)
        self.root.after(3000, self._poll_theme)
        self.root.after(MIC_REFRESH_MS, self._mic_tick)
        if os.environ.get("NPU_DICTATE_TEST_QUIT_AFTER"):  # test hook
            self.root.after(int(float(os.environ["NPU_DICTATE_TEST_QUIT_AFTER"]) * 1000), self.quit)
        for item in filter(None, os.environ.get("NPU_DICTATE_TEST_EVENTS", "").split(",")):  # test hook: "s:event"
            at, name = item.split(":")
            self.root.after(int(float(at) * 1000), lambda n=name: self.post((n,)))

    # ---- UI -----------------------------------------------------------------
    def _init_theme(self):
        """Fonts, then the colors of the current Windows theme, before any widget is built."""
        pick_fonts(self.root)
        self.themed = []  # widgets with apply_theme(tok), recolored on a theme switch
        self.dark = not win32.apps_use_light_theme()
        self.tok = TOKENS["dark" if self.dark else "light"]
        build_style(self.root, self.tok)
        self.root.configure(bg=self.tok["bg"])
        self.root.bind("<Map>", self._on_map, add="+")

    def _on_map(self, event):
        """The window frame (and so its title bar) exists once the window is shown."""
        if event.widget is self.root:
            win32.set_dark_title_bar(win32.toplevel_hwnd(self.root), self.dark)

    def _apply_theme(self, dark):
        """Switches every widget to the light or dark colors."""
        self.dark, self.tok = dark, TOKENS["dark" if dark else "light"]
        build_style(self.root, self.tok)
        self.root.configure(bg=self.tok["bg"])
        for t in self.themed:
            t.apply_theme(self.tok)
        win32.set_dark_title_bar(win32.toplevel_hwnd(self.root), dark)
        self._refresh()

    def _poll_theme(self):
        """Follows the Windows app mode (Settings > Personalization > Colors), read every 3 s."""
        if self.quitting:
            return
        try:
            dark = not win32.apps_use_light_theme()
            if dark != self.dark:
                self._apply_theme(dark)
        finally:
            if not self.quitting:
                self.root.after(3000, self._poll_theme)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=(px(20), px(16), px(20), px(14)))
        outer.pack(fill="both", expand=True)
        outer.bind("<Configure>", self._wrap)

        head = ttk.Frame(outer)
        head.pack(fill="x")
        ttk.Label(head, text=APP, style="Title.TLabel").pack(side="left")
        self.pill = StatusPill(head, self)
        self.pill.pack(side="right")
        self.hero = KeycapHero(outer, self)
        self.hero.pack(fill="x", pady=(px(14), 0))
        self.hint = ttk.Label(outer, text="", style="Muted.TLabel", wraplength=px(480), justify="left")
        self.hint.pack(fill="x", pady=(px(10), 0))
        self.progress = ttk.Progressbar(outer, maximum=1000)  # shown while the model downloads
        self.progress_shown = False

        # The footer is packed before the tabs, so a short window shrinks the tabs, never the footer.
        self.footer = ttk.Label(outer, text="", style="Muted.TLabel", wraplength=px(480), justify="left")
        self.footer.pack(side="bottom", fill="x", pady=(px(8), 0))
        ttk.Frame(outer, style="Divider.TFrame", height=px(1)).pack(side="bottom", fill="x", pady=(px(8), 0))

        nb = ttk.Notebook(outer, style="Flat.TNotebook")
        nb.pack(fill="both", expand=True, pady=(px(18), 0))
        page = ttk.Frame(nb)
        nb.add(page, text="Settings")
        self._build_settings(page)
        page = ttk.Frame(nb)
        nb.add(page, text="History", padding=(0, px(12), 0, 0))
        self.history_hint = ttk.Label(page, text="Click an entry to copy it.", style="Muted.TLabel")
        self.history_list = HistoryList(page, self, self._copy_history)
        self.history_list.pack(fill="both", expand=True)

    def _wrap(self, event):
        """The hint and the footer wrap to the window width (less the side padding and the label's border)."""
        for label in (self.hint, self.footer):
            label.configure(wraplength=max(px(100), event.width - 2 * px(20) - px(4)))

    def _build_settings(self, parent):
        """The Settings tab: group labels on the left, their settings on the right. Rows
        reflow when the window is narrow, and the tab scrolls when it is short."""
        area = ScrollFrame(parent, self)
        area.pack(fill="both", expand=True)
        f = area.inner
        f.configure(padding=(0, px(12), 0, px(12)))
        f.columnconfigure(0, minsize=px(96))
        f.columnconfigure(2, weight=1)

        def label(text, style="TLabel"):
            return ttk.Label(f, text=text, style=style)

        g_rec, mode_lbl = label("Recording", "Group.TLabel"), label("Mode")
        self.mode_var = tk.StringVar(value=self.settings["mode"])
        mode = Segmented(f, self, self.mode_var, [("toggle", "Press to start"), ("hold", "Hold to talk")],
                         self._changed)
        mic_lbl = label("Microphone")
        self.mic_var = tk.StringVar()
        self.mic_box = ttk.Combobox(f, textvariable=self.mic_var, state="readonly", width=16, font=font("text", 10),
                                    postcommand=self._fill_mics)
        self.mic_box.bind("<<ComboboxSelected>>", lambda e: self._changed())
        self._fill_mics()

        g_text, out_lbl = label("Text", "Group.TLabel"), label("Insert as")
        self.out_var = tk.StringVar(value=self.settings["output"])
        out = Segmented(f, self, self.out_var, [("paste", "Paste"), ("type", "Type"), ("clipboard", "Copy only")],
                        self._changed)
        paste_note = label("Paste keeps your clipboard", "Muted.TLabel")
        self.checks = {k: tk.BooleanVar(value=self.settings[k])
                       for k in ("trailing_space", "sounds", "overlay", "start_minimized")}
        space = ttk.Checkbutton(f, text="Add a space after the text", variable=self.checks["trailing_space"],
                                command=self._changed)

        g_mem = label("Memory", "Group.TLabel")
        uf = ttk.Frame(f)
        unload_lbl = ttk.Label(uf, text="Unload the model after")
        self.unload_var = tk.StringVar(value=str(self.settings["unload_after_s"]))
        sp = ttk.Spinbox(uf, from_=0, to=86400, increment=30, width=7, textvariable=self.unload_var,
                         font=font("text", 10), command=self._unload_changed)
        sp.bind("<Return>", self._unload_changed)
        sp.bind("<FocusOut>", self._unload_changed)
        idle_lbl = ttk.Label(uf, text="s idle")
        unload_note = label("0 keeps it loaded", "Muted.TLabel")

        g_app = label("App", "Group.TLabel")
        cf = ttk.Frame(f)
        self.autostart_var = tk.BooleanVar(value=autostart_enabled())
        boxes = [ttk.Checkbutton(cf, text="Start with Windows", variable=self.autostart_var,
                                 command=self._toggle_autostart)]
        boxes += [ttk.Checkbutton(cf, text=text, variable=self.checks[k], command=self._changed)
                  for k, text in (("start_minimized", "Start minimized"), ("sounds", "Play sounds"),
                                  ("overlay", "Show recording indicator"))]
        link = LinkButton(f, self, "Add Start menu and desktop shortcuts", self._make_shortcuts)

        # Rows: ("pair", label, control, sticky), ("note", widget, under the control), ("full", widget).
        groups = [(g_rec, [("pair", mode_lbl, mode, "w"), ("pair", mic_lbl, self.mic_box, "ew")]),
                  (g_text, [("pair", out_lbl, out, "w"), ("note", paste_note, True), ("full", space)]),
                  (g_mem, [("full", uf), ("note", unload_note, False)]),
                  (g_app, [("full", cf), ("full", link)])]
        dividers = [ttk.Frame(f, style="Divider.TFrame", height=px(1)) for _ in groups[1:]]
        shown = None  # what the current layout was made for

        def show_note(*_):
            if self.out_var.get() == "paste":
                paste_note.grid()
            else:
                paste_note.grid_remove()

        def layout(width):
            """Wide: sub-labels left of their controls, the unload sentence on one line,
            check boxes in two columns. Narrow: each of these on more lines."""
            nonlocal shown
            avail = width - px(96)
            wide = (max(w.winfo_reqwidth() for w in (mode_lbl, mic_lbl, out_lbl)) + px(12)
                    + max(mode.winfo_reqwidth(), out.winfo_reqwidth()) <= avail)
            two = sum(max(b.winfo_reqwidth() for b in boxes[i::2]) for i in (0, 1)) + px(16) <= avail
            one_line = sum(w.winfo_reqwidth() for w in (unload_lbl, sp, idle_lbl)) + px(12) <= avail
            key = (wide, two, one_line) + tuple(w.winfo_reqheight() for w in (g_rec, mode_lbl, mode, self.mic_box,
                                                                               out, sp, boxes[0]))
            if key == shown:
                return
            shown = key
            if one_line:
                unload_lbl.grid(row=0, column=0, columnspan=1, sticky="w", pady=0)
                sp.grid(row=0, column=1, sticky="w", padx=px(6), pady=0)
                idle_lbl.grid(row=0, column=2, sticky="w", pady=0)
            else:  # the spin box under the words
                unload_lbl.grid(row=0, column=0, columnspan=2, sticky="w", pady=0)
                sp.grid(row=1, column=0, sticky="w", padx=(0, px(6)), pady=(px(4), 0))
                idle_lbl.grid(row=1, column=1, sticky="w", pady=(px(4), 0))
            first_line = {cf: boxes[0], uf: sp if one_line else unload_lbl}  # what a group label lines up with
            for i, b in enumerate(boxes):
                r, c = divmod(i, 2) if two else (i, 0)
                b.grid(row=r, column=c, sticky="w", padx=(0, px(16) if two and c == 0 else 0),
                       pady=(px(4) if r else 0, 0))
            r = 0
            for n, (group, rows) in enumerate(groups):
                if n:
                    dividers[n - 1].grid(row=r, column=0, columnspan=3, sticky="ew", padx=0, pady=px(12))
                    r += 1
                for i, (kind, w, *more) in enumerate(rows):
                    top, r0 = 0 if i == 0 else px(2) if kind == "note" else px(8), r
                    if kind == "pair" and wide:
                        w.grid(row=r, column=1, columnspan=1, sticky="w", padx=(0, px(12)), pady=(top, 0))
                        more[0].grid(row=r, column=2, columnspan=1, sticky=more[1], padx=0, pady=(top, 0))
                        line = max(w.winfo_reqheight(), more[0].winfo_reqheight())
                    elif kind == "pair":
                        w.grid(row=r, column=1, columnspan=2, sticky="w", padx=0, pady=(top, 0))
                        r += 1
                        more[0].grid(row=r, column=1, columnspan=2, sticky=more[1], padx=0, pady=(px(4), 0))
                        line = w.winfo_reqheight()
                    elif kind == "note" and wide and more[0]:
                        w.grid(row=r, column=2, columnspan=1, sticky="w", padx=0, pady=(top, 0))
                    else:
                        w.grid(row=r, column=1, columnspan=2, sticky="w", padx=0, pady=(top, 0))
                        line = first_line.get(w, w).winfo_reqheight()
                    if i == 0:  # the group label lines up with the middle of its first line
                        group.grid(row=r0, column=0, sticky="nw", padx=0,
                                   pady=(max(0, (line - group.winfo_reqheight()) // 2), 0))
                    r += 1
            show_note()

        self.out_var.trace_add("write", show_note)
        area.on_width = layout
        layout(px(480))  # until the tab is shown and has its real width
        area.bind_wheel()

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
        # Hashing the files takes ~0.1 s, which a reload after an idle unload would add
        # before the microphone opens: once the model loaded, the files are trusted.
        if not self.model_ok and model_store.missing():
            return self._start_download()
        self.want_engine = True
        self.engine_state, self.engine_msg = "loading", "Loading the model onto the NPU…"
        self._spawn_if_possible()
        self._refresh()

    def _spawn_if_possible(self):
        gone = [e for e in self.closing if not e.alive()]
        self.closing = [e for e in self.closing if e not in gone]
        failed = next((e for e in gone if e.npu_started.is_set() and e.proc.exitcode), None)
        if failed is not None and self.engine_state != "error":  # it went on to the NPU after it was closed
            return self._engine_failed(f"The engine process exited (code {failed.proc.exitcode}) after it used "
                                       "the NPU. Quit and start the app again.")
        if not self.want_engine or self.engine is not None or self.engine_state == "error":
            return
        if self.closing:
            if not self._engine_stuck():
                self.engine_msg = "Waiting for the previous engine process to exit…"
            return
        now = win32.awake_time()
        if self.other_engine is not None and now - self.other_engine[1] < 1.0:
            return  # checked about once a second
        if win32.mutex_exists(engine.MUTEX):  # an engine process of an earlier run of the app is alive
            self.other_engine = (self.other_engine[0] if self.other_engine else now, now)
            if now - self.other_engine[0] > WATCHDOG_S:
                return self._engine_failed("An NPU engine from an earlier run is still running (possible NPU "
                                           "hang). Restart Windows, then start the app.")
            self.engine_msg = "Waiting for the previous engine process to exit…"
            return
        self.other_engine = None
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

    def _engine_stuck(self):
        """True while a closed engine process that Engine.close gave up on is alive:
        no new engine may start, so dictation stops for good (shown on the overlay)."""
        if not any(e.close_gave_up and e.alive() for e in self.closing):
            return False
        if self.engine_state != "error":
            self._engine_failed("The previous engine process is still stuck in the NPU driver. Restart Windows.")
        self.overlay.show("error", "Engine stuck: restart Windows", hold_s=3.0)
        return True

    def _unload_engine(self):
        """Idle for unload_after_s: the engine process exits to free its ~1.4 GB. Not a
        failure (no retry counting, no error shown): the next press loads the model
        again, through _start_model, while you already speak."""
        log(f"engine {self.engine_gen}: unloading the model after {self.settings['unload_after_s']} s idle")
        self._close_engine()
        self.engine_state = "unloaded"
        self.footer.configure(text=f"Model unloaded after {self.settings['unload_after_s']} s idle to free memory.")
        self._refresh()

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
                self.post(("download_done", None, True))
            except model_store.NoAutoRetry as e:  # a file that does not match, a full disk
                self.post(("download_done", str(e), False))
            except Exception as e:
                self.post(("download_done", f"{type(e).__name__}: {e}", True))

        threading.Thread(target=run, daemon=True, name="download").start()

    # ---- settings ---------------------------------------------------------------
    def _mic_tick(self):
        """Reads the devices again every MIC_REFRESH_MS while nothing records, so a
        microphone connected later is there at the next press without delaying it."""
        if self.quitting:
            return
        if not self.recording and self.recorder.idle():
            refresh_devices()
        self.root.after(MIC_REFRESH_MS, self._mic_tick)

    def _fill_mics(self, wait_s=0.0):
        """Fills the microphone list, read again from Windows when no stream is open or
        closing (one closing is waited for up to wait_s). A saved microphone that is
        missing is shown as not connected."""
        rec = getattr(self, "recorder", None)  # none yet while the window is built
        if rec is not None and not self.recording and rec.idle(wait_s):
            refresh_devices()
        try:
            names = input_devices()
        except Exception:
            names = []
        mic, values = self.settings["microphone"], ["System default"] + names
        if mic and mic not in names:  # device_index() uses the default instead
            mic += NOT_CONNECTED
            values.append(mic)
            if hasattr(self, "footer"):  # not yet while the window is built
                self.footer.configure(text=f"Microphone: {mic}.")
        self.mic_box["values"] = values
        self.mic_var.set(mic or "System default")

    def _changed(self):
        s = self.settings
        s["mode"], s["output"] = self.mode_var.get(), self.out_var.get()
        mic = self.mic_var.get()
        if mic != s["microphone"] + NOT_CONNECTED:  # that one keeps the saved name
            s["microphone"] = "" if mic == "System default" else mic
        for k, var in self.checks.items():
            s[k] = var.get()
        save_settings(s)
        self._refresh()

    def _unload_changed(self, event=None):
        """Spinbox arrows, Return or leaving the field: a number is clamped and saved,
        and the idle time counts from now; anything else is put back. Setting 0 while
        unloaded does not load the model: the next press does."""
        try:
            v = unload_seconds(int(float(self.unload_var.get())))
        except (ValueError, OverflowError):
            v = self.settings["unload_after_s"]
        self.unload_var.set(str(v))
        if v != self.settings["unload_after_s"]:
            self.settings["unload_after_s"] = v
            self.last_active = time.monotonic()
            save_settings(self.settings)
            self._refresh()

    def _refresh(self):
        s = self.settings
        key = win32.label(s["shortcut"])
        state = "recording" if self.recording else self.engine_state
        self.pill.set(state)
        self.hero.set(key.split("+"), "capturing" if self.capturing else "pressed" if self.recording
                      else "error" if self.engine_state == "error"
                      else "muted" if self.engine_state in ("loading", "download", "unloaded") else "normal")
        # Not registered: pressing the shortcut does nothing (after an NPU error the hint says to restart).
        taken = not self.shortcut_ok and not self.capturing and self.engine_state != "error"
        if taken:
            hint = f"{key} is used by another app: choose another shortcut."
        elif self.engine_state == "download":
            done, total = self.downloading
            hint = (f"Downloading the speech model, {done / 2**20:.0f} of {total / 2**20:.0f} MB. "
                    "This happens once; the app is ready when it is done.")
        elif self.engine_state == "loading":
            hint = (self.engine_msg + " (a few seconds). You can already dictate: speech is transcribed "
                    "once the model is ready.")
        elif self.engine_state in ("error", "waiting"):
            hint = self.engine_msg
        elif self.engine_state == "unloaded":
            hint = (f"The model is unloaded to free memory. Press {key} and just start speaking: it loads again "
                    "in a few seconds and your speech is transcribed once it is ready.")
        elif s["mode"] == "toggle":
            hint = f"Press {key}, speak, then press it again."
        else:
            hint = f"Hold {key} while you speak, then let go."
        self.hint.configure(text=hint)
        show = self.engine_state == "download"
        if show != self.progress_shown:
            self.progress_shown = show
            if show:
                self.progress.pack(fill="x", pady=(px(10), 0), after=self.hint)
            else:
                self.progress.pack_forget()
        if show:
            done, total = self.downloading
            self.progress["value"] = 1000 * done / max(total, 1)
        self._show_history()
        tray = getattr(self, "tray", None)
        title = f"{APP}: {key} is used by another app" if taken else f"{APP}: {STATES[state]}"
        if tray is not None and getattr(self, "_tray_state", None) != (state, title):
            self._tray_state = (state, title)
            tray.icon = icon_image(COLORS[state])
            tray.title = title

    def _show_history(self):
        """The history list, newest first; the empty text names the current shortcut."""
        key = win32.label(self.settings["shortcut"])
        self.history_list.set_items(self.history[::-1], f"Nothing dictated yet. Press {key} and speak; your text "
                                                        "also appears here. The list is kept only while the app runs.")
        if bool(self.history) != bool(self.history_hint.winfo_manager()):
            if self.history:
                self.history_hint.pack(anchor="w", pady=(0, px(8)), before=self.history_list)
            else:
                self.history_hint.pack_forget()

    def _copy_history(self, index):
        """A click on a history entry (0 = newest): the output thread copies it, as for a dictation."""
        self.out_q.put((None, self.history_list.items[index]["text"], "clipboard", None, False))
        self.history_list.flash(index)

    # ---- shortcut ------------------------------------------------------------
    def _register_shortcut(self, combo):
        ok = self.shortcut_ok = self.hotkey.set(combo)
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
        self.root.bind("<FocusOut>", self._on_capture_focus_out)
        self._refresh()

    def _on_capture_focus_out(self, event):
        """Capture ends, with the saved shortcut back, once focus has left the app.
        The root binding also sees focus move between widgets, hence the check later."""
        def check():
            try:
                gone = self.root.focus_get() is None
            except KeyError:  # a Combobox list has focus (older tkinter raises here)
                gone = False
            if self.capturing and gone:
                self._end_capture(self.settings["shortcut"])
        self.root.after(50, check)

    def _on_capture_key(self, event):
        if event.keysym == "Escape":
            return self._end_capture(self.settings["shortcut"])
        combo = win32.combo_from_event(event.keycode, event.state)
        if combo is None:
            return  # a modifier on its own: wait for the main key
        if not valid_shortcut(combo):
            self.footer.configure(text="Use Ctrl, Alt, Shift or Win with the key, or an F-key on its own.")
            return
        if self._register_shortcut(combo):
            self._end_capture(combo, registered=True)
            self.footer.configure(text=f"Shortcut set to {win32.label(combo)}.")

    def _end_capture(self, combo, registered=False):
        self.root.unbind("<KeyPress>")
        self.root.unbind("<FocusOut>")
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
        icon = sys.executable if FROZEN else self.icon_path
        parts, ok = [], True
        for name, csidl in (("Start menu", win32.CSIDL_PROGRAMS), ("desktop", win32.CSIDL_DESKTOPDIRECTORY)):
            try:
                make_shortcut(Path(win32.shell_folder(csidl)) / f"{APP}.lnk", icon)
                parts.append(f"{name} shortcut added")
            except Exception as e:
                parts.append(f"{name} shortcut failed: {str(e).rstrip('.')}")
                ok = False
        self.footer.configure(text="Shortcuts added to the Start menu and the desktop." if ok
                              else "; ".join(parts) + ".")

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
        if self.capturing:  # the saved shortcut works again while the window is hidden
            self._end_capture(self.settings["shortcut"])
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
        if self._engine_stuck():  # neither "loading" nor a recording while it lives
            return self._sound("error")
        if self.engine_state in ("waiting", "unloaded"):
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
        self.last_active = time.monotonic()
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
        self.last_active = time.monotonic()
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
            self._sound("error")  # the only one for this dictation, with or without text
            self.overlay.show("error", "Part of the dictation could not be transcribed", hold_s=3.0)
            self.footer.configure(text="A piece of the dictation failed on the CPU and is missing"
                                       + (" from the text below." if text else ".") + " Details in engine.log.")
        if not text:
            if not self.recording and not s.get("failed"):
                self.overlay.show("info", "No speech heard", hold_s=1.2)
            return
        latency = time.perf_counter() - (s["t_stop"] or time.perf_counter())
        self.out_q.put((session, text + (" " if self.settings["trailing_space"] else ""), self.settings["output"],
                        s["target"], s.get("failed")))
        self.history.append({"time": time.strftime("%H:%M"), "text": text})
        del self.history[:-50]
        self._show_history()
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
                session, text, mode, target, failed = self.out_q.get(timeout=0.05)
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
                            elif win32.modifiers_down():  # Shift, Alt or Win pressed since the wait above
                                win32.set_clipboard(text)  # left on the clipboard, as after the wait
                                done = "held"
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
                    elif win32.modifiers_down():  # with Alt held the characters could trigger menu shortcuts
                        win32.set_clipboard(text)
                        done = "held"
                    else:
                        win32.type_text(text)
                elif done == "clipboard":
                    win32.set_clipboard(text)
            except Exception as e:
                err = str(e)
            self.out_busy = False
            self.post(("output_done", session, done, err, failed))

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
        # in the middle of a piece does not count as an NPU hang. Closed engines are
        # timed too while in NPU work (a warm-up begun after a start timeout).
        now = win32.awake_time()
        late = self.busy is not None and self.busy[0] == self.engine_gen and now - self.busy[1] > WATCHDOG_S
        if late or self.engine_state != "error" and any(e.npu_overdue(now) for e in self.closing):
            self._engine_failed(f"The NPU did not answer within {WATCHDOG_S:.0f} s (possible NPU hang). Dictation "
                                "is stopped. Restart Windows before using the NPU again, then start the app.")
        eng = self.engine
        if eng is not None and self.busy is None and self.engine_state == "loading" and eng.npu_started.is_set():
            self.busy = (self.engine_gen, now)  # the warm-up began: timed even if its "npu" message is late
        if (eng is not None and self.engine_state == "loading" and not eng.npu_started.is_set()
                and now - eng.t0 > START_TIMEOUT_S):
            self._engine_start_failed(f"no answer within {START_TIMEOUT_S:.0f} s", retry=False)
        # Unload after unload_after_s idle: only with the model loaded and nothing
        # recorded, waiting for the engine, on the NPU or being inserted.
        u = self.settings["unload_after_s"]
        if (u > 0 and self.engine_state == "ready" and not self.recording and not self.sessions
                and not self.backlog and self.busy is None and self.out_q.empty() and not self.out_busy
                and time.monotonic() - self.last_active >= u):
            self._unload_engine()

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
                self.last_active = time.monotonic()
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
                self.download_fails = 0
                self._start_model()
            elif not ev[2]:  # trying again by itself cannot help: only a press does
                log(f"model download failed (press to retry): {ev[1]}")
                self.engine_state, self.retry_at = "waiting", None
                self.engine_msg = f"The model download failed: {ev[1]}. Press the shortcut to try again."
                self._notify(self.engine_msg)
                self._refresh()
            elif not self.quitting:
                log(f"model download failed: {ev[1]}")
                self.engine_state = "waiting"
                delay = RETRY_S[min(self.retries, len(RETRY_S) - 1)]
                self.retries += 1
                self.retry_at = time.monotonic() + delay
                self.engine_msg = (f"The model download failed ({ev[1]}). It tries again in {delay} s, "
                                   "or when you press the shortcut.")
                self.download_fails += 1
                if self.download_fails == 3:
                    self._notify("The model download failed 3 times. It keeps trying; open NPU Dictate "
                                 "from the tray for details.")
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
                self.last_active = time.monotonic()
                self.model_ok = True
                self.footer.configure(text=f"NPU engine ready in {msg[1]:.1f} s.")
            elif msg[0] == "start_failed":
                if msg[1].startswith("model files"):  # the engine found them damaged: check them again
                    self.model_ok = False
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
                self.last_active = time.monotonic()
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
            _, session, done, err, failed = ev
            self.last_active = time.monotonic()
            instead = {"moved": "the focused window changed before it was ready.",
                       "held": "Shift, Alt or Win was still held down."}.get(done)  # why it was only copied
            if session is None:  # copied from the history list
                self.footer.configure(text=f"Could not copy: {err}" if err else "Copied.")
            elif err:
                self.overlay.show("error", "Could not insert the text", hold_s=2.0)
                self.footer.configure(text=f"Insert failed: {err}. The text is in the history.")
            elif failed and instead:
                self.overlay.show("error", "Copied, part missing: press Ctrl+V", hold_s=3.0)
                self.footer.configure(text="A piece of the dictation failed on the CPU and is missing (details in "
                                      "engine.log). The rest was copied to the clipboard instead of inserted: "
                                      + instead)
            elif failed:  # the rest went in, but a piece failed on the CPU (_check_done set the footer)
                label = {"paste": "Inserted", "type": "Typed", "test": "Test output"}.get(done, "Copied")
                self.overlay.show("error", f"{label}, but part of the dictation is missing", hold_s=3.0)
            elif instead:
                self.overlay.show("info", "Copied: press Ctrl+V to paste", hold_s=2.5)
                self.footer.configure(text="The text was copied to the clipboard instead of inserted: " + instead)
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
