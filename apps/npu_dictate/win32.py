"""Win32 helpers (ctypes only): global hotkey, key state, clipboard, synthetic
input, and window styles for a click-through overlay that never takes focus."""

import ctypes
import threading
import time
from ctypes import wintypes as w

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

user32.GetAsyncKeyState.restype = ctypes.c_short
user32.GetKeyState.restype = ctypes.c_short
user32.GetClipboardData.restype = w.HANDLE
user32.SetClipboardData.argtypes = [w.UINT, w.HANDLE]
user32.SetClipboardData.restype = w.HANDLE
user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.GetWindowLongPtrW.argtypes = [w.HWND, ctypes.c_int]
user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.SetWindowLongPtrW.argtypes = [w.HWND, ctypes.c_int, ctypes.c_ssize_t]
user32.GetParent.restype = w.HWND
user32.GetParent.argtypes = [w.HWND]
user32.PostThreadMessageW.argtypes = [w.DWORD, w.UINT, w.WPARAM, w.LPARAM]
user32.SetWindowPos.argtypes = [w.HWND, w.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.UINT]
user32.ShowWindow.argtypes = [w.HWND, ctypes.c_int]
kernel32.GlobalAlloc.argtypes = [w.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = w.HANDLE
kernel32.GlobalLock.argtypes = [w.HANDLE]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [w.HANDLE]
kernel32.GlobalFree.argtypes = [w.HANDLE]
kernel32.CreateMutexW.restype = w.HANDLE

# -- key names -----------------------------------------------------------------
MODS = {"ctrl": 0x2, "alt": 0x1, "shift": 0x4, "win": 0x8}
MOD_VKS = {"ctrl": (0x11,), "alt": (0x12,), "shift": (0x10,), "win": (0x5B, 0x5C)}
MOD_ORDER = ("ctrl", "alt", "shift", "win")
KEYS = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "backspace": 0x08, "insert": 0x2D, "delete": 0x2E,
        "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22, "up": 0x26, "down": 0x28,
        "left": 0x25, "right": 0x27, "pause": 0x13, "capslock": 0x14, "scrolllock": 0x91,
        "`": 0xC0, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, "\\": 0xDC, ";": 0xBA, "'": 0xDE,
        ",": 0xBC, ".": 0xBE, "/": 0xBF}
KEYS.update({chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)})
KEYS.update({str(d): 0x30 + d for d in range(10)})
KEYS.update({f"f{i}": 0x6F + i for i in range(1, 25)})
KEYS.update({f"num{d}": 0x60 + d for d in range(10)})
VK_NAMES = {v: k for k, v in KEYS.items()}
MOD_KEY_VKS = {0x10, 0x11, 0x12, 0x5B, 0x5C, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5}


def parse(combo):
    """'ctrl+shift+space' -> (modifier flags, virtual key)."""
    parts = combo.lower().split("+")
    mods = 0
    for p in parts[:-1]:
        mods |= MODS[p]
    return mods, KEYS[parts[-1]]


def label(combo):
    return "+".join(p if len(p) == 1 else p.capitalize() for p in combo.split("+")).replace("Pageup", "PageUp") \
        .replace("Pagedown", "PageDown").replace("Capslock", "CapsLock")


def combo_from_event(vk, state):
    """Shortcut from a Tk key event (keycode is the Windows virtual key; state
    bits: Shift 0x1, Ctrl 0x4, Alt 0x20000). None for a modifier on its own or
    an unsupported key."""
    name = VK_NAMES.get(vk)
    if name is None or vk in MOD_KEY_VKS:
        return None
    held = {"ctrl": state & 0x4, "alt": state & 0x20000, "shift": state & 0x1,
            "win": user32.GetKeyState(0x5B) & 0x8000 or user32.GetKeyState(0x5C) & 0x8000}
    return "+".join([m for m in MOD_ORDER if held[m]] + [name])


def key_down(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def wait_modifiers_released(timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not any(key_down(v) for vks in MOD_VKS.values() for v in vks):
            return True
        time.sleep(0.01)
    return False


# -- global hotkey ---------------------------------------------------------------
WM_HOTKEY, WM_APP = 0x0312, 0x8000
MOD_NOREPEAT = 0x4000


class Hotkey(threading.Thread):
    """RegisterHotKey on a thread with its own message loop. on_press() runs on
    that thread; set(combo) re-registers (None unregisters) and returns whether
    Windows accepted the combination (False if another app owns it)."""

    def __init__(self, on_press):
        super().__init__(daemon=True, name="hotkey")
        self.on_press = on_press
        self.tid = None
        self._ready = threading.Event()
        self._done = threading.Event()
        self._pending = None
        self._ok = False
        self.start()
        self._ready.wait()

    def set(self, combo):
        self._pending = combo
        self._done.clear()
        user32.PostThreadMessageW(self.tid, WM_APP + 1, 0, 0)
        self._done.wait(2.0)
        return self._ok

    def run(self):
        self.tid = kernel32.GetCurrentThreadId()
        msg = w.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)  # create the message queue
        self._ready.set()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                self.on_press()
            elif msg.message == WM_APP + 1:
                user32.UnregisterHotKey(None, 1)
                self._ok = True
                if self._pending:
                    mods, vk = parse(self._pending)
                    self._ok = bool(user32.RegisterHotKey(None, 1, mods | MOD_NOREPEAT, vk))
                self._done.set()


# -- clipboard -------------------------------------------------------------------
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x2


def _open_clipboard():
    for _ in range(25):
        if user32.OpenClipboard(None):
            return True
        time.sleep(0.02)
    return False


def _global(data):
    h = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
    p = kernel32.GlobalLock(h)
    ctypes.memmove(p, data, len(data))
    kernel32.GlobalUnlock(h)
    return h


def clipboard_text():
    """(has_content, text): text is None if the clipboard holds something other than text."""
    if not _open_clipboard():
        return True, None
    try:
        if not user32.CountClipboardFormats():
            return False, None
        h = user32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return True, None
        p = kernel32.GlobalLock(h)
        try:
            return True, ctypes.wstring_at(p)
        finally:
            kernel32.GlobalUnlock(h)
    finally:
        user32.CloseClipboard()


def set_clipboard(text, history=True):
    """Put text on the clipboard. history=False keeps it out of Win+V clipboard history."""
    if not _open_clipboard():
        return False
    try:
        user32.EmptyClipboard()
        if text is not None:
            user32.SetClipboardData(CF_UNICODETEXT, _global((text + "\0").encode("utf-16-le")))
        if not history:
            for name in ("ExcludeClipboardContentFromMonitorProcessing", "CanIncludeInClipboardHistory"):
                fmt = user32.RegisterClipboardFormatW(name)
                user32.SetClipboardData(fmt, _global(b"\0\0\0\0"))
        return True
    finally:
        user32.CloseClipboard()


# -- synthetic input -------------------------------------------------------------
class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", w.WORD), ("wScan", w.WORD), ("dwFlags", w.DWORD), ("time", w.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", w.LONG), ("dy", w.LONG), ("mouseData", w.DWORD), ("dwFlags", w.DWORD),
                ("time", w.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", w.DWORD), ("u", _INPUTUNION)]


KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 0x2, 0x4


def _send(events):
    arr = (_INPUT * len(events))()
    for i, (vk, scan, flags) in enumerate(events):
        arr[i].type = 1
        arr[i].u.ki = _KEYBDINPUT(vk, scan, flags, 0, 0)
    return user32.SendInput(len(events), arr, ctypes.sizeof(_INPUT))


def send_paste():
    _send([(0x11, 0, 0), (0x56, 0, 0), (0x56, 0, KEYEVENTF_KEYUP), (0x11, 0, KEYEVENTF_KEYUP)])


def type_text(text):
    events = []
    for ch in text.replace("\r\n", "\n"):
        if ch == "\n":
            events += [(0x0D, 0, 0), (0x0D, 0, KEYEVENTF_KEYUP)]
            continue
        units = ch.encode("utf-16-le")
        for i in range(0, len(units), 2):
            u = int.from_bytes(units[i:i + 2], "little")
            events += [(0, u, KEYEVENTF_UNICODE), (0, u, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)]
    for i in range(0, len(events), 64):
        _send(events[i:i + 64])
        time.sleep(0.005)


# -- windows ---------------------------------------------------------------------
GWL_EXSTYLE = -20
WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW = 0x8, 0x20, 0x80
WS_EX_LAYERED, WS_EX_NOACTIVATE = 0x80000, 0x08000000


def toplevel_hwnd(tk_window):
    return user32.GetParent(tk_window.winfo_id())


def make_overlay(hwnd):
    """Click-through, never activated, not in the taskbar or Alt+Tab."""
    style = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style | WS_EX_TOPMOST | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
                             | WS_EX_LAYERED | WS_EX_NOACTIVATE)


def show_noactivate(hwnd, x, y):
    user32.SetWindowPos(hwnd, -1, x, y, 0, 0, 0x1 | 0x10 | 0x40)  # NOSIZE | NOACTIVATE | SHOWWINDOW


def hide(hwnd):
    user32.ShowWindow(hwnd, 0)


def work_area():
    """(left, top, right, bottom) of the primary monitor without the taskbar."""
    r = w.RECT()
    user32.SystemParametersInfoW(0x30, 0, ctypes.byref(r), 0)
    return r.left, r.top, r.right, r.bottom


def single_instance(name):
    """False if another process already holds the named mutex."""
    single_instance.handle = kernel32.CreateMutexW(None, False, name)
    return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS


def set_app_id(app_id):
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)


def set_dpi_aware():
    """Render at the real screen resolution instead of being bitmap-scaled by Windows."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system DPI aware
    except OSError:
        user32.SetProcessDPIAware()
