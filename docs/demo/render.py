"""Render a recording from record.py as an animated terminal SVG.

Program output keeps its recorded timing (token streaming runs at the real
NPU speed). Two edits are made for watchability, and the README says so:
waits longer than MAX_GAP (model loading) are shortened to MAX_GAP, and
typed input appears at TYPE_S per character. Full WAV paths are shown as
file names. Run: python docs/demo/render.py chat stt
"""

import json
import re
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "img"

COLS, CW, LH, FS = 100, 8.4, 20, 14
PAD_X, TOP = 18, 48
MAX_GAP, TYPE_S, HOLD_S = 1.2, 0.035, 5.0
FONT = "'Cascadia Mono', Consolas, 'SF Mono', Menlo, 'DejaVu Sans Mono', monospace"
C = dict(bg="#16161a", bar="#222228", text="#d6d6d0", muted="#8a8a84", user="#7fb3ff",
         npu="#5fd38d", stat="#e8b44a", prompt="#c79bff")

SCENES = {
    "chat": dict(title="LFM2.5-230M, every layer on the Ryzen AI NPU (real recording)",
                 command=r".\scripts\chat_lfm25_x8.ps1"),
    "stt": dict(title="Parakeet TDT 0.6B v2, encoder on the Ryzen AI NPU (real recording)",
                command=r".\scripts\parakeet_npu_demo.ps1 1089-134686-0021.wav 121-121726-0000.wav 1284-1180-0000.wav"),
}

# (pattern on the finished line, group to color, color)
STYLES = [
    (r"^(PS>)", 1, "prompt"),
    (r"^(You:)", 1, "user"),
    (r"^You: (.*)$", 1, "text"),
    (r"(NPU:)", 1, "npu"),
    (r"^(\(.*\))$", 1, "muted"),
    (r"^(\[.*?\])", 1, "stat"),
    (r"^(== .*)$", 1, "muted"),
    (r"^((loading|ready|programs|weights|LFM2.5-230M on).*)$", 1, "muted"),
]


def timeline(events, command):
    """Yield (time, text) with typing spread out and long waits shortened."""
    t = 0.0
    for ch in "PS> ":
        yield t, ch
    for ch in command:
        t += TYPE_S
        yield t, ch
    t += 0.4
    yield t, "\n"
    last = 0.0
    for when, kind, text in events:
        t += min(when - last, MAX_GAP)
        last = when
        if kind == "in":
            for ch in text:
                t += TYPE_S
                yield t, ch
            t += 0.25
            yield t, "\n"
        else:
            text = re.sub(r"(?:[\w.:-]+\\)+([\w-]+\.wav)", r"\1", text)
            yield t, text


def layout(stream):
    """Terminal emulation: returns {(row, col): (char, time)}."""
    cells, row, col, word = {}, 0, 0, 0  # word: column where the current word starts
    for t, text in stream:
        for ch in text:
            if ch == "\r":
                col = word = 0
            elif ch == "\n":
                row, col, word = row + 1, 0, 0
            else:
                if col >= COLS:
                    if ch == " ":
                        row, col, word = row + 1, 0, 0
                        continue
                    if 0 < word:  # move the unfinished word to the next row (word wrap)
                        moved = [cells.pop((row, c)) for c in range(word, col) if (row, c) in cells]
                        row, col = row + 1, 0
                        for cell in moved:
                            cells[(row, col)] = cell
                            col += 1
                    else:
                        row, col = row + 1, 0
                    word = 0
                if ch == " ":
                    word = col + 1
                if ch != " " or (row, col) in cells:
                    cells[(row, col)] = (ch, t)
                col += 1
    return cells


def colors(cells):
    rows = {}
    for (r, c), (ch, _) in cells.items():
        rows.setdefault(r, {})[c] = ch
    out = {}
    for r, line in rows.items():
        s = "".join(line.get(c, " ") for c in range(max(line) + 1))
        col = ["text"] * len(s)
        for pat, g, name in STYLES:
            m = re.search(pat, s)
            if m:
                for i in range(m.start(g), m.end(g)):
                    col[i] = name
        out[r] = col
    return out


def svg(name):
    scene = SCENES[name]
    events = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    cells = layout(timeline(events, scene["command"]))
    color = colors(cells)
    nrows = max(r for r, _ in cells) + 1
    end = max(t for _, t in cells.values()) + HOLD_S
    w = PAD_X * 2 + COLS * CW
    h = TOP + nrows * LH + 16

    runs = {}  # group consecutive cells with the same row, color and appear time
    for (r, c), (ch, t) in sorted(cells.items()):
        key = (r, color[r][c], round(t, 2))
        lst = runs.setdefault(key, [])
        if lst and lst[-1][0] + len(lst[-1][1]) == c:
            lst[-1][1] += ch
        else:
            lst.append([c, ch])

    body = []
    for (r, cname, t), pieces in sorted(runs.items(), key=lambda kv: kv[0][2]):
        f = min(max(t / end, 0.0), 0.999)
        anim = (f'<animate attributeName="opacity" values="0;1" keyTimes="0;{f:.4f}" '
                f'dur="{end:.2f}s" calcMode="discrete" repeatCount="indefinite"/>')
        for c, s in pieces:
            body.append(f'<text x="{PAD_X + c * CW:.1f}" y="{TOP + r * LH + 14}" fill="{C[cname]}" '
                        f'opacity="0">{escape(s)}{anim}</text>')
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h}" viewBox="0 0 {w:.0f} {h}" '
            f'font-family="{FONT}" font-size="{FS}" xml:space="preserve" role="img" aria-label="{escape(scene["title"])}">\n'
            f'<rect width="{w:.0f}" height="{h}" rx="10" fill="{C["bg"]}"/>\n'
            f'<path d="M0,10 Q0,0 10,0 H{w - 10:.0f} Q{w:.0f},0 {w:.0f},10 V32 H0 Z" fill="{C["bar"]}"/>\n'
            + "".join(f'<circle cx="{20 + 18 * i}" cy="16" r="5.5" fill="{col}"/>'
                      for i, col in enumerate(("#ff5f57", "#febc2e", "#28c840")))
            + f'\n<text x="{w / 2:.0f}" y="20.5" fill="{C["muted"]}" font-size="12" text-anchor="middle">'
              f'{escape(scene["title"])}</text>\n')
    return head + "\n".join(body) + "\n</svg>\n"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name in sys.argv[1:] or list(SCENES):
        path = OUT / f"demo_{name}.svg"
        path.write_text(svg(name), encoding="utf-8")
        print(path, f"{path.stat().st_size / 1024:.0f} KB")
