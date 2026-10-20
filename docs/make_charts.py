"""Draw the README result charts as plain SVG (no dependencies).

Numbers come from experiments/010_x8_engine.md and
experiments/012_all_npu_encoder.md. Run: python docs/make_charts.py
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent / "img"

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781",
                  grid="#e1e0d9", base="#c3c2b7", accent="#2a78d6", other="#c9c8bf"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", base="#383835", accent="#3987e5", other="#55544f"),
}

CHARTS = {
    "llm_decode": dict(
        title="LFM2.5-230M decode speed, one stream",
        subtitle="tokens per second, higher is better · Ryzen 9 8945HS",
        unit=" tok/s",
        ticks=[0, 40, 80, 120, 160],
        rows=[
            ("CPU · llama.cpp Q8_0", 160, False),
            ("CPU · llama.cpp F16", 91, False),
            ("NPU · 8-core engine (exp 010)", 58, True),
            ("CPU · PyTorch BF16, 8 threads", 33, False),
            ("NPU · first all-NPU path (exp 006)", 2.2, False),
        ],
    ),
    "stt_encoder": dict(
        title="Parakeet TDT 0.6B v2 encoder, 10 s of audio",
        subtitle="milliseconds, lower is better · same WER on NPU and CPU FP32",
        unit=" ms",
        ticks=[0, 100, 200, 300, 400, 500],
        rows=[
            ("NPU + CPU hybrid (exp 011)", 520, False),
            ("CPU · ONNX Runtime FP32, 8 cores", 496, False),
            ("CPU · ONNX Runtime dynamic int8", 411, False),
            ("NPU · whole encoder (exp 012)", 185, True),
            ("GPU · RTX 4060 FP32", 26, False),
        ],
    ),
}

W, LABEL_W, RIGHT = 760, 260, 70
TOP, ROW_H, BAR_H = 78, 38, 18
FONT = "'Segoe UI', -apple-system, 'Helvetica Neue', Arial, sans-serif"


def fmt(v):
    return f"{v:g}" if v >= 10 else f"{v:.1f}"


def bar(x, y, w, h, r, fill):
    # Rounded data end, square end on the baseline.
    r = min(r, w)
    return (f'<path d="M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} '
            f'V{y + h - r} Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z" fill="{fill}"/>')


def svg(chart, t):
    rows = chart["rows"]
    h = TOP + ROW_H * len(rows) + 34
    plot_w = W - LABEL_W - RIGHT
    vmax = chart["ticks"][-1]
    sx = lambda v: LABEL_W + plot_w * v / vmax
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
           f'font-family="{FONT}" role="img" aria-label="{chart["title"]}">',
           f'<rect width="{W}" height="{h}" rx="10" fill="{t["surface"]}"/>',
           f'<text x="20" y="32" font-size="17" font-weight="600" fill="{t["text"]}">{chart["title"]}</text>',
           f'<text x="20" y="54" font-size="13" fill="{t["text2"]}">{chart["subtitle"]}</text>']
    y_end = TOP + ROW_H * len(rows)
    for tick in chart["ticks"][1:]:
        x = sx(tick)
        out.append(f'<line x1="{x:.1f}" y1="{TOP - 6}" x2="{x:.1f}" y2="{y_end}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{y_end + 18}" font-size="11" fill="{t["muted"]}" text-anchor="middle">{tick}</text>')
    out.append(f'<text x="{sx(0):.1f}" y="{y_end + 18}" font-size="11" fill="{t["muted"]}" text-anchor="middle">0</text>')
    for i, (label, v, hi) in enumerate(rows):
        y = TOP + ROW_H * i + (ROW_H - BAR_H) / 2
        weight = "600" if hi else "400"
        out.append(f'<text x="{LABEL_W - 12}" y="{y + BAR_H / 2 + 4.5:.1f}" font-size="13" font-weight="{weight}" '
                   f'fill="{t["text"] if hi else t["text2"]}" text-anchor="end">{label}</text>')
        out.append(bar(LABEL_W, y, max(sx(v) - LABEL_W, 2), BAR_H, 4, t["accent"] if hi else t["other"]))
        out.append(f'<text x="{sx(v) + 8:.1f}" y="{y + BAR_H / 2 + 4.5:.1f}" font-size="13" font-weight="{weight}" '
                   f'fill="{t["text"]}">{fmt(v)}{chart["unit"]}</text>')
    out.append(f'<line x1="{LABEL_W}" y1="{TOP - 6}" x2="{LABEL_W}" y2="{y_end}" stroke="{t["base"]}" stroke-width="1"/>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, chart in CHARTS.items():
        for mode, t in THEMES.items():
            (OUT / f"{name}_{mode}.svg").write_text(svg(chart, t), encoding="utf-8")
            print(OUT / f"{name}_{mode}.svg")
