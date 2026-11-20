"""Experiment 015: can a user program reach the NPU's hidden column 0?

Phoenix has five hardware columns; MLIR-AIE exposes four. This script builds
two tiny int32 "+1" designs and patched copies of their xclbin partition
metadata, then (one step per process) creates hardware contexts and runs them.

  build                 compile both designs, write partition variants
  ctx  VARIANT          create a hardware context only (no kernel run),
                        print the driver's partition report, exit
  run  VARIANT          create the context, run once, check the output

Designs (IRON columns are relative to the partition start):
  col1  shim (1,0) -> core (1,2) -> shim (1,0)
  col0  shim (1,0) -> core (0,2) -> shim (1,0); in a 5-wide partition that
        starts at hardware column 0, core (0,2) is in the hidden column.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache" / "015"
XCLBINUTIL = ROOT / "cache" / "iron" / "sdk" / "xrt_sdk" / "xrt" / "xclbinutil.exe"
XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
N, M = 1024, 256

# variant -> (design, column_width, start_columns); None keeps the compiled value
VARIANTS = {
    "col1_base": ("col1", None, None),
    "col1_start1": ("col1", 4, [1]),
    "col1_w5": ("col1", 5, [0]),
    "col0_base": ("col0", None, None),
    "col0_w5": ("col0", 5, [0]),
    "id1_base": ("id1", None, None),
    "id1_w5": ("id1", 5, [0]),
    "id0_base": ("id0", None, None),
    "id0_w5": ("id0", 5, [0]),
    "full0_base": ("full0", None, None),
    "full0_w5": ("full0", 5, [0]),
}
KDIR = Path(__file__).resolve().parent / "kernels" / "015"


def designs():
    import aie.iron as iron
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config

    T = np.ndarray[(M,), np.dtype[np.int32]]
    V = np.ndarray[(N,), np.dtype[np.int32]]
    tap = TensorAccessPattern((N,), 0, [1, 1, 1, N], [0, 0, 0, 1])

    def make(core_col, name):
        @iron.jit(tag=f"hc015_{name}_v1")
        def design(a: In, b: Out, *, tag: CompileTime[str]):
            fin = ObjectFifo(T, name=f"{name}_in", depth=2)
            fout = ObjectFifo(T, name=f"{name}_out", depth=2)

            def core(i, o):
                for _ in range_(N // M):
                    x = i.acquire(1)
                    y = o.acquire(1)
                    for j in range_(M):
                        y[j] = x[j] + 1
                    i.release(1)
                    o.release(1)

            w = Worker(core, [fin.cons(), fout.prod()], tile=Tile(core_col, 2))

            def seq(a, b, pin, pout):
                g = TaskGroup()
                pout.drain(b, tap=tap, group=g, wait=True)
                pin.fill(a, tap=tap, group=g)
                g.finish()

            rt = Runtime(seq, [V, V, fin.prod(tile=Tile(1, 0)), fout.cons(tile=Tile(1, 0))])
            return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

        return design

    def make_id(core_col, name):
        @iron.jit(tag=f"hc015_{name}_v1")
        def design(a: In, b: Out, *, tag: CompileTime[str]):
            fin = ObjectFifo(T, name=f"{name}_in", depth=2)
            fout = ObjectFifo(T, name=f"{name}_out", depth=2)
            k = ExternalFunction("hc_id", source_file=str(KDIR / "hc_id.cc"),
                                 arg_types=[T, T, np.int32],
                                 include_dirs=[config.cxx_header_path()])

            def core(i, o, fn):
                for _ in range_(N // M):
                    x = i.acquire(1)
                    y = o.acquire(1)
                    fn(x, y, M)
                    i.release(1)
                    o.release(1)

            w = Worker(core, [fin.cons(), fout.prod(), k], tile=Tile(core_col, 2))

            def seq(a, b, pin, pout):
                g = TaskGroup()
                pout.drain(b, tap=tap, group=g, wait=True)
                pin.fill(a, tap=tap, group=g)
                g.finish()

            rt = Runtime(seq, [V, V, fin.prod(tile=Tile(1, 0)), fout.cons(tile=Tile(1, 0))])
            return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

        return design

    V4 = np.ndarray[(4 * N,), np.dtype[np.int32]]

    def make_full(name):
        """Cores at (0,2)..(0,5), each with its own input and output stream.
        Rows 2-4 are fed from shims (1,0) and (2,0) directly; row 5's input
        is staged through memory tile (0,1)."""
        @iron.jit(tag=f"hc015_{name}_v1")
        def design(a: In, b: Out, *, tag: CompileTime[str]):
            k = ExternalFunction("hc_id", source_file=str(KDIR / "hc_id.cc"),
                                 arg_types=[T, T, np.int32],
                                 include_dirs=[config.cxx_header_path()])
            fins = [ObjectFifo(T, name=f"{name}_in{r}", depth=2) for r in range(4)]
            fouts = [ObjectFifo(T, name=f"{name}_out{r}", depth=2) for r in range(4)]
            staged = fins[3].cons().forward(tile=Tile(0, 1), name=f"{name}_mt")

            def core(i, o, fn):
                for _ in range_(N // M):
                    x = i.acquire(1)
                    y = o.acquire(1)
                    fn(x, y, M)
                    i.release(1)
                    o.release(1)

            ws = []
            for r in range(4):
                src = staged.cons() if r == 3 else fins[r].cons()
                ws.append(Worker(core, [src, fouts[r].prod(), k], tile=Tile(0, 2 + r)))
            shim = [Tile(1, 0), Tile(1, 0), Tile(2, 0), Tile(2, 0)]

            def seq(a, b, *ends):
                pins, pouts = ends[:4], ends[4:]
                g = TaskGroup()
                for r in range(4):
                    pouts[r].drain(b, tap=TensorAccessPattern((4 * N,), r * N, [1, 1, 1, N], [0, 0, 0, 1]),
                                   group=g, wait=True)
                for r in range(4):
                    pins[r].fill(a, tap=tap, group=g)
                g.finish()

            rt = Runtime(seq, [V, V4] + [fins[r].prod(tile=shim[r]) for r in range(4)]
                         + [fouts[r].cons(tile=shim[r]) for r in range(4)])
            return Program(iron.get_current_device(), rt, workers=ws).resolve_program()

        return design

    return {"col1": make(1, "col1"), "col0": make(0, "col0"), "full0": make_full("full0"),
            "id1": make_id(1, "id1"), "id0": make_id(0, "id0")}


def patch(src, dst, width, starts):
    # Start from the build's own partition JSON: its PDI entry names the PDI
    # file by path, which xclbinutil needs to rebuild the section.
    part = OUT / (dst.stem + ".part.json")
    doc = json.loads((Path(src).parent / "partition_main.json").read_text())
    p = doc["aie_partition"]["partition"]
    if width is not None:
        p["column_width"] = width
    if starts is not None:
        p["start_columns"] = list(starts)
    part.write_text(json.dumps(doc, indent=1))
    subprocess.run([str(XCLBINUTIL), "--input", str(src), "--remove-section", "AIE_PARTITION",
                    "--add-section", f"AIE_PARTITION:JSON:{part}", "--output", str(dst), "--force"],
                   check=True, capture_output=True)
    return p


def build():
    import npu_direct as nd
    OUT.mkdir(parents=True, exist_ok=True)
    built = {}
    for name, d in designs().items():
        xclbin, insts = nd.compile_design(d)
        built[name] = (xclbin, insts)
        print(name, xclbin, insts, flush=True)
    manifest = {}
    for v, (dname, width, starts) in VARIANTS.items():
        xclbin, insts = built[dname]
        dst = OUT / f"{v}.xclbin"
        p = patch(xclbin, dst, width, starts)
        manifest[v] = {"xclbin": str(dst), "insts": str(insts), "partition": p}
        print(v, json.dumps(p), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))


def report():
    r = subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True)
    return "\n".join(l for l in r.stdout.splitlines() if l.strip() and not l.startswith("---"))


def ctx_or_run(variant, run):
    import pyxrt
    import npu_direct as nd
    m = json.loads((OUT / "manifest.json").read_text())[variant]
    print("variant", variant, json.dumps(m["partition"]), flush=True)
    try:
        prog = nd.Program(m["xclbin"])
    except Exception as e:  # driver refused the context
        print("CONTEXT REFUSED:", repr(e), flush=True)
        nd.finish(2)
    print("context created", flush=True)
    print(report(), flush=True)
    if run:
        entry = prog.entry(m["insts"])
        a = nd.Buffer(N, np.int32)
        nb = 4 * N if variant.startswith("full") else N
        b = nd.Buffer(nb, np.int32)
        src = np.arange(N, dtype=np.int32) * 7 - 300
        a.write(src)
        b.write(np.zeros(nb, dtype=np.int32))
        r = entry.start(a, b)
        state = r.wait()
        print("run state", state, flush=True)
        got = np.array(b.from_device())
        want = np.tile(src + 1, nb // N)
        if variant.startswith(("id", "full")):
            ids = got[::M].copy()
            want[::M] = got[::M]
            print("CORE IDS", [hex(int(i)) for i in ids], flush=True)
        ok = bool(np.array_equal(got, want))
        print("RESULT", "correct" if ok else "WRONG", got[:8].tolist(), flush=True)
    nd.finish(0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd in ("ctx", "run"):
        ctx_or_run(sys.argv[2], cmd == "run")
