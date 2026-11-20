# Experiment 015: running code on the NPU's hidden column 0

Phoenix and Hawk Point NPUs have five hardware columns, but MLIR-AIE and
IRON expose four. AMD's Linux driver blocks the leftmost column
(`npu1_regs.c`: `.first_col = 1`; only developer builds can override it with
`start_col_index`), and MLIR-AIE removed its five-column `npu1` device in
2025 after users hit that check ([mlir-aie #2333, #2371,
#2397](https://github.com/Xilinx/mlir-aie/pull/2397)). This experiment asks
whether the Windows driver allows it and whether the column works.

## Result

**It works on Windows.** An IRON program whose xclbin declares a 5-column
partition gets all five columns, and cores in hardware column 0 run code and
return correct results. Each core proves where it ran by writing its
hardware ID (`get_coreid()`, encoded as `column << 16 | row`) into its
output.

| Run | Partition declared | Columns the driver reports | Core placed at (IRON) | Core ID returned | Output |
| --- | --- | --- | --- | --- | --- |
| `id1_base` | 4 wide (as compiled) | 1, 2, 3, 4 | (1, 2) | `0x20002`: column 2, row 2 | correct |
| `id0_base` | 4 wide (as compiled) | 1, 2, 3, 4 | (0, 2) | `0x10002`: column 1, row 2 | correct |
| `id1_w5` | 5 wide | **0, 1, 2, 3, 4** | (1, 2) | `0x10002`: column 1, row 2 | correct |
| `id0_w5` | 5 wide | **0, 1, 2, 3, 4** | (0, 2) | **`0x00002`: column 0, row 2** | correct |
| `full0_w5` | 5 wide | **0, 1, 2, 3, 4** | (0, 2) … (0, 5), row 5 fed through memory tile (0, 1) | **`0x2`, `0x3`, `0x4`, `0x5`** | correct |

`id0_base` and `id0_w5` run the same compiled program: only the partition
metadata in the xclbin differs. The core ID moves from column 1 to column 0.
`full0_w5` uses every compute tile of the hidden column at once, plus its
memory tile, with input streams from shims in columns 1 and 2. All runs
completed normally (`ERT_CMD_STATE_COMPLETED`); there were no timeouts or
device resets.

Two facts that earlier experiments had only inferred are now measured:

* IRON's columns 0-3 are hardware columns 1-4 (the core IDs above).
* The hidden column's compute tiles and memory tile are regular, working
  tiles.

## How it is done

1. Compile a normal IRON design (MLIR-AIE 1.4.3, `npu1` = 4 columns). The
   column-0 designs place cores at IRON column 0 and take their data from
   shims in IRON columns 1 and 2, never from the column-0 shim.
2. Rewrite the xclbin's `AIE_PARTITION` section with `xclbinutil`:
   `column_width` 4 → 5, `start_columns` `[0]`. Column addresses in the
   program are relative to the partition start, so with a partition starting
   at hardware column 0, IRON column *c* becomes hardware column *c*.
3. Load it with pyxrt as usual. `xrt-smi examine -r aie-partitions` shows the
   context holding columns `[0, 1, 2, 3, 4]`.

The partition's `start_columns` value does not choose the placement: AMD's
own overlays in `C:\Windows\System32\AMD` declare `[1]` for 4-wide
partitions, MLIR-AIE emits `[0]`, and both land on columns 1-4. A partition
5 columns wide can only start at column 0.

## What this does and does not give

* **One more column of compute: 20 cores instead of 16**, the count behind
  AMD's 10 TOPS figure for Phoenix (16 TOPS for Hawk Point). Compute-bound
  work could use up to 25% more cores.
* **No more DDR bandwidth.** Column 0's bottom tile is a shim without a DMA
  (MLIR-AIE's old model called it a Shim PL tile "not connected to
  anything"), so data still enters through the 8 DMA channels of columns 1-4.
  The LFM2.5 decode engine (experiment 010) is limited by DDR bandwidth and
  would not get faster. The Parakeet encoder (experiment 012) is partly
  compute-bound and might.
* **Not supported by AMD.** The Linux driver blocks this on purpose, a
  5-column partition leaves no columns for any other program, and a future Windows driver may refuse it. The column-0 shim was not
  touched; writing its registers is untested.

## Safety

Every NPU step ran in its own process under
[`scripts/run_npu_timeout.ps1`](../scripts/run_npu_timeout.ps1) (60 s limit),
in order of risk: context creation only, then the regular-column design in
both partitions, then column-0 designs. No step hung.

## Run it

```powershell
& .\scripts\iron_python.ps1 experiments\015_hidden_column.py build
& .\scripts\run_npu_timeout.ps1 -Script experiments\015_hidden_column.py -ScriptArgs "ctx col1_w5" -Seconds 60
& .\scripts\run_npu_timeout.ps1 -Script experiments\015_hidden_column.py -ScriptArgs "run id0_w5" -Seconds 60
& .\scripts\run_npu_timeout.ps1 -Script experiments\015_hidden_column.py -ScriptArgs "run full0_w5" -Seconds 60
```

Variants: `col1_base`, `col1_start1`, `col1_w5`, `col0_base`, `col0_w5`
(plain +1 designs), `id1_*`, `id0_*` (+1 with the core ID in element 0 of each
256-value block), `full0_base`, `full0_w5` (four cores in one column).
Patched xclbins go to the ignored `cache/015/`. The kernel is
[`kernels/015/hc_id.cc`](kernels/015/hc_id.cc).

Environment: Ryzen 9 8945HS, NPU driver 32.0.20101.3760, firmware
1.5.5.391, XRT 2.21.0, MLIR-AIE 1.4.3, Windows 11, AC power, NPU power
mode Default.
