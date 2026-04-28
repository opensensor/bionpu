# bert_int8_matmul — source manifest (v0.4 alpha)

INT8 matmul kernel — the workhorse op for every BERT body matmul in
the DNABERT-Epi AIE2P scorer port. v0 is a single-tile correctness
path; memtile streaming + multi-tile cascade is the v0.5 follow-up
(see `docs/aie2p-scorer-port-design.md` § "Concrete first task").

## Op contract — bionpu Python API (`bert_int8_matmul_head`)

Input | shape | dtype | layout
---|---|---|---
`x`               | M × K | int8 | row-major (token-major)
`w`               | N × K | int8 | output-channel-major (each row = one output channel's weights)
`scales_combined` | N + 1 | float32 | first N entries = `(scales_in * scales_w[n]) / scales_out`; trailing entry reserved for bias (unused in v0)
Output | shape | dtype | layout
---|---|---|---
`y` | M × N | int8 | row-major

## Op contract — AIE2P silicon wire format (compute-tile DMA)

The CoreTile only has 2 input + 2 output DMA channels, so the v0
topology cannot afford 3 separate input ObjectFifos. The host runner
packs `w` and `scales_combined` into a single byte buffer:

```
ws_buf = w.tobytes() + scales_combined.tobytes()  # padded to multiple of 4
```

The kernel slices it back via byte offsets:

```c
const int8_t *w      = ws_buf;
const float  *scales = (const float *)(ws_buf + N * K);
```

This is an implementation detail of the silicon dispatch path. The
bionpu Python API stays unchanged.

Computes:

```
acc[m, n] = Σ_k int32(x[m, k]) * int32(w[n, k])         # i32 accumulator
y[m, n]   = saturate_int8(round(acc[m, n] * combined[n]))
combined[n] = (scales_in * scales_w[n]) / scales_out
```

Symmetric quantization (no zero-point). `combined[n]` is a fused
scale precomputed by the host before dispatch.

## v0 pinned shape (classifier-head specialization)

| field | value | rationale |
|---|---|---|
| `M` | 47 | DNABERT-3 token-pair length |
| `K` | 768 | BERT-base hidden size |
| `N` | 2 | binary off-target classifier head |

Total tile memory at this shape: 47·768 + 768·2 + 2·4 + 47·2 ≈ 37.5 KB
of 64 KB DM — fits a single AIE compute tile.

## Future instances (same kernel C++; different M/K/N constants per build)

| build | M | K | N | weight bytes | tile-fit? |
|---|---|---|---|---|---|
| `bert_int8_matmul_head`   | 47 | 768 | 2    | 1.5 KB | yes (this v0) |
| `bert_int8_matmul_qkvo`   | 47 | 768 | 768  | 576 KB | no — needs memtile streaming (v0.5) |
| `bert_int8_matmul_ffn1`   | 47 | 768 | 3072 | 2.3 MB | no — needs memtile streaming + shim DMA from DDR |
| `bert_int8_matmul_ffn2`   | 47 | 3072 | 768 | 2.3 MB | same |

The Python topology and kernel C++ are parametric in M/K/N; future
instances will be additional `Makefile` targets that lower the same
`.py` with different `--M --K --N` flags.

## Files

| name | role |
|---|---|
| `bert_int8_matmul.py` | IRON-Python lowering. Emits MLIR-AIE for a single compute tile + 4 ObjectFifos (x in, w in, scales in, y out). |
| `bert_int8_matmul.cc` | AIE2P C++ kernel — AIE-API vectorized i8×i8 → i32 dot-product path with scalar host fallback, FP32 fused-scale requantization, INT8 saturate-and-pack. |
| `runner.cpp` | Host C++ XRT runner — file-backed I/O for x/w/scales/y; consumes a small byte format compatible with `bionpu.scoring.quantize.QuantizationPassport.weights.npz` indices. |
| `Makefile` | Builds head specialization (M=47, K=768, N=2) + the host runner. |
| `__init__.py` | Python `NpuOp` registration: `bert_int8_matmul_head`. |
| `MANIFEST.md` | This file. |

## Tile-memory budget (v0 head specialization)

AIE2P: ~64 KiB DM per tile, ~16 KiB program memory per tile.

| resident | bytes | % of 64 KiB |
|---|---|---|
| x (47 × 768 int8) | 36,096 | 55.0% |
| w (2 × 768 int8) | 1,536 | 2.3% |
| scales_in / scales_w / scales_out / combined | 4 + 2·4 + 4 + 2·4 = 24 | <0.1% |
| y (47 × 2 int8) | 94 | 0.1% |
| working / scratch | ~256 | 0.4% |
| **total** | **~38 KB** | **~58%** |

Headroom for `qkvo` / `ffn1` / `ffn2` requires either (a) memtile DMA
of the weight matrix in chunks (compute tile holds K rows of weight at
a time) or (b) multi-tile fan-out with each tile owning a slice of the
N axis. v0.5 picks one based on whichever PRD-1 v0.4 latency budget
allows.

## Build provenance

- **NPU2 flag**: `1` (REQUIRED — AIE2P only).
- **Bring-up env**:
  - XRT 2.23.0 at `/opt/xilinx/xrt/`
  - amdxdna 2.23.0 / NPU firmware 1.1.2.64
  - Architecture `aie2p` (8×4 = 32 compute tiles)
  - `<ironenv>` with Python 3.11+, mlir-aie, llvm-aie

## Status

✅ **Working on AIE2P silicon (v0.4-alpha, first kernel).** Built,
deployed, and verified byte-equivalent vs the numpy host-emulation
reference on the ProArt host (2026-04-28):

```
$ ./bert_int8_matmul --xclbin build/head/final.xclbin \
                     --insts  build/head/insts.bin \
                     --x x.bin --ws ws.bin --out y_npu.bin
$ python3 -c "..."
y_ref bytes: 94, y_npu bytes: 96  (96 = 94 padded to multiple of 4)
first 8 ref: [ 3  1  0 -1  0  1  0  2]
first 8 npu: [ 3  1  0 -1  0  1  0  2]
max |ref - npu|: 0
BYTE-EQUAL: True
```

This validates: the IRON Python topology lowers, the Peano AIE2P
clang++ compiles the scalar kernel, the NPU instruction stream runs
on real silicon, and the per-channel-symmetric INT8 arithmetic
matches the host-emulation reference exactly.

## Build artifacts (v0.4-alpha, head specialization)

| file | bytes | role |
|---|---|---|
| `build/head/aie.mlir`  |  2,938 | IRON-lowered MLIR-AIE topology |
| `build/head/bert_int8_matmul.o` | 1,752 | Peano AIE2P kernel object |
| `build/head/final.xclbin` | 10,314 | NPU-loadable kernel binary |
| `build/head/insts.bin` | 420 | NPU instruction stream |
| `bert_int8_matmul` | 34,160 | XRT host runner |

## v0.4-beta — qkvo specialization (M=47, K=768, N=768)

✅ **Working on AIE2P silicon (2026-04-28).** Built, deployed, and
verified byte-equivalent vs the numpy host-emulation reference on the
ProArt host. The qkvo kernel is the workhorse for the four BERT body
projections (Q, K, V, O), each 47×768 @ 768×768, weight 576 KB INT8.

```
$ ./bert_int8_matmul --variant qkvo \
                     --xclbin build/qkvo/final.xclbin \
                     --insts  build/qkvo/insts.bin \
                     --xs xs.bin --w w.bin --out y_npu.bin
$ python3 -c "<compare row-major output>"
y_ref bytes: 36096, y_npu bytes: 36864  (48 padded rows)
first 8 ref:        [ 1  1  0 -1 -1  0  0 -2]
first 8 npu:        [ 1  1  0 -1 -1  0  0 -2]
max |ref - npu|: 0
mismatch count : 0 / 36096
BYTE-EQUAL: True
```

### Topology

v0.4-beta used N-axis fan-out and produced slice-major output. v0.4-rc
switches to M-axis fan-out: each compute tile owns 12 padded token rows
and all 768 output channels. Because each producer emits a contiguous
row slab, the memtile `join()` flat-concat directly produces row-major
output.

```
shim ──xs──→ broadcast to 4 compute tiles (1 shim MM2S)
shim ──w───→ broadcast to 4 compute tiles (1 shim MM2S)
4 cores ──row slabs──→ memtile ──join(M)──→ shim (1 shim S2MM)
```

### Per-tile DM utilization (v0.4-rc design)

| resident | bytes | role |
|---|---|---|
| qkvo_acc (i32 accumulator) | 36,864 | 12×768, persistent across K loop |
| w_to_core slab (i8) | 6,144 | one 768×8 slab |
| xs_in (i8) | 3,452 | 47×8 x_chunk + 769 fp32 scales |
| y_part out (i8) | 9,216 | one 12×768 row slab |
| qkvo_scales (fp32) | 3,076 | full 768+1 scale vector |
| stack | 1,024 | crt0 |
| **total** | **59,776** | fits 64 KiB tile DM |

The row-slab topology deliberately trades more K chunks for a memtile
join that is already row-major. `K_CHUNK=8` keeps the full 768-output
row-slab state inside tile DM.

The build passes `aiecc --alloc-scheme=basic-sequential` intentionally:
large contiguous buffers such as `qkvo_acc` and the head `x_in` span
multiple 16 KiB banks. Bank-aware allocation only places one logical
buffer inside one bank, so it warns and falls back to the same
sequential layout.

### Build artifacts (v0.4-beta, qkvo specialization)

| file | bytes | role |
|---|---|---|
| `build/qkvo/aie.mlir` | 11,120 | IRON-lowered MLIR-AIE topology |
| `build/qkvo/bert_int8_matmul.o` | 4,680 | Peano AIE2P kernel object (4 entry points) |
| `build/qkvo/final.xclbin` | 29,929 | NPU-loadable kernel binary (4 compute tiles + 1 memtile) |
| `build/qkvo/insts.bin` | 420 | NPU instruction stream |

### Wire format for qkvo silicon dispatch

The IRON topology streams two file-backed buffers; they're chunk-major
to match the K-chunked weight stream:

* `xs.bin` — `K_CHUNKS × xs_chunk_size` = `96 × 3452` =
  **331,392 bytes**.
  Each xs_chunk is `[ M × K_CHUNK int8 x slab | (N+1) fp32 scales ]`.
  The host duplicates the scales prefix on every chunk to keep the
  ObjectFifo element type uniform-sized (the kernel re-stashes scales
  into its tile-local `qkvo_scales` Buffer on every call; only the last
  call's stash is read by finalize, which is correct because the input
  scales never change within a launch).
* `w.bin` — `K_CHUNKS × (N × K_CHUNK)` = `96 × 6144` =
  **589,824 bytes**.
  Each chunk is N rows × K_CHUNK cols of int8 weights, N-major. Host
  must lay this out so that bytes `[c*N*K_CHUNK + n*K_CHUNK ..
  c*N*K_CHUNK + (n+1)*K_CHUNK)` = `w[n, c*K_CHUNK:(c+1)*K_CHUNK]`.

Output:

* `y.bin` — `M_PAD * N` = `48 × 768` = **36,864 bytes**,
  row-major layout. The public API returns `y[:47, :]`.

### Known issues / followups

1. **Closed in v0.4-rc: slice-major output.** The qkvo/ffn group
   topology now splits M rows instead of N channels, so memtile
   flat-concat directly produces row-major output.

2. **Per-tile DM at 99.6%.** Adding any new per-tile state (e.g., a
   bias term, dropout mask) will not fit. v0.5 will need to drop one
   of: the scales-suffix-on-every-chunk pattern (savings ~5 KB on
   xs_in), or shrink the i32 accumulator via M-chunking
   (savings up to half of the 36 KB acc_buf).

3. **Tight memtile budget.** w_to_core split currently uses 4/6 MM2S
   on the memtile. Future scaling to ffn1/ffn2 (N=3072 → 16 N-slices
   of 192 each) won't fit memtile DMA channels. ffn1/ffn2 will likely
   need a different topology (e.g., serialised split of weight slabs
   over time with a 4-tile-fan-out at smaller N, or DDR-streamed
   weight via shim with the memtile only for relay).

## v0.4-rc — ffn1 / ffn2 DDR-streamed weight ABI

✅ **Host dispatch ABI landed.** The Python NpuOps
`bert_int8_matmul_ffn1` and `bert_int8_matmul_ffn2` are registered and
share the qkvo-sized group kernel discipline:

* `ffn1`: `M<=47, K=768, N=3072`. Host streams four 768-output groups.
* `ffn2`: `M<=47, K=3072, N=768`. Host streams one 768-output group
  across 384 K-chunks.

The group wire format keeps each compute tile at a 12-row slab:

* `xs` is group-major, then K-chunk-major. Each element is
  `[M x K_CHUNK int8 x slab | 769 fp32 group scales]`.
* `w` is group-major, then K-chunk-major. Each element is
  `768 x K_CHUNK int8`, N-major within the group.
* `y` is row-major with padded M (`M_PAD=48` for M=47). Host dispatch
  trims padding only.

This is the DDR-streamed weight contract needed for FFN without
placing the 2.3 MB weight in memtile or tile data memory. Hardware
artifact builds use:

```
make NPU2=1 ffn1   # M=47 K=768  N=768 group kernel
make NPU2=1 ffn2   # M=47 K=3072 N=768 group kernel
```

`ffn1` consumes the group kernel four times inside the host-side ABI;
`ffn2` consumes one group with a longer K stream.

## Next sub-tasks (per design doc)

- **v0.4-alpha**: ✅ done — head shape (47 × 768 × 2), single-tile.
- **v0.4-beta**: ✅ done — qkvo shape (47 × 768 × 768), 4-tile + K-chunked.
- **v0.4-rc**: ✅ host ABI and registered ops landed for `ffn1` /
  `ffn2` via DDR-streamed weights. Remaining hardware work is to build
  and ratify the group xclbins under `build/{ffn1,ffn2}/`.
- **v0.4-final**: wire the silicon dispatch (pyxrt host runner adapter)
  for both bert_int8_matmul_head and bert_int8_matmul_qkvo NpuOps so
  `bionpu score --device npu` runs end-to-end on real silicon (the
  NotImplementedError raised by both NpuOps when artifacts are present
  is the v0.4-final landing surface).
- **v0.5**: ✅ source landed — inner dot products use guarded
  AIE-API `::aie::vector` / `aie::mac` intrinsics with the scalar
  loop kept as the host-build fallback. Remaining work is Peano build
  + silicon ratification for the >10× perf target.
