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
| `bert_int8_matmul.cc` | AIE2P C++ kernel — scalar i8×i8 → i32 inner loop, FP32 fused-scale requantization, INT8 saturate-and-pack. **No vector intrinsics in v0** — correctness first; the AIE-API vectorization is the v0.5 perf pass. |
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

## Next sub-tasks (per design doc)

- **v0.4-alpha**: this skeleton + verification on the head shape
  (47 × 768 × 2). Uses scalar intrinsics; perf is irrelevant.
- **v0.4-beta**: scale to `qkvo` shape (47 × 768 × 768) via
  memtile-resident weight or N-axis tile fan-out.
- **v0.4-rc**: scale to `ffn1` / `ffn2` (47 × 768 × 3072) via DDR
  weight streaming. Becomes the latency bottleneck of the port.
- **v0.5**: switch from scalar to AIE-API ::aie::vector intrinsics
  for the inner loop; targeting >10× speedup based on basecalling
  track precedent.
