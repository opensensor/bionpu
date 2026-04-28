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
$ python3 -c "<unscramble slice-major to row-major; compare>"
y_ref bytes: 36096, y_npu bytes: 36096
first 8 ref:        [ 1  1  0 -1 -1  0  0 -2]
first 8 npu (unsr): [ 1  1  0 -1 -1  0  0 -2]
max |ref - npu|: 0
mismatch count : 0 / 36096
BYTE-EQUAL: True
```

### Topology

4 compute tiles, N-axis fan-out (192 channels per tile), K-chunked
along the reduction axis (K_CHUNK=64 → 12 chunks per launch). All
per-tile state is held in IRON `Buffer` objects (the i32 accumulator
and the per-tile fp32 scales-stash) to escape the tight `.bss` budget
that static C++ globals would face.

```
shim ──xs──→ broadcast to 4 compute tiles (1 shim MM2S)
shim ──w───→ memtile ──split(N)──→ 4 compute tiles (1 shim MM2S, memtile resident relay)
4 cores ──y_part──→ memtile ──join(N)──→ shim (1 shim S2MM)
```

### Per-tile DM utilization (actual, observed in linker layout)

| resident | bytes | role |
|---|---|---|
| qkvo_acc (i32 accumulator) | 36,096 | persistent across K loop |
| w_to_core slice (i8) | 12,288 | one 192×64 slab |
| xs_in (i8) | 6,084 | x_chunk + scales prefix |
| y_part out (i8) | 9,024 | one 47×192 partial |
| qkvo_scales (fp32, per-tile slice) | 772 | this tile's 192+1 entries |
| stack | 1,024 | crt0 |
| **total** | **65,288** | **~99.6% of 64 KiB tile DM** |

The design doc's predicted budget of 51 KB underestimated by ~13 KB
because (a) scales are packed into xs_chunk via 3076-byte tail bytes
duplicated per chunk, growing xs_chunk from 3008 → 6084 B, and (b) the
per-tile fp32 scales-stash + 1024-byte stack weren't counted. The
tighter 99.6% utilization is acceptable for v0.4-beta; v0.4-rc / v0.5
will need to either drop the scales-suffix-on-every-chunk pattern (if
shim DMA programming gets a 3rd MM2S slot via memtile-routing) or move
the i32 accumulator to a tile-shared memtile slot.

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

* `xs.bin` — `K_CHUNKS × xs_chunk_size` = `12 × 6084` = **73,008 bytes**.
  Each xs_chunk is `[ M × K_CHUNK int8 x slab | (N+1) fp32 scales ]`.
  The host duplicates the scales prefix on every chunk to keep the
  ObjectFifo element type uniform-sized (the kernel re-stashes scales
  into its tile-local `qkvo_scales` Buffer on every call; only the last
  call's stash is read by finalize, which is correct because the input
  scales never change within a launch).
* `w.bin` — `K_CHUNKS × (N × K_CHUNK)` = `12 × 49152` = **589,824 bytes**.
  Each chunk is N rows × K_CHUNK cols of int8 weights, N-major. Host
  must lay this out so that bytes `[c*N*K_CHUNK + n*K_CHUNK ..
  c*N*K_CHUNK + (n+1)*K_CHUNK)` = `w[n, c*K_CHUNK:(c+1)*K_CHUNK]`.

Output:

* `y.bin` — `M * N` = `47 × 768` = **36,096 bytes**, **slice-major
  layout**: tile i contributes the contiguous byte block at
  `[i * M * N_PER_TILE .. (i+1) * M * N_PER_TILE)`, which is row-major
  47×192 within the slice. To reconstruct row-major `y[M, N]`:
  ```python
  y_full = np.empty((M, N), dtype=np.int8)
  for i in range(N_TILES):
      sl = y_npu[i*M*N_PER_TILE:(i+1)*M*N_PER_TILE].reshape(M, N_PER_TILE)
      y_full[:, i*N_PER_TILE:(i+1)*N_PER_TILE] = sl
  ```

### Known issues / followups

1. **Slice-major output requires host post-processing.** The memtile
   join lays four 47×192 partials end-to-end; the natural row-major
   reconstruction is a Python-side reshape+stitch. v0.5 should
   either (a) program the memtile join with strided offsets that
   produce row-major directly, or (b) move the unscramble into the
   host runner.

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

## Next sub-tasks (per design doc)

- **v0.4-alpha**: ✅ done — head shape (47 × 768 × 2), single-tile.
- **v0.4-beta**: ✅ done — qkvo shape (47 × 768 × 768), 4-tile + K-chunked.
- **v0.4-rc**: scale to `ffn1` / `ffn2` (47 × 768 × 3072) via DDR
  weight streaming. Becomes the latency bottleneck of the port.
- **v0.4-final**: wire the silicon dispatch (pyxrt host runner adapter)
  for both bert_int8_matmul_head and bert_int8_matmul_qkvo NpuOps so
  `bionpu score --device npu` runs end-to-end on real silicon (the
  NotImplementedError raised by both NpuOps when artifacts are present
  is the v0.4-final landing surface).
- **v0.5**: switch from scalar to AIE-API ::aie::vector intrinsics
  for the inner loop; targeting >10× speedup based on basecalling
  track precedent.
