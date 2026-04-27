# LSTM cell on AIE2P — design notes

This document records the design choices behind
`bionpu/kernels/basecalling/lstm_cell/`. Required by the task brief: any
sharding decision needs to live in DESIGN.md so future revisions can
reason about why we shipped a specific structure.

## Op being lowered

One unidirectional `nn.LSTM(input=96, hidden=96, num_layers=1,
bidirectional=False, batch_first=False)` cell over a fixed time
sequence of length L=334. Reverse-direction LSTMs in Bonito's
alternating-direction pattern are implemented host-side as
`flip(input, dim=0) → forward LSTM → flip(output, dim=0)`, matching
's `dorado_fast.py` `LSTMLayer` and the pinned ONNX export.

PyTorch state-dict layout (gate-major, ih before hh):
- `weight_ih_l0`: `(4*96, 96) = (384, 96)` FP32 — gates `i, f, g, o`.
- `weight_hh_l0`: `(4*96, 96) = (384, 96)` FP32 — same gate ordering.
- `bias_ih_l0`,  `bias_hh_l0`: `(4*96,) = (384,)` FP32 each.

Total params per layer: `4*(96*96 + 96*96) + 4*(96 + 96) = 74,496` FP32 =
**291 KiB** of weights/biases per LSTM cell.

## Hardware constraint: 64 KiB tile data memory

AIE2P has 6×8 = 48 compute tiles. Each compute tile has **64 KiB** of
data memory (4 banks × 16 KiB) and **16 KiB** of program memory. The
74,496-FP32 weight set is **far above** any single tile's data memory
budget. We must shard.

## DMA-channel constraint: 2 in, 2 out per CoreTile

Equally hard: a single AIE2P CoreTile has **2 input DMA channels and 2
output DMA channels** (IRON allocates one per ObjectFifo end). With 4
logical fifos (input, weight, bias, output) we run out of channels. We
fold biases into the weight stream as a 768-FP32 prefix on every
weight chunk; the kernel reads them once on `(t==0, g==0, chunk==0)`
into a static `bias_cache[768]` array and skips the prefix on every
subsequent call. This costs `(L * 4 * 4) * 768 = 16,400` FP32 of
host-side bloat per LSTM call (~64 KB) — negligible.

See `` for the upstream bug we'd file against IRON
to lift the per-fifo channel allocation; the workaround is documented
there too.

## Sharding strategy: half-gate streaming

Per timestep, per gate, the kernel acquires **four** half-gate weight
slabs in this order:
1. `W_ih_h0`: `(HIDDEN, HALF_IN) = (96, 48)` FP32 — covers `x_t[0:48]`.
2. `W_ih_h1`: `(96, 48)` FP32 — covers `x_t[48:96]`.
3. `W_hh_h0`: `(96, 48)` FP32 — covers `h[0:48]`.
4. `W_hh_h1`: `(96, 48)` FP32 — covers `h[48:96]`.

Each half is 4608 FP32 = **18 KiB** uncompressed. With each chunk also
carrying the 768-FP32 bias prefix, the on-the-wire chunk is **5376 FP32
= 21 KiB**.

The half-along-input-dim split is the right granularity for v1
because:
- One half (18 KiB) fits comfortably with depth-2 ObjectFifo
  buffering (2×21 KiB = 42 KiB) within the 64 KiB tile budget.
- The matrix-vector product decomposes cleanly along the input
  dimension: `z[oc] = (W[:, 0:48] @ src[0:48]) + (W[:, 48:96] @ src[48:96])`.
- Doubling to full-gate (36 KiB) chunks would require `depth=1`
  ObjectFifo (giving up ping-pong overlap of compute and DMA), and
  with 4 chunks per gate × 4 gates = 16 stalls per timestep.

The kernel maintains accumulators in a static `gate_acc[4][96]` array
across chunk calls within a single timestep. After the **last chunk
of the last gate** (`g==3, chunk_idx==3`), the kernel applies the
sigmoid/tanh nonlinearities, updates the static `h_state[96]` and
`c_state[96]` arrays, and writes `y_t = h_new` to the output fifo.

## Tile-memory walkthrough

Parsed from `build/aie_L334.mlir.prj/main_core_0_2.ld.script`:

| Buffer                       | Size (B) | Notes                          |
|------------------------------|---------:|---------------------------------|
| stack                        |    1024 | crt0 stack                      |
| `weight_in_cons_buff_0`      |   21504 | depth-2 weight chunk (5376 FP32) |
| `weight_in_cons_buff_1`      |   21504 | ...                             |
| `output_out_buff_{0,1}`      |   2×384 | depth-2 output (96 FP32)        |
| `input_in_cons_buff_{0,1}`   |   2×384 | depth-2 input (96 FP32)         |
| `.bss` (h_state, c_state, gate_acc, bias_cache) |  ~5400 | static state |
| **total (≈)**                |  **51 KiB** | within 64 KiB budget    |

Headroom: ~13 KiB for compiler scratch + alignment slack.

## DMA volume per LSTM-layer call

Per call:
- Input: `L * 96 = 32,064` FP32 = **125 KB** (host → device, once)
- Weight: `L * 4 * 4 * 5376 = 28,720,896` FP32 = **110 MB** (host →
  device, once; the host pre-materialises the per-timestep cycle).
   should investigate BD-level repeat patterns to eliminate the
  redundant weight DMA — the underlying weight set is only ~291 KiB.
- Output: `L * 96 = 32,064` FP32 = **125 KB** (device → host, once)

Measured wall-clock per call: **~1.77 seconds** (driven by the 110 MB
weight DMA at ~70 MB/s effective sustained bandwidth — much of it
sequential serialization). 's primary perf target is to drop the
weight DMA volume by two orders of magnitude with bf16 vector
intrinsics + on-tile weight residency.

## Numerical fidelity (FP32, scalar)

The kernel uses a 3rd-order Padé approximation for `tanh` and the
identity `sigmoid(x) = 0.5 * (tanh(0.5x) + 1)` for the gate
nonlinearities. Max error of the `tanh` approximation on `[-3, 3]`:
~3e-3. This is the dominant source of LSTM-vs-CPU drift, ~1e-1 to ~1e0
absolute over the 5-layer stack on Dorado weights.

 should replace with `aie::tanh<float>` or `aie::tanh<bfloat16>`
vector intrinsics (already available in `aie_kernels/aie2p/silu.cc`,
`gelu.cc`); that path is expected to bring the per-block bound down to
the 1e-4 task-brief gate.

## Why not multi-tile sharding for v1

A multi-tile design (e.g. one tile per gate, or one tile per hidden-dim
slab) would distribute the weight footprint and avoid the 110 MB DMA.
We don't ship it for because:
- Multi-tile coordination adds a second axis of bug surface; v1's
  goal is correctness-first.
- The per-call latency is **not** the validation gate — the gate is
  per-block FP32 max-abs error vs CPU. Latency optimization is .
- The IRON multi-tile examples (`programming_examples/basic/
  matrix_multiplication/whole_array/`) target GEMM, not RNN with
  recurrent state propagation. Adapting them to an LSTM cell is non-
  trivial — gate accumulators and h/c need to flow between tiles
  per timestep, which IRON's ObjectFifo-driven dataflow doesn't
  natively support without inter-tile fifos. See ``
   for the absent IRON RNN/recurrent-multi-tile pattern.
