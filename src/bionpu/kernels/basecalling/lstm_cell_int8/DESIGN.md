# INT8 LSTM cell on AIE2P — design notes

This document records the design of
`bionpu/kernels/basecalling/lstm_cell_int8/` — the INT8 sibling of
's bf16 `lstm_cell_bf16` kernel. closes (the
"no AIE2P INT8 LSTM kernel" surface 's INT8 sweep needed) and
gates Phase 2's §3.2 ratification and the in-process dispatch
wall.

## What changed vs (bf16)

Same op (one `nn.LSTM(input=96, hidden=96, num_layers=1,
bidirectional=False)` cell over L=334 timesteps). Same DMA topology
(input + output + per-chunk weight stream with prefix folded in).
**The wire format and the math change.**

| aspect | (bf16) | (int8) |
|---------------------|------------------------------|------------------------------------------|
| input wire          | bf16                         | int8 (per-tensor symmetric)              |
| output wire         | bf16                         | int8 (per-tensor symmetric)              |
| weight wire         | bf16                         | int8 (per-channel symmetric)             |
| bias wire           | bf16 (folded into prefix)    | fp32 (folded into prefix)                |
| prefix per chunk    | 1536 B (768 bf16)            | 3136 B (784 fp32 padded for align)       |
| weight slab/chunk   | 10752 B (HIDDEN×HALF_IN bf16)| 4608 B (HIDDEN×HALF_IN int8)             |
| total chunk size    | 10752 B                      | 7744 B                                   |
| MAC accumulator     | accfloat (FP32)              | acc32 (INT32)                            |
| gate accumulator    | bf16 storage                 | INT32 storage                            |
| recurrent state     | bf16 / FP32| **FP32**|
| sigmoid / tanh      | hardware bf16                | hardware bf16 (via INT32→FP32→bf16 dequant)|
| AM020 ops/tile/cyc  | 128 (bf16 + FP32 acc)        | 256 (INT8 path, AM020 Table 14)          |

## Per-channel calibration: where it lives

INT8 LSTM with per-output-channel symmetric weight scales is the
standard PTQ pattern ('s `bionpu/quant/calibrate.py`'s
`per_channel` strategy). The **placement** of the scales is the
load-bearing design choice:

1. **Per-channel scale broadcast at the inner MAC** would force a
   FP32 scale lane-load per output channel inside the int8 inner
   loop, dropping AIE2P's 256 ops/tile/cycle int8 path back to
   ~bf16 throughput. Documented as .
2. **Per-channel scale folded into the bias** preserves the fast int8
   inner loop. The host calibrator picks a per-gate reference scale
   `s_ref[g]` (the mean of the per-channel scales) and rescales each
   weight row's int8 values so they share `s_ref[g]`'s grid. The
   per-channel ratio `s_w[g][oc] / s_ref[g]` is absorbed into the
   int8 weight values by a row-wise rescale + saturating round.

Choice **2** is what this kernel ships. The runner's `expand_wb`
function does the rescale; the on-tile dequant is a single per-gate
FP32 multiply (`per_gate_scale_x[g]`), applied once per timestep at
gate-finalization time.

The cost is a single round-to-nearest step on the row rescale,
producing small per-channel quantization noise. 's PASSPORT
records the achieved per-cell max-abs honestly. The test gate is
< 0.1 per-cell vs bf16 reference on synthetic weights.

## Recurrent state precision: the lesson

Per 's  entry , narrowing `h_t` / `c_t` from
FP32 accumulator output to a smaller storage type per timestep
compounds quantization noise across 334 × 5 ≈ 1670 narrowings. 
shipped at bf16 and measured 2.458 end-to-end max-abs on Dorado fast
weights. switched to FP32 storage and reduced to 2.076 (16%
improvement, less than the cross-walk's predicted 10-100× because
the bf16 multiplier input narrowing is the dominant residual wall).

 INT8 storage of `h_t` / `c_t` would be strictly worse than
's bf16 storage — quantization noise per step ~ 1/256 = 4e-3
vs bf16's 2^-8 = 4e-3 (same magnitude, but the saturating clip on
each store amplifies error on outlier values). The kernel therefore
**holds h_state and c_state at FP32 in tile DM**,
narrowing to int8 only on the W_hh matmul input and the y_t output
writeback.

## Dataflow

Per timestep, per gate, the kernel acquires four chunks (same
half-gate streaming pattern as ; the bias-into-weight-prefix
folding from still applies because INT8 doesn't free a
DMA channel):

1. `W_ih_h0`: `(HIDDEN, HALF_IN) = (96, 48)` int8 — covers `x_t[0:48]`
2. `W_ih_h1`: `(96, 48)` int8 — covers `x_t[48:96]`
3. `W_hh_h0`: `(96, 48)` int8 — covers `h[0:48]` (FP32→int8 narrowing)
4. `W_hh_h1`: `(96, 48)` int8 — covers `h[48:96]` (FP32→int8 narrowing)

16 chunks per timestep × 334 timesteps = 5344 chunks per LSTM call.
Each chunk on the wire is 7744 B (3136 prefix + 4608 weight slab),
total weight-stream volume per call: 5344 × 7744 = 39.7 MB (vs
's 57 MB bf16 — 30% smaller).

## DMA-channel constraint: unchanged

INT8 doesn't free a DMA channel either. The same 2-input limit on
AIE2P CoreTile drives the same prefix-into-weight-stream folding.
The kernel reads scales + biases once on `(t==0, g==0, chunk==0)`
into static caches (`per_gate_scale_*`, `bias_cache`); subsequent
chunks see the same prefix bytes and skip the cache load.

## Tile memory walkthrough

Estimated from the bf16 baseline + per-element-size deltas:

- stack: 0x400 = 1024 B
- `weight_in_cons_buff_{0,1}`: 2 × 0x1E40 = 15488 B (depth-2,
  7744 B chunk)
- `output_out_buff_{0,1}`: 2 × 0x60 = 192 B (depth-2, 96 int8)
- `input_in_cons_buff_{0,1}`: 2 × 0x60 = 192 B (depth-2, 96 int8)
- subtotal (buffers + stack): ~16480 B = 16 KiB
- static `.bss`:
  - `h_state` (96 × 4) + `c_state` (96 × 4): 768 B (FP32)
  - `gate_acc[4][96]` (INT32): 1536 B
  - `bias_cache[768]` (FP32): 3072 B
  - `per_gate_scale_*[4]` + `h_scale` + `y_scale`: 40 B
  - **subtotal**: ~5.4 KiB
- **total ≈ 22 KiB on a 64 KiB AIE2P tile** (similar to 's
  22.3 KiB — INT8 wire savings offset by FP32 bias slab)

The actual numbers will be parsed from
`build/aie_L334.mlir.prj/main_core_*.ld.script` after the first
build; PASSPORT.json's `tile_memory_used_bytes` field will be
updated post-canary.

## What this task DOES NOT do

- **Multi-cell stack**: ships only the single-cell op
  `dorado_fast_lstm_cell_int8`. A 5-layer `lstm_stack_int8`
  composite (sister to 's `lstm_stack_bf16`) is 's
  scope — the §3.2 ratification sweep wires up the int8 stack
  through the encoder and measures end-to-end accuracy.
- **Fork primitives**: uses existing IRON `Worker` +
  `ObjectFifo` (per the task brief — INT8 LSTM is the
  ratify-§3.2 kernel, not the precision-recovery kernel).
   / 's CascadeFifo / AccumFifo are -cascade's
  surface.
- **In-process dispatch**: ships the subprocess-based runner
  (sister to 's pattern). closes the dispatch wall by
  replacing subprocess calls with in-process pyxrt 3.14 calls;
  's NPU op surface is wire-format-compatible with 's
  closure.

## Validation gates (per task brief)

- **Per-cell INT8 max-abs vs bf16 reference < 0.1** on synthetic
  weights (small INT8 quant noise expected; per-channel calibration
  absorbs most). Honest floor documented if exceeded on Dorado
  fast trained weights.
- **End-to-end LSTM stack INT8 max-abs vs FP32 within
  passport-tracked bounds** — bound is the achieved measurement;
  's §3.2 sweep cites this as the INT8 strategy's measured floor.
- **Determinism**: bit-identical output across runs.
- Tests at `tests/test_lstm_int8_npu.py`, marked `pytest.mark.npu`
  for the runtime tests.

## Honest deviation policy

If INT8 quantization noise on Dorado's trained weights exceeds 0.1
max-abs (likely per 's finding that bf16 multiplier-input
narrowing already costs precision on Dorado weights), document the
achieved bound honestly in PASSPORT.json and  entry
. Do NOT relax the test to make it pass — the floor IS
the publishable result per PRD §3 success criterion 3.
