# linear_projection — call-fusion design (post-cascade speedup lever)

**Status (2026-04-26):** sketch, not implemented. Filed as part of
`bionpu/kernels/basecalling/conv_stem_layers_2_3/` .
This document describes what the kernel *should* look like to deliver
the post-cascade encoder throughput we need; the next concrete step is
to build it and silicon-test against the existing artifact.

## Why

Pre-cascade Dorado-`fast` measurements (`results/basecalling/b-m6/
measurements.json`):

```
linear_projection: 564 ms  /  1336 calls  =  ~422 µs / call
```

Each call is a 96 × 64 GEMM (~6 k MACs). At AIE2P 50 TOPS that math
should take ~120 ns; the rest of the 422 µs is per-call setup —
ObjectFifo acquire, weight chunk DMA, output BD release, kernel-frame
prologue. With LSTM cascade landing, this op (and `conv_stem_layer3`
with the same shape pathology) becomes ~99 % of NPU encoder wall time.

## Current shape (recap)

`linear_projection.py` declares per-call ObjectFifos:

- `in_step_ty = (HIDDEN=96,) fp32` — one timestep input per acquire
- `weight_chunk_ty = (OC_GROUP_SIZE=64 × HIDDEN=96 = 6144,) fp32` — one
  output-group weight slab per acquire (24 KiB, fp32)
- `out_chunk_ty = (OC_GROUP_SIZE=64,) fp32` — one output group per release

Inner loop (`core_body`):

```python
for t in range_(L):                      # 334 timesteps
    elem_in = of_input.acquire(1)
    for g in range_(N_OC_GROUPS):        # 4 output groups
        elem_w   = of_weight.acquire(1)
        elem_out = of_output.acquire(1)
        linear_fn(elem_in, elem_w, elem_out)   # ← 1336 total invocations
        of_weight.release(1)
        of_output.release(1)
    of_input.release(1)
```

The per-call setup cost is paid 1336 times per dispatch.

## Target shape — single-call streaming kernel

Tile L1 capacity: ~64 KiB on AIE2P CoreTile. Full weight tensor
geometry:

```
W = OUT_DIM × HIDDEN = 256 × 96 = 24576 elements
  fp32:  98304 B  =   96 KiB   ✗ does not fit
  bf16:  49152 B  =   48 KiB   ✓ fits with ~16 KiB headroom
```

Per-timestep working set: 96 inputs (384 B fp32 / 192 B bf16) + 256
outputs (1 KiB / 512 B) + accumulators. Comfortable.

Plan:

1. **Convert weights to bf16 at host build time.** This is precision-
   neutral on AIE2P (multiplier already narrows fp32 inputs to bf16 per
   AM020 Appendix A; see root-cause analysis).

2. **Single weight acquire per dispatch.** Replace the streamed
   per-group weight ObjectFifo with one acquire of the full bf16
   weight slab (48 KiB) at kernel start, kept in a static array on
   tile DM.

3. **Internal timestep loop in the kernel.** The core call becomes:

   ```python
   def core_body(of_input, of_weight, of_output, linear_fn):
       elem_w_full = of_weight.acquire(1)            # 48 KiB once
       for t in range_(L):                           # 334 timesteps
           elem_in  = of_input.acquire(1)            # 96 bf16
           elem_out = of_output.acquire(1)           # 256 bf16
           linear_fn(elem_in, elem_w_full, elem_out) # one call/ts
           of_input.release(1)
           of_output.release(1)
       of_weight.release(1)
   ```

   Total kernel function calls: **334** (one per timestep) instead of
   1336 — a 4× call-count reduction with no math change.

4. **Optional further fusion: pull L into the kernel.** Stage 4 is
   one-call-per-dispatch:

   ```python
   def core_body(of_input, of_weight, of_output, linear_fn):
       linear_fn(of_input, of_weight, of_output, L)  # one call/dispatch
   ```

   The kernel internally walks the fifos for L timesteps. Total kernel
   function calls: **1** per dispatch — ~1336× call-count reduction.

5. **Kernel body change** (`linear_projection.cc`): replace the
   single-tuple computation with the timestep loop. Weight is bf16
   tile-DM static; input/output are streamed via the acquired fifo
   handles.

## Estimated wall-time after each stage

These are bounded above by GPU's 84 µs (single dense matmul) and
below by our token-order microtest's ~3.4 µs/timestep infrastructure
floor at 334 production-shaped timesteps:

| Stage | Calls/dispatch | Per-call setup share | Estimated wall time |
|------:|---------------:|---------------------:|--------------------:|
| current (b-m6) | 1336 | ~422 µs each | 564 ms |
| stage 3 (per-ts call) | 334 | ~422 µs each | ~141 ms (4× speedup) |
| stage 4 (per-dispatch call) | 1 | one setup amortized over 334 ts | **~few ms** |

Stage 4 is the production target. Stage 3 is a useful intermediate if
the kernel-internal-fifo-walk pattern needs separate validation.

## Risks / open questions

1. **Weight ObjectFifo depth + acquire semantics with a 48 KiB single
   chunk.** Upstream `programming_examples/basic/matrix_multiplication/
   single_core` uses a similar one-shot weight load. Verify the IRON
   ObjectFifo lowering produces a single shim DMA → tile-DM transfer
   (not a chunked one) for a 48 KiB element.
2. **bf16 weights + fp32 inputs.** The existing kernel takes fp32
   weights. Converting host-side and adjusting the kernel's accumulator
   precision has to preserve the existing accuracy gate (encoder
   max-abs vs FP32 reference). Compare against the LSTM bf16-mixed-fp32
   kernel which already does this; same precision contract should hold.
3. **Clamp-to-[-5, 5] is the existing kernel's tail step.** Keep it
   inside the timestep loop; it's free.

## Validation plan

1. Build stage 3 (per-timestep fused call) artifact alongside the
   existing per-group artifact. Both register under different op
   names so encoder.py can pick.
2. Silicon-test stage 3 against synthetic random inputs; assert
   byte-equal to the existing artifact (after bf16 conversion of
   weights). Measure wall time. Expected: ~141 ms; if higher, the
   per-call setup share was overestimated.
3. If stage 3 lands clean, build stage 4 (single-call kernel-internal
   loop). Same validation; expected wall time in low-ms.
4. Update `results/basecalling/b-m6/measurements.json` with the new
   per-block latency for the chosen stage; the gap entry 
   in `conv_stem_layers_2_3/` flips from `severity: performance`
   to `severity: workaround` or `severity: closed` based on outcome.

## Out of scope for this kernel

- Cross-op fusion (e.g. fusing linear_projection with the LSTM cascade
  tail) is a separate larger redesign — reasonable to consider in a
   production rebuild but not necessary for this gap.
- `conv_stem_layer3` has the same call-count pathology (2004 calls /
  dispatch) but a more constrained memory budget — its weight is 117 KiB
  fp32 / ~58 KiB bf16, which does fit a single tile in bf16 but with
  much less headroom, and the stride-6 input-access pattern is more
  intricate. Filed under as the second target; design lives
  in that kernel's directory, not here.
