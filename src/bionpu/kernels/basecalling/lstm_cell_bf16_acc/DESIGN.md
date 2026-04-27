# Mixed-precision LSTM cell on AIE2P — design notes

This document records the design of
`bionpu/kernels/basecalling/lstm_cell_bf16_acc/` — the mixed-precision
sibling of bf16 `lstm_cell_bf16` kernel. scope is
the **AM020 cross-walk follow-up** to : per
`docs/.md` §, 's "wash" finding
(end-to-end max-abs 2.458 vs Padé baseline 2.303) is misdiagnosed
in . The actual root cause is the bf16 recurrent-
state writeback between timesteps, not the multiplier-input
narrowing.

## Cross-walk diagnosis (AM020 §)

> "Native FP32 (supported through emulation using bfloat16)" — Removed
> in AIE-ML.  
> [Appendix A, p. 79, AM020]

> "All floating-point additions are done in one go ... with 23 bits of
> fractional bits."  
> [Ch. 4, p. 65, AM020 — Floating-Point Vector Unit]

The AIE-ML / AIE2P bf16 multiplier accumulates in **FP32** (23
mantissa bits, full single-precision). The MAC isn't the precision
wall. cumulative drift to 2.458 max-abs is consistent with:
- 5 LSTMs × ~200 timesteps = ~1000 stateful steps
- per step: h_t and c_t narrowed from FP32 accumulator → bf16 storage
  (8 mantissa bits) → re-loaded as bf16 input next step
- per narrowing: ~5e-3 quant noise; cumulative across 1000 steps ~ 2.5

## What changed vs 

Same op (one `nn.LSTM(input=96, hidden=96, num_layers=1,
bidirectional=False)` cell over L=334 timesteps). Same DMA topology,
same wire format (bf16 input/weights/output), same sharding,
same activations (`aie::tanh<bfloat16>` + sigmoid identity). **The
only thing that changes is the on-tile storage discipline for h, c,
and gate_acc.**

| aspect | (lstm_cell_bf16) | (lstm_cell_bf16_acc) |
|-----------------|----------------------------------|--------------------------------------|
| h_state | `bfloat16[96]` | `float[96]` (FP32 across timesteps) |
| c_state | `bfloat16[96]` | `float[96]` (FP32 across timesteps) |
| gate_acc | `bfloat16[4][96]` | `float[4][96]` (FP32 per chunk) |
| matmul input | bf16 | bf16 (unchanged — 8-bit mantissa OK) |
| MAC accumulator | FP32 accfloat (silicon mandate) | FP32 accfloat (unchanged) |
| activations | bf16 hw tanh + sigmoid identity | bf16 hw tanh + sigmoid identity |
| writeback | h_t/c_t narrowed to bf16 store | h_t/c_t stored as FP32 |
| weight DMA/call | ~57 MB (half of FP32) | ~57 MB (unchanged) |
| tile-mem buffers| 22.3 KiB (DMA) + ~2 KiB .bss | 22.3 KiB (DMA) + ~3.7 KiB .bss |

Hardware delta: zero. The bf16 multiplier path (40× perf win) is
preserved verbatim; the only change is the post-MAC store dtype for
the persistent state.

## Hardware-supported design vs implementation reality

### What AM020 documents (Ch. 4 p. 67)

The **ideal** design persists h, c in BM accumulator registers
(1024-bit) across timesteps via the AM-to-AM register move primitive
(512 bits/cycle), and cascade-streams layer-N's accumulator state into
layer-N+1's input (512 bits inter-tile, no precision loss).

### What IRON on AIE2P actually exposes

Neither primitive is surfaced at the IRON lowering layer. The
Worker / ObjectFifo abstraction models per-tile compute as a stateless
function with persistent state expressed via static C++ storage in
the kernel object. Inter-tile communication is via ObjectFifo (DMA-
mediated), not via cascade stream.

This is documented as **** + **** in ``:
real RQ4 findings — hardware supports it; toolchain doesn't expose it.

### Precision-equivalent fallback

The `acc.to_vector<float>` conversion path is **hardware-free** on
AIE-ML / AIE2P (AM020 Ch. 4 p. 65 — the FP32 lane width matches the
accumulator's 23 mantissa bits). Holding h, c as FP32 static arrays
in tile DM preserves the same precision invariant as the BM register
path: 23 mantissa bits across all timesteps.

The only delta vs the ideal AM-resident path is cycle-level: an extra
DM round-trip per timestep. Functional precision is identical.

## Storage discipline (lstm_cell_bf16_acc.cc)

```c++
static float h_state[HIDDEN]; // FP32 across timesteps
static float c_state[HIDDEN]; // FP32 across timesteps
static float gate_acc[N_GATES][HIDDEN]; // FP32 per-gate sum
static bfloat16 bias_cache[768]; // bf16 (load-once; 8-bit mantissa OK)
```

Per-step flow:

1. **bf16 multiplier inputs**: `x_t` and `h_state` (the latter narrowed
   from FP32 to bf16 via the `acc.to_vector<bfloat16>` path).
2. **FP32 accumulator MAC**: `gate_acc[g][oc] += sum(bf16_mac)`.
3. **FP32 state update**: `c_t = f * c_old + i * g` computed via
   accumulator; stored as FP32. `h_t = o * tanh(c_t)` same.
4. **bf16 narrowing** only at output stream boundary (the host wants
   bf16 on the wire) and at activation inputs (sigmoid/tanh take
   bf16 vectors). The store paths for h, c, and gate_acc stay FP32.

## Codegen detail: avoiding 's 

's first-pass kernel held `gate_acc` as FP32 stack arrays mixed
with bf16 lane temporaries; Peano AIE2P backend rejected this with
"immediate operand value -184 not multiple of 32". 's
workaround was to demote `gate_acc` to bf16 storage.

 **must** hold `gate_acc` as FP32 (the load-bearing precision
change). This is achieved by:
- All FP32 stores go through `aie::store_v` of `aie::vector<float, VEC>`
  values (16-lane FP32 vector stores; 64-byte boundary-aligned).
- Bias prefix is loaded once, promoted to FP32 via
  `aie::accum<accfloat,VEC>` round-trip, then stored as FP32 into
  gate_acc.
- The chunk MAC reads FP32 gate_acc, adds the FP32 sum from the
  current chunk's lane accumulator, writes FP32 back.

The pattern that triggered was small bf16 stack arrays
interleaved with large FP32 static arrays. pattern is
**uniform FP32 storage** for all state arrays — no mixed-precision
stack temporaries.

## Tile memory walkthrough

Parsed from `build/aie_L334.mlir.prj/main_core_0_2.ld.script`:
- stack: 0x400 = 1024 B
- `weight_in_cons_buff_{0,1}`: 2 × 0x2A00 = 21504 B (depth-2, 5376 bf16)
- `output_out_buff_{0,1}`: 2 × 0xC0 = 384 B
- `input_in_cons_buff_{0,1}`: 2 × 0xC0 = 384 B
- subtotal (DMA buffers + stack): **22784 B = 22.3 KiB** (same as )
- static `.bss` (h_state + c_state in FP32, gate_acc in FP32, bias_cache
  in bf16): 0xf00 = **3840 B** (vs ~2 KiB at bf16 storage)
- **total ≈ 26 KiB on a 64 KiB AIE2P tile budget** (vs 51 KiB FP32
  scalar; vs ~24.5 KiB bf16-only)

The 1.5 KiB increase in .bss is well within the 32 KiB slack 
freed. doesn't reduce the slack budget for future fusion.

## Validation gates (per task brief)

- **Per-cell**: max-abs vs FP32 reference < 1e-3 (1 timestep) — predicted
  by cross-walk for trained weights; expected to be looser on
  synthetic Gaussian weights where multiplier-input drift dominates.
- **Per-stack** (5-LSTM): max-abs < 1e-2 (cumulative).
- **End-to-end encoder**: max-abs strictly less than min('s 2.303,
  's 2.458) by at least 10× (target < 0.23). **Honest reporting
  if the achieved bound is different — the cross-walk's diagnosis is
  falsifiable.**

The end-to-end test on real Dorado weights (vs PyTorch FP32 reference)
is the load-bearing test. The cross-walk predicts ~10-100× improvement
from 2.458 baseline; observed numbers either confirm or
refute that prediction. Per the task brief: "If doesn't
dramatically beat the bf16 baseline, the cross-walk's diagnosis is
wrong and that itself is a real RQ4 finding worth documenting honestly."

## What this task DOES NOT do (deferred)

- **AM-to-AM register-resident state**: requires an IRON
  lowering that exposes the AM020 Ch. 4 p. 67 primitive. Documented
  in as . Cycle-level optimization on top of
  's precision-equivalent fallback.
- **Cascade-stream layer chaining**: requires IRON exposure
  of the cascade primitive (AM020 Ch. 4 p. 67). Documented in
   as . Cycle-level + cross-layer precision
  optimization on top of per-cell FP32 state.
- ** / **: independent (NaN window in stem_L2 / weight DMA
  reduction). Not addressed by .
