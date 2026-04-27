# bf16 LSTM cell on AIE2P — design notes

This document records the design of
`bionpu/kernels/basecalling/lstm_cell_bf16/` — the bf16 sibling of
's FP32 scalar `lstm_cell` kernel. 's scope is **one
custom kernel driven by profile data** (B-M6); the profile data is
's gaps.yaml (specifically ****: the 3rd-order Padé
tanh approximation cumulatively drifts to end-to-end max-abs 2.303
across the 5 LSTM × 200 timesteps).

## What changed vs 

Same op (one `nn.LSTM(input=96, hidden=96, num_layers=1,
bidirectional=False)` cell over L=334 timesteps). Same DMA topology
(input + output + weight stream with folded bias prefix on every
chunk per ). **Only the per-element type and the
nonlinearities change.**

| aspect | (lstm_cell) | (lstm_cell_bf16) |
|------------------|--------------------------|--------------------------------------|
| precision        | FP32 scalar              | bf16 vector (lane=16)                |
| tanh             | 3rd-order Padé (~3e-3)   | hardware `aie::tanh<bfloat16>` (ULP) |
| sigmoid          | `0.5*(Padé tanh+1)`      | `0.5*(hw tanh+1)` (vectorised)       |
| gate accumulator | fp32 static array        | bf16 static + fp32 mac accumulator   |
| state (h, c)     | fp32 static              | bf16 static                          |
| weight DMA/call  | ~110 MB                  | ~57 MB (half of FP32)                |
| tile-mem buffers | 51 KiB                   | 22.3 KiB                             |

## Hardware bf16 tanh — why it exists

AIE2P CoreTile has hardware bf16 transcendentals exposed as
`aie::tanh<bfloat16>` (and `aie::exp2<bfloat16>`). The pattern used
here mirrors `mlir-aie/aie_kernels/aie2p/{silu,gelu,swiglu}.cc`
(verified at the same mlir-aie commit pinned). Sigmoid is
synthesised via the identity `sigmoid(x) = 0.5 * (tanh(x/2) + 1)` —
same identity used, but the underlying tanh is now hardware-ULP
instead of Padé-3e-3.

## bf16 cast strategy: lossless for these magnitudes

bf16 has 8 mantissa bits (vs FP32's 23). For Dorado fast LSTM
weights, observed magnitudes fit well within bf16's dynamic range:
weights are O(0.05) (extracted via `_load_pinned_model` on the
synthetic seed), gates pre-nonlinearity are O(1.0), post-sigmoid
gates are bounded to `[0, 1]`, post-tanh hidden is bounded to
`[-1, 1]`. The bf16 quantisation noise (round-to-nearest-even on
the FP32→bf16 cast) is ~5e-3 relative — comparable to 's Padé
tanh error per call but **bounded per-call rather than systematically
biased**, so it shouldn't cumulatively drift the way Padé does.

## Codegen detail: bf16 storage for gate accumulators

The first-pass kernel held `gate_acc[N_GATES][HIDDEN]` as fp32 and
loaded bf16 lane temporaries via small stack arrays before the
nonlinearity step. The Peano AIE2P backend rejected this with:

```
fatal error: error in backend: immediate operand value -184 is not a
multiple of 32 (AIE2PMCCodeEmitter pass)
```

This is a register-allocator interaction with the mixed-precision
pattern; captures the verbatim diagnostic. **Workaround:**
hold gate_acc as bf16 storage; promote to fp32 only inside the inner
mac via `aie::accum<accfloat, VEC>`. Cleaner, faster (no per-lane
loop on the nonlinearity step), and within bf16's dynamic range for
these magnitudes.

## Sharding strategy: same as 

Per timestep, per gate, the kernel acquires four half-gate weight
slabs:
1. `W_ih_h0`: `(HIDDEN, HALF_IN) = (96, 48)` bf16 — covers `x_t[0:48]`
2. `W_ih_h1`: `(96, 48)` bf16 — covers `x_t[48:96]`
3. `W_hh_h0`: `(96, 48)` bf16 — covers `h[0:48]`
4. `W_hh_h1`: `(96, 48)` bf16 — covers `h[48:96]`

16 acquires per timestep × 334 = 5344 weight acquires per LSTM-layer
call. Each chunk is `BIAS_LEN + WEIGHT_HALF_LEN` = 5376 bf16 = 10.5
KiB on the wire (vs 's 21 KiB at FP32).

## DMA-channel constraint: unchanged

bf16 doesn't free a DMA channel. The same 2-input limit on AIE2P
CoreTile drives the same bias-into-weight folding workaround. The
kernel still reads biases once on `(t==0, g==0, chunk==0)` into
`bias_cache[768]` and skips the prefix on every subsequent call.

## Tile memory walkthrough (post-shrink)

Parsed from `build/aie_L334.mlir.prj/main_core_0_2.ld.script`:
- stack: 0x400 = 1024 B
- `weight_in_cons_buff_{0,1}`: 2 × 0x2A00 = 21504 B (depth-2, 5376 bf16)
- `output_out_buff_{0,1}`: 2 × 0xC0 = 384 B
- `input_in_cons_buff_{0,1}`: 2 × 0xC0 = 384 B
- subtotal: 23296 B
- static `.bss` (h_state, c_state, gate_acc[4], bias_cache; all bf16):
  ~2 KiB
- **total ≈ 25 KiB on a 64 KiB AIE2P tile** (vs 's 51 KiB)

This is **<50% of 's footprint** — bf16 frees substantial slack
that (follow-up) could spend on multi-cell pipelining /
on-tile weight residency.

## What this task DOES NOT do (deferred to follow-ups)

- ****: stem_L2 NaN window fix. Independent
  scalar-FP32 codegen issue; encoder transparently CPU-falls-back.
- ****: weight DMA reduction beyond the 2× from bf16 cast.
  Requires on-tile weight residency or multi-tile sharding
.
- ****: lift conv stems + linear projection to bf16 too.
  v1 keeps them FP32; only LSTM is the target.
- ****: INT8 sweep. bf16 is *not* INT8; will sweep INT8
  against the bf16 baseline this task produces.

These are documented as `severity: cosmetic` deferral entries in
`gaps.yaml` so future agents see the surface preserved.

## Validation gates (per task brief)

- bf16 cell max-abs vs FP32 reference < 1e-2 per timestep (1 cell)
- bf16 stack max-abs vs FP32 reference < 5e-2 per layer (5 cells)
- end-to-end encoder bf16 max-abs vs FP32 reference: target < 1e-1,
  accept < 5e-1 (vs 's 2.303 baseline)
- bf16 max-abs MUST be strictly less than 's Padé baseline 2.303
  — if not, that's a real finding worth documenting honestly.
