# lstm_cell_bf16_acc_cascade — design + kernel-author hand-off (T7-IRON)

**Status (2026-04-26 post-Followup-G + retirement of the keepalive
hypothesis)**: remains OPEN. Fix C is silicon-EXECUTED and
silicon-FALSIFIED — achieves the Followup E Stage 1-validated
`0 jnz around vmov mcd/scd` per-tile-ELF invariant but does NOT lift
the wedge. The follow-on "cascade-port active-call ratio is the
load-bearing variable" explanation (Stage 1 reproducer 100 % active = PASS;
Fix C production ~6–12 % active = WEDGE) is now recorded as an
**over-fit on N=2**, not a proven root cause — the same kind of
single-PASS-vs-single-WEDGE inference that Followup C made about
`jnz`-around-cascade and that Fix C subsequently falsified. 
has been **re-scoped**: it no longer prescribes a cascade-port
keepalive; it now tracks a contract/topology audit (AccumFifo
acquire/release token ordering, producer/consumer balance, and
horizontal-hop (0,2)→(1,2) validity under production-shaped traffic).

Status timeline:

- 2026-04-25 post-T1-swarm: design + Python topology landed; C++
  kernel authored; xclbin builds; ** (the AIE2P 5-slot
  host-arg ABI cap blocker) is closed** via consolidated weight DMA
  through memtile-split-by-offset (4+1 layout shown below). Silicon
  ABI ratification: PASS (host call submits cleanly with 3 BOs — was
  7). Functional silicon test surfaces a NEW firmware-timeout issue
 — distinct
  from 's ABI scope.
- 2026-04-26 Followup C: MLIR + ELF diff localises the wedge to
  peano-emitted ``jnz`` instructions around ``vmov mcd`` / ``vmov scd``
  (cascade-port intrinsics) on cascade-configured AIE2P tiles. The
  wedge correlates 1:1 with branched cascade-port emission; the
  hypothesis is an AIE2P firmware-side cascade-port watchdog with an
  idle-ratio threshold somewhere in (10 %, 94 %].
- 2026-04-26 Followup E Stage 1: silicon-validates Fix B
  (collapse 16-call IRON loop to 1 call/ts, emit cascade ops
  unconditionally) on Item 4's minimal reproducer
  (``tests/cascade_bisection/topology_5tile_lstm_call_pattern/``,
  branch ``followup/E-fix-b-collapse-lstm-cascade`` commit
  ``f875ebe``): per-tile cascade-fn jnz drops from 3 to 0; silicon
  PASS in 1890us with element-by-element correct cascade round-trip
  across L=334 timesteps × 96 floats/ts. Hypothesis confirmed on the
  math-free reproducer.
- 2026-04-26 Followup G: Fix C applied to the production LSTM
  (``lstm_layer_cascade.cc`` split into 8 ``extern "C"`` functions
  per cascade-presence; ``lstm_cell_bf16_acc_cascade.py`` IRON loop
  unrolls inner ``g``/``chunk`` loops at IRON-build-time and statically
  selects which kernel symbol to call at each of the 16 (g, chunk)
  call-sites per ts). Per-tile ELF disasm verifies all 5 cascade-firing
  functions on all 5 tiles have 0 jnz; only one math-only function
  (``_first_math`` on tile 0_5, 3 jnz) contains any jnz and that
  function has 0 cascade ops — same per-tile-ELF invariant Followup E
  silicon-validated. **Silicon outcome: WEDGE** with identical
  fingerprint (DPU PC=0xffffffff, Context PC=0x28b060ad,
  msg=0x1d000001, ~5s per-iter wall, output all-zero, dmesg
  ``aie2_dump_ctx``). Evidence at
  ``state/followup-g/silicon-20260426T025820Z.json`` and the per-tile
  ELF disasm/jnz-invariant snapshots
  (``state/followup-g/jnz-invariant-20260426T025506Z.json``).

**Refined posture (post-Followup-G, post-keepalive-retirement)**: the
Followup E "0-jnz-around-cascade" invariant is necessary but not
sufficient. Beyond that, Followup G's two-point comparison (Stage 1
reproducer at 100 % cascade-active vs Fix C production at ~6–12 %
cascade-active) is **not strong enough to prescribe a fix** — the two
samples differ in primitive contract, token ordering, and topology, so
the active-call ratio is a possible correlate, not a verified cause.
The naive "emit ``vmov mcd, scd`` keepalive in every ``_math`` call"
patch is also **not semantics-preserving**: the production cascade
schedule has downstream tiles ``get_scd``-blocking on what should be
real ``put_mcd`` traffic, and injecting keepalive tokens into those
slots feeds dummy data where layer input is expected.

**Production-side status (Fix C silicon-tested 2026-04-26)**: WEDGE.
Fix C is filed-and-falsified. Two known forward paths:
- **Fix B production**: collapse 16-call IRON
  loop to 1 call/ts AT THE PRODUCTION SCALE — requires the L1 /
  memtile weight-DMA restructure that Stage 2 scoping rejected as
  infeasible on AIE2P CoreTile L1 (~64 KB) / single-memtile (~512 KB)
  capacity. Gated on first identifying a correct cascade
  schedule.
- **Contract/topology audit**: test
  the actual silicon contract before any "fix" is prescribed —
  AccumFifo acquire/release token ordering, producer/consumer balance,
  and whether the horizontal hop (0,2)→(1,2) is valid under the same
  stream/DMA pressures as production. **Replaces** the retired
  cascade-passthrough/keepalive hypothesis.

The cross-walk's < 0.23 silicon-level prediction remains UNTESTED on
silicon — closing still requires the audit to
identify a correct cascade schedule (and, if applicable, Fix B
production at that schedule).

See ``gaps.yaml`` for the full + + 
narrative and ``bionpu/iron_extensions/INVENTORY.md`` for the original
toolchain finding this is the silicon-level engineering follow-up of.

## What this kernel is

A 5-layer cascade-stream-driven LSTM stack for Dorado fast@v5.0.0,
testing the AM020 cross-walk's strong silicon-level prediction
("encoder max-abs < 0.23 with cascade-stream inter-layer plumbing,
vs IRON-level fallback's 2.076"). The Python-level topology
extension (``bionpu/iron_extensions/cascade_stream.py``) was the
T7-IRON deliverable; this kernel is the engineering hand-off that
turns that toolchain finding into a measurable silicon result.

Math contract: identical to (bf16 multiplier inputs, FP32
accumulator state). Difference: where 's stack chains 5 cells
via host-side dispatch (each layer's bf16 output round-trips through
XRT), this variant chains 5 cells via **inter-tile cascade stream**:
layer N's output FP32 accumulator is cascaded directly to layer N+1's
input on a vertically-adjacent CoreTile in the same column.

## Topology — 4+1 layout (post- + post-)

The original DESIGN above (5 vertically-adjacent CoreTiles in column
0, rows 2..6) is silicon-infeasible on AIE2P: NPU2's column has only
4 CoreTile rows. The realised topology spans 2 columns:

```
   col 0 row 5:  Layer 0  (FIRST role,  put_only)   — bottom of col 0
   col 0 row 4:  Layer 1  (MIDDLE role, put_get)
   col 0 row 3:  Layer 2  (MIDDLE role, put_get)
   col 0 row 2:  Layer 3  (MIDDLE role, put_get)    — top of col 0
   col 1 row 2:  Layer 4  (LAST role,   get_only)   — horizontal hop
```

4 cascade hops (all via ``aie.iron.AccumFifo`` lowering to
``aie.cascade_flow``):

- (0,5) → (0,4): vertical-up (south-of-source → dest)
- (0,4) → (0,3): vertical-up
- (0,3) → (0,2): vertical-up
- (0,2) → (1,2): horizontal-east (west-of-source → dest)

Direction constraint enforced by ``AIELowerCascadeFlows.cpp:63``
(source must be South-of OR West-of dest). The chain runs upward
then turns east. Bonito's alternating-direction pattern is preserved
by data layout (even-indexed layers see input in forward time order;
odd-indexed layers see input reversed; flip is host-side at input
preparation time).

## Host-visible buffer contract (post- — 3 BOs total)

```
  bo_input    : memref<32064xbf16>      = L * INPUT_DIM (334 * 96)
  bo_weights  : memref<143646720xbf16>  = L * N_GATES * 4 * N_LAYERS * CHUNK_LEN
                                        = 334 * 4 * 4 * 5 * 5376
  bo_output   : memref<32064xbf16>      = L * HIDDEN (334 * 96)
```

The consolidated weights buffer is built host-side by
``__init__.py::_pack_consolidated_wb_per_layer``. On-wire layout
is layer-minor / chunk-major: for each chunk-frame f in
``[0, n_weight_chunks)`` (where ``n_weight_chunks = L * N_GATES * 4
= 5344``), N_LAYERS chunks back-to-back
(``[L0_chunk_f, L1_chunk_f, ..., L4_chunk_f]``).

The IRON-emitted memtile-split splits each parent frame by per-layer
offset and dispatches into 5 per-cell L2→L1 sub-fifos:

```
  aie.objectfifo.link [@weight_in_all]
    -> [@weight_in_L0, @weight_in_L1, @weight_in_L2, @weight_in_L3, @weight_in_L4]
       ([] [0, 5376, 10752, 16128, 21504])
```

Each cascade-stage core acquires from its own ``weight_in_L<i>``
sub-fifo presenting one ``chunk_with_bias_ty`` (``memref<5376xbf16>``)
per acquire — unchanged from the pre- per-layer wire format,
so the existing per-role C++ kernels work unmodified.

This pattern mirrors the upstream
``programming_examples/basic/matrix_multiplication/cascade/cascade.py``
reference design (one shim DMA per col, fanned out at the memtile
via ``object_fifo_link``).

## Max-abs threshold (cross-walk hypothesis)

Per the cross-walk (``docs/.md`` ):

- 's IRON-level baseline: encoder max-abs ≈ 2.076 vs FP32
- Cross-walk strong prediction (cascade closes the recurrent-state
  writeback narrowing wall): max-abs **< 0.23** on synthetic weights
- Cross-walk hard floor (cascade isn't supposed to ADD drift): ≥ 
  / 1.5 ≈ 3.1 would refute both the cross-walk + multi-tile design

**As of T1-swarm closure (2026-04-25)** the silicon-level kernel
itself wedges the firmware, so the cross-walk's strong
prediction remains UNTESTED on silicon. The 3-BO host call submits
cleanly but the multi-tile cascade kernel
never produces output. Decisive measurement requires 
resolution first.

## Three C++ kernel variants needed

Per the upstream ``aie_kernels/aie2/cascade_mm.cc`` pattern:

### Variant 1: ``lstm_layer_cascade_put_only`` (FIRST role / row 2)

```c++
// FIRST role — Layer 0 of the stack. Receives input from ObjectFifo
// (from the conv stem / projection, the upstream block in the
// encoder), computes its own LSTM forward, and emits the post-cell
// FP32 hidden state to the cascade.
extern "C" void
dorado_fast_lstm_layer_cascade_put_only(bfloat16 *x_in, bfloat16 *w_in,
                                         /* no out via ObjectFifo */) {
  // ... standard LSTM forward over L timesteps with FP32
  //     accumulator state (h, c) ...
  for (int t = 0; t < L; ++t) {
    // ... gate computations (4 gates × 4 chunks each, shape) ...
    // ... compute new h, c at FP32 accumulator precision ...
    aie::accum<accfloat, 16> h_acc = ...;
    aie::accum<accfloat, 16> c_acc = ...;

    // Cascade out — 512 bits per put_mcd, two halves of HIDDEN=96
    // requires 6 puts for h_acc (96/16) and 6 for c_acc.
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      put_mcd(h_state_lane[v]);  // h slice
      put_mcd(c_state_lane[v]);  // c slice
    }
  }
}
```

### Variant 2: ``lstm_layer_cascade_put_get`` (MIDDLE role / rows 3..5)

```c++
// MIDDLE role — Layers 1..3 of the stack. Receives input via
// cascade from the previous layer, computes its own LSTM forward,
// and emits to the next layer via cascade.
extern "C" void
dorado_fast_lstm_layer_cascade_put_get(bfloat16 *w_in,
                                        /* no input ObjectFifo: cascade is the input */
                                        /* no output ObjectFifo: cascade is the output */) {
  for (int t = 0; t < L; ++t) {
    // Cascade in — read 512 bits at a time. The previous layer's
    // FP32 hidden state is the input to this layer's matmul.
    aie::vector<float, 16> h_in_lanes[HIDDEN_VECS];
    aie::vector<float, 16> c_in_lanes[HIDDEN_VECS];
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      h_in_lanes[v] = get_scd_v16int32();  // reinterpret as float
      c_in_lanes[v] = get_scd_v16int32();
    }
    // Note: this layer's c_state is its own (LSTMs don't share c
    // across layers); the cascaded c is unused here. We only need h
    // as the input to the matmul; c is per-layer-private.
    // (For an alternating-direction stack the previous layer's
    // direction-reversed h IS the input — Bonito's pattern lets us
    // use h-only cascade and keep c local.)

    // Narrow h_in (FP32 accumulator) to bf16 for matmul input.
    // Bf16 multiplier inputs match AM020 Ch. 4 p. 65; the FP32
    // hidden state is preserved across the cascade boundary
    // (no narrowing in transit). The narrowing happens HERE,
    // ONCE, before the matmul — same as 's within-layer
    // recurrence, but ZERO bf16 round-trips between layers.
    aie::vector<bfloat16, 16> x_bf16 = to_bf16(h_in_lanes[v]);

    // ... gate computations + activations ...

    // Cascade out for next layer.
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      put_mcd(h_lane[v]);
      put_mcd(c_lane[v]);
    }
  }
}
```

### Variant 3: ``lstm_layer_cascade_get_only`` (LAST role / row 6)

```c++
// LAST role — Layer 4 of the stack (the output of the encoder). Receives
// input via cascade from layer 3, computes its own LSTM forward, and
// emits the result via ObjectFifo (the standard output channel to the
// linear projection / CRF that follows the encoder).
extern "C" void
dorado_fast_lstm_layer_cascade_get_only(bfloat16 *w_in, bfloat16 *y_out) {
  for (int t = 0; t < L; ++t) {
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      h_in_lanes[v] = get_scd_v16int32();
      c_in_lanes[v] = get_scd_v16int32();
    }
    // ... gate computations ...
    // Output: bf16 (the encoder's downstream consumers want bf16
    // anyway, so the final narrowing is preserved).
    for (int v = 0; v < HIDDEN_VECS; ++v) {
      y_out[t * HIDDEN + v * VEC : ...] = to_bf16(h_lane[v]);
    }
  }
}
```

## Per-worker weight DMA topology

Each worker has its own LSTM-layer weights (the 5 layers do not share
weights). The weight DMA topology that uses (single
ObjectFifo of bf16 weights+bias chunks per layer) replicates per
worker:

```
   row 2: weight_in_L0 (ObjectFifo, bf16 chunks, shape)
   row 3:  weight_in_L1  (ObjectFifo, bf16 chunks)
   row 4:  weight_in_L2  (ObjectFifo, bf16 chunks)
   row 5:  weight_in_L3  (ObjectFifo, bf16 chunks)
   row 6:  weight_in_L4  (ObjectFifo, bf16 chunks)
```

5 weight ObjectFifos, each ~110 MB / call, total ~550 MB / call.
This is 5× 's per-call DMA volume — the cycle savings from
cascade need to outpace the increased DMA latency. The cross-walk
notes (§) the path that closes this — sparse weight
compression — is a separate kernel direction (deferred );
the cascade variant should establish the precision win first, then
the compression follow-up addresses the throughput regression.

Input ObjectFifo is only on row 2; output ObjectFifo is only on
row 6. The middle three workers (rows 3..5) consume from cascade
and emit to cascade; their `Worker.fn_args` consume only the
weight ObjectFifo + the cascade primitives implicit in the kernel
variant.

## Kernel-author hand-off checklist

The Python-side topology + design lands today. The remaining
engineering for an end-to-end silicon-level test:

- [ ] **C++ kernel** at ``lstm_layer_cascade.cc`` (~400-500 LoC):
  three role variants (put_only, put_get, get_only) + the shared
  gate-computation helpers. Reuse the bf16 multiplier intrinsics
  + FP32-accumulator-store pattern from 's
  ``lstm_cell_bf16_acc.cc``.

- [ ] **IRON topology file** at
  ``lstm_cell_bf16_acc_cascade.py`` (~150-200 LoC): three
  ``Kernel`` declarations, three ``core_fn`` factories
  (parameterized by role), call to
  ``bionpu.iron_extensions.cascade_stream_chain`` to build the
  workers, ``Runtime`` + ``Program`` glue.

- [ ] **Makefile / build script** at ``Makefile``: `aiecc.py
  --no-xchesscc --no-xbridge` against the multi-worker design;
  produces single ``final.xclbin`` + ``insts.bin``.

- [ ] **Host runner** at ``runner.cpp``: 6 buffer setup (1 input + 5
  weight buffers + 1 output buffer), single xclbin run. Otherwise
  identical to 's ``runner.cpp``.

- [ ] **Integration test** at
  ``tests/test_t7_iron_cascade_silicon_npu.py`` (NPU-marked):
  `dispatch("dorado_fast_lstm_stack_bf16_acc_cascade", ...)` with
  real Dorado fast weights; assert encoder max-abs is recorded
  honestly (the < 0.23 strong-prediction threshold is recorded
  but not asserted — same falsifiability discipline as 's
  test).

- [ ] **Measurements** at
  ``results/basecalling/b-m6d-cascade/measurements.json``: real
  encoder max-abs vs 's 2.076 baseline (silicon-level
  cross-walk hypothesis test result).

- [ ] **Update** ``bionpu/iron_extensions/INVENTORY.md`` §6
  ("Cross-walk falsifiability") with the silicon-level result.

Estimated kernel-author work: ~3-5 days with NPU access.

## Not in this design

- **Layer-fan-in via memtile**: not used. The 5-layer stack is a
  single column. The cross-walk's separate "memtile-mediated 4-into-1
  fan-in" follow-up (§) is unrelated.
- **Sparse weight decompression**: not used. The cross-walk's §G-
   weight-DMA-volume reduction is the deferred-
  surface, orthogonal to T7-IRON's cascade test.
- **Variable-rate / packet-switched ObjectFifo**: not used. T7-
  IRON's investigation flagged this as RED (extension path NOT
  viable; needs source rebuild). The cascade variant doesn't need
  variable rates — the 5-layer pipeline is a fixed-stride
  producer-consumer chain.
- **AM-to-AM register-resident ``h_state``**: not used (yet). The
  C++ kernel falls back to FP32 tile-DM static storage for `h_state`
  /`c_state` within each layer, identical to 's discipline.
  T7-IRON's INVENTORY §4 marks AM-resident state as YELLOW
  (extension-path viable but not load-bearing — the spill behavior
  produces the same code path).

## References

- AM020 Ch. 4 p. 67 — cascade stream + AM-to-AM register move
  primitives.
- AM020 Appendix A p. 80 Figure 45 — vertical + horizontal cascade
  topology.
- ``~/xdna-bringup/mlir-aie/programming_examples/basic/matrix_multiplication/cascade/cascade.py``
  — upstream IRON-level cascade reference design.
- ``~/xdna-bringup/mlir-aie/aie_kernels/aie2/cascade_mm.cc`` —
  upstream C++ cascade kernel with the `put_mcd` / `get_scd`
  intrinsics in context.
- ``bionpu/iron_extensions/INVENTORY.md`` §3 — cascade-stream
  primitive feasibility verdict.
- ``bionpu/iron_extensions/cascade_stream.py`` — the IRON
  helper this kernel consumes.
- ``bionpu/kernels/basecalling/lstm_cell_bf16_acc/`` —
  the IRON-level fallback this cascade variant is the silicon-level
  test of.
