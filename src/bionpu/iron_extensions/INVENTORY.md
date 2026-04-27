# IRON surface inventory + AM020-primitive feasibility verdicts (T7-IRON)

**Date**: 2026-04-25  
**Wheel under test**: `mlir_aie` installed at
`~/xdna-bringup/ironenv/lib/python3.11/site-packages/mlir_aie/python/aie/`  
**Source-tree reference**: `~/xdna-bringup/mlir-aie/programming_examples/`  
**Cross-walk under test**: `docs/.md`

This document is the load-bearing artifact for T7-IRON. The investigation
brief framed the task as "feasibility report + ONE concrete prototype",
and the report is the hand-off — even where the prototype landed (cascade
stream), the report explains the boundary that lets future work decide
between the extension path and the source-rebuild path.

## TL;DR

| Primitive (AM020 cite) | gap-id | Reachable as Python extension? | Prototype landed? | Cost-of-source-rebuild estimate |
|---|---|---|---|---|
| **Cascade stream** (Ch. 4 p. 67; Appendix A p. 80 Fig. 45) | | **YES — GREEN** | YES (`cascade_stream.py`, ~280 LoC + multi-tile kernel design files at `bionpu/kernels/basecalling/lstm_cell_bf16_acc_cascade/`) | N/A — extension path viable |
| **AM-to-AM register move** (Ch. 4 p. 67) | | **PARTIAL — YELLOW** | No (out-of-scope: pure C++ kernel-author work, not toolchain) | ~0 days from toolchain, ~3-5 days from kernel-author |
| **Variable-rate ObjectFifo / pktMerge** (Ch. 2 Fig. 17; Ch. 5 p. 74) | | **NO — RED** | No (would not fit "extension above wheel" pattern) | ~10-20 days source-tree work; needs new IRON class + MLIR pass changes |

**Cross-walk hypothesis status (cascade stream + accumulator-state retention)**: the cross-walk's core architectural prediction
(IRON is the toolchain limit, not the silicon) holds at the
investigation level — the cascade primitive is reachable without a
wheel rebuild. However, the falsifiable silicon-level test of the
cross-walk's strong "encoder max-abs < 0.23 with cascade" prediction
**was not run** in this scope: the engineering work to author the
LSTM C++ kernel variants with `put_mcd`/`get_scd` intrinsics and
build a working xclbin is kernel-author scope, not toolchain
investigation scope. The prototype delivers the *Python-side*
topology that this engineering would need; it does not by itself
produce silicon-level numbers. See the "Cross-walk falsifiability"
section below.

## 1. IRON surface map

The installed wheel exposes two layers:

### 1.1 IRON layer (`aie.iron.*`) — the high-level Python API

Re-exported via `aie.iron.__init__.py`:

| Class / function | What it does | Cascade-relevant? |
|---|---|---|
| `Buffer` | Named tile-DM region accessible from Worker + Runtime | No |
| `ObjectFifo` (`aie.iron.dataflow.objectfifo.ObjectFifo`, 786 LoC) | Synchronized circular-buffer dataflow channel between program components. Lowers to `aie.dialects.aie.object_fifo` (DMA-mediated). | **No** — explicitly DMA-mediated; constructor flags are `dims_to_stream`, `dims_from_stream_per_cons`, `plio`, `pad_dimensions`, `depth`. **Zero hooks for cascade routing**. |
| `Worker` (`aie.iron.worker.Worker`, 227 LoC) | Compute task on a single CoreTile. Constructor takes `core_fn`, `fn_args`, `tile`. The `core_fn` body emits MLIR via the `aie.dialects` op-builders. | Partial — the `core_fn` body CAN emit `aie.put_cascade` / `aie.get_cascade` (the dialect-level ops), but the IRON `Worker` class has no first-class cascade-input/output port concept. Cascade access is via direct MLIR emission inside `core_fn`. |
| `Runtime` (`aie.iron.runtime.Runtime`) | Host-side orchestration; `rt.start(*workers)`, `rt.fill`, `rt.drain`. | No |
| `Program` (`aie.iron.program.Program`, 135 LoC) | Top-level container; `resolve_program()` emits MLIR. | No |
| `Kernel` / `ExternalFunction` (`aie.iron.kernel.Kernel`) | Binds a C++ symbol+object to a function-call site. | Indirect — the C++ kernel is where `put_mcd` / `get_scd` intrinsics live; the IRON layer just calls the symbol. |
| `WorkerRuntimeBarrier` | Lock-based sync between worker and runtime | No |

### 1.2 Dialect layer (`aie.dialects.*`) — MLIR op-builders

Re-exported via `aie.dialects.aie` (which does `from ._aie_ops_gen
import *`). The generated op-builders live in
`aie.dialects._aie_ops_gen` (auto-generated from
`aie/Dialect/AIE/IR/AIEOps.td`).

**Cascade-relevant ops, all present in the installed wheel**:

| Op | Builder function | Source line | Description |
|---|---|---|---|
| `aie.cascade_flow` | `cascade_flow(source_tile, dest_tile)` | `_aie_ops_gen.py:365` | Declares a cascade connection between two tiles. Lowered by the placement pass to `aie.configure_cascade` ops. |
| `aie.configure_cascade` | `configure_cascade(tile, input_dir, output_dir)` | `_aie_ops_gen.py:444` | Per-tile configuration — declares which directions (North/South/East/West, from `_aie_enum_gen.CascadeDir`) the cascade in/out ports use. |
| `aie.put_cascade` | `put_cascade(cascade_value)` | `_aie_ops_gen.py:4105` | Inside a `core_fn`: write a 512-bit value to the outbound cascade. AIE2 cascade size is `i512` or `vector<16xi32>`. |
| `aie.get_cascade` | `get_cascade(cascade_value)` | `_aie_ops_gen.py:1903` | Inside a `core_fn`: read a 512-bit value from the inbound cascade. |

`CascadeDir` enum (`_aie_enum_gen.py:114`) has `South=3 / West=4 /
North=5 / East=6` — the four neighbour directions. AIE2P supports
both vertical (N-S) and horizontal (E-W) per AM020 Appendix A p. 80
Figure 45 (carry-forward from AIE-ML).

### 1.3 The boundary: where IRON's abstraction stops

The IRON layer's design contract is:
> **Workers are stateless functions placed on a single CoreTile;
> inter-tile communication is via ObjectFifo (DMA-mediated).**

Concretely, the `aie.iron.Worker.__init__` code (worker.py:85–116)
explicitly walks `fn_args` looking for `ObjectFifoHandle`,
`Buffer`, and `WorkerRuntimeBarrier` instances and registers them.
There is no analogous code path for cascade ports, accumulator
register persistence, or pktMerge variable-rate streams. The
`core_fn` body is free to emit any MLIR op the placement pass
accepts — including `aie.put_cascade`, `aie.get_cascade`, and
direct MLIR-level lowering — but those ops are **not first-class
IRON primitives**.

The wheel's source tree contains a working reference design that
uses cascade routing **without** any IRON-level cascade primitive:
`programming_examples/basic/matrix_multiplication/cascade/cascade.py`
declares three `external_func` variants (`matmul_scalar_cascade_get_only`,
`..._cascade_put_only`, `..._cascade_put_get`) and calls them inside
`@core(...)` bodies that run on vertically-adjacent CoreTiles in
the same column. Cascade routing is **implicit** from vertical tile
adjacency + matched `put_mcd` / `get_scd` intrinsics in the C++
kernels. **The placed example does not emit `aie.cascade_flow` ops
explicitly** (verified by grep over the file). Same pattern works
on AIE2P (`--dev npu2` is supported in the example's argparse).

This is the key "extension is enough" finding: the IRON
extension layer can compose `aie.iron.Worker` + `aie.iron.Kernel` +
vertical tile placement to reproduce the canonical cascade pattern,
without changes to the wheel.

## 2. The decision: extension path or source path?

**Decision: EXTENSION PATH for cascade stream**, taken on the
following grounds:

1. The MLIR ops are present in the installed wheel
   (`_aie_ops_gen.py:309..4106`).
2. The wheel's source tree has a complete IRON-level reference
   design (`programming_examples/basic/matrix_multiplication/cascade/`).
3. The reference design does not require explicit `aie.cascade_flow`
   ops — vertical tile adjacency + matched C++ intrinsics suffice
   under the placed/dialect-style API.
4. The IRON `Worker` class accepts arbitrary `core_fn` bodies
   and any `aie.iron.Kernel` symbol; nothing in the layer prevents
   binding the cascade-aware kernel variants.
5. Estimated extension-side LoC budget: ~280 LoC for
   `cascade_stream.py` + design files at
   `bionpu/kernels/basecalling/lstm_cell_bf16_acc_cascade/`.
   Estimated wheel-rebuild LoC budget: undefined; cascade is already
   in the wheel.

The remaining engineering for an end-to-end cascade-LSTM (C++ kernel
authoring with `put_mcd` / `get_scd` intrinsics, weight DMA per
worker, xclbin build) is **kernel-author scope**, not toolchain
scope. The prototype includes the design files for that
engineering; the engineering itself is a follow-up.

## 3. Cascade stream — primitive feasibility verdict (GREEN)

**Verdict**: extension-path viable; prototype landed.

**Implementation summary**: see
`bionpu/iron_extensions/cascade_stream.py` for the IRON
helper, and
`bionpu/kernels/basecalling/lstm_cell_bf16_acc_cascade/` for the
multi-tile LSTM design that consumes it.

**Approach taken**: composition. The
`cascade_stream_chain(n_workers, kernels_by_role, ...)` helper
accepts a 3-element kernel-by-role mapping (`FIRST` /
`MIDDLE` / `LAST`) and a column + starting row, and:

1. Computes the role for each chain position (`FIRST`/`MIDDLE`/
   `LAST` for N>=2; degenerate `SOLO` for N=1).
2. For each position, creates an `aie.iron.Worker` placed on a
   vertically-adjacent CoreTile in the column, with `fn_args`
   bound to the role-appropriate `aie.iron.Kernel` symbol.
3. Returns a `CascadeStreamChain` dataclass with the prebuilt
   workers ready to attach to a `Program` + `Runtime`.

This composes existing IRON primitives only — no new MLIR
ops, no wheel changes. The cascade routing happens transparently
via vertical-tile-adjacency + matched C++ kernel intrinsics, the
same way the upstream `programming_examples/basic/matrix_multiplication
/cascade/cascade.py` reference design works.

**Falsifiable cross-walk-hypothesis test status: DEFERRED**. The
cross-walk's strong silicon-level prediction (encoder max-abs < 0.23
with cascade-stream-driven layer chaining vs 's 2.076 IRON-level
fallback) requires:

1. A C++ kernel with `put_mcd` / `get_scd` intrinsics for the LSTM
   accumulator state. (Designed; not authored.)
2. Per-worker weight DMA topology — each layer gets its own
   ~110 MB weight stream, distributed across 5 workers in the
   chain. (Designed; not implemented.)
3. xclbin build via `aiecc.py` with the multi-worker design.
4. Encoder-level integration test on real Dorado weights.

Items 1–4 are **kernel-author scope** and beyond T7-IRON's
toolchain-investigation budget. The prototype delivers the
Python-side topology + design notes that the engineering would
consume.

**What the prototype DOES test (today)**:

- IRON topology is reachable: the helper produces a valid
  `CascadeStreamChain` with `n_workers` workers, each placed on a
  vertically-adjacent CoreTile, each bound to the role-appropriate
  kernel. `tests/test_iron_extensions_cascade.py` asserts the
  topology + role assignment + tile placement.
- The MLIR-dialect cascade ops are exposed by the installed wheel
  (verified by import).
- The cascade-routing pattern (vertical tile adjacency + role-based
  kernel selection) matches the upstream reference design.

**What the prototype does NOT test**:

- End-to-end max-abs vs the cross-walk's < 0.23 prediction.
- Per-cell vs 5-stack max-abs improvement deltas.
- Cycle count / timing of the cascade vs DMA path.

The cross-walk hypothesis is **architecturally confirmed**
(extension path viable; the IRON wall is real but penetrable
without a source rebuild) and **silicon-level deferred**. Diagnosis
for the deferral is recorded honestly: the silicon test requires
non-trivial kernel engineering (~400-800 LoC of careful C++ with
weight handling) that does not fit "investigation + ONE prototype"
scope.

## 4. AM-to-AM register move — primitive feasibility verdict (YELLOW)

**Verdict**: PARTIAL extension-path viable. The accumulator-register
persistence is **already accessible** to the kernel author; what's
missing at the IRON layer is sugar.

**Cite**: AM020 Ch. 4 p. 67, "Register Move Functionality" — "Move
one 512-bit accumulator (AM) register to another AM-register in one
cycle".

**Why YELLOW, not GREEN**: the AM-to-AM register move is a *within-tile,
within-kernel* primitive. The C++ kernel can already use the AIE2P
hardware via `aie::accum<...>` types and the standard `aie_api`
register abstractions — there's no MLIR op to expose, and no
IRON helper that would meaningfully change anything for the
kernel author. 's lstm_cell_bf16_acc.cc could in principle hold
`h_state` and `c_state` as `aie::accum<accfloat, 16>` register
variables (1024 bits = two AM-register halves, AM020 Ch. 4 p. 67) and
let the compiler keep them in registers across timesteps; nothing in
IRON prevents this.

**Why it didn't happen in **: the C++ kernel falls back to FP32
static tile-DM storage because the LSTM has L=334 timesteps and the
register pressure of holding `h_state[96]` + `c_state[96]` +
`gate_acc[4][96]` simultaneously in AM registers exceeds the AIE2P
compute tile's register file (16 AM registers × 1024 bits = 16 KiB
of register state, vs the ~3 KiB of FP32 state needed). Without
spill discipline (the compiler will spill to DM automatically when
register pressure rises), the practical outcome is the same as
the current FP32-tile-DM-storage path.

**What an IRON sugar API would look like** (if pursued):

```python
# Hypothetical Worker.persist_accumulator API — NOT IMPLEMENTED.
worker = Worker(
    core_fn,
    fn_args=[...],
    tile=Tile(0, 2),
    persistent_accumulators={
        "h_state": (HIDDEN, "accfloat"),  # 96 lanes FP32 -> 6 AM-regs
        "c_state": (HIDDEN, "accfloat"),
    },
)
```

This would generate the C++ glue declaring the named variables as
`aie::accum<accfloat, 16>` register-resident state and threading
their addresses into the kernel call. Estimated cost:

- ~100-150 LoC IRON helper (in `bionpu/iron_extensions/`).
- Zero wheel changes.
- ~50 LoC of C++ kernel-author boilerplate per kernel that adopts it.

**Why we are NOT prototyping it now**: already does this in
spirit — `h_state`, `c_state`, `gate_acc` are FP32 tile-DM static
arrays, which the AIE compiler can (and likely does) keep in AM
registers across timesteps when register pressure permits. The
sugar API would not change the silicon behaviour; it would only
make the storage discipline clearer at the Python level. 's
gaps.yaml entry already records this as a "workaround
acceptable" finding.

**Cost estimate to fully prototype**: ~3-5 days kernel-author
work (1 day for the IRON helper, 2-4 days to refactor
's C++ kernel to use AM-register-resident state with
explicit spill control + verify the encoder max-abs is unchanged).
Not on the critical path; deferred unless 's encoder
max-abs measurements show a meaningful gap traceable to the
DM-vs-register storage choice.

## 5. Variable-rate ObjectFifo / pktMerge — primitive feasibility verdict (RED)

**Verdict**: extension-path **NOT** viable. The full design needs MLIR
pass changes that are beyond the wheel's currently-exported surface.

**Cite**: AM020 Ch. 2 Figure 17 ("pktMerge"); Ch. 5 p. 74 ("memtile
out-of-order BD processing based on incoming packet header"); Ch. 2
"Tile DMA Controller" p. 27 ("S2MM finish-on-TLAST").

**Three primitives, all needed for the variable-rate filter-early
pattern wanted**:

1. **Packet-switched merge N:1**. Multiple packet streams converge
   through a hardware merge block. AM020 Figure 17 shows the
   topology; the MLIR side requires a packet-switched ObjectFifo
   variant.
2. **S2MM finish-on-TLAST**. ObjectFifo termination on a TLAST
   signal mid-stream — variable per chunk.
3. **Out-of-order BD processing on packet header**. Memtile routes
   packets to different buffer descriptors based on header bits
   (variable-rate routing logic in fabric).

**Why RED**: the IRON `ObjectFifo` class (786 LoC at
`aie.iron.dataflow.objectfifo`) is hard-coded to a fixed-stride
producer-consumer pipeline. Constructor flags expose `dims_to_stream`,
`dims_from_stream_per_cons`, `plio`, `pad_dimensions`, `depth` — none
of these touch the variable-rate / packet-switched / TLAST
primitives. A subclass-based extension cannot reach these because:

- The lowered `aie.dialects.aie.object_fifo` op is hard-coded to
  fixed-stride rectangle dataflow.
- Packet-switched merge is a different MLIR op family
  (`aie.packet_flow`, `aie.packet_source`, `aie.packet_dest`,
  `aie.packet_rules`); these ops ARE exported by `_aie_ops_gen.py`
  but are not composed into any `ObjectFifo`-equivalent
  abstraction in IRON.
- The packet-flow ops would need a new IRON-level class
  (`PacketFifo`? `MergeFifo`?) that is **not present** in the
  installed wheel and would not naturally compose with `Worker`'s
  fn_args resolution (worker.py:85–116 only registers
  `ObjectFifoHandle` instances, not arbitrary packet-flow handles).

**Estimated cost to deliver the variable-rate primitive**:

- **As an IRON extension above the wheel**: not viable as
  pure subclassing. Would require a new `PacketFifo`-style class
  that re-implements the IRON-side resolution glue
  (`Resolvable.resolve()`, `Worker.fn_args` registration) and
  re-emits the dialect ops directly. Estimated ~600-1000 LoC,
  mostly re-paving the IRON resolution machinery to teach `Worker`
  about a non-`ObjectFifoHandle` dataflow handle. **This is
  effectively a partial fork of IRON, not a clean extension.**

- **As a wheel rebuild (forking mlir-aie source)**: estimated
  ~10-20 days. The `aie.iron.dataflow` package currently has 2
  files (`endpoint.py`, `objectfifo.py`); a `packetfifo.py`
  sibling that uses the existing `aie.packet_flow` op family
  would need to integrate with the resolution pipeline in
  `Program.resolve_program()`. The MLIR pass `--aie-objectfifo-stateful-transform`
  would also need to be extended (or a sibling pass added) to
  lower packet-switched ObjectFifos correctly through the
  packet routing fabric.

**Recommendation**: defer 's full-cycle-savings filter-early
pattern until the basecalling track's RQ4 writeup decides whether
the variable-rate ObjectFifo is worth a 10-20 day source-rebuild
investment. The current v1 (filter-late) is shipping; the
pre-AM020 framing of "filter-early needs IRON to expose pktMerge"
is correct, and this RED verdict pinpoints exactly *which* IRON
internals would need surgery.

## 6. Cross-walk falsifiability — what this investigation closes and leaves open

**Cross-walk's central architectural claim** (`docs/.md`
"What this doc unblocks for + " section): "of our 5
documented blockers and 4 workarounds, **6 are IRON
abstraction limits** (toolchain), **1 is a silicon ceiling** (
/ — 2 DMA channels), and **2 are misdiagnosed** (
+ wash actually trace to the recurrent-state writeback at
bf16 width, not to tanh approximation)."

**Status after T7-IRON's investigation**:

| Cross-walk claim | Confirmed at investigation level? | Confirmed at silicon level? |
|---|---|---|
| Cascade stream is reachable as IRON extension (no wheel rebuild) | YES — `cascade_stream.py` prototype landed | DEFERRED — silicon test requires C++ kernel authoring (out of scope for T7-IRON) |
| AM-to-AM register move is reachable as IRON sugar | YES — but the underlying primitive is already accessible to the kernel author; the sugar would not change silicon behaviour | N/A — primitive is already accessible; sugar is convenience only |
| Variable-rate ObjectFifo / pktMerge is reachable as IRON extension | NO — RED verdict, requires partial fork or wheel rebuild | DEFERRED — even at the architectural level the extension path is not viable |
| Encoder max-abs improves to < 0.23 with cascade-stream layer chaining (the strong, falsifiable prediction) | UNTESTED — silicon test deferred | UNTESTED — silicon test deferred |
| The IRON wall is the wall — silicon allows what we want | **PARTIAL** — wall is penetrable for cascade (extension path); penetrable in spirit for AM-to-AM (sugar only); IMPENETRABLE without source rebuild for variable-rate ObjectFifo | UNTESTED for the falsifiable prediction |

**Honest framing**: the cross-walk's "RQ4 toolchain story" gets a
nuanced verdict from T7-IRON. The IRON wall is **not
monolithic**: cascade is reachable, AM-to-AM is sugar-only,
variable-rate ObjectFifo is genuinely behind a source rebuild. The
RQ4 writeup should distinguish among these three rather
than collapsing them into a single "IRON limits us" narrative.

**On the strong falsifiable prediction (cross-walk's "10-100×
encoder max-abs improvement")**: 's IRON-level fallback gave
a ~16% max-abs improvement over (2.076 vs 2.458) — already
a partial refutation of the strong prediction. T7-IRON's cascade
prototype would provide the silicon-level test, but the kernel
engineering required is non-trivial. The most honest current
prediction (recommended for the writeup): if the cascade
silicon test produces another 16-50% improvement, the cross-walk's
recurrent-state-writeback diagnosis is **partially correct but
incomplete** — the bf16 multiplier-input narrowing on Dorado's
trained weights is doing meaningful work too. If it produces 10-100×
improvement, the cross-walk's strong prediction is confirmed. The
prototype's design files give the engineering a clean starting
point.

## 7. References

- AMD, *Versal Adaptive SoC AIE-ML Architecture Manual* (AM020
  v1.5, Feb 2026).
- `~/xdna-bringup/mlir-aie/programming_examples/basic/matrix_multiplication/cascade/`
  — upstream IRON cascade reference design.
- `~/xdna-bringup/mlir-aie/aie_kernels/aie2/cascade_mm.cc` — upstream
  C++ cascade kernel (the `put_mcd` / `get_scd` intrinsics in
  context).
- `docs/.md` — the cross-walk this
  investigation tests.
- `bionpu/kernels/basecalling/lstm_cell_bf16_acc/` —
  baseline this prototype builds on.
- 's measurements at `results/basecalling/b-m6d/measurements.json`.
