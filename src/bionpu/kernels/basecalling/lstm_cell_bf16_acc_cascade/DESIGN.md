# `lstm_cell_bf16_acc_cascade` — design note

A cascade-stream LSTM cell for AIE2P that hands the FP32
accumulator state across vertically-adjacent CoreTiles through the
hardware cascade channel rather than narrowing to bf16 / int8 for
storage between cells.

## Architectural references

- **AM020 Ch. 4 p. 67** — cascade stream is a 512-bit physical
  channel between adjacent CoreTiles, carrying one accumulator
  register per cycle.
- **AM020 Appendix A p. 80 Figure 45** — vertical / horizontal cascade
  topology on AIE-ML; AIE2P inherits the same routing.
- **`aie::iron::AccumFifo`** (upstream `mlir-aie`) — the FP32
  inter-tile accumulator hand-off primitive this kernel consumes.
- **`aie::iron::CascadeFifo`** (upstream `mlir-aie`) — first-class
  cascade-stream `ObjectFifo` subclass used reflexively at
  construction time.

## Design rationale

A bf16-multiplier / FP32-accumulator LSTM stack hits a precision wall
when the recurrent state (`h`, `c`) is forced to round-trip through
a memref between cells: every storage write narrows the accumulator
to bf16 and the cumulative drift over `L` timesteps grows as
`O(L · 2⁻⁷)`. Holding `h` / `c` in the cascade channel at full FP32
precision drops the drift to `O(L · 2⁻²³)` — the difference between
the bf16-mantissa floor and the FP32-mantissa floor.

The IRON layer of this kernel emits an `aie.cascade_flow` between
each pair of (layer N, layer N+1) tiles via `CascadeFifo` /
`AccumFifo`. The C++ kernel side owns the actual cascade reads /
writes via the `put_mcd` / `get_scd_v16int32` intrinsics inside the
core-fn body.

## Topology

```
host (shim)
   │
   ▼
tile (col, 0)  layer 0  ── cascade ──▶ tile (col, 1)  layer 1
                                                   │
                                                ── cascade ──▶ tile (col, 2)  layer 2
                                                                            │
                                                                         ── cascade ──▶ ...
```

- One cascade per `(layer N → layer N+1)` boundary.
- Vertical adjacency required (same column, adjacent rows).
  Horizontal-hop cascades are valid by AM020 routing but less
  exercised.
- The shim DMA delivers the input embedding to layer 0 and drains the
  layer-N output back to host; intermediate accumulator state never
  leaves the cascade fabric.

## Files in this directory

| File | Role |
|---|---|
| `lstm_cell_bf16_acc_cascade.py` | IRON Python topology — places workers, wires `CascadeFifo` / `AccumFifo` between layers, emits the runtime sequence. |
| `lstm_layer_cascade.cc` | AIE2P C++ kernel — bf16 matmul + FP32 accumulator update + cascade put/get inside the core-fn. |
| `runner.cpp` | Host XRT runner. |
| `Makefile` | Builds the xclbin + host runner. |
| `__init__.py` | Python `NpuOp` registration. |

## Known limitations

- Cascade-stream behaviour on AIE2P silicon under low cascade-port
  duty-cycle is sensitive to firmware-side watchdog thresholds. The
  production kernel emits cascade ops unconditionally per timestep
  (no branched / guarded cascade emission) to avoid tripping
  watchdog edge cases observed in earlier development.
- Horizontal-hop cascade routing is a documented AM020 capability
  but should be silicon-validated for any new column geometry before
  shipping numbers from it.
