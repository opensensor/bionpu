# `bionpu.iron_extensions`

Helper module that wraps the AIE-ML / AIE2P **cascade stream** primitive
(AM020 Ch. 4 p. 67; Appendix A p. 80 Figure 45) at the IRON Python
layer.

## Status

The motivating feasibility study for this module concluded that the
cascade-stream primitive should be exposed as a first-class IRON
abstraction (analogous to `aie.iron.ObjectFifo`). That work landed
upstream in `mlir-aie` as the `CascadeFifo` class —
[Xilinx/mlir-aie #3039][upstream-pr]. New code should prefer
`from aie.iron import CascadeFifo` over this module.

`bionpu.iron_extensions.cascade_stream` is kept as a back-compat shim
for designs written against the pre-upstream API. It will be removed
once `mlir-aie` ≥ the version that includes the `CascadeFifo` merge
is the floor in `pyproject.toml`.

[upstream-pr]: https://github.com/Xilinx/mlir-aie/pull/3039

## What's in this package

| File | Purpose |
|---|---|
| `cascade_stream.py` | Builds the cascade-routing topology from a list of cascade-aware Workers placed on vertically-adjacent CoreTiles. Emits implicit cascade connections plus the three kernel-variant convention (`..._cascade_get_only`, `..._cascade_put_get`, `..._cascade_put_only`) the C++ kernels use to express their position in the chain. |

## Architectural references

- AM020 Ch. 4 p. 67 — cascade stream is a 512-bit physical channel
  between adjacent CoreTiles.
- AM020 Appendix A p. 80 Figure 45 — vertical and horizontal cascade
  topology on AIE-ML; AIE2P inherits the same routing.
- `aie.put_cascade` / `aie.get_cascade` — the cascade write/read MLIR
  ops; the operand type must match the cascade size (i512 or
  vector<16xi32>).
- `aie.cascade_flow` — declarative connection between two tiles that
  the placer lowers to per-tile `aie.configure_cascade` ops.
