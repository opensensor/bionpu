# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Cascade-stream IRON topology helper (T7-IRON, AM020 Ch. 4 p. 67).

Module purpose
--------------

Wraps the AIE2P **cascade stream** primitive (512-bit
accumulator-to-accumulator inter-tile transfer per cycle, no precision
loss; AM020 Ch. 4 p. 67 + Appendix A p. 80 Figure 45 for the
vertical+horizontal grid topology) at the IRON level.

The upstream ``mlir-aie`` wheel exposes the underlying MLIR ops
(``aie.put_cascade``, ``aie.get_cascade``, ``aie.cascade_flow``,
``aie.configure_cascade``) at the *dialect* level, and a working
reference design at
``programming_examples/basic/matrix_multiplication/cascade/cascade.py``
in the wheel's source tree. The IRON *abstraction* layer
(``aie.iron.{Worker,ObjectFifo,Runtime}``) however does **not** surface
the cascade stream as a first-class primitive: there is no
``Worker.cascade_in()``, no ``ObjectFifo(use_cascade=True)``, and the
``Worker`` class strictly models per-tile compute with ObjectFifo-only
dataflow neighbours.

This module fills that gap with a single helper,
:func:`cascade_stream_chain`, which accepts a list of cascade-aware
workers (each placed on a vertically-adjacent ``CoreTile`` in the same
column) and emits the cascade-routing topology the upstream example
uses: implicit cascade connections via vertical placement, plus the
three kernel-variant convention (``..._cascade_get_only``,
``..._cascade_put_get``, ``..._cascade_put_only``) that the C++ kernels
use to express their position in the chain.

What it does NOT do
-------------------

- It does not generate the C++ kernel variants. Those must be authored
  by the user (the cascade reads happen via ``put_mcd(value)`` /
  ``get_scd_v16int32()`` intrinsics inside the kernel body — see the
  upstream ``aie_kernels/aie2/cascade_mm.cc`` reference).
- It does not perform tile placement validation. AM020 Ch. 4 p. 67 +
  Appendix A p. 80 confirms cascade is available vertical AND
  horizontal on AIE-ML; AIE2P inheritance is **assumed-but-unverified**
  (``docs/.md`` flags this as a chip-generation
  caveat). On AIE2P we have only verified vertical (column-wise);
  horizontal cascade was not measured in T7-IRON's investigation.
- It does not fix the IRON "Worker is single-tile, single-call"
  abstraction limit for non-cascade-stream multi-tile patterns
  (e.g. AM-to-AM register move within a tile, variable-rate
  ObjectFifo). See ``INVENTORY.md`` for those primitives' feasibility
  verdicts.

Design rationale: extension path vs source rebuild
--------------------------------------------------

The investigation in ``INVENTORY.md`` concluded the cascade primitive
is reachable at the **IRON extension layer** without a wheel
rebuild, because:

1. The MLIR ops (``aie.put_cascade``, ``aie.get_cascade``,
   ``aie.cascade_flow``) are already exported by
   ``aie.dialects._aie_ops_gen`` in the installed wheel.
2. The lower-level ``aie.dialects.aie`` placed-style API (used by
   ``cascade.py`` in the wheel's ``programming_examples/``) supports
   cascade routing today via vertical tile placement + matched C++
   kernel intrinsics — no changes to the wheel needed.
3. The IRON layer above can compose these primitives by
   declaring three Worker variants (one per chain role) on
   vertically-adjacent CoreTiles in the same column.

The remaining engineering work (authoring the C++ kernel variants for
the LSTM use case, building the xclbin, hooking up the host runner)
is **kernel-author scope**, not toolchain scope. The "T7-IRON
extension path is viable" finding lets that engineering proceed
without first taking on a wheel rebuild.

Usage sketch
------------

.. code-block:: python

    from aie.iron import Kernel, Program, Runtime, Worker, ObjectFifo
    from aie.iron.device import NPU2
    from bionpu.iron_extensions import cascade_stream_chain, CascadeRole

    n_layers = 5

    # Three kernel variants, one per role in the chain. The C++ side
    # uses put_mcd() / get_scd_v16int32() intrinsics; the IRON layer
    # only needs the symbol names.
    k_first = Kernel("lstm_layer_cascade_put_only",  "lstm_cascade.o", [...])
    k_middle = Kernel("lstm_layer_cascade_put_get",  "lstm_cascade.o", [...])
    k_last  = Kernel("lstm_layer_cascade_get_only",  "lstm_cascade.o", [...])

    chain = cascade_stream_chain(
        n_workers=n_layers,
        kernels_by_role={
            CascadeRole.FIRST:  k_first,
            CascadeRole.MIDDLE: k_middle,
            CascadeRole.LAST:   k_last,
        },
        accumulator_dtype="accfloat",  # FP32 accumulator (AM020 Ch. 4 p. 65)
        column=0,                       # AIE2P column 0; vertical chain
        starting_row=2,                 # First CoreTile row on AIE2P
    )

    # `chain.workers` is a list[Worker] ready to attach to your Program;
    # cascade routing between them is implicit via vertical placement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

# Cascade-stream wire width on AIE-ML / AIE2P (per AM020 Ch. 4 p. 67).
# The bit width is identical to one accumulator register slice. The
# canonical view in C++ is ``v16int32`` (16 lanes of int32) or ``v16
# accfloat`` (16 lanes of FP32 accumulator) — see
# ``aie_api/adf/stream.hpp`` ``cascade_stream_helper<accfloat,N>`` for
# the FP32 variant.
CASCADE_BITS: int = 512
CASCADE_LANES_INT32: int = 16
CASCADE_LANES_FP32: int = 16   # accfloat lanes per cascade transfer

# Supported AIE2P CoreTile starting row. AM020 Ch. 1 p. 6 (carried
# forward to AIE2P): row 0 = ShimTile, row 1 = MemTile, rows 2..N =
# CoreTiles. The cascade-stream chain must start at row >= 2.
_DEFAULT_CASCADE_STARTING_ROW: int = 2

class CascadeRole(IntEnum):
    """Position of a worker in the cascade chain.

    The role determines which C++ kernel variant the worker calls:

    - ``FIRST``: emits a cascade output but never reads from cascade.
      C++ kernel uses ``put_mcd(value)`` only.
    - ``MIDDLE``: reads from cascade input AND emits to cascade output.
      C++ kernel uses ``get_scd_v16int32()`` + ``put_mcd(value)``.
    - ``LAST``: reads from cascade input but never emits to cascade.
      C++ kernel uses ``get_scd_v16int32()`` only.
    - ``SOLO``: a chain of length 1 has no cascade neighbours; this
      role exists for callers building parameterized chains where N=1
      is a degenerate case. The C++ kernel is the standalone variant
      with no cascade ops.
    """

    SOLO = 0
    FIRST = 1
    MIDDLE = 2
    LAST = 3

@dataclass(frozen=True)
class _CascadeWorkerSpec:
    """Per-worker placement+role record in a cascade chain."""

    role: CascadeRole
    column: int
    row: int
    kernel_name: str  # informational; the actual Kernel obj is held on the worker

@dataclass(frozen=True)
class CascadeStreamChain:
    """A declarative description of a cascade-stream chain ready to be
    composed into an IRON :class:`aie.iron.Program`.

    Use :func:`cascade_stream_chain` to construct.

    Attributes
    ----------
    n_workers
        Number of vertically-adjacent CoreTiles in the chain. Must be
        >= 1. Length 1 produces a single :attr:`CascadeRole.SOLO`
        worker (no cascade routing, useful as a degenerate fallback
        when the caller's logic is parameterized over N).
    workers
        Ready-made list of ``aie.iron.Worker`` instances, one per
        chain position, with ``tile`` set to the appropriate CoreTile
        coordinates and the ``core_fn`` parameterized to call the
        kernel variant matching its :class:`CascadeRole`. Attach
        these to a :class:`aie.iron.Runtime` via ``rt.start(*chain.
        workers)`` (or equivalent).
    accumulator_dtype
        AIE accumulator type tag used by the chain. The MLIR-level
        cascade-flow ops carry this implicitly via the C++ kernel's
        ``put_mcd``/``get_scd`` width; this field is informational
        for the caller and gets recorded on each spec for debugging.
        Common values: ``"accfloat"`` (FP32 accumulator), ``"acc32"``
        (int32 accumulator), ``"acc48"`` (AIE1 only — not supported
        on AIE2P).
    cascade_flow_ops
        List of ``(source_spec, dest_spec)`` pairs representing the
        explicit ``aie.cascade_flow`` connections this chain implies.
        On AIE2P the upstream
        ``programming_examples/basic/matrix_multiplication/cascade/``
        relies on **implicit** cascade routing from vertical tile
        adjacency — no explicit ``cascade_flow`` ops are needed in
        that placed-style example. This field is provided for
        debuggability and for callers who want to lower the chain
        through an alternative path that DOES emit
        ``aie.cascade_flow`` explicitly.
    specs
        Underlying placement+role records (one per worker). Mostly
        useful for tests + INVENTORY-style topology dumps.
    """

    n_workers: int
    workers: list[Any] = field(default_factory=list)  # list[aie.iron.Worker]
    accumulator_dtype: str = "accfloat"
    cascade_flow_ops: list[tuple[_CascadeWorkerSpec, _CascadeWorkerSpec]] = field(
        default_factory=list
    )
    specs: list[_CascadeWorkerSpec] = field(default_factory=list)

    def role_of(self, idx: int) -> CascadeRole:
        """Role of the worker at chain position ``idx``."""
        return self.specs[idx].role

    def vertically_adjacent(self) -> bool:
        """Confirm specs[i] and specs[i+1] are vertically adjacent in
        the same column. Used by tests."""
        if len(self.specs) < 2:
            return True
        for a, b in zip(self.specs[:-1], self.specs[1:], strict=True):
            if a.column != b.column:
                return False
            if b.row - a.row != 1:
                return False
        return True

def _resolve_role(idx: int, n: int) -> CascadeRole:
    if n == 1:
        return CascadeRole.SOLO
    if idx == 0:
        return CascadeRole.FIRST
    if idx == n - 1:
        return CascadeRole.LAST
    return CascadeRole.MIDDLE

def cascade_stream_chain(
    *,
    n_workers: int,
    kernels_by_role: Mapping[CascadeRole, Any],
    core_fn_factories_by_role: Mapping[CascadeRole, Any] | None = None,
    accumulator_dtype: str = "accfloat",
    column: int = 0,
    starting_row: int = _DEFAULT_CASCADE_STARTING_ROW,
    fn_args_by_role: Mapping[CascadeRole, Sequence[Any]] | None = None,
    emit_explicit_cascade_flow: bool = False,
) -> CascadeStreamChain:
    """Build a cascade-stream chain of ``n_workers`` vertically-adjacent
    CoreTile workers.

    Parameters
    ----------
    n_workers
        Number of workers in the chain. Must be >= 1.
    kernels_by_role
        Mapping from :class:`CascadeRole` to the
        ``aie.iron.Kernel`` instance the worker at that role should
        call. The expected mapping for a chain of N >= 2 contains
        keys ``FIRST``, ``MIDDLE`` (if N >= 3), ``LAST``. For a
        chain of N == 1, only ``SOLO`` is required.
    core_fn_factories_by_role
        Optional mapping from role -> callable building the per-role
        ``core_fn`` (the body that gets compiled into the AIE core).
        If omitted, no Worker objects are constructed and
        :attr:`CascadeStreamChain.workers` is left empty (useful for
        tests that only need to inspect the placement topology, or
        for callers using the placed-style ``@core(...)`` decorator
        directly).
    accumulator_dtype
        Recorded on each spec for debugging. Default ``"accfloat"``
        (FP32 accumulator path; the right setting for the
        -cascade LSTM use case per AM020 Ch. 4 p. 65).
    column
        AIE2P column to place the chain in. Default 0. Single-column
        chains are sufficient for vertical cascade.
    starting_row
        First CoreTile row in the chain. Default 2 (AIE2P CoreTiles
        start at row 2). The chain occupies rows
        ``starting_row..starting_row+n_workers-1``.
    fn_args_by_role
        Optional mapping from role -> argument tuple to pass to the
        per-role ``core_fn``. Forwarded to :class:`aie.iron.Worker`.
    emit_explicit_cascade_flow
        If True, populate :attr:`CascadeStreamChain.cascade_flow_ops`
        with the source/dest pairs the chain implies. The
        unplaced/IRON-level emission of ``aie.cascade_flow`` ops is
        TODO; the upstream cascade reference design does not require
        them, and the placement pass infers cascade routing from
        vertical tile adjacency.

    Returns
    -------
    CascadeStreamChain
        Topology + (optionally) ready-to-use Worker list.

    Raises
    ------
    ValueError
        If ``n_workers < 1``, if a required role is missing from
        ``kernels_by_role``, or if ``starting_row + n_workers - 1``
        exceeds the AIE2P device row count (assumed safe up to row 5
        on a 4-CoreTile column; not validated against the actual
        ``Device``).
    """
    if n_workers < 1:
        raise ValueError(
            f"cascade_stream_chain: n_workers must be >= 1, got {n_workers}"
        )

    # Validate the role -> kernel mapping covers the chain.
    if n_workers == 1:
        required_roles = {CascadeRole.SOLO}
    elif n_workers == 2:
        required_roles = {CascadeRole.FIRST, CascadeRole.LAST}
    else:
        required_roles = {
            CascadeRole.FIRST,
            CascadeRole.MIDDLE,
            CascadeRole.LAST,
        }
    missing = required_roles - set(kernels_by_role.keys())
    if missing:
        raise ValueError(
            f"cascade_stream_chain: kernels_by_role missing required roles "
            f"{sorted(int(r) for r in missing)} for n_workers={n_workers}"
        )

    specs: list[_CascadeWorkerSpec] = []
    for idx in range(n_workers):
        role = _resolve_role(idx, n_workers)
        kernel = kernels_by_role[role]
        kernel_name = getattr(kernel, "name", None) or getattr(
            kernel, "_name", None
        ) or repr(kernel)
        specs.append(
            _CascadeWorkerSpec(
                role=role,
                column=column,
                row=starting_row + idx,
                kernel_name=str(kernel_name),
            )
        )

    cascade_flow_ops: list[tuple[_CascadeWorkerSpec, _CascadeWorkerSpec]] = []
    if emit_explicit_cascade_flow and len(specs) >= 2:
        cascade_flow_ops = list(zip(specs[:-1], specs[1:], strict=True))

    workers: list[Any] = []
    if core_fn_factories_by_role is not None:
        # Defer the IRON Worker construction until the caller actually
        # wants it — otherwise the test path (no NPU) doesn't need to
        # import aie.iron at all.
        from aie.iron import Worker  # type: ignore
        from aie.iron.device import Tile  # type: ignore

        for spec in specs:
            factory = core_fn_factories_by_role.get(spec.role)
            if factory is None:
                raise ValueError(
                    f"cascade_stream_chain: core_fn_factories_by_role missing "
                    f"role {spec.role!r}"
                )
            kernel = kernels_by_role[spec.role]
            args = list(fn_args_by_role.get(spec.role, [])) if fn_args_by_role else []
            args = args + [kernel]
            worker = Worker(
                factory(spec),
                fn_args=args,
                tile=Tile(spec.column, spec.row),
            )
            workers.append(worker)

    return CascadeStreamChain(
        n_workers=n_workers,
        workers=workers,
        accumulator_dtype=accumulator_dtype,
        cascade_flow_ops=cascade_flow_ops,
        specs=specs,
    )
