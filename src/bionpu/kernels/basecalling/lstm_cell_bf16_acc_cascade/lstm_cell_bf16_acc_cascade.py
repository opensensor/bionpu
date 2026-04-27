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

# lstm_cell_bf16_acc_cascade.py — IRON lowering for the 5-layer cascade-stream
#                                   LSTM stack -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Phase 2 / — promote Phase 1's wrapper-level cascade prototype
# (``bionpu/iron_extensions/cascade_stream.py``) to the fork's first-class
# IRON primitives:
#
#   - aie.iron.AccumFifo: the inter-tile FP32 accumulator hand-off,
#     replacing Phase 1's "implicit-vertical-placement + matched intrinsics"
#     convention. AccumFifo lowers to ``aie.cascade_flow`` between distinct
#     producer/consumer tiles (AM020 Ch. 4 p. 67).
#   - aie.iron.CascadeFifo: used reflexively at construction time as
#     the import-trigger that proves the fork wheel is loaded; the
#     load-bearing primitive for the LSTM stack is AccumFifo (FP32 state),
#     but importing both keeps the contract honest with the task brief.
#
# ** (Fix C) refactor — 2026-04-26**:
# Per the silicon-validated  prescription, the per-tile
# kernel functions must emit `vmov mcd` / `vmov scd` cascade-port
# intrinsics UNCONDITIONALLY (no peano-emitted `jnz` around them). The
# kernel side splits each role into 8 extern "C" functions
# (`_first_math` + `_first_putonly`, `_middle_math` + `_middle_getonly`
# + `_middle_putonly`, `_last_math` + `_last_getonly` + `_last_writeout`);
# this IRON file unrolls the inner `for g in range(N_GATES): for chunk
# in range(4)` loops at IRON-build time using plain Python `range(...)`
# (NOT `range_(...)`) and statically chooses which kernel function to
# call at each of the 16 (g, chunk) call-sites per ts. The per-call
# selection therefore happens at MLIR-emit time, not at runtime —
# producing 16 distinct `func.call` ops per ts per worker, each calling
# its appropriate per-cascade-presence kernel.
#
# The outer `for t in range_(L)` loop remains a runtime SCF loop so the
# per-ts work doesn't unroll L=334 times per worker.
#
# Topology (5 vertically-adjacent CoreTiles in column 0, rows 2..6):
#
#   row 2  Layer 0  FIRST   put_only  — bf16 input via ObjectFifo
#                                       FP32 h cascaded out via AccumFifo
#   row 3  Layer 1  MIDDLE  put_get   — FP32 h from AccumFifo upstream
#                                       FP32 h cascaded out via AccumFifo
#   row 4  Layer 2  MIDDLE  put_get
#   row 5  Layer 3  MIDDLE  put_get
#   row 6  Layer 4  LAST    get_only  — FP32 h from AccumFifo upstream
#                                       bf16 y_t output via ObjectFifo
#
# Wire format (host-visible — closure: 3 host BOs total):
# - input ObjectFifo at row 2: bf16 (L=334) × 96 lanes (= 's
#     input wire format, single-layer entry point)
#   - **single consolidated weight ObjectFifo** (shim → memtile): one
#     parent frame per chunk-slot carrying N_LAYERS chunks back-to-back
#     in cascade-consume order [L0_chunk, L1_chunk, ..., L4_chunk].
#     The memtile splits each parent frame by per-layer offset into 5
#     per-cell sub-fifos via the IRON ``ObjectFifoHandle.split`` API
#     (lowers to ``aie.objectfifo.link`` with ``dst_offsets``). Reduces
#     the host BO count from the original 7 (1 in + 5 wb + 1 out) to
#     3 (1 in + 1 wb_consolidated + 1 out), fitting the AIE2P 5-slot
# kernel ABI cap. See in  + the upstream
#     ``programming_examples/basic/matrix_multiplication/cascade/cascade.py``
#     pattern reference.
#   - output ObjectFifo at row 6: bf16 (L=334) × 96 lanes
#   - inter-row AccumFifo: 4 cascade hops (rows 2→3, 3→4, 4→5, 5→6),
#     each carries 6 cascade words (HIDDEN_VECS) per timestep =
#     1 * 6 * 16 * 4 bytes = 384 B per timestep. FP32 accumulator dtype.
#
# **Bonito alternating directions**: not handled in the fabric. Bonito's
# alternating forward/reverse pattern (layers 4,6,8 forward; 5,7 reverse)
# is implemented host-side: the `weights_per_layer` argument is rebuilt
# with reverse layers' weights pre-flipped, and the dispatch input/output
# is flipped time-axis around the stack boundary. The cascade fabric
# always runs forward-time (this matches 's stack contract — the
# alternation is data layout, not kernel direction).

from __future__ import annotations

import argparse
import sys

import numpy as np

from bionpu.kernels.basecalling.lstm_cell_bf16_acc.lstm_cell_bf16_acc import (
    _BF16,
    BIAS_LEN,
    HIDDEN,
    INPUT_DIM,
    N_GATES,
    N_HALVES_PER_GATE,
    WEIGHT_HALF_LEN,
)

N_LAYERS = 5

# Cascade chain placement: column 0, rows 2..6.
COLUMN = 0
STARTING_ROW = 2

def my_dorado_fast_lstm_stack_bf16_acc_cascade(
    dev,
    L: int,
    *,
    full_chain_probe_placement: bool = False,
):
    """Emit MLIR for the 5-layer cascade-stream LSTM stack.

    The emit pass:
    1. Constructs eight role-specific Kernel objects bound to
       ``lstm_layer_cascade.o`` symbols ( Fix C; was three).
    2. Constructs the 5 vertically-adjacent CoreTile placements.
    3. Builds the 4 inter-row AccumFifos (FP32 cascade hand-off).
    4. Builds the input ObjectFifo (terminating at row 2) and output
       ObjectFifo (originating at row 6) plus 5 per-row weight
       ObjectFifos.
    5. Builds 5 Workers, one per row, with per-role core_fns wiring
       the appropriate ObjectFifos + AccumFifo handles. Per-call kernel
       selection is now STATIC (build-time Python conditional on the
       (g, chunk) index) — 's Fix C invariant.
    6. Builds the Runtime fill/drain sequence for input + 5 weight
       buffers + output.
    7. Resolves the Program.

    Returns the resolved MLIR module (as Program.resolve_program would).
    """
    # Late imports — touching aie.iron transitively imports MLIR ops; we
    # want pure-Python tests to be able to import this module without
    # requiring the wheel to be available at every import.
    from aie.iron import ( # noqa: F401 -- contract
        AccumFifo,
        CascadeFifo,
        Kernel,
        ObjectFifo,
        Program,
        Runtime,
        Worker,
    )
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    # closed upstream by fork PR feature/-fifohandle-fix:
    # AccumFifoHandle now overrides all_of_endpoints() to return endpoint-
    # typed objects directly (no longer trips on the bypassed
    # ObjectFifoHandle.__init__'s missing _object_fifo). The earlier outer-
    # repo monkey-patch (per-instance all_of_endpoints lambda binding) is
    # no longer required and has been removed.

    # ---- Type declarations (per-cell shape; identical to ) ----
    in_step_ty = np.ndarray[(INPUT_DIM,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    chunk_with_bias_ty = np.ndarray[
        (BIAS_LEN + WEIGHT_HALF_LEN,), np.dtype[_BF16]
    ]
    chunk_len = BIAS_LEN + WEIGHT_HALF_LEN  # 768 + 4608 = 5376 bf16

    n_weight_chunks = L * N_GATES * N_HALVES_PER_GATE  # per layer

    in_total_ty = np.ndarray[(L * INPUT_DIM,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]

    # ---- closure: consolidated-weight memtile-split layout ----
    # The L3→L2 (shim → memtile) ObjectFifo carries one *chunk-frame*
    # per parent transfer: N_LAYERS chunks concatenated back-to-back in
    # cascade-consume order [layer0_chunk, layer1_chunk, ..., layer4_chunk].
    # The memtile splits each parent frame by per-layer offset into 5
    # per-cell L2→L1 sub-fifos (one per cascade-stage core), each
    # presenting one ``chunk_with_bias_ty`` per acquire — exactly what
    # the existing per-role kernels already consume.
    #
    # Mirrors the upstream pattern in
    #   ``programming_examples/basic/matrix_multiplication/cascade/cascade.py``
    # (one shim DMA per col, fanned out at the memtile via
    # ``object_fifo_link`` / ``ObjectFifoHandle.split(of_offsets, ...)``).
    chunk_x_layers_ty = np.ndarray[
        (N_LAYERS * chunk_len,), np.dtype[_BF16]
    ]
    weight_total_ty = np.ndarray[
        (n_weight_chunks * N_LAYERS * chunk_len,), np.dtype[_BF16]
    ]
    per_layer_chunk_offsets = [i * chunk_len for i in range(N_LAYERS)]

    # ---- Per-role Kernel declarations ( Fix C: 8 kernels) ----
    #
    # Each role used to be a single monolithic kernel (FIRST `_put_only`,
    # MIDDLE `_put_get`, LAST `_get_only`) with cascade put/get gated by
    # `(g==3,chunk_idx==3)` / `(g==0,chunk_idx==0)` runtime branches. That
    # pattern wedged the AIE2P firmware-side cascade-port watchdog
    #.  silicon-validated that splitting into
    # cascade-presence-specific functions (and emitting cascade ops
    # unconditionally inside any function that contains them) lifts the
    # wedge. Fix C achieves this WITHOUT the per-layer L1 / memtile
    # capacity wall that Fix B production hit.
    #
    # Per-call kernel selection is done STATICALLY in the IRON core_fn
    # below (plain Python `for g in range(N_GATES): for chunk in
    # range(4)` unrolling, with `if (g, chunk) == (3, 3): ... else: ...`
    # selecting the kernel symbol). The MLIR emit pass produces 16
    # distinct `func.call` ops per ts per worker, each calling its
    # appropriate per-cascade-presence kernel.

    # FIRST role (Layer 0) — math-only + cascade-put-at-(3,3).
    first_math = Kernel(
        "dorado_fast_lstm_layer_cascade_first_math",
        "lstm_layer_cascade.o",
        [in_step_ty, chunk_with_bias_ty, np.int32, np.int32, np.int32],
    )
    first_putonly = Kernel(
        "dorado_fast_lstm_layer_cascade_first_putonly",
        "lstm_layer_cascade.o",
        [in_step_ty, chunk_with_bias_ty, np.int32, np.int32, np.int32],
    )
    # MIDDLE role (Layers 1-3) — math-only + cascade-get-at-(0,0) +
    # cascade-put-at-(3,3). No x_t buffer (cascade is the input).
    middle_math = Kernel(
        "dorado_fast_lstm_layer_cascade_middle_math",
        "lstm_layer_cascade.o",
        [chunk_with_bias_ty, np.int32, np.int32, np.int32],
    )
    middle_getonly = Kernel(
        "dorado_fast_lstm_layer_cascade_middle_getonly",
        "lstm_layer_cascade.o",
        [chunk_with_bias_ty, np.int32, np.int32, np.int32],
    )
    middle_putonly = Kernel(
        "dorado_fast_lstm_layer_cascade_middle_putonly",
        "lstm_layer_cascade.o",
        [chunk_with_bias_ty, np.int32, np.int32, np.int32],
    )
    # LAST role (Layer 4) — math-only + cascade-get-at-(0,0) +
    # writeout-at-(3,3). y_t arg passed to all three for ABI uniformity
    # (so the IRON core_fn passes the same args to all three kernels).
    last_math = Kernel(
        "dorado_fast_lstm_layer_cascade_last_math",
        "lstm_layer_cascade.o",
        [chunk_with_bias_ty, out_step_ty, np.int32, np.int32, np.int32],
    )
    last_getonly = Kernel(
        "dorado_fast_lstm_layer_cascade_last_getonly",
        "lstm_layer_cascade.o",
        [chunk_with_bias_ty, out_step_ty, np.int32, np.int32, np.int32],
    )
    last_writeout = Kernel(
        "dorado_fast_lstm_layer_cascade_last_writeout",
        "lstm_layer_cascade.o",
        [chunk_with_bias_ty, out_step_ty, np.int32, np.int32, np.int32],
    )

    # ---- Tile placements ----
    # AIE2P / Strix has only 4 CoreTiles per column (rows 2..5; AM020
    # Ch. 1 p. 6 carry-forward: row 0 = ShimTile, row 1 = MemTile,
    # rows 2..5 = CoreTiles). The original DESIGN.md assumed rows 2..6
    # (5-deep), which is silicon-infeasible per
    # ``aie.iron.device.NPU2`` validation: rows = 6 (indices 0..5).
    # Filed as ().
    #
    # Cascade-direction constraint (AIELowerCascadeFlows.cpp): source
    # tile must be South of OR West of the destination tile. So the
    # chain direction must run **upwards** (high row → low row) or
    # **rightwards** (low col → high col). Vertical-down (row 2→3)
    # is rejected.
    #
    # Topology: 4-deep cascade in column 0 going UPWARD (row 5 → row 2)
    # for layers 0..3, then a horizontal hop **across** to column 1
    # for layer 4. The hop direction is West→East: src=(0, row),
    # dst=(1, row). Per AccumFifo's `_check_vertical_adjacency` an
    # informational warning is raised; AM020 Appendix A p. 80 Figure
    # 45 documents horizontal cascade routing as supported.
    #
    # Layer 0 (FIRST role) → bottom-most: (0, 5)
    # Layer 1 (MIDDLE)              :   (0, 4)
    # Layer 2 (MIDDLE)              :   (0, 3)
    # Layer 3 (MIDDLE)              :   (0, 2)
    # Layer 4 (LAST role) → horizontal: (1, 2)
    tiles = (
        [
            Tile(0, 5),  # Layer 0  — FIRST
            Tile(1, 5),  # Layer 1  — MIDDLE (west-to-east first hop)
            Tile(1, 4),  # Layer 2  — MIDDLE
            Tile(1, 3),  # Layer 3  — MIDDLE
            Tile(1, 2),  # Layer 4  — LAST
        ]
        if full_chain_probe_placement
        else [
            Tile(0, 5),  # Layer 0  — FIRST  (bottom of col 0)
            Tile(0, 4),  # Layer 1  — MIDDLE
            Tile(0, 3),  # Layer 2  — MIDDLE
            Tile(0, 2),  # Layer 3  — MIDDLE (top of col 0)
            Tile(1, 2),  # Layer 4  — LAST   (horizontal hop into col 1)
        ]
    )

    # ---- AccumFifos (FP32 inter-tile state hand-off) ----
    # 4 cascade hops: row5→row4, row4→row3, row3→row2, row2→(col1,row2).
    # 16 lanes of accfloat = one 512-bit cascade word; the C++ kernel
    # issues 6 puts/gets per timestep (HIDDEN_VECS).
    accum_fifos = []
    for i in range(N_LAYERS - 1):
        af = AccumFifo(
            producer=tiles[i],
            consumer=tiles[i + 1],
            dtype="accfloat",
            lanes=16,
            name=f"h_cascade_{i}_{i+1}",
        )
        accum_fifos.append(af)

    # ---- ObjectFifos: 1 input + 1 consolidated weight + 1 output ----
    # closure: the host sees exactly 3 BOs (was 7 before).
    # Input (shim → row 5 / Layer 0 FIRST):
    of_input = ObjectFifo(in_step_ty, name="input_in", depth=2)

    # Consolidated weights (shim → memtile). Each parent frame =
    # N_LAYERS × chunk_len bf16 elements concatenated in cascade order:
    # [L0_chunk, L1_chunk, ..., L(N_LAYERS-1)_chunk].
    of_weights_consolidated = ObjectFifo(
        chunk_x_layers_ty, name="weight_in_all", depth=2
    )

    # Memtile-split: 5 per-layer L2→L1 sub-fifos, each presenting
    # chunk_with_bias_ty objects (single per-layer chunks). The
    # ``split`` API lowers to ``aie.objectfifo.link`` with
    # ``dst_offsets = per_layer_chunk_offsets``. Each per-layer
    # consumer core (cascade FIRST/MIDDLE/LAST) acquires from its
    # per-layer sub-fifo unchanged from the pre- topology.
    of_weights = of_weights_consolidated.cons().split(
        per_layer_chunk_offsets,
        obj_types=[chunk_with_bias_ty] * N_LAYERS,
        names=[f"weight_in_L{i}" for i in range(N_LAYERS)],
    )

    # Output (LAST cascade core → shim):
    of_output = ObjectFifo(out_step_ty, name="output_out", depth=2)

    # ---- Per-role core_fn factories ( Fix C: static call-site selection) ----
    #
    # Each core_fn's outer `for t in range_(L_)` is a runtime SCF loop
    # (compiled to MLIR's scf.for). The inner `for g in range(N_GATES):
    # for chunk in range(4)` use plain Python `range` so they're
    # UNROLLED at IRON-emit time — producing 16 distinct func.call ops
    # per ts in the lowered MLIR. This lets us select the kernel symbol
    # at each call-site based on the loop-invariant (g, chunk) pair,
    # producing per-tile peano-emitted ELF where:
    #
    #   - 15 (or 14) of the 16 calls go to the `_math` kernel (no
    #     cascade intrinsics) — these have ZERO `vmov mcd` / `vmov scd`
    #     instructions and therefore ZERO `jnz` around them.
    #   - 1 (or 2) calls go to the cascade-firing `_putonly` /
    #     `_getonly` / `_writeout` kernels — these contain their cascade
    #     intrinsic UNCONDITIONALLY at function entry/exit (no `jnz`
    #     around them in the per-tile ELF).
    #
    # This achieves 's silicon-validated invariant ("0 jnz
    # around vmov mcd / vmov scd in any function on any tile") for the
    # production LSTM kernel, without the L1 / memtile capacity wall
    # that Fix B production hit.

    def make_core_first(L_):
        def core_body(of_input_h, of_weight_h, accum_out_h,
                       k_math, k_putonly):
            from aie.dialects import arith
            from aie.ir import IntegerType
            i32 = IntegerType.get_signless(32)
            for t in range_(L_):
                elem_in = of_input_h.acquire(1)
                t_i32 = arith.index_cast(i32, t)
                # 16 statically-unrolled call-sites. (g, chunk) is a
                # build-time constant pair, lowered as arith.constant
                # i32 ops into the func.call args.
                for g_static in range(N_GATES):
                    for chunk_static in range(N_HALVES_PER_GATE):
                        w_chunk = of_weight_h.acquire(1)
                        g_const = arith.constant(i32, g_static)
                        chunk_const = arith.constant(i32, chunk_static)
                        if g_static == N_GATES - 1 and chunk_static == N_HALVES_PER_GATE - 1:
                            # (g==3, chunk==3): cascade-put + final state update.
                            k_putonly(elem_in, w_chunk, g_const, t_i32, chunk_const)
                        else:
                            # All other (g, chunk): math only.
                            k_math(elem_in, w_chunk, g_const, t_i32, chunk_const)
                        of_weight_h.release(1)
                of_input_h.release(1)
                # The cascade transfer is implicit per-cycle on the
                # AccumFifo wire — the IRON-level acquire/release is
                # a no-op (cascade has no buffer depth). We still call
                # them so the worker fn_args registers the handle for
                # the placement pass.
                accum_out_h.acquire(1)
                accum_out_h.release(1)
        return core_body

    def make_core_middle(L_):
        def core_body(accum_in_h, of_weight_h, accum_out_h,
                       k_math, k_getonly, k_putonly):
            from aie.dialects import arith
            from aie.ir import IntegerType
            i32 = IntegerType.get_signless(32)
            for t in range_(L_):
                accum_in_h.acquire(1)
                t_i32 = arith.index_cast(i32, t)
                for g_static in range(N_GATES):
                    for chunk_static in range(N_HALVES_PER_GATE):
                        w_chunk = of_weight_h.acquire(1)
                        g_const = arith.constant(i32, g_static)
                        chunk_const = arith.constant(i32, chunk_static)
                        if g_static == 0 and chunk_static == 0:
                            # (g==0, chunk==0): cascade-get + math.
                            k_getonly(w_chunk, g_const, t_i32, chunk_const)
                        elif g_static == N_GATES - 1 and chunk_static == N_HALVES_PER_GATE - 1:
                            # (g==3, chunk==3): math + final state + cascade-put.
                            k_putonly(w_chunk, g_const, t_i32, chunk_const)
                        else:
                            # All other (g, chunk): math only.
                            k_math(w_chunk, g_const, t_i32, chunk_const)
                        of_weight_h.release(1)
                accum_in_h.release(1)
                accum_out_h.acquire(1)
                accum_out_h.release(1)
        return core_body

    def make_core_last(L_):
        def core_body(accum_in_h, of_weight_h, of_output_h,
                       k_math, k_getonly, k_writeout):
            from aie.dialects import arith
            from aie.ir import IntegerType
            i32 = IntegerType.get_signless(32)
            for t in range_(L_):
                accum_in_h.acquire(1)
                elem_out = of_output_h.acquire(1)
                t_i32 = arith.index_cast(i32, t)
                for g_static in range(N_GATES):
                    for chunk_static in range(N_HALVES_PER_GATE):
                        w_chunk = of_weight_h.acquire(1)
                        g_const = arith.constant(i32, g_static)
                        chunk_const = arith.constant(i32, chunk_static)
                        if g_static == 0 and chunk_static == 0:
                            # (g==0, chunk==0): cascade-get + math.
                            k_getonly(w_chunk, elem_out, g_const, t_i32, chunk_const)
                        elif g_static == N_GATES - 1 and chunk_static == N_HALVES_PER_GATE - 1:
                            # (g==3, chunk==3): math + final state + writeout.
                            k_writeout(w_chunk, elem_out, g_const, t_i32, chunk_const)
                        else:
                            # All other (g, chunk): math only (y_t arg unused).
                            k_math(w_chunk, elem_out, g_const, t_i32, chunk_const)
                        of_weight_h.release(1)
                accum_in_h.release(1)
                of_output_h.release(1)
        return core_body

    # ---- Workers (one per row) ----
    workers = []

    # FIRST role — row 5 (bottom of col 0)
    workers.append(Worker(
        make_core_first(L),
        fn_args=[
            of_input.cons(),
            of_weights[0].cons(),
            accum_fifos[0].prod(),
            first_math,
            first_putonly,
        ],
        tile=tiles[0],
    ))

    # MIDDLE roles — rows 4..2 (Layer 1, Layer 2, Layer 3)
    for i in range(1, N_LAYERS - 1):
        workers.append(Worker(
            make_core_middle(L),
            fn_args=[
                accum_fifos[i - 1].cons(),
                of_weights[i].cons(),
                accum_fifos[i].prod(),
                middle_math,
                middle_getonly,
                middle_putonly,
            ],
            tile=tiles[i],
        ))

    # LAST role — col 1 row 2 (horizontal hop from col 0 row 2)
    workers.append(Worker(
        make_core_last(L),
        fn_args=[
            accum_fifos[-1].cons(),
            of_weights[-1].cons(),
            of_output.prod(),
            last_math,
            last_getonly,
            last_writeout,
        ],
        tile=tiles[-1],
    ))

    # ---- AccumFifo / Worker tile-identity reconcile (workaround) ----
    # ``Worker.__init__`` calls ``tile.copy()`` on its tile arg, so the
    # Worker's ``_tile`` is a *distinct* :class:`Tile` instance from the
    # original ``tiles[i]`` even though col/row match. The AccumFifo
    # captured the originals at construction (via ``_coerce_to_tile``
    # which returns the input unchanged). At program-resolve time the
    # device's per-coord merge map (``Device._resolved_coords``) is
    # only consulted via ``id(tile)``-keyed lookup, which means the
    # AccumFifo's stored ``_prod_tile`` / ``_cons_tile`` never have
    # their ``.op`` set, and ``AccumFifo.resolve`` raises
    # ``ValueError: Cannot get op before it is set``.
    #
    # Workaround (bionpu-side, no fork edit): after Workers are built,
    # rebind each AccumFifo's tile fields to the Worker's copy so the
    # AccumFifo and its consumer Worker share Tile identity. The
    # device's ``resolve_tile`` is then guaranteed to set ``.op`` on
    # the same Tile instance the AccumFifo holds. Filed as a
    # **fork-side** follow-up under (the existing closure
    # for the *prior* AccumFifo bypass bug); the proper fix is to make
    # ``Tile.op`` look through ``_resolved_coords`` when ``_op`` is
    # None, but that's a fork-side patch we cannot land here without
    # colliding with T4's worktree.
    for i, af in enumerate(accum_fifos):
        af._prod_tile = workers[i]._tile
        af._cons_tile = workers[i + 1]._tile

    # ---- Runtime sequence ----
    # Host buffer order for the consolidated weights tensor:
    #   for each chunk-frame f in [0, n_weight_chunks):
    #     for each layer l in [0, N_LAYERS):
    #       chunk_len bf16 elements = layer l's chunk-f payload
    #
    # i.e. layer-minor / chunk-major. The host runner is responsible
    # for pre-interleaving per-layer expanded weight slabs into this
    # layout before shipping.
    rt = Runtime()
    with rt.sequence(
        in_total_ty,
        weight_total_ty,
        out_total_ty,
    ) as (X, W_all, Y):
        rt.start(*workers)
        rt.fill(of_input.prod(), X)
        rt.fill(of_weights_consolidated.prod(), W_all)
        rt.drain(of_output.cons(), Y, wait=True)

    return Program(dev, rt).resolve_program()

def my_dorado_fast_lstm_stack_bf16_acc_cascade_weight_stream_chain(dev, L: int):
    """Emit a split-free production-weight-frame stream-chain bisection.

    This keeps the production consolidated weight ABI but removes the
    memtile ``ObjectFifoHandle.split`` fanout. Each tile forwards the
    current five-layer chunk frame to the next tile before it computes
    that chunk. The first frame of a timestep is forwarded before the
    tile waits on cascade, so downstream can keep receiving weights while
    upstream computes toward its final cascade put. This avoids both the
    memtile split/fanout and the 16-frame cache that exceeded BD limits.
    """
    from aie.iron import AccumFifo, Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    in_step_ty = np.ndarray[(INPUT_DIM,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    chunk_len = BIAS_LEN + WEIGHT_HALF_LEN
    frame_ty = np.ndarray[(N_LAYERS * chunk_len,), np.dtype[_BF16]]
    n_weight_chunks = L * N_GATES * N_HALVES_PER_GATE
    in_total_ty = np.ndarray[(L * INPUT_DIM,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]
    weight_total_ty = np.ndarray[
        (n_weight_chunks * N_LAYERS * chunk_len,), np.dtype[_BF16]
    ]

    first_math = Kernel(
        "dorado_prod_frame_first_math",
        "lstm_layer_cascade.o",
        [in_step_ty, frame_ty, np.int32, np.int32, np.int32],
    )
    first_putonly = Kernel(
        "dorado_prod_frame_first_putonly",
        "lstm_layer_cascade.o",
        [in_step_ty, frame_ty, np.int32, np.int32, np.int32],
    )
    middle_math = Kernel(
        "dorado_prod_frame_middle_math",
        "lstm_layer_cascade.o",
        [frame_ty, np.int32, np.int32, np.int32, np.int32],
    )
    middle_getonly = Kernel(
        "dorado_prod_frame_middle_getonly",
        "lstm_layer_cascade.o",
        [frame_ty, np.int32, np.int32, np.int32, np.int32],
    )
    middle_putonly = Kernel(
        "dorado_prod_frame_middle_putonly",
        "lstm_layer_cascade.o",
        [frame_ty, np.int32, np.int32, np.int32, np.int32],
    )
    last_math = Kernel(
        "dorado_prod_frame_last_math",
        "lstm_layer_cascade.o",
        [frame_ty, out_step_ty, np.int32, np.int32, np.int32],
    )
    last_getonly = Kernel(
        "dorado_prod_frame_last_getonly",
        "lstm_layer_cascade.o",
        [frame_ty, out_step_ty, np.int32, np.int32, np.int32],
    )
    last_writeout = Kernel(
        "dorado_prod_frame_last_writeout",
        "lstm_layer_cascade.o",
        [frame_ty, out_step_ty, np.int32, np.int32, np.int32],
    )
    forward_frame = Kernel(
        "dorado_prod_weight_frame_forward",
        "lstm_layer_cascade.o",
        [frame_ty, frame_ty],
    )

    tiles = [
        Tile(0, 5),
        Tile(0, 4),
        Tile(0, 3),
        Tile(0, 2),
        Tile(1, 2),
    ]
    accum_fifos = [
        AccumFifo(
            producer=tiles[i],
            consumer=tiles[i + 1],
            dtype="accfloat",
            lanes=16,
            name=f"prod_stream_h_{i}_{i+1}",
        )
        for i in range(N_LAYERS - 1)
    ]
    of_input = ObjectFifo(in_step_ty, name="prod_stream_input", depth=2)
    of_weight = ObjectFifo(frame_ty, name="prod_stream_weight_in", depth=1)
    weight_chain = [
        ObjectFifo(frame_ty, name=f"prod_stream_weight_{i}_{i+1}", depth=1)
        for i in range(N_LAYERS - 1)
    ]
    of_output = ObjectFifo(out_step_ty, name="prod_stream_output", depth=2)

    def _const_i32(value: int):
        from aie.dialects import arith
        from aie.ir import IntegerType

        return arith.constant(IntegerType.get_signless(32), value)

    def first_body(of_input_h, of_weight_h, of_weight_out_h, accum_out_h,
                   k_math, k_putonly, k_forward):
        from aie.dialects import arith
        from aie.ir import IntegerType

        i32 = IntegerType.get_signless(32)
        for t in range_(L):
            elem_in = of_input_h.acquire(1)
            t_i32 = arith.index_cast(i32, t)
            for g_static in range(N_GATES):
                for chunk_static in range(N_HALVES_PER_GATE):
                    f_in = of_weight_h.acquire(1)
                    f_out = of_weight_out_h.acquire(1)
                    k_forward(f_in, f_out)
                    of_weight_out_h.release(1)
                    g_const = _const_i32(g_static)
                    chunk_const = _const_i32(chunk_static)
                    if (
                        g_static == N_GATES - 1
                        and chunk_static == N_HALVES_PER_GATE - 1
                    ):
                        k_putonly(elem_in, f_in, g_const, t_i32, chunk_const)
                    else:
                        k_math(elem_in, f_in, g_const, t_i32, chunk_const)
                    of_weight_h.release(1)
            of_input_h.release(1)
            accum_out_h.acquire(1)
            accum_out_h.release(1)

    def middle_body(accum_in_h, of_weight_h, of_weight_out_h, accum_out_h,
                    layer, k_math, k_getonly, k_putonly, k_forward):
        from aie.dialects import arith
        from aie.ir import IntegerType

        i32 = IntegerType.get_signless(32)
        layer_const = _const_i32(layer)
        for t in range_(L):
            t_i32 = arith.index_cast(i32, t)
            for g_static in range(N_GATES):
                for chunk_static in range(N_HALVES_PER_GATE):
                    f_in = of_weight_h.acquire(1)
                    f_out = of_weight_out_h.acquire(1)
                    k_forward(f_in, f_out)
                    of_weight_out_h.release(1)
                    if g_static == 0 and chunk_static == 0:
                        accum_in_h.acquire(1)
                    g_const = _const_i32(g_static)
                    chunk_const = _const_i32(chunk_static)
                    if g_static == 0 and chunk_static == 0:
                        k_getonly(
                            f_in, layer_const, g_const, t_i32, chunk_const,
                        )
                    elif (
                        g_static == N_GATES - 1
                        and chunk_static == N_HALVES_PER_GATE - 1
                    ):
                        k_putonly(
                            f_in, layer_const, g_const, t_i32, chunk_const,
                        )
                    else:
                        k_math(f_in, layer_const, g_const, t_i32, chunk_const)
                    of_weight_h.release(1)
            accum_in_h.release(1)
            accum_out_h.acquire(1)
            accum_out_h.release(1)

    def last_body(accum_in_h, of_weight_h, of_output_h,
                  k_math, k_getonly, k_writeout):
        from aie.dialects import arith
        from aie.ir import IntegerType

        i32 = IntegerType.get_signless(32)
        for t in range_(L):
            elem_out = of_output_h.acquire(1)
            t_i32 = arith.index_cast(i32, t)
            for g_static in range(N_GATES):
                for chunk_static in range(N_HALVES_PER_GATE):
                    f_in = of_weight_h.acquire(1)
                    if g_static == 0 and chunk_static == 0:
                        accum_in_h.acquire(1)
                    g_const = _const_i32(g_static)
                    chunk_const = _const_i32(chunk_static)
                    if g_static == 0 and chunk_static == 0:
                        k_getonly(f_in, elem_out, g_const, t_i32, chunk_const)
                    elif (
                        g_static == N_GATES - 1
                        and chunk_static == N_HALVES_PER_GATE - 1
                    ):
                        k_writeout(f_in, elem_out, g_const, t_i32, chunk_const)
                    else:
                        k_math(f_in, elem_out, g_const, t_i32, chunk_const)
                    of_weight_h.release(1)
            accum_in_h.release(1)
            of_output_h.release(1)

    workers = [
        Worker(
            first_body,
            fn_args=[
                of_input.cons(),
                of_weight.cons(),
                weight_chain[0].prod(),
                accum_fifos[0].prod(),
                first_math,
                first_putonly,
                forward_frame,
            ],
            tile=tiles[0],
        )
    ]
    for i in range(1, N_LAYERS - 1):
        workers.append(Worker(
            middle_body,
            fn_args=[
                accum_fifos[i - 1].cons(),
                weight_chain[i - 1].cons(),
                weight_chain[i].prod(),
                accum_fifos[i].prod(),
                i,
                middle_math,
                middle_getonly,
                middle_putonly,
                forward_frame,
            ],
            tile=tiles[i],
        ))
    workers.append(Worker(
        last_body,
        fn_args=[
            accum_fifos[-1].cons(),
            weight_chain[-1].cons(),
            of_output.prod(),
            last_math,
            last_getonly,
            last_writeout,
        ],
        tile=tiles[-1],
    ))

    for i, af in enumerate(accum_fifos):
        af._prod_tile = workers[i]._tile
        af._cons_tile = workers[i + 1]._tile

    rt = Runtime()
    with rt.sequence(in_total_ty, weight_total_ty, out_total_ty) as (X, W, Y):
        rt.start(*workers)
        rt.fill(of_input.prod(), X)
        rt.fill(of_weight.prod(), W)
        rt.drain(of_output.cons(), Y, wait=True)

    return Program(dev, rt).resolve_program()

def _production_direct_stream_edges():
    """Construct route markers for the production-weight suffix stream chain."""
    from aie.iron import ObjectFifo

    stream_route_marker_ty = np.ndarray[(1,), np.dtype[np.int32]]
    return [
        ObjectFifo(
            stream_route_marker_ty,
            name=f"prod_ds_stream_{i}_{i + 1}",
            depth=2,
            aie_stream=2,
            aie_stream_port=0,
        )
        for i in range(N_LAYERS - 1)
    ]

def _lower_production_direct_stream_flows(mlir: str) -> str:
    """Replace direct-stream ObjectFifo markers with explicit Core:0 flows."""
    routes = [
        ("%logical_core", "%logical_core_0"),
        ("%logical_core_0", "%logical_core_1"),
        ("%logical_core_1", "%logical_core_2"),
        ("%logical_core_2", "%logical_core_3"),
    ]
    for i, (src, dst) in enumerate(routes):
        old = (
            f"    aie.objectfifo @prod_ds_stream_{i}_{i + 1}({src}, "
            f"{{{dst}}}, 2 : i32) {{aie_stream = 2 : i32, "
            "aie_stream_port = 0 : i32} : !aie.objectfifo<memref<1xi32>> \n"
        )
        new = f"    aie.flow({src}, Core : 0, {dst}, Core : 0)\n"
        if old not in mlir:
            raise RuntimeError(
                "production direct-stream flow lowering did not find "
                f"expected marker prod_ds_stream_{i}_{i + 1}"
            )
        mlir = mlir.replace(old, new)
    return mlir

def my_dorado_fast_lstm_stack_bf16_acc_cascade_weight_direct_stream(dev, L: int):
    """Emit production-weight delivery with direct streamed suffixes.

    The first tile receives the existing production five-layer weight frame.
    It consumes layer 0 locally and sends layers 1..4 over Core stream 0.
    Each middle tile receives the suffix, buffers only its own 5376-bf16
    chunk, forwards the remaining tail, and then computes. This keeps the
    production host ABI but removes the full-frame inter-core ObjectFifo
    pressure that exceeded tile memory in the stream-chain bisection.
    """
    from aie.iron import AccumFifo, Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    in_step_ty = np.ndarray[(INPUT_DIM,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    chunk_len = BIAS_LEN + WEIGHT_HALF_LEN
    frame_ty = np.ndarray[(N_LAYERS * chunk_len,), np.dtype[_BF16]]
    n_weight_chunks = L * N_GATES * N_HALVES_PER_GATE
    in_total_ty = np.ndarray[(L * INPUT_DIM,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]
    weight_total_ty = np.ndarray[
        (n_weight_chunks * N_LAYERS * chunk_len,), np.dtype[_BF16]
    ]

    first_math = Kernel(
        "dorado_prod_stream_first_math",
        "lstm_layer_cascade.o",
        [in_step_ty, frame_ty, np.int32, np.int32, np.int32],
    )
    first_putonly = Kernel(
        "dorado_prod_stream_first_putonly",
        "lstm_layer_cascade.o",
        [in_step_ty, frame_ty, np.int32, np.int32, np.int32],
    )
    middle_math = Kernel(
        "dorado_prod_stream_middle_math",
        "lstm_layer_cascade.o",
        [np.int32, np.int32, np.int32, np.int32],
    )
    middle_getonly = Kernel(
        "dorado_prod_stream_middle_getonly",
        "lstm_layer_cascade.o",
        [np.int32, np.int32, np.int32, np.int32],
    )
    middle_putonly = Kernel(
        "dorado_prod_stream_middle_putonly",
        "lstm_layer_cascade.o",
        [np.int32, np.int32, np.int32, np.int32],
    )
    last_math = Kernel(
        "dorado_prod_stream_last_math",
        "lstm_layer_cascade.o",
        [out_step_ty, np.int32, np.int32, np.int32],
    )
    last_getonly = Kernel(
        "dorado_prod_stream_last_getonly",
        "lstm_layer_cascade.o",
        [out_step_ty, np.int32, np.int32, np.int32],
    )
    last_writeout = Kernel(
        "dorado_prod_stream_last_writeout",
        "lstm_layer_cascade.o",
        [out_step_ty, np.int32, np.int32, np.int32],
    )

    tiles = [
        Tile(0, 5),
        Tile(0, 4),
        Tile(0, 3),
        Tile(0, 2),
        Tile(1, 2),
    ]
    accum_fifos = [
        AccumFifo(
            producer=tiles[i],
            consumer=tiles[i + 1],
            dtype="accfloat",
            lanes=16,
            name=f"prod_ds_h_{i}_{i + 1}",
        )
        for i in range(N_LAYERS - 1)
    ]
    of_input = ObjectFifo(in_step_ty, name="prod_ds_input", depth=2)
    of_weight = ObjectFifo(frame_ty, name="prod_ds_weight_in", depth=1)
    of_output = ObjectFifo(out_step_ty, name="prod_ds_output", depth=2)
    stream_edges = _production_direct_stream_edges()

    def _const_i32(value: int):
        from aie.dialects import arith
        from aie.ir import IntegerType

        return arith.constant(IntegerType.get_signless(32), value)

    def first_body(of_input_h, of_weight_h, _stream_out, accum_out_h,
                   k_math, k_putonly):
        from aie.dialects import arith
        from aie.ir import IntegerType

        i32 = IntegerType.get_signless(32)
        for t in range_(L):
            elem_in = of_input_h.acquire(1)
            t_i32 = arith.index_cast(i32, t)
            for g_static in range(N_GATES):
                for chunk_static in range(N_HALVES_PER_GATE):
                    frame = of_weight_h.acquire(1)
                    g_const = _const_i32(g_static)
                    chunk_const = _const_i32(chunk_static)
                    if (
                        g_static == N_GATES - 1
                        and chunk_static == N_HALVES_PER_GATE - 1
                    ):
                        k_putonly(elem_in, frame, g_const, t_i32, chunk_const)
                    else:
                        k_math(elem_in, frame, g_const, t_i32, chunk_const)
                    of_weight_h.release(1)
            of_input_h.release(1)
            accum_out_h.acquire(1)
            accum_out_h.release(1)

    def middle_body(accum_in_h, _stream_in, _stream_out, accum_out_h,
                    layers_remaining, k_math, k_getonly, k_putonly):
        from aie.dialects import arith
        from aie.ir import IntegerType

        i32 = IntegerType.get_signless(32)
        layers_const = _const_i32(layers_remaining)
        for t in range_(L):
            t_i32 = arith.index_cast(i32, t)
            for g_static in range(N_GATES):
                for chunk_static in range(N_HALVES_PER_GATE):
                    if g_static == 0 and chunk_static == 0:
                        accum_in_h.acquire(1)
                    g_const = _const_i32(g_static)
                    chunk_const = _const_i32(chunk_static)
                    if g_static == 0 and chunk_static == 0:
                        k_getonly(
                            layers_const, g_const, t_i32, chunk_const,
                        )
                    elif (
                        g_static == N_GATES - 1
                        and chunk_static == N_HALVES_PER_GATE - 1
                    ):
                        k_putonly(
                            layers_const, g_const, t_i32, chunk_const,
                        )
                    else:
                        k_math(layers_const, g_const, t_i32, chunk_const)
            accum_in_h.release(1)
            accum_out_h.acquire(1)
            accum_out_h.release(1)

    def last_body(accum_in_h, _stream_in, of_output_h,
                  k_math, k_getonly, k_writeout):
        from aie.dialects import arith
        from aie.ir import IntegerType

        i32 = IntegerType.get_signless(32)
        for t in range_(L):
            elem_out = of_output_h.acquire(1)
            t_i32 = arith.index_cast(i32, t)
            for g_static in range(N_GATES):
                for chunk_static in range(N_HALVES_PER_GATE):
                    if g_static == 0 and chunk_static == 0:
                        accum_in_h.acquire(1)
                    g_const = _const_i32(g_static)
                    chunk_const = _const_i32(chunk_static)
                    if g_static == 0 and chunk_static == 0:
                        k_getonly(elem_out, g_const, t_i32, chunk_const)
                    elif (
                        g_static == N_GATES - 1
                        and chunk_static == N_HALVES_PER_GATE - 1
                    ):
                        k_writeout(elem_out, g_const, t_i32, chunk_const)
                    else:
                        k_math(elem_out, g_const, t_i32, chunk_const)
            accum_in_h.release(1)
            of_output_h.release(1)

    workers = [
        Worker(
            first_body,
            fn_args=[
                of_input.cons(),
                of_weight.cons(),
                stream_edges[0].prod(),
                accum_fifos[0].prod(),
                first_math,
                first_putonly,
            ],
            tile=tiles[0],
        )
    ]
    for i in range(1, N_LAYERS - 1):
        workers.append(Worker(
            middle_body,
            fn_args=[
                accum_fifos[i - 1].cons(),
                stream_edges[i - 1].cons(),
                stream_edges[i].prod(),
                accum_fifos[i].prod(),
                N_LAYERS - i,
                middle_math,
                middle_getonly,
                middle_putonly,
            ],
            tile=tiles[i],
        ))
    workers.append(Worker(
        last_body,
        fn_args=[
            accum_fifos[-1].cons(),
            stream_edges[-1].cons(),
            of_output.prod(),
            last_math,
            last_getonly,
            last_writeout,
        ],
        tile=tiles[-1],
    ))

    for i, af in enumerate(accum_fifos):
        af._prod_tile = workers[i]._tile
        af._cons_tile = workers[i + 1]._tile

    rt = Runtime()
    with rt.sequence(in_total_ty, weight_total_ty, out_total_ty) as (X, W, Y):
        rt.start(*workers)
        rt.fill(of_input.prod(), X)
        rt.fill(of_weight.prod(), W)
        rt.drain(of_output.cons(), Y, wait=True)

    return Program(dev, rt).resolve_program()

def my_dorado_prod_weight_direct_stream_probe(dev, L: int):
    """Emit a no-cascade direct-stream probe for production weight frames."""
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    in_step_ty = np.ndarray[(INPUT_DIM,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    chunk_len = BIAS_LEN + WEIGHT_HALF_LEN
    frame_ty = np.ndarray[(N_LAYERS * chunk_len,), np.dtype[_BF16]]
    n_weight_chunks = L * N_GATES * N_HALVES_PER_GATE
    in_total_ty = np.ndarray[(L * INPUT_DIM,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]
    weight_total_ty = np.ndarray[
        (n_weight_chunks * N_LAYERS * chunk_len,), np.dtype[_BF16]
    ]

    first_kernel = Kernel(
        "dorado_prod_stream_probe_first",
        "lstm_layer_cascade.o",
        [frame_ty],
    )
    middle_kernel = Kernel(
        "dorado_prod_stream_probe_middle",
        "lstm_layer_cascade.o",
        [np.int32],
    )
    last_kernel = Kernel(
        "dorado_prod_stream_probe_last",
        "lstm_layer_cascade.o",
        [out_step_ty],
    )

    tiles = [
        Tile(0, 5),
        Tile(0, 4),
        Tile(0, 3),
        Tile(0, 2),
        Tile(1, 2),
    ]
    of_input = ObjectFifo(in_step_ty, name="prod_ds_probe_input", depth=2)
    of_weight = ObjectFifo(frame_ty, name="prod_ds_probe_weight_in", depth=1)
    of_output = ObjectFifo(out_step_ty, name="prod_ds_probe_output", depth=2)
    stream_edges = _production_direct_stream_edges()

    def _const_i32(value: int):
        from aie.dialects import arith
        from aie.ir import IntegerType

        return arith.constant(IntegerType.get_signless(32), value)

    def first_body(of_input_h, of_weight_h, _stream_out, k_first):
        for _ in range_(L):
            _elem_in = of_input_h.acquire(1)
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    frame = of_weight_h.acquire(1)
                    k_first(frame)
                    of_weight_h.release(1)
            of_input_h.release(1)

    def middle_body(_stream_in, _stream_out, layers_remaining, k_middle):
        layers_const = _const_i32(layers_remaining)
        for _ in range_(L):
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    k_middle(layers_const)

    def last_body(_stream_in, of_output_h, k_last):
        for _ in range_(L):
            elem_out = of_output_h.acquire(1)
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    k_last(elem_out)
            of_output_h.release(1)

    workers = [
        Worker(
            first_body,
            fn_args=[
                of_input.cons(),
                of_weight.cons(),
                stream_edges[0].prod(),
                first_kernel,
            ],
            tile=tiles[0],
        )
    ]
    for i in range(1, N_LAYERS - 1):
        workers.append(Worker(
            middle_body,
            fn_args=[
                stream_edges[i - 1].cons(),
                stream_edges[i].prod(),
                N_LAYERS - i,
                middle_kernel,
            ],
            tile=tiles[i],
        ))
    workers.append(Worker(
        last_body,
        fn_args=[
            stream_edges[-1].cons(),
            of_output.prod(),
            last_kernel,
        ],
        tile=tiles[-1],
    ))

    rt = Runtime()
    with rt.sequence(in_total_ty, weight_total_ty, out_total_ty) as (X, W, Y):
        rt.start(*workers)
        rt.fill(of_input.prod(), X)
        rt.fill(of_weight.prod(), W)
        rt.drain(of_output.cons(), Y, wait=True)

    return Program(dev, rt).resolve_program()

def my_dorado_prod_weight_direct_layer_probe(dev, L: int):
    """Emit a five-weight-BO direct layer ingress probe.

    This removes both memtile split and tile-to-tile weight forwarding.
    Each layer receives its own chunk stream directly from runtime. The
    runner keeps the existing consolidated weight file format and splits it
    into five BOs at launch.
    """
    from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    in_step_ty = np.ndarray[(INPUT_DIM,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    chunk_len = BIAS_LEN + WEIGHT_HALF_LEN
    chunk_ty = np.ndarray[(chunk_len,), np.dtype[_BF16]]
    n_weight_chunks = L * N_GATES * N_HALVES_PER_GATE
    in_total_ty = np.ndarray[(L * INPUT_DIM,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]
    weight_layer_ty = np.ndarray[(n_weight_chunks * chunk_len,), np.dtype[_BF16]]

    touch_kernel = Kernel(
        "dorado_prod_direct_layer_probe_touch",
        "lstm_layer_cascade.o",
        [chunk_ty],
    )
    last_kernel = Kernel(
        "dorado_prod_direct_layer_probe_last",
        "lstm_layer_cascade.o",
        [chunk_ty, out_step_ty],
    )

    tiles = [
        Tile(0, 5),
        Tile(0, 4),
        Tile(0, 3),
        Tile(0, 2),
        Tile(1, 2),
    ]
    of_input = ObjectFifo(in_step_ty, name="prod_dl_probe_input", depth=2)
    of_weights = [
        ObjectFifo(chunk_ty, name=f"prod_dl_probe_weight_l{i}", depth=1)
        for i in range(N_LAYERS)
    ]
    of_output = ObjectFifo(out_step_ty, name="prod_dl_probe_output", depth=2)

    def first_body(of_input_h, of_weight_h, k_touch):
        for _ in range_(L):
            _elem_in = of_input_h.acquire(1)
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    k_touch(chunk)
                    of_weight_h.release(1)
            of_input_h.release(1)

    def middle_body(of_weight_h, k_touch):
        for _ in range_(L):
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    k_touch(chunk)
                    of_weight_h.release(1)

    def last_body(of_weight_h, of_output_h, k_last):
        for _ in range_(L):
            elem_out = of_output_h.acquire(1)
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    k_last(chunk, elem_out)
                    of_weight_h.release(1)
            of_output_h.release(1)

    workers = [
        Worker(
            first_body,
            fn_args=[of_input.cons(), of_weights[0].cons(), touch_kernel],
            tile=tiles[0],
        )
    ]
    for i in range(1, N_LAYERS - 1):
        workers.append(Worker(
            middle_body,
            fn_args=[of_weights[i].cons(), touch_kernel],
            tile=tiles[i],
        ))
    workers.append(Worker(
        last_body,
        fn_args=[of_weights[-1].cons(), of_output.prod(), last_kernel],
        tile=tiles[-1],
    ))

    rt = Runtime()
    with rt.sequence(
        in_total_ty,
        weight_layer_ty,
        weight_layer_ty,
        weight_layer_ty,
        weight_layer_ty,
        weight_layer_ty,
        out_total_ty,
    ) as (X, W0, W1, W2, W3, W4, Y):
        rt.start(*workers)
        rt.fill(of_input.prod(), X)
        rt.fill(of_weights[0].prod(), W0)
        rt.fill(of_weights[1].prod(), W1)
        rt.fill(of_weights[2].prod(), W2)
        rt.fill(of_weights[3].prod(), W3)
        rt.fill(of_weights[4].prod(), W4)
        rt.drain(of_output.cons(), Y, wait=True)

    return Program(dev, rt).resolve_program()

def my_dorado_prod_weight_direct_group_probe(dev, L: int):
    """Emit a three-weight-BO grouped layer ingress probe."""
    return _my_dorado_prod_weight_direct_group_probe(
        dev,
        L,
        touch_symbol="dorado_prod_direct_layer_probe_touch",
        last_symbol="dorado_prod_direct_layer_probe_last",
        name_prefix="prod_dg_probe",
    )

def my_dorado_prod_weight_direct_group_math_probe(dev, L: int):
    """Emit grouped weight ingress plus tile-local AIE math pressure."""
    return _my_dorado_prod_weight_direct_group_probe(
        dev,
        L,
        touch_symbol="dorado_prod_direct_group_math_probe_touch",
        last_symbol="dorado_prod_direct_group_math_probe_last",
        name_prefix="prod_dgm_probe",
    )

def my_dorado_prod_weight_direct_group_cascade_probe(dev, L: int):
    """Emit grouped weight ingress plus one vertical cascade edge."""
    return _my_dorado_prod_weight_direct_group_probe(
        dev,
        L,
        touch_symbol="dorado_prod_direct_group_math_probe_touch",
        last_symbol="dorado_prod_direct_group_math_probe_last",
        name_prefix="prod_dgc_probe",
        cascade_first_symbol="dorado_prod_direct_group_cascade_probe_first",
        cascade_second_symbol="dorado_prod_direct_group_cascade_probe_second",
    )

def my_dorado_prod_weight_direct_group_vertical_chain_probe(dev, L: int):
    """Emit grouped weight ingress plus a 3-hop vertical cascade chain."""
    return _my_dorado_prod_weight_direct_group_probe(
        dev,
        L,
        touch_symbol="dorado_prod_direct_group_math_probe_touch",
        last_symbol="dorado_prod_direct_group_cascade_probe_last",
        name_prefix="prod_dgv_probe",
        cascade_first_symbol="dorado_prod_direct_group_cascade_probe_first",
        cascade_second_symbol="dorado_prod_direct_group_cascade_probe_middle",
        cascade_last_symbol="dorado_prod_direct_group_cascade_probe_last",
        cascade_start_layer=1,
    )

def my_dorado_prod_weight_consolidated_vertical_chain_probe(dev, L: int):
    """Emit one consolidated weight BO plus a 3-hop vertical cascade chain."""
    return _my_dorado_prod_weight_consolidated_chain_probe(
        dev,
        L,
        name_prefix="prod_cv_probe",
        horizontal_last_hop=False,
    )

def my_dorado_prod_weight_consolidated_horizontal_probe(dev, L: int):
    """Emit consolidated weights plus a final horizontal cascade hop."""
    return _my_dorado_prod_weight_consolidated_chain_probe(
        dev,
        L,
        name_prefix="prod_ch_probe",
        horizontal_last_hop=True,
    )

def my_dorado_prod_weight_consolidated_full_chain_probe(dev, L: int):
    """Emit one consolidated weight BO plus a full 5-layer cascade chain."""
    return _my_dorado_prod_weight_consolidated_chain_probe(
        dev,
        L,
        name_prefix="prod_cf_probe",
        horizontal_last_hop=False,
        cascade_start_layer=0,
    )

def my_dorado_prod_weight_consolidated_full_chain_lstm_probe(dev, L: int):
    """Emit production LSTM kernels on the passing full-chain placement."""
    return my_dorado_fast_lstm_stack_bf16_acc_cascade(
        dev,
        L,
        full_chain_probe_placement=True,
    )

def _my_dorado_prod_weight_consolidated_chain_probe(
    dev,
    L: int,
    *,
    name_prefix: str,
    horizontal_last_hop: bool,
    cascade_start_layer: int = 1,
):
    from aie.iron import AccumFifo, Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    in_step_ty = np.ndarray[(INPUT_DIM,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    chunk_len = BIAS_LEN + WEIGHT_HALF_LEN
    chunk_ty = np.ndarray[(chunk_len,), np.dtype[_BF16]]
    chunk_x_layers_ty = np.ndarray[(N_LAYERS * chunk_len,), np.dtype[_BF16]]
    n_weight_chunks = L * N_GATES * N_HALVES_PER_GATE
    in_total_ty = np.ndarray[(L * INPUT_DIM,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]
    weight_total_ty = np.ndarray[
        (n_weight_chunks * N_LAYERS * chunk_len,), np.dtype[_BF16]
    ]

    touch_kernel = Kernel(
        "dorado_prod_direct_group_math_probe_touch",
        "lstm_layer_cascade.o",
        [chunk_ty],
    )
    start_kernel = Kernel(
        "dorado_prod_direct_group_cascade_probe_first",
        "lstm_layer_cascade.o",
        [chunk_ty],
    )
    middle_kernel = Kernel(
        "dorado_prod_direct_group_cascade_probe_middle",
        "lstm_layer_cascade.o",
        [chunk_ty],
    )
    last_kernel = Kernel(
        "dorado_prod_direct_group_cascade_probe_last",
        "lstm_layer_cascade.o",
        [chunk_ty, out_step_ty],
    )

    tiles = (
        [
            Tile(0, 5),
            Tile(1, 5),
            Tile(1, 4),
            Tile(1, 3),
            Tile(2, 3),
        ]
        if horizontal_last_hop
        else [
            Tile(0, 5),
            Tile(1, 5),
            Tile(1, 4),
            Tile(1, 3),
            Tile(1, 2),
        ]
    )
    of_input = ObjectFifo(in_step_ty, name=f"{name_prefix}_input", depth=2)
    of_weights_all = ObjectFifo(
        chunk_x_layers_ty, name=f"{name_prefix}_weight_all", depth=1
    )
    of_weights = of_weights_all.cons().split(
        [i * chunk_len for i in range(N_LAYERS)],
        obj_types=[chunk_ty] * N_LAYERS,
        names=[f"{name_prefix}_weight_l{i}" for i in range(N_LAYERS)],
    )
    of_weights = [of_w.cons() for of_w in of_weights]
    of_output = ObjectFifo(out_step_ty, name=f"{name_prefix}_output", depth=2)
    accum_fifos = [
        AccumFifo(
            producer=tiles[layer],
            consumer=tiles[layer + 1],
            dtype="accfloat",
            lanes=16,
            name=f"{name_prefix}_cascade_{layer}_{layer + 1}",
        )
        for layer in range(cascade_start_layer, N_LAYERS - 1)
    ]

    def first_body(of_input_h, of_weight_h, k_touch):
        for _ in range_(L):
            _elem_in = of_input_h.acquire(1)
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    k_touch(chunk)
                    of_weight_h.release(1)
            of_input_h.release(1)

    def start_body(of_weight_h, accum_out_h, k_start):
        for _ in range_(L):
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    k_start(chunk)
                    of_weight_h.release(1)
                    accum_out_h.acquire(1)
                    accum_out_h.release(1)

    def first_cascade_body(of_input_h, of_weight_h, accum_out_h, k_start):
        for _ in range_(L):
            _elem_in = of_input_h.acquire(1)
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    k_start(chunk)
                    of_weight_h.release(1)
                    accum_out_h.acquire(1)
                    accum_out_h.release(1)
            of_input_h.release(1)

    def middle_body(of_weight_h, accum_in_h, accum_out_h, k_middle):
        for _ in range_(L):
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    accum_in_h.acquire(1)
                    k_middle(chunk)
                    accum_in_h.release(1)
                    of_weight_h.release(1)
                    accum_out_h.acquire(1)
                    accum_out_h.release(1)

    def last_body(of_weight_h, accum_in_h, of_output_h, k_last):
        for _ in range_(L):
            elem_out = of_output_h.acquire(1)
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    accum_in_h.acquire(1)
                    k_last(chunk, elem_out)
                    accum_in_h.release(1)
                    of_weight_h.release(1)
            of_output_h.release(1)

    if cascade_start_layer == 0:
        workers = [
            Worker(
                first_cascade_body,
                fn_args=[
                    of_input.cons(),
                    of_weights[0],
                    accum_fifos[0].prod(),
                    start_kernel,
                ],
                tile=tiles[0],
            ),
            Worker(
                middle_body,
                fn_args=[
                    of_weights[1],
                    accum_fifos[0].cons(),
                    accum_fifos[1].prod(),
                    middle_kernel,
                ],
                tile=tiles[1],
            ),
            Worker(
                middle_body,
                fn_args=[
                    of_weights[2],
                    accum_fifos[1].cons(),
                    accum_fifos[2].prod(),
                    middle_kernel,
                ],
                tile=tiles[2],
            ),
            Worker(
                middle_body,
                fn_args=[
                    of_weights[3],
                    accum_fifos[2].cons(),
                    accum_fifos[3].prod(),
                    middle_kernel,
                ],
                tile=tiles[3],
            ),
            Worker(
                last_body,
                fn_args=[
                    of_weights[4],
                    accum_fifos[3].cons(),
                    of_output.prod(),
                    last_kernel,
                ],
                tile=tiles[4],
            ),
        ]
    else:
        workers = [
            Worker(
                first_body,
                fn_args=[of_input.cons(), of_weights[0], touch_kernel],
                tile=tiles[0],
            ),
            Worker(
                start_body,
                fn_args=[of_weights[1], accum_fifos[0].prod(), start_kernel],
                tile=tiles[1],
            ),
            Worker(
                middle_body,
                fn_args=[
                    of_weights[2],
                    accum_fifos[0].cons(),
                    accum_fifos[1].prod(),
                    middle_kernel,
                ],
                tile=tiles[2],
            ),
            Worker(
                middle_body,
                fn_args=[
                    of_weights[3],
                    accum_fifos[1].cons(),
                    accum_fifos[2].prod(),
                    middle_kernel,
                ],
                tile=tiles[3],
            ),
            Worker(
                last_body,
                fn_args=[
                    of_weights[4],
                    accum_fifos[2].cons(),
                    of_output.prod(),
                    last_kernel,
                ],
                tile=tiles[4],
            ),
        ]
    for af in accum_fifos:
        for worker in workers:
            if worker._tile.col == af._prod_tile.col and worker._tile.row == af._prod_tile.row:
                af._prod_tile = worker._tile
            if worker._tile.col == af._cons_tile.col and worker._tile.row == af._cons_tile.row:
                af._cons_tile = worker._tile

    rt = Runtime()
    with rt.sequence(in_total_ty, weight_total_ty, out_total_ty) as (X, W, Y):
        rt.start(*workers)
        rt.fill(of_input.prod(), X)
        rt.fill(of_weights_all.prod(), W)
        rt.drain(of_output.cons(), Y, wait=True)

    return Program(dev, rt).resolve_program()

def _my_dorado_prod_weight_direct_group_probe(
    dev,
    L: int,
    *,
    touch_symbol: str,
    last_symbol: str,
    name_prefix: str,
    cascade_first_symbol: str | None = None,
    cascade_second_symbol: str | None = None,
    cascade_last_symbol: str | None = None,
    cascade_start_layer: int = 0,
):
    from aie.iron import AccumFifo, Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    in_step_ty = np.ndarray[(INPUT_DIM,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    chunk_len = BIAS_LEN + WEIGHT_HALF_LEN
    chunk_ty = np.ndarray[(chunk_len,), np.dtype[_BF16]]
    group2_ty = np.ndarray[(2 * chunk_len,), np.dtype[_BF16]]
    n_weight_chunks = L * N_GATES * N_HALVES_PER_GATE
    in_total_ty = np.ndarray[(L * INPUT_DIM,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]
    weight_group2_ty = np.ndarray[
        (n_weight_chunks * 2 * chunk_len,), np.dtype[_BF16]
    ]
    weight_group1_ty = np.ndarray[(n_weight_chunks * chunk_len,), np.dtype[_BF16]]

    touch_kernel = Kernel(
        touch_symbol,
        "lstm_layer_cascade.o",
        [chunk_ty],
    )
    last_kernel = Kernel(
        last_symbol,
        "lstm_layer_cascade.o",
        [chunk_ty, out_step_ty],
    )
    cascade_first_kernel = (
        Kernel(cascade_first_symbol, "lstm_layer_cascade.o", [chunk_ty])
        if cascade_first_symbol is not None
        else None
    )
    cascade_second_kernel = (
        Kernel(cascade_second_symbol, "lstm_layer_cascade.o", [chunk_ty])
        if cascade_second_symbol is not None
        else None
    )
    cascade_last_kernel = (
        Kernel(cascade_last_symbol, "lstm_layer_cascade.o", [chunk_ty, out_step_ty])
        if cascade_last_symbol is not None
        else None
    )

    if cascade_start_layer == 1:
        tiles = [
            Tile(0, 5),
            Tile(1, 5),
            Tile(1, 4),
            Tile(1, 3),
            Tile(1, 2),
        ]
    else:
        tiles = [
            Tile(0, 5),
            Tile(0, 4),
            Tile(1, 5),
            Tile(1, 4),
            Tile(1, 3),
        ]
    of_input = ObjectFifo(in_step_ty, name=f"{name_prefix}_input", depth=2)
    of_w01 = ObjectFifo(group2_ty, name=f"{name_prefix}_weight_l01", depth=1)
    of_w23 = ObjectFifo(group2_ty, name=f"{name_prefix}_weight_l23", depth=1)
    of_w4 = ObjectFifo(chunk_ty, name=f"{name_prefix}_weight_l4", depth=1)
    of_w0, of_w1 = of_w01.cons().split(
        [0, chunk_len],
        obj_types=[chunk_ty, chunk_ty],
        names=[f"{name_prefix}_weight_l0", f"{name_prefix}_weight_l1"],
    )
    of_w2, of_w3 = of_w23.cons().split(
        [0, chunk_len],
        obj_types=[chunk_ty, chunk_ty],
        names=[f"{name_prefix}_weight_l2", f"{name_prefix}_weight_l3"],
    )
    of_weights = [
        of_w0.cons(),
        of_w1.cons(),
        of_w2.cons(),
        of_w3.cons(),
        of_w4.cons(),
    ]
    of_output = ObjectFifo(out_step_ty, name=f"{name_prefix}_output", depth=2)
    accum_fifos = []
    if cascade_first_kernel is not None:
        cascade_last_layer = N_LAYERS - 1 if cascade_last_kernel is not None else 1
        for layer in range(cascade_start_layer, cascade_last_layer):
            accum_fifos.append(AccumFifo(
                producer=tiles[layer],
                consumer=tiles[layer + 1],
                dtype="accfloat",
                lanes=16,
                name=f"{name_prefix}_cascade_{layer}_{layer + 1}",
            ))

    def first_body(of_input_h, of_weight_h, k_touch):
        for _ in range_(L):
            _elem_in = of_input_h.acquire(1)
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    k_touch(chunk)
                    of_weight_h.release(1)
            of_input_h.release(1)

    def first_cascade_body(of_input_h, of_weight_h, accum_out_h, k_first):
        for _ in range_(L):
            _elem_in = of_input_h.acquire(1)
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    k_first(chunk)
                    of_weight_h.release(1)
                    accum_out_h.acquire(1)
                    accum_out_h.release(1)
            of_input_h.release(1)

    def middle_body(of_weight_h, k_touch):
        for _ in range_(L):
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    k_touch(chunk)
                    of_weight_h.release(1)

    def middle_cascade_body(of_weight_h, accum_in_h, k_second):
        for _ in range_(L):
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    accum_in_h.acquire(1)
                    k_second(chunk)
                    accum_in_h.release(1)
                    of_weight_h.release(1)

    def start_cascade_put_body(of_weight_h, accum_out_h, k_first):
        for _ in range_(L):
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    k_first(chunk)
                    of_weight_h.release(1)
                    accum_out_h.acquire(1)
                    accum_out_h.release(1)

    def middle_cascade_put_body(of_weight_h, accum_in_h, accum_out_h, k_middle):
        for _ in range_(L):
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    accum_in_h.acquire(1)
                    k_middle(chunk)
                    accum_in_h.release(1)
                    of_weight_h.release(1)
                    accum_out_h.acquire(1)
                    accum_out_h.release(1)

    def last_cascade_body(of_weight_h, accum_in_h, of_output_h, k_last):
        for _ in range_(L):
            elem_out = of_output_h.acquire(1)
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    accum_in_h.acquire(1)
                    k_last(chunk, elem_out)
                    accum_in_h.release(1)
                    of_weight_h.release(1)
            of_output_h.release(1)

    def last_body(of_weight_h, of_output_h, k_last):
        for _ in range_(L):
            elem_out = of_output_h.acquire(1)
            for _g_static in range(N_GATES):
                for _chunk_static in range(N_HALVES_PER_GATE):
                    chunk = of_weight_h.acquire(1)
                    k_last(chunk, elem_out)
                    of_weight_h.release(1)
            of_output_h.release(1)

    if cascade_first_kernel is None:
        workers = [
            Worker(
                first_body,
                fn_args=[of_input.cons(), of_weights[0], touch_kernel],
                tile=tiles[0],
            )
        ]
        middle_start = 1
    elif cascade_start_layer == 0:
        workers = [
            Worker(
                first_cascade_body,
                fn_args=[
                    of_input.cons(),
                    of_weights[0],
                    accum_fifos[0].prod(),
                    cascade_first_kernel,
                ],
                tile=tiles[0],
            ),
            Worker(
                middle_cascade_body,
                fn_args=[
                    of_weights[1],
                    accum_fifos[0].cons(),
                    cascade_second_kernel,
                ],
                tile=tiles[1],
            ),
        ]
        middle_start = 2
    else:
        workers = [
            Worker(
                first_body,
                fn_args=[of_input.cons(), of_weights[0], touch_kernel],
                tile=tiles[0],
            ),
            Worker(
                start_cascade_put_body,
                fn_args=[
                    of_weights[1],
                    accum_fifos[0].prod(),
                    cascade_first_kernel,
                ],
                tile=tiles[1],
            ),
            Worker(
                middle_cascade_put_body,
                fn_args=[
                    of_weights[2],
                    accum_fifos[0].cons(),
                    accum_fifos[1].prod(),
                    cascade_second_kernel,
                ],
                tile=tiles[2],
            ),
            Worker(
                middle_cascade_put_body,
                fn_args=[
                    of_weights[3],
                    accum_fifos[1].cons(),
                    accum_fifos[2].prod(),
                    cascade_second_kernel,
                ],
                tile=tiles[3],
            ),
        ]
        middle_start = N_LAYERS - 1
    for i in range(middle_start, N_LAYERS - 1):
        workers.append(Worker(
            middle_body,
            fn_args=[of_weights[i], touch_kernel],
            tile=tiles[i],
        ))
    if cascade_start_layer == 1:
        workers.append(Worker(
            last_cascade_body,
            fn_args=[
                of_weights[-1],
                accum_fifos[-1].cons(),
                of_output.prod(),
                cascade_last_kernel,
            ],
            tile=tiles[-1],
        ))
    else:
        workers.append(Worker(
            last_body,
            fn_args=[of_weights[-1], of_output.prod(), last_kernel],
            tile=tiles[-1],
        ))
    for af in accum_fifos:
        for worker in workers:
            if worker._tile.col == af._prod_tile.col and worker._tile.row == af._prod_tile.row:
                af._prod_tile = worker._tile
            if worker._tile.col == af._cons_tile.col and worker._tile.row == af._cons_tile.row:
                af._cons_tile = worker._tile

    rt = Runtime()
    with rt.sequence(
        in_total_ty,
        weight_group2_ty,
        weight_group2_ty,
        weight_group1_ty,
        out_total_ty,
    ) as (X, W01, W23, W4, Y):
        rt.start(*workers)
        rt.fill(of_input.prod(), X)
        rt.fill(of_w01.prod(), W01)
        rt.fill(of_w23.prod(), W23)
        rt.fill(of_w4.prod(), W4)
        rt.drain(of_output.cons(), Y, wait=True)

    return Program(dev, rt).resolve_program()

def my_dorado_hidden_state_cascade_only(
    dev,
    L: int,
    *,
    math_pressure: bool = False,
    weight_pressure: bool = False,
    weight_chain: bool = False,
):
    """Emit the post-CRISPR Dorado hidden-state handoff bisection.

    This is deliberately not the production LSTM. It preserves the
    production cascade payload shape (96 bf16 input lanes converted to
    six 512-bit FP32 accumulator words, then passed across the same
    five-tile vertical+horizontal chain), while removing the
    consolidated weight ObjectFifo/memtile split. With
    ``math_pressure=False`` it also removes heavy LSTM gate math. With
    ``math_pressure=True`` it adds tile-local static state and nonlinear
    ``aie_api`` math pressure while preserving the same no-weight ABI.
    With ``weight_pressure=True`` it adds ordinary weight ObjectFifo
    pressure while still avoiding memtile split/fanout. ``weight_chain``
    forwards the same per-timestep weight frame across the five cores via
    four ordinary ObjectFifos so every tile consumes weight traffic while
    the host ABI stays input + one weights buffer + output.

    Use this as the first silicon probe before reintroducing:
      1. static LSTM gate/state math,
      2. simple per-tile weight delivery,
      3. consolidated memtile-split weight fanout.
    """
    from aie.iron import AccumFifo, Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    in_step_ty = np.ndarray[(INPUT_DIM,), np.dtype[_BF16]]
    out_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    weight_step_ty = np.ndarray[(HIDDEN,), np.dtype[_BF16]]
    in_total_ty = np.ndarray[(L * INPUT_DIM,), np.dtype[_BF16]]
    out_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]
    weight_total_ty = np.ndarray[(L * HIDDEN,), np.dtype[_BF16]]
    if weight_chain and not weight_pressure:
        raise ValueError("weight_chain requires weight_pressure")

    first_kernel = Kernel(
        (
            "dorado_hidden_weight_first"
            if weight_pressure
            else "dorado_hidden_math_first"
            if math_pressure
            else "dorado_hidden_stream_first"
        ),
        "lstm_layer_cascade.o",
        [in_step_ty, weight_step_ty]
        if weight_pressure
        else [in_step_ty],
    )
    middle_kernel = Kernel(
        "dorado_hidden_weight_chain_middle_compute"
        if weight_chain
        else "dorado_hidden_math_middle"
        if math_pressure
        else "dorado_hidden_stream_middle",
        "lstm_layer_cascade.o",
        [weight_step_ty] if weight_chain else [],
    )
    weight_forward_kernel = (
        Kernel(
            "dorado_hidden_weight_forward",
            "lstm_layer_cascade.o",
            [weight_step_ty, weight_step_ty],
        )
        if weight_chain
        else None
    )
    last_kernel = Kernel(
        "dorado_hidden_weight_chain_last"
        if weight_chain
        else "dorado_hidden_math_last"
        if math_pressure
        else "dorado_hidden_stream_last",
        "lstm_layer_cascade.o",
        [weight_step_ty, out_step_ty] if weight_chain else [out_step_ty],
    )

    tiles = [
        Tile(0, 5),
        Tile(0, 4),
        Tile(0, 3),
        Tile(0, 2),
        Tile(1, 2),
    ]
    accum_fifos = []
    for i in range(N_LAYERS - 1):
        accum_fifos.append(AccumFifo(
            producer=tiles[i],
            consumer=tiles[i + 1],
            dtype="accfloat",
            lanes=16,
            name=f"hidden_only_cascade_{i}_{i+1}",
        ))

    of_input = ObjectFifo(in_step_ty, name="hidden_only_input", depth=2)
    of_output = ObjectFifo(out_step_ty, name="hidden_only_output", depth=2)
    of_weight = (
        ObjectFifo(weight_step_ty, name="hidden_weight_in", depth=2)
        if weight_pressure
        else None
    )
    weight_chain_fifos = [
        ObjectFifo(weight_step_ty, name=f"hidden_weight_chain_{i}_{i+1}", depth=2)
        for i in range(N_LAYERS - 1)
    ] if weight_chain else []

    def first_body(of_input_h, accum_out_h, kernel):
        for _ in range_(L):
            elem_in = of_input_h.acquire(1)
            kernel(elem_in)
            of_input_h.release(1)
            accum_out_h.acquire(1)
            accum_out_h.release(1)

    def first_weight_body(of_input_h, of_weight_h, accum_out_h, kernel):
        for _ in range_(L):
            elem_in = of_input_h.acquire(1)
            elem_w = of_weight_h.acquire(1)
            kernel(elem_in, elem_w)
            of_weight_h.release(1)
            of_input_h.release(1)
            accum_out_h.acquire(1)
            accum_out_h.release(1)

    def first_weight_chain_body(
        of_input_h,
        of_weight_h,
        of_weight_out_h,
        accum_out_h,
        kernel,
        forward_kernel,
    ):
        for _ in range_(L):
            elem_in = of_input_h.acquire(1)
            elem_w = of_weight_h.acquire(1)
            elem_w_out = of_weight_out_h.acquire(1)
            forward_kernel(elem_w, elem_w_out)
            of_weight_out_h.release(1)
            kernel(elem_in, elem_w)
            of_weight_h.release(1)
            of_input_h.release(1)
            accum_out_h.acquire(1)
            accum_out_h.release(1)

    def middle_body(accum_in_h, accum_out_h, kernel):
        for _ in range_(L):
            accum_in_h.acquire(1)
            kernel()
            accum_in_h.release(1)
            accum_out_h.acquire(1)
            accum_out_h.release(1)

    def middle_weight_chain_body(
        accum_in_h,
        of_weight_h,
        of_weight_out_h,
        accum_out_h,
        kernel,
        forward_kernel,
    ):
        for _ in range_(L):
            elem_w = of_weight_h.acquire(1)
            elem_w_out = of_weight_out_h.acquire(1)
            forward_kernel(elem_w, elem_w_out)
            of_weight_out_h.release(1)
            accum_in_h.acquire(1)
            kernel(elem_w)
            of_weight_h.release(1)
            accum_in_h.release(1)
            accum_out_h.acquire(1)
            accum_out_h.release(1)

    def last_body(accum_in_h, of_output_h, kernel):
        for _ in range_(L):
            accum_in_h.acquire(1)
            elem_out = of_output_h.acquire(1)
            kernel(elem_out)
            accum_in_h.release(1)
            of_output_h.release(1)

    def last_weight_chain_body(accum_in_h, of_weight_h, of_output_h, kernel):
        for _ in range_(L):
            accum_in_h.acquire(1)
            elem_w = of_weight_h.acquire(1)
            elem_out = of_output_h.acquire(1)
            kernel(elem_w, elem_out)
            of_weight_h.release(1)
            accum_in_h.release(1)
            of_output_h.release(1)

    first_args = [of_input.cons()]
    first_core = first_body
    if weight_chain:
        first_args.extend([of_weight.cons(), weight_chain_fifos[0].prod()])
        first_core = first_weight_chain_body
    elif of_weight is not None:
        first_args.append(of_weight.cons())
        first_core = first_weight_body
    first_args.extend([accum_fifos[0].prod(), first_kernel])
    if weight_chain:
        first_args.append(weight_forward_kernel)
    workers = [
        Worker(first_core, fn_args=first_args, tile=tiles[0]),
    ]
    for i in range(1, N_LAYERS - 1):
        if weight_chain:
            workers.append(Worker(
                middle_weight_chain_body,
                fn_args=[
                    accum_fifos[i - 1].cons(),
                    weight_chain_fifos[i - 1].cons(),
                    weight_chain_fifos[i].prod(),
                    accum_fifos[i].prod(),
                    middle_kernel,
                    weight_forward_kernel,
                ],
                tile=tiles[i],
            ))
            continue
        workers.append(Worker(
            middle_body,
            fn_args=[
                accum_fifos[i - 1].cons(),
                accum_fifos[i].prod(),
                middle_kernel,
            ],
            tile=tiles[i],
        ))
    if weight_chain:
        workers.append(Worker(
            last_weight_chain_body,
            fn_args=[
                accum_fifos[-1].cons(),
                weight_chain_fifos[-1].cons(),
                of_output.prod(),
                last_kernel,
            ],
            tile=tiles[-1],
        ))
    else:
        workers.append(Worker(
            last_body,
            fn_args=[accum_fifos[-1].cons(), of_output.prod(), last_kernel],
            tile=tiles[-1],
        ))

    for i, af in enumerate(accum_fifos):
        af._prod_tile = workers[i]._tile
        af._cons_tile = workers[i + 1]._tile

    rt = Runtime()
    if weight_pressure:
        with rt.sequence(in_total_ty, weight_total_ty, out_total_ty) as (X, W, Y):
            rt.start(*workers)
            rt.fill(of_input.prod(), X)
            rt.fill(of_weight.prod(), W)
            rt.drain(of_output.cons(), Y, wait=True)
    else:
        with rt.sequence(in_total_ty, out_total_ty) as (X, Y):
            rt.start(*workers)
            rt.fill(of_input.prod(), X)
            rt.drain(of_output.cons(), Y, wait=True)

    return Program(dev, rt).resolve_program()

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", required=True, dest="device",
                   help="AIE Device (npu/npu2)")
    p.add_argument("-L", "--seq", type=int, default=334,
                   help="LSTM sequence length (default 334)")
    p.add_argument(
        "--experiment",
        choices=(
            "production",
            "hidden-state-only",
            "hidden-state-math",
            "hidden-state-weight",
            "hidden-state-weight-chain",
            "production-weight-cache-chain",
            "production-weight-stream-chain",
            "production-weight-direct-stream",
            "production-weight-direct-stream-probe",
            "production-weight-direct-layer-probe",
            "production-weight-direct-group-probe",
            "production-weight-direct-group-math-probe",
            "production-weight-direct-group-cascade-probe",
            "production-weight-direct-group-vertical-chain-probe",
            "production-weight-consolidated-vertical-chain-probe",
            "production-weight-consolidated-horizontal-probe",
            "production-weight-consolidated-full-chain-probe",
            "production-weight-consolidated-full-chain-lstm-probe",
        ),
        default="production",
        help="emit the production cascade or the hidden-state-only bisection",
    )
    return p.parse_args(argv)

def _select_device(name: str):
    from aie.iron.device import NPU2, NPU1Col1
    if name == "npu":
        return NPU1Col1()
    if name == "npu2":
        return NPU2()
    raise ValueError(f"[ERROR] Device name {name!r} is unknown")

if __name__ == "__main__":
    opts = _parse_args(sys.argv[1:])
    dev = _select_device(opts.device)
    if opts.experiment == "hidden-state-only":
        module = my_dorado_hidden_state_cascade_only(dev, L=opts.seq)
    elif opts.experiment == "hidden-state-math":
        module = my_dorado_hidden_state_cascade_only(
            dev, L=opts.seq, math_pressure=True,
        )
    elif opts.experiment == "hidden-state-weight":
        module = my_dorado_hidden_state_cascade_only(
            dev, L=opts.seq, math_pressure=True, weight_pressure=True,
        )
    elif opts.experiment == "hidden-state-weight-chain":
        module = my_dorado_hidden_state_cascade_only(
            dev,
            L=opts.seq,
            math_pressure=True,
            weight_pressure=True,
            weight_chain=True,
        )
    elif opts.experiment == "production-weight-cache-chain":
        module = my_dorado_fast_lstm_stack_bf16_acc_cascade_weight_stream_chain(
            dev, L=opts.seq,
        )
    elif opts.experiment == "production-weight-stream-chain":
        module = my_dorado_fast_lstm_stack_bf16_acc_cascade_weight_stream_chain(
            dev, L=opts.seq,
        )
    elif opts.experiment == "production-weight-direct-stream":
        module = (
            my_dorado_fast_lstm_stack_bf16_acc_cascade_weight_direct_stream(
                dev, L=opts.seq,
            )
        )
        module = _lower_production_direct_stream_flows(str(module))
    elif opts.experiment == "production-weight-direct-stream-probe":
        module = my_dorado_prod_weight_direct_stream_probe(dev, L=opts.seq)
        module = _lower_production_direct_stream_flows(str(module))
    elif opts.experiment == "production-weight-direct-layer-probe":
        module = my_dorado_prod_weight_direct_layer_probe(dev, L=opts.seq)
    elif opts.experiment == "production-weight-direct-group-probe":
        module = my_dorado_prod_weight_direct_group_probe(dev, L=opts.seq)
    elif opts.experiment == "production-weight-direct-group-math-probe":
        module = my_dorado_prod_weight_direct_group_math_probe(dev, L=opts.seq)
    elif opts.experiment == "production-weight-direct-group-cascade-probe":
        module = my_dorado_prod_weight_direct_group_cascade_probe(dev, L=opts.seq)
    elif opts.experiment == "production-weight-direct-group-vertical-chain-probe":
        module = my_dorado_prod_weight_direct_group_vertical_chain_probe(
            dev, L=opts.seq,
        )
    elif opts.experiment == "production-weight-consolidated-vertical-chain-probe":
        module = my_dorado_prod_weight_consolidated_vertical_chain_probe(
            dev, L=opts.seq,
        )
    elif opts.experiment == "production-weight-consolidated-horizontal-probe":
        module = my_dorado_prod_weight_consolidated_horizontal_probe(
            dev, L=opts.seq,
        )
    elif opts.experiment == "production-weight-consolidated-full-chain-probe":
        module = my_dorado_prod_weight_consolidated_full_chain_probe(
            dev, L=opts.seq,
        )
    elif opts.experiment == "production-weight-consolidated-full-chain-lstm-probe":
        module = my_dorado_prod_weight_consolidated_full_chain_lstm_probe(
            dev, L=opts.seq,
        )
    else:
        module = my_dorado_fast_lstm_stack_bf16_acc_cascade(dev, L=opts.seq)
    print(module)
