# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# bert_int8_matmul — IRON-Python lowering for the BERT body INT8 matmul.
#
# The workhorse kernel of the DNABERT-Epi AIE2P scorer port. Two
# variants are supported:
#
# * ``head`` (default; v0.4-alpha):
#   single compute tile, weights resident on tile. Works for the
#   classifier-head specialization (M=47, K=768, N=2 — total tile DM
#   ~38 KB / 64 KB).
#
# * ``qkvo`` (v0.4-beta):
#   four compute tiles, M-axis fan-out (12 padded token rows per tile),
#   K-chunked along the reduction axis. Both x and w stream directly
#   from DDR/shim to all compute tiles; memtile flat-concats the four
#   row slabs into a row-major output. This replaces the earlier N-axis
#   fan-out whose memtile join produced slice-major output.
#
# * ``ffn1`` / ``ffn2`` (v0.4-rc):
#   DDR-streamed feed-forward projections. The generated AIE program is
#   the qkvo-sized 768-output group kernel; host dispatch streams the
#   larger FFN weight matrix in row-major 768-output groups. This keeps
#   per-tile state at a qkvo-sized row slab while allowing
#   the 3072-wide FFN expansion weight to live in DDR.
#
# Shapes (per CLI flags):
#   x:           M × K  int8           (token-major)
#   w:           N × K  int8           (output-channel-major)
#   scales:      N + 1  fp32           (combined per-output FP32 scales)
#   y:           M × N  int8           (token-major)
#
# Topology (v0.4-alpha — head):
#   shim ──x── compute_tile ──y── shim
#   shim ──ws── compute_tile
#
# Topology (v0.4-beta — qkvo):
#   shim ──xs_chunk──→ broadcast to 4 compute tiles (1 shim MM2S)
#       (xs = M×K_CHUNK x slab + (N+1) fp32 scales prefix; same scales
#        prefix is duplicated on every chunk to keep ObjectFifo elements
#        uniform-sized while still fitting NPU2Col1's 2-MM2S shim budget)
#   shim ──w_chunk──→ memtile ──split(N)──→ 4 compute tiles (1 shim MM2S)
#   compute_0..3 ──y_partial──→ memtile ──join(N)──→ shim (1 shim S2MM)

from __future__ import annotations

import argparse
import sys

import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2Col1


def _round4(n: int) -> int:
    """Round up to a multiple of 4 (AIE2P shim DMA requirement)."""
    return ((n + 3) // 4) * 4


def emit_mlir_head(M: int, K: int, N: int, dev) -> str:
    """v0.4-alpha single-tile path. Weights tile-resident."""
    # ObjectFifo element types — flat byte arrays per IRON convention.
    #
    # AIE2P CoreTiles only have 2 input + 2 output DMA channels, so
    # we cannot afford 3 separate input ObjectFifos (x, w, s). The
    # smaller two (w + scales) are concatenated into a single byte
    # buffer; the kernel C++ slices it back into typed pointers via
    # known byte offsets. Layout (head shape):
    #   ws[0 .. N*K - 1]                  = int8 weights (N rows × K cols)
    #   ws[N*K .. N*K + (N+1)*4 - 1]      = float32 fused scales (N+1 entries)
    # AIE2P shim DMA requires transfer length to be a multiple of 4
    # bytes (`aie.dma_bd` op constraint). Pad each fifo size up to the
    # next multiple of 4 so the lowering succeeds. The kernel C++
    # writes only the useful prefix; the host runner reads the same
    # useful prefix and ignores any trailing pad bytes.
    x_size = _round4(M * K)
    ws_bytes = _round4(N * K + (N + 1) * 4)   # int8 weights + float32 scales
    y_size = _round4(M * N)
    x_ty  = np.ndarray[(x_size,), np.dtype[np.int8]]
    ws_ty = np.ndarray[(ws_bytes,), np.dtype[np.int8]]
    y_ty  = np.ndarray[(y_size,), np.dtype[np.int8]]

    # depth=1 in v0 — there's only one chunk per launch and the
    # 36 KB x buffer is large enough that depth=2 (72 KB) blows the
    # 64 KB tile DM cap.
    of_x  = ObjectFifo(x_ty,  name="x_in",  depth=1)
    of_ws = ObjectFifo(ws_ty, name="ws_in", depth=1)
    of_y  = ObjectFifo(y_ty,  name="y_out", depth=1)

    # Compute kernel — AIE builds take the vectorized INT8 dot-product
    # path; host builds keep the scalar fallback. Symbol is
    # `bert_int8_matmul_<N>` so the same .py / .cc compile to multiple
    # specializations via the Makefile.
    matmul_sym = f"bert_int8_matmul_{N}"
    matmul = Kernel(
        matmul_sym,
        "bert_int8_matmul.o",
        [x_ty, ws_ty, y_ty],
    )

    # one chunk in / one chunk out per launch. The Kernel must be passed
    # *into* the body as an arg (not closed over) so the IRON resolver
    # can bind the symbol at lower-time.
    def core_body(of_x, of_ws, of_y, matmul_kernel):
        x_buf  = of_x.acquire(1)
        ws_buf = of_ws.acquire(1)
        y_buf  = of_y.acquire(1)
        matmul_kernel(x_buf, ws_buf, y_buf)
        of_x.release(1)
        of_ws.release(1)
        of_y.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_x.cons(), of_ws.cons(), of_y.prod(), matmul],
    )

    rt = Runtime()
    with rt.sequence(x_ty, ws_ty, y_ty) as (x, ws, y):
        rt.start(worker)
        rt.fill(of_x.prod(), x)
        rt.fill(of_ws.prod(), ws)
        rt.drain(of_y.cons(), y, wait=True)

    return Program(dev, rt).resolve_program()


def emit_mlir_qkvo(M: int, K: int, N: int, K_CHUNK: int, dev) -> str:
    """v0.4-beta/rc row-major multi-tile + K-chunked path.

    Topology:
        shim ──xs_chunk──→ broadcast to 4 compute tiles (1 shim MM2S)
        shim ──w_chunk───→ broadcast to 4 compute tiles (1 shim MM2S)
        4 cores ──row slabs──→ memtile ──join(M)──→ shim (1 shim S2MM)

    Each tile owns a contiguous slab of padded token rows and all output
    channels. The memtile join is therefore a flat row-slab concat,
    which directly produces row-major output and removes the host-side
    slice-major unscramble step.

    Per-tile DM budget (actual, observed in linker layout, depth=1):

      qkvo_acc i32          :  12 × 768 × 4   = 36,864 B
      w_to_core i8          : 768 × 8         =  6,144 B
      y_part out i8         :  12 × 768       =  9,216 B
      xs_in i8 (+scales)    : 47×8 + 769×4    =  3,452 B
      qkvo_scales fp32      : 768+1 fp32      =  3,076 B
      stack                 : —               =  1,024 B
      ──────────────────────────────────────────
      total                                   = 59,776 B

    Shim budget (out of 2 MM2S + 2 S2MM):
      MM2S: xs (1) + w (1)                   = 2/2 channels
      S2MM: y_out (1)                        = 1/2 channels
    """
    if K % K_CHUNK != 0:
        raise ValueError(
            f"qkvo variant requires K divisible by K_CHUNK, got K={K}, "
            f"K_CHUNK={K_CHUNK}"
        )

    N_TILES = 4
    M_PAD = ((M + N_TILES - 1) // N_TILES) * N_TILES
    M_PER_TILE = M_PAD // N_TILES
    K_CHUNKS = K // K_CHUNK

    # ─── Whole-runtime tensor types (host-visible flat buffers) ───
    # Streaming layout — host writes them in chunk-major order so the
    # shim DMA stride math stays linear-1D:
    #   xs_total : K_CHUNKS * (M*K_CHUNK + (N+1)*4) bytes — each chunk is
    #              one M×K_CHUNK slab of x followed by the (redundantly
    #              repeated) scales fp32 array. Trick lets us collapse 3
    #              shim MM2S streams (x, w, s) into 2 (xs, w), fitting
    #              the NPU2Col1 shim's 2-MM2S budget.
    #   w_total  : K_CHUNKS * (N * K_CHUNK) bytes — one N×K_CHUNK slab
    #              per chunk.
    #   y_total  : M_PAD * N int8 — final row-major output, joined as
    #              contiguous row slabs from the four compute tiles.
    w_total_bytes      = _round4(K_CHUNKS * N * K_CHUNK)
    y_total_bytes      = _round4(M_PAD * N)

    w_total_ty      = np.ndarray[(w_total_bytes,), np.dtype[np.int8]]
    y_total_ty      = np.ndarray[(y_total_bytes,), np.dtype[np.int8]]

    # ─── Per-chunk types (visible to memtile + compute tiles) ───
    # xs_chunk: x slab + scales packing (see big block comment further down
    # in the ObjectFifos section for rationale).
    xs_chunk_size      = _round4(M * K_CHUNK + (N + 1) * 4)
    xs_total_bytes     = _round4(K_CHUNKS * xs_chunk_size)
    # 49152 B at qkvo shape (768×64): full w_chunk on memtile.
    w_chunk_size       = _round4(N * K_CHUNK)
    y_part_size        = _round4(M_PER_TILE * N)

    xs_chunk_ty      = np.ndarray[(xs_chunk_size,),      np.dtype[np.int8]]
    xs_total_ty      = np.ndarray[(xs_total_bytes,),     np.dtype[np.int8]]
    w_chunk_ty       = np.ndarray[(w_chunk_size,),       np.dtype[np.int8]]
    y_part_ty        = np.ndarray[(y_part_size,),        np.dtype[np.int8]]

    # ─── Compute kernels — three symbols for the K-chunked accumulator pattern.
    # The IRON `range_` loop yields an `index`-typed loop variable that
    # would clash with the kernel's `i32` arg (operand type mismatch
    # at MLIR verify time), so rather than pass `k_iter` we split the
    # work into three lifecycle stages and pass tile-local persistent
    # state through IRON `Buffer` objects (placed on the tile by the
    # Worker that consumes them):
    #
    #   bert_int8_matmul_qkvo_init     — zero the i32 accumulator
    #   bert_int8_matmul_qkvo_acc      — i32 += i8 × i8 over the K_CHUNK,
    #                                     stashes scales (from xs tail) on
    #                                     the first call
    #   bert_int8_matmul_qkvo_finalize — fused-scale + INT8 saturate → y
    #
    # The accumulator and scales-stash are tile-local Buffers. They occupy
    # tile DM allocated by IRON (sized appropriately, not via the linker
    # script's tight `data` region — Buffers map onto AIE buffer ops which
    # live in their own DM segment). This is the canonical pattern for
    # tile-resident scratch state; static C++ arrays would land in `.bss`
    # which is squeezed by the ObjectFifo buffer placements.
    acc_ty        = np.ndarray[(M_PER_TILE * N,), np.dtype[np.int32]]
    scales_buf_ty = np.ndarray[((N + 1),),        np.dtype[np.float32]]

    matmul_init = Kernel(
        "bert_int8_matmul_qkvo_init",
        "bert_int8_matmul.o",
        [acc_ty],                  # i32 accumulator only (zeroes it)
    )
    matmul_acc = Kernel(
        "bert_int8_matmul_qkvo_acc",
        "bert_int8_matmul.o",
        [
            xs_chunk_ty,           # x slab for this K_CHUNK + trailing scales
            w_chunk_ty,            # full N x K_CHUNK weight slab
            acc_ty,                # tile-local i32 accumulator
            scales_buf_ty,         # tile-local fp32 scales-stash (acc_k
                                   # writes the per-tile slice every call —
                                   # last write wins; only the last call's
                                   # values are read by finalize, so this is
                                   # correct without a "first-call" gate)
            np.int32,              # tile_idx (0..3) — selects row slab
        ],
    )
    matmul_finalize = Kernel(
        "bert_int8_matmul_qkvo_finalize",
        "bert_int8_matmul.o",
        [
            acc_ty,                # tile-local i32 accumulator
            scales_buf_ty,         # tile-local fp32 scales-stash (per-tile slice)
            y_part_ty,             # int8 output partial
        ],
    )

    # ─── ObjectFifos ───
    #
    # Resource budget (NPU2Col1):
    #   shim   : 2 source (MM2S) + 2 dest (S2MM)  channels
    #   memtile: 6 MM2S + 6 S2MM  channels        (one memtile total)
    #   compute: 2 MM2S + 2 S2MM  channels each
    #
    # Naive layout uses 3 shim MM2S streams (x, w, scales) — exceeds
    # the 2/2 shim MM2S budget. To fit, we PACK scales into the trailing
    # bytes of every x_chunk slab (x_chunk_with_scales = x_chunk +
    # scales). Cost: 12× duplicated 3076-byte scales = 36 KB extra
    # host→shim transfer per launch (negligible vs. 576 KB weights).
    # The kernel reads scales out of the FIRST chunk only and stashes
    # them in static memory for the finalize stage.
    #
    # This collapses the shim MM2S budget to 2 (x_with_scales + w),
    # frees a memtile MM2S slot, and keeps the kernel logic simple
    # (no k_iter scalar arg required).
    #
    # Element layout per x_chunk transfer (3008 + 3076 = 6084 B, padded
    # to mult of 4 → 6084 already aligned):
    #   xs_chunk[0          .. M*K_CHUNK - 1]                  = x int8
    #   xs_chunk[M*K_CHUNK  .. M*K_CHUNK + (N+1)*4 - 1]        = scales fp32
    #
    # Shim budget after packing:
    #   MM2S: xs (1) + w (1) = 2/2  ✓
    #   S2MM: y_out (1)      = 1/2  ✓
    #
    # Memtile budget:
    #   S2MM: w_in (1) + 4× y_part_i (4)            = 5/6
    #   MM2S: 4× w_to_core_i (4) + y_out (1)        = 5/6
    #   Memory: w_chunk (49152) + 4× y_part (9024)
    #         + y_total (36096) + scratch          ≈ 122 KB / 512 KB

    # xs: shim → 4 compute tiles (direct switchbox broadcast).
    of_xs = ObjectFifo(xs_chunk_ty, name="xs_in", depth=1)

    # w_chunk: shim → 4 compute tiles (direct switchbox broadcast).
    of_w_in  = ObjectFifo(w_chunk_ty, name="w_in",  depth=1)
    of_y_out = ObjectFifo(y_total_ty, name="y_out", depth=1)

    # y output: 4 compute tiles → memtile → join along M row slabs → shim.
    # Each tile's M_PER_TILE×N partial lands at byte offset i*y_part_size
    # in the joined M_PAD×N output, which is already row-major.
    y_join_offsets = [i * y_part_size for i in range(N_TILES)]
    of_y_partials = of_y_out.prod().join(
        y_join_offsets,
        obj_types=[y_part_ty] * N_TILES,
        names=[f"y_part_{i}" for i in range(N_TILES)],
    )

    # ─── Per-tile worker body ───
    # Each worker has its own pair of tile-local Buffers (acc + scales)
    # passed in as fn_args. They are persistent across the K loop:
    #   1. init zeroes the i32 accumulator
    #   2. K loop accumulates; first call also stashes scales out of
    #      the xs_chunk tail into the scales_buf
    #   3. finalize applies fused-scale + INT8 saturate using scales_buf
    def core_body(of_xs, of_w, of_y, acc_buf, scales_buf, init_k, acc_k, fin_k, tile_idx):
        init_k(acc_buf)
        for _ in range_(K_CHUNKS):
            xs_buf = of_xs.acquire(1)
            w_buf  = of_w.acquire(1)
            acc_k(xs_buf, w_buf, acc_buf, scales_buf, tile_idx)
            of_xs.release(1)
            of_w.release(1)
        y_buf = of_y.acquire(1)
        fin_k(acc_buf, scales_buf, y_buf)
        of_y.release(1)

    workers = []
    for i in range(N_TILES):
        # Per-tile Buffer for the i32 accumulator and the fp32 scales
        # stash. IRON places these on the Worker's tile when consumed
        # via fn_args (Worker._buffers handling). Distinct names so the
        # MLIR remains tile-unique.
        acc_buf = Buffer(type=acc_ty, name=f"qkvo_acc_{i}")
        scales_buf = Buffer(type=scales_buf_ty, name=f"qkvo_scales_{i}")
        workers.append(Worker(
            core_body,
            fn_args=[
                of_xs.cons(),                 # direct shim-broadcast xs consumer
                of_w_in.cons(),               # direct shim-broadcast w consumer
                of_y_partials[i].prod(),      # per-tile output producer (memtile-join)
                acc_buf,
                scales_buf,
                matmul_init,
                matmul_acc,
                matmul_finalize,
                int(i),                       # tile_idx 0..3
            ],
        ))

    # ─── Runtime sequence (shim-side DMAs) ───
    # The host pre-stages:
    #   x_total: K_CHUNKS contiguous slabs of M×K_CHUNK each (chunk-major)
    #   w_total: K_CHUNKS contiguous slabs of N×K_CHUNK each (chunk-major,
    #            with N-major stride within each slab so memtile-split works)
    #   scales : (N+1) fp32 in one shot
    # Runtime sequence: shim drives 2 input streams and 1 output stream.
    #   xs_total : K_CHUNKS × xs_chunk_size bytes (x_chunk + scales)
    #   w_total  : K_CHUNKS × w_chunk_size bytes  (full N×K_CHUNK slabs)
    #   y_total  : M_PAD × N bytes (row-major joined output)
    rt = Runtime()
    with rt.sequence(xs_total_ty, w_total_ty, y_total_ty) as (
        xs_total, w_total, y_total,
    ):
        rt.start(*workers)
        rt.fill(of_xs.prod(),   xs_total)
        rt.fill(of_w_in.prod(), w_total)
        rt.drain(of_y_out.cons(), y_total, wait=True)

    return Program(dev, rt).resolve_program()


def emit_mlir(
    M: int,
    K: int,
    N: int,
    target: str = "npu2",
    variant: str = "head",
    K_CHUNK: int = 8,
) -> str:
    """Emit IRON Python -> MLIR-AIE for the given matmul shape.

    Returns the MLIR-AIE module as a string. Caller pipes to
    aiecc to produce the xclbin.

    Args:
        M, K, N: matmul shape.
        target: "npu" (NPU1, Phoenix) or "npu2" (NPU2 / AIE2P / Strix).
        variant: "head" (single-tile, weights resident),
                 "qkvo" (4-tile fan-out + K-chunked memtile streaming),
                 or "ffn1"/"ffn2" (same qkvo-sized group kernel used by
                 the host-side DDR-streamed FFN dispatcher).
        K_CHUNK: K-axis chunk size for the qkvo/ffn variants. Default
                 8 keeps the full N=768 row-slab state inside tile DM.
    """
    if target == "npu2":
        dev = NPU2Col1()
    elif target == "npu":
        dev = NPU1Col1()
    else:
        raise ValueError(f"unknown device target: {target!r}")

    if variant == "head":
        return emit_mlir_head(M=M, K=K, N=N, dev=dev)
    elif variant == "qkvo":
        return emit_mlir_qkvo(M=M, K=K, N=N, K_CHUNK=K_CHUNK, dev=dev)
    elif variant in ("ffn1", "ffn2"):
        if N != 768:
            raise ValueError(
                f"{variant} lowering emits one DDR-streamed 768-output "
                f"group per dispatch; pass N=768 for the group kernel, "
                f"got N={N}"
            )
        return emit_mlir_qkvo(M=M, K=K, N=N, K_CHUNK=K_CHUNK, dev=dev)
    else:
        raise ValueError(
            f"unknown variant: {variant!r} "
            f"(expected 'head', 'qkvo', 'ffn1', or 'ffn2')"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--target", choices=["npu", "npu2"], default="npu2")
    p.add_argument("--M", type=int, default=47)
    p.add_argument("--K", type=int, default=768)
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--variant", choices=["head", "qkvo", "ffn1", "ffn2"], default="head")
    p.add_argument("--K-chunk", type=int, default=8,
                   help="K-axis chunk size for qkvo/ffn variants (default 8).")
    args = p.parse_args()
    sys.stdout.write(str(emit_mlir(
        M=args.M, K=args.K, N=args.N, target=args.target,
        variant=args.variant, K_CHUNK=args.K_chunk,
    )))


if __name__ == "__main__":
    main()
