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

# match_singletile.py — IRON lowering for the single-tile CRISPR mismatch
# kernel.
#
# Per CRISPR PRD §4.1–4.2 and plan / split: this is the
# *single-tile* correctness foundation. Multi-tile streaming + tile-local
# thresholding is .
#
# Layout:
#   - One AIE compute tile holds:
#       * 128 guides × 5 packed bytes = 640 bytes (resident; loaded once)
#       * a window-chunk × 5 bytes (streamed in via ObjectFifo)
#       * a window-chunk × 128 bytes (streamed out)
#     Total tile-memory budget per chunk: 640 + chunk*5 + chunk*128 bytes.
#   - The host runtime walks the full window batch by re-firing this kernel
# across as many chunks as needed. For we ship a single shape:
#     128 guides × 4096 windows in one launch, partitioned into
#     `n_chunks = 4096 / WINDOWS_PER_CHUNK` ObjectFifo iterations.
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Matt Davis. Derived from
# mlir-aie/programming_examples/basic/vector_scalar_mul/vector_scalar_mul.py.

import argparse
import sys

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, NPU1Col1

# Pinned shape for . Multi-tile / streaming with different shapes is .
N_GUIDES = 128
SPACER_BYTES = 5  # 20 nt × 2 bits / 8 = 5 bytes
N_WINDOWS = 4096
# WINDOWS_PER_CHUNK: how many windows the tile processes per ObjectFifo
# acquire. Picked so per-chunk on-tile memory stays well under the 64 KiB
# AIE2P data-memory budget:
#   guides      : 128 * 5                    =     640 bytes  (resident)
#   window slot : WINDOWS_PER_CHUNK * 5      =     320 bytes  (in)
#   output slot : WINDOWS_PER_CHUNK * 128    =   8192 bytes   (out)
#   ──────────────────────────────────────────────────────────────────
#   totals      : ~9152 bytes per double-buffered side ≈ 18.3 KiB
# Comfortably fits.
WINDOWS_PER_CHUNK = 64
N_CHUNKS = N_WINDOWS // WINDOWS_PER_CHUNK  # 64 chunks

def crispr_match_singletile(dev):
    """Build the IRON program for the single-tile CRISPR mismatch kernel."""
    # tensor sizes (in elements; uint8 → bytes)
    guides_size = N_GUIDES * SPACER_BYTES  # 640
    windows_size = N_WINDOWS * SPACER_BYTES  # 20480
    out_size = N_WINDOWS * N_GUIDES  # 524288 — window-major

    chunk_windows_size = WINDOWS_PER_CHUNK * SPACER_BYTES  # 320
    chunk_out_size = WINDOWS_PER_CHUNK * N_GUIDES  # 8192

    # Top-level tensor types (host-visible).
    guides_ty = np.ndarray[(guides_size,), np.dtype[np.uint8]]
    windows_ty = np.ndarray[(windows_size,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(out_size,), np.dtype[np.uint8]]

    # Tile-local types (per-chunk).
    guides_tile_ty = np.ndarray[(guides_size,), np.dtype[np.uint8]]
    windows_tile_ty = np.ndarray[(chunk_windows_size,), np.dtype[np.uint8]]
    out_tile_ty = np.ndarray[(chunk_out_size,), np.dtype[np.uint8]]

    # External kernel function — the C++ in match_kernel.cc.
    match_fn = Kernel(
        "crispr_match_singletile",
        "match_kernel.o",
        [guides_tile_ty, windows_tile_ty, out_tile_ty, np.int32],
    )

    # AIE-array data movement.
    of_guides = ObjectFifo(guides_tile_ty, name="guides")
    of_windows = ObjectFifo(windows_tile_ty, name="windows")
    of_out = ObjectFifo(out_tile_ty, name="out")

    def core_body(of_guides, of_windows, of_out, match_fn):
        # Acquire the resident guide batch once.
        elem_guides = of_guides.acquire(1)
        # Iterate across the 64 chunks of 64 windows.
        for _ in range_(N_CHUNKS):
            elem_win = of_windows.acquire(1)
            elem_out = of_out.acquire(1)
            match_fn(elem_guides, elem_win, elem_out, WINDOWS_PER_CHUNK)
            of_windows.release(1)
            of_out.release(1)
        of_guides.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_guides.cons(), of_windows.cons(), of_out.prod(), match_fn],
    )

    # Runtime sequence (host → tile movement).
    rt = Runtime()
    with rt.sequence(guides_ty, windows_ty, out_ty) as (G, W, O):
        rt.start(worker)
        rt.fill(of_guides.prod(), G)
        rt.fill(of_windows.prod(), W)
        rt.drain(of_out.cons(), O, wait=True)

    return Program(dev, rt).resolve_program()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", required=True, dest="device", help="AIE device")
    opts = p.parse_args(sys.argv[1:])

    if opts.device == "npu":
        dev = NPU1Col1()
    elif opts.device == "npu2":
        dev = NPU2()
    else:
        raise ValueError(f"[ERROR] unknown device: {opts.device!r}")

    print(crispr_match_singletile(dev))

if __name__ == "__main__":
    main()
