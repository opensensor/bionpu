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

"""bert_int8_matmul — AIE2P INT8 matmul op (DNABERT-Epi scorer port).

Registers two specializations in :data:`bionpu.dispatch.npu.NPU_OPS`
at import time:

* ``bert_int8_matmul_head`` (v0.4-alpha; M=47, K=768, N=2) — single
  compute tile, weights resident on tile. Runs on real AIE2P silicon
  byte-equivalent to the host-emulation reference.

* ``bert_int8_matmul_qkvo`` (v0.4-beta; M=47, K=768, N=768) — four
  compute tiles, N-axis fan-out (192 channels per tile), K-chunked
  along the reduction axis (K_CHUNK=64). Memtile splits the per-K-chunk
  weight tile into 4 N-slices and joins the 4 tile outputs back into
  a single 47×768 INT8 result.

Future ffn1 / ffn2 specializations (N=3072) will land here too as
their xclbins are produced.

Op contract (both variants)
----------------------------

* ``x`` (np.ndarray): shape ``(M, K)`` int8, row-major.
* ``w`` (np.ndarray): shape ``(N, K)`` int8, row-major (each row =
  one output channel's quantized weights).
* ``scales_combined`` (np.ndarray): shape ``(N + 1,)`` float32 — first
  N entries are the per-output-channel fused scale
  ``(scales_in * scales_w[n]) / scales_out``; trailing entry reserved
  for an optional bias term.
* Returns: ``np.ndarray`` shape ``(M, N)`` int8.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,
    NpuOp,
    register_npu_op,
)


# ─── Common emulation reference ──────────────────────────────────────

# Pinned shapes for the silicon builds (see MANIFEST.md). The emulation
# reference accepts any M ≤ _MAX_M and any K=_K; the N constraint is
# variant-specific and enforced per-NpuOp.
_MAX_M = 47
_K = 768
_HEAD_N = 2
_QKVO_N = 768


def _emulate(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    expected_n: int,
) -> np.ndarray:
    """Host-emulation reference — byte-equal to the silicon path.

    Same i32 accumulator matmul + per-output FP32 fused-scale + INT8
    saturate that the AIE2P kernel performs. Variant-agnostic: the
    silicon multi-tile + K-chunked path produces the same i32
    accumulator before fused-scale, so the byte output matches.

    Args:
        x: (M, K) int8 — token-major activation.
        w: (expected_n, K) int8 — output-channel-major weights.
        scales_combined: (expected_n + 1,) float32 — fused per-output
            scale; trailing entry reserved for bias (unused in v0).
        expected_n: pinned N for the variant (e.g., 2 for head,
            768 for qkvo).

    Returns:
        (M, expected_n) int8 — fused-scale + saturated output.
    """
    if x.dtype != np.int8 or w.dtype != np.int8:
        raise TypeError("bert_int8_matmul: x and w must be int8")
    if scales_combined.dtype != np.float32:
        raise TypeError("bert_int8_matmul: scales_combined must be float32")
    if x.ndim != 2 or x.shape[1] != _K or x.shape[0] > _MAX_M:
        raise ValueError(
            f"x shape mismatch: expected (M, {_K}) with M <= {_MAX_M}, "
            f"got {x.shape}"
        )
    if w.shape != (expected_n, _K):
        raise ValueError(
            f"w shape mismatch: expected ({expected_n}, {_K}), got {w.shape}"
        )
    if scales_combined.shape != (expected_n + 1,):
        raise ValueError(
            f"scales_combined shape mismatch: expected ({expected_n + 1},), "
            f"got {scales_combined.shape}"
        )

    # i32 accumulator matmul, then per-output FP32 fused-scale + INT8 saturate.
    acc = np.einsum("mk,nk->mn", x.astype(np.int32), w.astype(np.int32))
    fy = acc.astype(np.float32) * scales_combined[:expected_n]
    y = np.clip(np.round(fy), -128, 127).astype(np.int8)
    return y


# ─── Head specialization (v0.4-alpha) ────────────────────────────────


class _BertInt8MatmulHead(NpuOp):
    """``bert_int8_matmul_head`` op (M=47, K=768, N=2)."""

    name = "bert_int8_matmul_head"

    def __call__(
        self,
        *,
        x: np.ndarray,
        w: np.ndarray,
        scales_combined: np.ndarray,
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_int8_matmul_head(
            x, w, scales_combined, artifacts_dir=artifacts_dir
        )


def bert_int8_matmul_head(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """Dispatch the head-specialization INT8 matmul (M=47, K=768, N=2).

    When the AIE2P xclbin is built and present at
    ``bionpu/dispatch/_npu_artifacts/bert_int8_matmul_head/``, this
    runs on silicon. When absent, falls through to :func:`_emulate`
    and produces output byte-equal to the silicon path.
    """
    artifacts_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else Path(__file__).resolve().parents[3]
        / "dispatch" / "_npu_artifacts" / "bert_int8_matmul_head"
    )
    have_silicon = (
        artifacts_dir.is_dir()
        and (artifacts_dir / "final.xclbin").is_file()
        and (artifacts_dir / "insts.bin").is_file()
    )
    if not have_silicon:
        return _emulate(x, w, scales_combined, expected_n=_HEAD_N)

    raise NotImplementedError(
        "bert_int8_matmul_head silicon dispatch not yet wired; "
        "v0.4-alpha exposes only the host-emulation reference."
    )


# ─── qkvo specialization (v0.4-beta) ─────────────────────────────────


class _BertInt8MatmulQkvo(NpuOp):
    """``bert_int8_matmul_qkvo`` op (M=47, K=768, N=768).

    Used for the BERT body's Q / K / V / O projection matmuls, all four
    of which share this exact shape. Compute is N-axis-fanned across
    four AIE2P compute tiles; the IRON topology + kernel C++ live at
    `bionpu/kernels/scoring/bert_int8_matmul/{bert_int8_matmul.py,
    bert_int8_matmul.cc}` (variant=qkvo).
    """

    name = "bert_int8_matmul_qkvo"

    def __call__(
        self,
        *,
        x: np.ndarray,
        w: np.ndarray,
        scales_combined: np.ndarray,
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_int8_matmul_qkvo(
            x, w, scales_combined, artifacts_dir=artifacts_dir
        )


def bert_int8_matmul_qkvo(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """Dispatch the qkvo-specialization INT8 matmul (M=47, K=768, N=768).

    When the AIE2P xclbin is built and present at
    ``bionpu/dispatch/_npu_artifacts/bert_int8_matmul_qkvo/``, this
    runs on silicon. When absent, falls through to :func:`_emulate`
    and produces output byte-equal to the silicon path.

    Same op contract as ``bert_int8_matmul_head`` but pinned at N=768.
    The silicon path is multi-tile + K-chunked; the host-emulation
    reference IS still a single einsum because the i32 accumulator
    semantics are associative — the silicon's per-tile 192-channel
    partials joined N-axis produce the same i32 result before scaling,
    and the fused-scale + saturate is per-output-channel and order-
    independent.
    """
    artifacts_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else Path(__file__).resolve().parents[3]
        / "dispatch" / "_npu_artifacts" / "bert_int8_matmul_qkvo"
    )
    have_silicon = (
        artifacts_dir.is_dir()
        and (artifacts_dir / "final.xclbin").is_file()
        and (artifacts_dir / "insts.bin").is_file()
    )
    if not have_silicon:
        return _emulate(x, w, scales_combined, expected_n=_QKVO_N)

    raise NotImplementedError(
        "bert_int8_matmul_qkvo silicon dispatch not yet wired; "
        "v0.4-beta produces the xclbin (build/qkvo/final.xclbin) but "
        "the pyxrt host-side runner adapter has not yet been written. "
        "Use the host-emulation path (delete the artifacts_dir or pass "
        "artifacts_dir=None to a non-existent path) until v0.4-rc lands "
        "the silicon dispatch."
    )


# ─── Op registration ─────────────────────────────────────────────────

register_npu_op("bert_int8_matmul_head", _BertInt8MatmulHead())
register_npu_op("bert_int8_matmul_qkvo", _BertInt8MatmulQkvo())
