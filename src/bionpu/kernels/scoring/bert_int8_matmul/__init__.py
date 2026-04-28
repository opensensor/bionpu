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

"""bert_int8_matmul — AIE2P INT8 matmul op (DNABERT-Epi scorer port v0.4 alpha).

Registers the head specialization (M=47, K=768, N=2) in
:data:`bionpu.dispatch.npu.NPU_OPS` at import time. Future
specializations (qkvo / ffn1 / ffn2) will register additional
NpuOp entries here as their xclbins land.

Op contract
-----------

* ``x`` (np.ndarray): shape ``(M, K)`` int8, row-major.
* ``w`` (np.ndarray): shape ``(N, K)`` int8, row-major (each row =
  one output channel's quantized weights).
* ``scales_combined`` (np.ndarray): shape ``(N + 1,)`` float32 — first
  N entries are the per-output-channel fused scale
  ``(scales_in * scales_w[n]) / scales_out``; trailing entry reserved
  for an optional bias term.
* Returns: ``np.ndarray`` shape ``(M, N)`` int8.

Status
------

🚧 **Skeleton scaffold (v0.4-alpha).** The IRON topology, kernel
C++, host runner, and Makefile are committed but the kernel has not
yet been built or run on silicon. See ``MANIFEST.md`` § Status for
the next-concrete-step list.

The host-emulation fallback (the dispatch path used by every other
NPU op when xclbins are absent) is NOT yet wired for this op — that's
part of the v0.4-alpha verification step. Until then, requesting
this op via ``bionpu.dispatch.lookup_npu_op('bert_int8_matmul_head')``
raises :class:`bionpu.dispatch.npu.NpuArtifactsMissingError` cleanly
with the rebuild instructions.
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


# Pinned head shape (see MANIFEST.md).
_M = 47
_K = 768
_N = 2


def _emulate(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
) -> np.ndarray:
    """Host-emulation reference: same arithmetic as the AIE2P kernel.

    Used by the dispatch layer when the silicon xclbin isn't built;
    produces output byte-equal to the silicon path by construction.
    """
    if x.dtype != np.int8 or w.dtype != np.int8:
        raise TypeError("bert_int8_matmul: x and w must be int8")
    if scales_combined.dtype != np.float32:
        raise TypeError("bert_int8_matmul: scales_combined must be float32")
    if x.shape != (_M, _K):
        raise ValueError(f"x shape mismatch: expected ({_M}, {_K}), got {x.shape}")
    if w.shape != (_N, _K):
        raise ValueError(f"w shape mismatch: expected ({_N}, {_K}), got {w.shape}")
    if scales_combined.shape != (_N + 1,):
        raise ValueError(
            f"scales_combined shape mismatch: expected ({_N + 1},), got {scales_combined.shape}"
        )

    # i32 accumulator matmul, then per-output FP32 fused-scale + INT8 saturate.
    acc = np.einsum("mk,nk->mn", x.astype(np.int32), w.astype(np.int32))
    fy = acc.astype(np.float32) * scales_combined[:_N]
    y = np.clip(np.round(fy), -128, 127).astype(np.int8)
    return y


def bert_int8_matmul_head(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """Dispatch the head-specialization INT8 matmul.

    When the AIE2P xclbin is built and present at
    ``bionpu/dispatch/_npu_artifacts/bert_int8_matmul_head/``, this
    runs on silicon. When absent, falls through to :func:`_emulate`
    and produces output byte-equal to the silicon path.

    The signature mirrors the eventual silicon dispatch so the
    bionpu.scoring.dnabert_epi NPU backend can swap implementations
    transparently.
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
        # Host-emulation path is byte-equal to the silicon path by
        # construction (same i32 accumulator + same FP32 fused
        # scale + same INT8 saturate). The v0.4-beta build will
        # produce real artifacts; until then, emulation IS the answer.
        return _emulate(x, w, scales_combined)

    # When artifacts land, dispatch via the host runner. Currently
    # raises so the absent-artifacts path doesn't silently take a
    # potentially slower or wrong code path.
    raise NotImplementedError(
        "bert_int8_matmul_head silicon dispatch not yet wired; "
        "v0.4-alpha exposes only the host-emulation reference."
    )


register_npu_op("bert_int8_matmul_head", _BertInt8MatmulHead())
