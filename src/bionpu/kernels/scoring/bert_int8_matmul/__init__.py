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
  compute tiles, M-axis fan-out (12 padded token rows per tile),
  K-chunked along the reduction axis (K_CHUNK=8). The memtile joins
  four row slabs back into a row-major 48×768 INT8 result; the host
  only trims the padded final row.

* ``bert_int8_matmul_ffn1`` / ``bert_int8_matmul_ffn2`` (v0.4-rc) —
  BERT feed-forward projections. ffn1 is 47×768 @ 3072×768; ffn2 is
  47×3072 @ 768×3072. The host wire format streams weights from DDR as
  768-output groups so the tile-local accumulator remains qkvo-sized.

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

from pathlib import Path

import numpy as np

from bionpu.dispatch.npu import (
    NpuOp,
    default_backend,
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
_FFN1_K = 768
_FFN1_N = 3072
_FFN2_K = 3072
_FFN2_N = 768
_K_CHUNK = 8
_QKVO_TILES = 4
_FFN_GROUP_N = 768
_ARG_X = 3
_ARG_WS = 4
_ARG_Y = 5


def _round4(n: int) -> int:
    return ((int(n) + 3) // 4) * 4


def _padded_m(m: int) -> int:
    return ((int(m) + _QKVO_TILES - 1) // _QKVO_TILES) * _QKVO_TILES


def _default_artifacts_dir(variant: str) -> Path:
    dispatch_dir = (
        Path(__file__).resolve().parents[3]
        / "dispatch" / "_npu_artifacts" / f"bert_int8_matmul_{variant}"
    )
    if (dispatch_dir / "final.xclbin").is_file() and (
        dispatch_dir / "insts.bin"
    ).is_file():
        return dispatch_dir
    return Path(__file__).resolve().parent / "build" / variant


def _has_silicon_artifacts(path: Path) -> bool:
    return (path / "final.xclbin").is_file() and (path / "insts.bin").is_file()


def _pad_rows(x: np.ndarray, *, cols: int) -> np.ndarray:
    if x.shape[0] == _MAX_M:
        return np.ascontiguousarray(x, dtype=np.int8)
    out = np.zeros((_MAX_M, cols), dtype=np.int8)
    out[: x.shape[0], :] = x
    return out


def _pad_bytes(payload: bytes, size: int) -> bytes:
    if len(payload) > size:
        raise ValueError(f"payload is {len(payload)} bytes; expected <= {size}")
    return payload + bytes(size - len(payload))


def _pack_head_ws(w: np.ndarray, scales_combined: np.ndarray) -> bytes:
    size = _round4(_HEAD_N * _K + (_HEAD_N + 1) * 4)
    return _pad_bytes(w.tobytes() + scales_combined.tobytes(), size)


def _pack_grouped_xs(
    x: np.ndarray,
    scales_combined: np.ndarray,
    *,
    k: int,
    group_n: int,
) -> bytes:
    x_full = _pad_rows(x, cols=k)
    k_chunks = k // _K_CHUNK
    n_groups = scales_combined.shape[0] - 1
    if n_groups % group_n != 0:
        raise ValueError(
            f"grouped matmul requires N divisible by {group_n}; got N={n_groups}"
        )
    chunk_size = _round4(_MAX_M * _K_CHUNK + (group_n + 1) * 4)
    chunks: list[bytes] = []
    for group0 in range(0, n_groups, group_n):
        group_scales = np.empty((group_n + 1,), dtype=np.float32)
        group_scales[:group_n] = scales_combined[group0 : group0 + group_n]
        group_scales[group_n] = scales_combined[n_groups]
        scales_payload = group_scales.tobytes()
        for c in range(k_chunks):
            x_chunk = np.ascontiguousarray(
                x_full[:, c * _K_CHUNK : (c + 1) * _K_CHUNK],
                dtype=np.int8,
            )
            chunks.append(_pad_bytes(x_chunk.tobytes() + scales_payload, chunk_size))
    return b"".join(chunks)


def _pack_grouped_w(w: np.ndarray, *, group_n: int) -> bytes:
    n, k = w.shape
    if n % group_n != 0:
        raise ValueError(
            f"grouped matmul requires N divisible by {group_n}; got N={n}"
        )
    chunks: list[bytes] = []
    for group0 in range(0, n, group_n):
        wg = w[group0 : group0 + group_n, :]
        for c in range(k // _K_CHUNK):
            slab = np.ascontiguousarray(
                wg[:, c * _K_CHUNK : (c + 1) * _K_CHUNK],
                dtype=np.int8,
            )
            chunks.append(slab.tobytes())
    return b"".join(chunks)


def _pack_qkvo_xs(x: np.ndarray, scales_combined: np.ndarray) -> bytes:
    return _pack_grouped_xs(x, scales_combined, k=_K, group_n=_QKVO_N)


def _pack_qkvo_w(w: np.ndarray) -> bytes:
    return _pack_grouped_w(w, group_n=_QKVO_N)


def _read_row_major(raw: bytes, *, n: int) -> np.ndarray:
    m_pad = _padded_m(_MAX_M)
    y = np.frombuffer(raw, dtype=np.int8, count=m_pad * n)
    return y.reshape(m_pad, n)


def _emulate(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    expected_n: int,
    expected_k: int = _K,
) -> np.ndarray:
    """Host-emulation reference — byte-equal to the silicon path.

    Same i32 accumulator matmul + per-output FP32 fused-scale + INT8
    saturate that the AIE2P kernel performs. Variant-agnostic: the
    silicon multi-tile + K-chunked path produces the same i32
    accumulator before fused-scale, so the byte output matches.

    Args:
        x: (M, expected_k) int8 — token-major activation.
        w: (expected_n, expected_k) int8 — output-channel-major weights.
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
    if x.ndim != 2 or x.shape[1] != expected_k or x.shape[0] > _MAX_M:
        raise ValueError(
            f"x shape mismatch: expected (M, {expected_k}) with M <= {_MAX_M}, "
            f"got {x.shape}"
        )
    if w.shape != (expected_n, expected_k):
        raise ValueError(
            f"w shape mismatch: expected ({expected_n}, {expected_k}), got {w.shape}"
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
    reference = _emulate(x, w, scales_combined, expected_n=_HEAD_N)
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else (
        _default_artifacts_dir("head")
    )
    if not _has_silicon_artifacts(artifacts_dir):
        return reference

    x_full = _pad_rows(x, cols=_K)
    x_size = _round4(_MAX_M * _K)
    y_size = _round4(_MAX_M * _HEAD_N)
    raw, _, _, _ = default_backend().run_xclbin(
        xclbin=artifacts_dir / "final.xclbin",
        insts=artifacts_dir / "insts.bin",
        kernel_name="MLIR_AIE",
        in_buffers=[
            (_pad_bytes(x_full.tobytes(), x_size), _ARG_X),
            (_pack_head_ws(np.ascontiguousarray(w), scales_combined), _ARG_WS),
        ],
        out_size=y_size,
        out_arg_index=_ARG_Y,
    )
    y = np.frombuffer(raw, dtype=np.int8, count=_MAX_M * _HEAD_N)
    return y.reshape(_MAX_M, _HEAD_N)[: x.shape[0], :].copy()


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
    reference = _emulate(x, w, scales_combined, expected_n=_QKVO_N)
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else (
        _default_artifacts_dir("qkvo")
    )
    if not _has_silicon_artifacts(artifacts_dir):
        return reference

    xs = _pack_qkvo_xs(x, scales_combined)
    w_chunks = _pack_qkvo_w(np.ascontiguousarray(w))
    y_size = _round4(_padded_m(_MAX_M) * _QKVO_N)
    raw, _, _, _ = default_backend().run_xclbin(
        xclbin=artifacts_dir / "final.xclbin",
        insts=artifacts_dir / "insts.bin",
        kernel_name="MLIR_AIE",
        in_buffers=[
            (xs, _ARG_X),
            (w_chunks, _ARG_WS),
        ],
        out_size=y_size,
        out_arg_index=_ARG_Y,
    )
    return _read_row_major(raw, n=_QKVO_N)[: x.shape[0], :].copy()


class _BertInt8MatmulFfn1(NpuOp):
    """``bert_int8_matmul_ffn1`` op (M=47, K=768, N=3072)."""

    name = "bert_int8_matmul_ffn1"

    def __call__(
        self,
        *,
        x: np.ndarray,
        w: np.ndarray,
        scales_combined: np.ndarray,
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_int8_matmul_ffn1(
            x, w, scales_combined, artifacts_dir=artifacts_dir
        )


class _BertInt8MatmulFfn2(NpuOp):
    """``bert_int8_matmul_ffn2`` op (M=47, K=3072, N=768)."""

    name = "bert_int8_matmul_ffn2"

    def __call__(
        self,
        *,
        x: np.ndarray,
        w: np.ndarray,
        scales_combined: np.ndarray,
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_int8_matmul_ffn2(
            x, w, scales_combined, artifacts_dir=artifacts_dir
        )


def _bert_int8_matmul_ffn(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    expected_k: int,
    expected_n: int,
    variant: str,
    artifacts_dir: Path | str | None,
) -> np.ndarray:
    """Dispatch a feed-forward matmul using the DDR-streamed grouped ABI."""
    reference = _emulate(
        x,
        w,
        scales_combined,
        expected_n=expected_n,
        expected_k=expected_k,
    )
    artifacts_path = Path(artifacts_dir) if artifacts_dir is not None else (
        _default_artifacts_dir(variant)
    )
    if not _has_silicon_artifacts(artifacts_path):
        return reference

    xs = _pack_grouped_xs(
        x,
        scales_combined,
        k=expected_k,
        group_n=_FFN_GROUP_N,
    )
    w_groups = _pack_grouped_w(np.ascontiguousarray(w), group_n=_FFN_GROUP_N)
    y_size = _round4(_padded_m(_MAX_M) * expected_n)
    raw, _, _, _ = default_backend().run_xclbin(
        xclbin=artifacts_path / "final.xclbin",
        insts=artifacts_path / "insts.bin",
        kernel_name="MLIR_AIE",
        in_buffers=[
            (xs, _ARG_X),
            (w_groups, _ARG_WS),
        ],
        out_size=y_size,
        out_arg_index=_ARG_Y,
    )
    return _read_row_major(raw, n=expected_n)[: x.shape[0], :].copy()


def bert_int8_matmul_ffn1(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """Dispatch BERT FFN expansion matmul (M<=47, K=768, N=3072).

    The silicon ABI streams weight groups from DDR as four 768-output
  groups. Each group reuses the qkvo row-slab layout, so the tile
  accumulator never grows beyond 12×768 int32.
    """
    return _bert_int8_matmul_ffn(
        x,
        w,
        scales_combined,
        expected_k=_FFN1_K,
        expected_n=_FFN1_N,
        variant="ffn1",
        artifacts_dir=artifacts_dir,
    )


def bert_int8_matmul_ffn2(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """Dispatch BERT FFN contraction matmul (M<=47, K=3072, N=768)."""
    return _bert_int8_matmul_ffn(
        x,
        w,
        scales_combined,
        expected_k=_FFN2_K,
        expected_n=_FFN2_N,
        variant="ffn2",
        artifacts_dir=artifacts_dir,
    )


# ─── Op registration ─────────────────────────────────────────────────

register_npu_op("bert_int8_matmul_head", _BertInt8MatmulHead())
register_npu_op("bert_int8_matmul_qkvo", _BertInt8MatmulQkvo())
register_npu_op("bert_int8_matmul_ffn1", _BertInt8MatmulFfn1())
register_npu_op("bert_int8_matmul_ffn2", _BertInt8MatmulFfn2())
