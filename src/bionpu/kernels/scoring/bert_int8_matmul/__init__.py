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

Registers BERT-base (hidden=768) specializations in
:data:`bionpu.dispatch.npu.NPU_OPS` at import time:

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

Step 0.3b (PRD-dnabert-epi v0.1.4) adds BERT-mini hidden=256 variants
that share the qkvo group-kernel discipline at smaller K and group-N:

* ``bert_int8_matmul_qkvo_h256``  (M=47, K=256, N=256)
* ``bert_int8_matmul_ffn1_h256``  (M=47, K=256, N=1024)
* ``bert_int8_matmul_ffn2_h256``  (M=47, K=1024, N=256)

These reuse the same packing helpers as the BERT-base variants; the
hidden=256 entry points pin K, group_n, and the artifact-directory
suffix at call time so a single kernel C++/IRON-Python source pair
compiles to multiple specialisations driven by Makefile flags. K and
group_n are no longer hardcoded module-level constants on the helper
functions — see SCOPE-3 in PRD-dnabert-epi-on-xdna v0.1.4 §1.5
step 0.3b.

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
# reference accepts any M ≤ _MAX_M, any K, and any N — the per-variant
# entry-point functions below pin K + N + group_n at call time so
# multiple hidden-size specialisations (BERT-base 768, BERT-mini 256)
# share the same packing/emulation helpers.
_MAX_M = 47
_K_CHUNK = 8
_QKVO_TILES = 4
_ARG_X = 3
_ARG_WS = 4
_ARG_Y = 5

# BERT-base (hidden=768) shape constants. Kept as named module-level
# values for the existing entry points; new entry points pass shapes
# in directly rather than read these.
_K = 768                  # BERT-base hidden / qkvo K and ffn1 K
_HEAD_N = 2
_QKVO_N = 768
_FFN1_K = 768
_FFN1_N = 3072
_FFN2_K = 3072
_FFN2_N = 768
_FFN_GROUP_N = 768        # BERT-base group kernel output width

# BERT-mini (hidden=256) shape constants for step 0.3b variants.
_K_H256 = 256             # BERT-mini hidden / qkvo K and ffn1 K
_QKVO_N_H256 = 256
_FFN1_K_H256 = 256
_FFN1_N_H256 = 1024
_FFN2_K_H256 = 1024
_FFN2_N_H256 = 256
_FFN_GROUP_N_H256 = 256   # BERT-mini group kernel output width (= hidden)


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


# ─── Activation quantization helpers (step 0.3d) ─────────────────────
#
# Step 0.3d (PRD-dnabert-epi v0.1.5 §1.5) host-side fix for the two
# activations whose per-tensor symmetric INT8 scale collapses their
# distribution: softmax-output (sparse, one peak per row → per-token
# scale preserves the soft-attention tail) and GELU-output (long-tailed
# outliers compress the bulk → 99.9th percentile clip preserves bulk
# resolution). Both are pure numpy; no kernel changes required because
# the silicon's i32 accumulator is independent of the input scale and
# the host re-applies the per-row dequant after the kernel completes.


def quantise_per_token_sym_int8(
    x_fp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-token (per-row) symmetric INT8 quantisation of a 2D activation.

    Each row of ``x_fp`` is quantised against its own max-abs, so a
    sparse softmax row whose peak is 0.99 and whose tail is ~1e-3 keeps
    full INT8 resolution on the tail; whereas the per-tensor variant
    would scale the entire tensor against the global max (~1) and
    collapse the tail to a single INT8 bin.

    Args:
        x_fp: ``(M, K)`` float32 activation.

    Returns:
        Tuple of:
          * ``x_int8`` — ``(M, K)`` int8, each row independently scaled.
          * ``scale_x_per_row`` — ``(M,)`` float32, scale_x[m] = max(|x_fp[m]|) / 127.
          * ``scale_x_unif`` — float, ``max(scale_x_per_row)``. Used as
            the unified ``scale_x`` for the kernel's ``scales_combined``
            so the saturation budget is sized for the worst-case row.

    Dequant convention (mirrored in the BertMiniBlock helpers):

      ``y_fp[m, n] = y_int8[m, n] * scale_y * (scale_x_per_row[m] / scale_x_unif)``

    Derivation: the silicon computes
    ``y_int8[m, n] = sat(round(int_acc[m, n] * sc_comb[n]))``; the host
    sets ``sc_comb[n] = scale_x_unif * scale_w[n] / scale_y``, so
    ``int_acc[m, n] ≈ y_int8[m, n] * scale_y / (scale_x_unif * scale_w[n])``.
    Since the underlying FP product is
    ``y_fp[m, n] = scale_x_per_row[m] * scale_w[n] * int_acc[m, n]``,
    substituting gives the formula above. When all rows share the same
    scale, this degenerates to the per-tensor case
    (``scale_x_per_row[m] / scale_x_unif == 1``).
    """
    x_fp = np.ascontiguousarray(x_fp, dtype=np.float32)
    if x_fp.ndim != 2:
        raise ValueError(f"quantise_per_token_sym_int8: expected 2D, got {x_fp.shape}")
    row_max = np.maximum(np.abs(x_fp).max(axis=1), 1e-12)  # (M,)
    scale_x_per_row = (row_max / 127.0).astype(np.float32)
    x_int8 = np.clip(
        np.round(x_fp / scale_x_per_row[:, None]),
        -128, 127,
    ).astype(np.int8)
    scale_x_unif = float(scale_x_per_row.max())
    return x_int8, scale_x_per_row, scale_x_unif


def quantise_per_tensor_percentile_sym_int8(
    x_fp: np.ndarray,
    *,
    percentile: float = 99.9,
) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric INT8 quantisation using a percentile clip.

    Replaces ``max(|x|)`` with ``np.percentile(|x|, percentile)`` so a
    single GELU outlier doesn't compress the bulk distribution to a
    handful of INT8 bins. Values beyond the percentile are clipped at
    ``±127``; we trade tail-clipping (a few bf16-precision-bounded
    elements) for ~3-10× better resolution on the bulk distribution.

    Args:
        x_fp: float32 array of any shape.
        percentile: percentile of |x| used as the upper magnitude.
            99.9 by default — clamps roughly 0.1% of elements at most.

    Returns:
        Tuple of (x_int8, scale_x). ``x_int8`` clipped+rounded; values
        whose absolute value exceeds the percentile are saturated to
        ``±127`` instead of mapping to overflow values.
    """
    x_fp = np.ascontiguousarray(x_fp, dtype=np.float32)
    abs_x = np.abs(x_fp)
    clip_mag = float(np.percentile(abs_x, percentile))
    clip_mag = max(clip_mag, 1e-12)
    scale = clip_mag / 127.0
    x_int8 = np.clip(np.round(x_fp / scale), -128, 127).astype(np.int8)
    return x_int8, float(scale)


def _pack_head_ws(w: np.ndarray, scales_combined: np.ndarray) -> bytes:
    size = _round4(_HEAD_N * _K + (_HEAD_N + 1) * 4)
    return _pad_bytes(w.tobytes() + scales_combined.tobytes(), size)


def _pack_head_ws_param(
    w: np.ndarray, scales_combined: np.ndarray, *, k: int, n: int
) -> bytes:
    """Parameterised head-style ws packer used by step-0.3b qkt/sv variants."""
    size = _round4(n * k + (n + 1) * 4)
    return _pad_bytes(w.tobytes() + scales_combined.tobytes(), size)


def _pad_rows_param(x: np.ndarray, *, m_pad: int, cols: int) -> np.ndarray:
    """Parameterised row-pad helper used by step-0.3b qkt/sv variants."""
    if x.shape[0] == m_pad:
        return np.ascontiguousarray(x, dtype=np.int8)
    out = np.zeros((m_pad, cols), dtype=np.int8)
    out[: x.shape[0], :] = x
    return out


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


def _pack_qkvo_xs_param(
    x: np.ndarray, scales_combined: np.ndarray, *, k: int, group_n: int
) -> bytes:
    """Parameterised qkvo xs packer used by step-0.3b hidden=256 variants."""
    return _pack_grouped_xs(x, scales_combined, k=k, group_n=group_n)


def _pack_qkvo_w_param(w: np.ndarray, *, group_n: int) -> bytes:
    """Parameterised qkvo weight packer used by step-0.3b hidden=256 variants."""
    return _pack_grouped_w(w, group_n=group_n)


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
    return _bert_int8_matmul_qkvo_param(
        x,
        w,
        scales_combined,
        expected_k=_K,
        expected_n=_QKVO_N,
        variant="qkvo",
        artifacts_dir=artifacts_dir,
    )


def _bert_int8_matmul_qkvo_param(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    expected_k: int,
    expected_n: int,
    variant: str,
    artifacts_dir: Path | str | None,
) -> np.ndarray:
    """Parameterised qkvo dispatch shared by BERT-base and BERT-mini variants.

    SCOPE-3 of step 0.3b: thread (K, group_n=N, variant suffix) through
    the qkvo dispatch so the same packing + read-row-major logic serves
    both hidden=768 (`variant="qkvo"`) and hidden=256
    (`variant="qkvo_h256"`).
    """
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

    xs = _pack_qkvo_xs_param(
        x, scales_combined, k=expected_k, group_n=expected_n
    )
    w_chunks = _pack_qkvo_w_param(
        np.ascontiguousarray(w), group_n=expected_n
    )
    y_size = _round4(_padded_m(_MAX_M) * expected_n)
    raw, _, _, _ = default_backend().run_xclbin(
        xclbin=artifacts_path / "final.xclbin",
        insts=artifacts_path / "insts.bin",
        kernel_name="MLIR_AIE",
        in_buffers=[
            (xs, _ARG_X),
            (w_chunks, _ARG_WS),
        ],
        out_size=y_size,
        out_arg_index=_ARG_Y,
    )
    return _read_row_major(raw, n=expected_n)[: x.shape[0], :].copy()


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
    group_n: int = _FFN_GROUP_N,
) -> np.ndarray:
    """Dispatch a feed-forward matmul using the DDR-streamed grouped ABI.

    ``group_n`` defaults to the BERT-base value (768) for backward compat;
    BERT-mini hidden=256 callers pass ``group_n=256`` so the host-side
    output-group iteration and the tile-resident accumulator both stay
    sized to the (smaller) hidden dim.
    """
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
        group_n=group_n,
    )
    w_groups = _pack_grouped_w(np.ascontiguousarray(w), group_n=group_n)

    # The xclbin's runtime_sequence consumes exactly one ``group_n``
    # output group per launch; iterate host dispatch over groups and
    # assemble the N-axis output. The single-group case (e.g., ffn2 at
    # BERT-base, both ffn variants at BERT-mini hidden=256) degenerates
    # to one launch.
    n_groups = expected_n // group_n
    m_pad = _padded_m(_MAX_M)
    y_per_group = _round4(m_pad * group_n)
    xs_bytes_per_group = len(xs) // n_groups
    w_bytes_per_group = len(w_groups) // n_groups

    y_full = np.zeros((m_pad, expected_n), dtype=np.int8)
    backend = default_backend()
    for g in range(n_groups):
        xs_g = xs[g * xs_bytes_per_group : (g + 1) * xs_bytes_per_group]
        w_g = w_groups[g * w_bytes_per_group : (g + 1) * w_bytes_per_group]
        raw, _, _, _ = backend.run_xclbin(
            xclbin=artifacts_path / "final.xclbin",
            insts=artifacts_path / "insts.bin",
            kernel_name="MLIR_AIE",
            in_buffers=[
                (xs_g, _ARG_X),
                (w_g, _ARG_WS),
            ],
            out_size=y_per_group,
            out_arg_index=_ARG_Y,
        )
        y_g = _read_row_major(raw, n=group_n)
        y_full[:, g * group_n : (g + 1) * group_n] = y_g

    return y_full[: x.shape[0], :].copy()


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


# ─── BERT-mini hidden=256 specialisations (step 0.3b) ───────────────


class _BertInt8MatmulQkvoH256(NpuOp):
    """``bert_int8_matmul_qkvo_h256`` — qkvo at BERT-mini hidden (M=47, K=256, N=256)."""

    name = "bert_int8_matmul_qkvo_h256"

    def __call__(
        self,
        *,
        x: np.ndarray,
        w: np.ndarray,
        scales_combined: np.ndarray,
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_int8_matmul_qkvo_h256(
            x, w, scales_combined, artifacts_dir=artifacts_dir
        )


def bert_int8_matmul_qkvo_h256(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """qkvo INT8 matmul at BERT-mini hidden=256 (M=47, K=256, N=256).

    Uses the same multi-tile + K-chunked discipline as the BERT-base
    qkvo, just at K=256 / N=256 instead of K=768 / N=768. K=256 is
    divisible by ``_K_CHUNK=8`` (32 chunks), so the existing IRON
    lowering applies directly via Makefile shape overrides.
    """
    return _bert_int8_matmul_qkvo_param(
        x,
        w,
        scales_combined,
        expected_k=_K_H256,
        expected_n=_QKVO_N_H256,
        variant="qkvo_h256",
        artifacts_dir=artifacts_dir,
    )


class _BertInt8MatmulFfn1H256(NpuOp):
    """``bert_int8_matmul_ffn1_h256`` — ffn1 at BERT-mini (M=47, K=256, N=1024)."""

    name = "bert_int8_matmul_ffn1_h256"

    def __call__(
        self,
        *,
        x: np.ndarray,
        w: np.ndarray,
        scales_combined: np.ndarray,
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_int8_matmul_ffn1_h256(
            x, w, scales_combined, artifacts_dir=artifacts_dir
        )


def bert_int8_matmul_ffn1_h256(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """BERT-mini FFN expansion (M<=47, K=256, N=1024).

    Streams four 256-output groups (1024 / 256 = 4) using the qkvo-
    sized group kernel, mirroring the BERT-base ffn1 host-side ABI.
    """
    return _bert_int8_matmul_ffn(
        x,
        w,
        scales_combined,
        expected_k=_FFN1_K_H256,
        expected_n=_FFN1_N_H256,
        variant="ffn1_h256",
        artifacts_dir=artifacts_dir,
        group_n=_FFN_GROUP_N_H256,
    )


class _BertInt8MatmulFfn2H256(NpuOp):
    """``bert_int8_matmul_ffn2_h256`` — ffn2 at BERT-mini (M=47, K=1024, N=256)."""

    name = "bert_int8_matmul_ffn2_h256"

    def __call__(
        self,
        *,
        x: np.ndarray,
        w: np.ndarray,
        scales_combined: np.ndarray,
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_int8_matmul_ffn2_h256(
            x, w, scales_combined, artifacts_dir=artifacts_dir
        )


def bert_int8_matmul_ffn2_h256(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """BERT-mini FFN contraction (M<=47, K=1024, N=256). Single output group."""
    return _bert_int8_matmul_ffn(
        x,
        w,
        scales_combined,
        expected_k=_FFN2_K_H256,
        expected_n=_FFN2_N_H256,
        variant="ffn2_h256",
        artifacts_dir=artifacts_dir,
        group_n=_FFN_GROUP_N_H256,
    )


# ─── Per-head Q·Kᵀ / scores·V (step 0.3b SCOPE-1) ────────────────────
#
# Q·Kᵀ per head:   (M=47, K=64, N=47)
# scores·V per head: (M=47, K=47, N=64)  — K=47 is the unaligned shape
#                                          that motivated SCOPE-1.
#
# Both reuse the head-style single-tile lowering. The head path streams
# full K resident on one tile (no K_CHUNK requirement), so K=47 needs
# no padding or new IRON lowering. Per-tile DM at scores·V shape:
#   x:        47×47 = 2,209 B
#   ws:       64×47 + 65×4 = 3,268 B
#   y:        47×64 = 3,008 B
#   total:    ~8.5 KB (~13% of 64 KiB DM)
# Trivial fit; the qkvo K%K_CHUNK==0 contract is bypassed entirely.


def _bert_int8_matmul_head_param(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    expected_k: int,
    expected_n: int,
    variant: str,
    artifacts_dir: Path | str | None,
) -> np.ndarray:
    """Parameterised head-style dispatch shared by the BERT-base classifier
    head (K=768, N=2) and the step-0.3b per-head Q·Kᵀ / scores·V variants.
    """
    reference = _emulate(
        x, w, scales_combined,
        expected_n=expected_n, expected_k=expected_k,
    )
    artifacts_path = Path(artifacts_dir) if artifacts_dir is not None else (
        _default_artifacts_dir(variant)
    )
    if not _has_silicon_artifacts(artifacts_path):
        return reference

    x_full = _pad_rows_param(x, m_pad=_MAX_M, cols=expected_k)
    x_size = _round4(_MAX_M * expected_k)
    y_size = _round4(_MAX_M * expected_n)
    raw, _, _, _ = default_backend().run_xclbin(
        xclbin=artifacts_path / "final.xclbin",
        insts=artifacts_path / "insts.bin",
        kernel_name="MLIR_AIE",
        in_buffers=[
            (_pad_bytes(x_full.tobytes(), x_size), _ARG_X),
            (_pack_head_ws_param(
                np.ascontiguousarray(w), scales_combined,
                k=expected_k, n=expected_n,
            ), _ARG_WS),
        ],
        out_size=y_size,
        out_arg_index=_ARG_Y,
    )
    y = np.frombuffer(raw, dtype=np.int8, count=_MAX_M * expected_n)
    return y.reshape(_MAX_M, expected_n)[: x.shape[0], :].copy()


_QKT_K = 64
_QKT_N = 47
_SV_K = 47
_SV_N = 64


class _BertInt8MatmulQkt(NpuOp):
    """``bert_int8_matmul_qkt`` — Q·Kᵀ per-head (M=47, K=64, N=47)."""

    name = "bert_int8_matmul_qkt"

    def __call__(
        self,
        *,
        x: np.ndarray,
        w: np.ndarray,
        scales_combined: np.ndarray,
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_int8_matmul_qkt(
            x, w, scales_combined, artifacts_dir=artifacts_dir
        )


def bert_int8_matmul_qkt(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """Per-head Q·Kᵀ INT8 matmul (M=47, K=64, N=47).

    K (head_dim=64) is the reduction; N (=47=M) is the second token's
    attention-target index. Output is the per-head (47, 47) attention
    score matrix in INT8 (host-side dequantises and softmax-scales).
    """
    return _bert_int8_matmul_head_param(
        x, w, scales_combined,
        expected_k=_QKT_K, expected_n=_QKT_N,
        variant="qkt",
        artifacts_dir=artifacts_dir,
    )


class _BertInt8MatmulSv(NpuOp):
    """``bert_int8_matmul_sv`` — scores·V per-head (M=47, K=47, N=64)."""

    name = "bert_int8_matmul_sv"

    def __call__(
        self,
        *,
        x: np.ndarray,
        w: np.ndarray,
        scales_combined: np.ndarray,
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_int8_matmul_sv(
            x, w, scales_combined, artifacts_dir=artifacts_dir
        )


def bert_int8_matmul_sv(
    x: np.ndarray,
    w: np.ndarray,
    scales_combined: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """Per-head scores·V INT8 matmul (M=47, K=47, N=64).

    K (=47=M) is the attention-token reduction; N (head_dim=64) is the
    output projection per head. The INT8 zero-pad-to-multiple-of-8
    workaround the qkvo path would need is sidestepped by using the
    head-style single-tile lowering (no K%K_CHUNK constraint).
    """
    return _bert_int8_matmul_head_param(
        x, w, scales_combined,
        expected_k=_SV_K, expected_n=_SV_N,
        variant="sv",
        artifacts_dir=artifacts_dir,
    )


# ─── Op registration ─────────────────────────────────────────────────

register_npu_op("bert_int8_matmul_head", _BertInt8MatmulHead())
register_npu_op("bert_int8_matmul_qkvo", _BertInt8MatmulQkvo())
register_npu_op("bert_int8_matmul_ffn1", _BertInt8MatmulFfn1())
register_npu_op("bert_int8_matmul_ffn2", _BertInt8MatmulFfn2())

# step 0.3b: BERT-mini hidden=256 variants for single-block composition.
register_npu_op("bert_int8_matmul_qkvo_h256", _BertInt8MatmulQkvoH256())
register_npu_op("bert_int8_matmul_ffn1_h256", _BertInt8MatmulFfn1H256())
register_npu_op("bert_int8_matmul_ffn2_h256", _BertInt8MatmulFfn2H256())

# step 0.3b SCOPE-1: per-head Q·Kᵀ + scores·V (head-style single-tile).
register_npu_op("bert_int8_matmul_qkt", _BertInt8MatmulQkt())
register_npu_op("bert_int8_matmul_sv", _BertInt8MatmulSv())
