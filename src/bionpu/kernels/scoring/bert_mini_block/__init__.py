# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# bert_mini_block — AIE2P softmax + LayerNorm kernels for BERT-mini.
#
# Phase 0 step 0.3 (PRD-dnabert-epi v0.1.3 §3.8) adds two AIE-resident
# kernels alongside the existing bert_int8_matmul_* set:
#
#   * bert_mini_attention_softmax  — per-row softmax over (M, M) attention
#     scores; FP16 (bf16 actually — AIE2P hardware-native; see findings)
#     reduction with /sqrt(head_dim) scaling assumed pre-applied by host.
#
#   * bert_mini_layer_norm         — per-row LayerNorm over hidden dim;
#     bf16 reductions, γ/β streamed once per launch and held tile-resident.
#
# Both ops register in :data:`bionpu.dispatch.npu.NPU_OPS` at import time.
# Numpy reference fallback exists when the silicon xclbin is missing.

from __future__ import annotations

from pathlib import Path

import numpy as np

from bionpu.dispatch.npu import NpuOp, default_backend, register_npu_op


# Pinned shape constants — tracked against PRD-dnabert-epi v0.1.3 §3.8.
_M = 47
_HIDDEN = 256
_HEAD_DIM = 64
_NUM_HEADS = 4
_SOFTMAX_PAD = 64        # pad M=47 to multiple of bf16 vector lane (32)
_GELU_HIDDEN = 1024      # BERT-mini FFN expansion dim (= 4 * _HIDDEN)

_ARG_X = 3
_ARG_GB = 4
_ARG_Y = 5

_NEG_SENTINEL_BF16 = np.float32(-65000.0)


def _round4(n: int) -> int:
    return ((int(n) + 3) // 4) * 4


def _default_artifacts_dir(variant: str) -> Path:
    dispatch_dir = (
        Path(__file__).resolve().parents[3]
        / "dispatch" / "_npu_artifacts" / f"bert_mini_{variant}"
    )
    if (dispatch_dir / "final.xclbin").is_file() and (
        dispatch_dir / "insts.bin"
    ).is_file():
        return dispatch_dir
    return Path(__file__).resolve().parent / "build" / variant


def _has_silicon_artifacts(path: Path) -> bool:
    return (path / "final.xclbin").is_file() and (path / "insts.bin").is_file()


# ─── bf16 round-trip helpers ────────────────────────────────────────
#
# Reference path round-trips floats through bf16 to match the silicon
# kernel's accumulator-storage type. Without this, the byte-equal
# compare against silicon would fail on the low mantissa bits.

def _f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-half-to-even f32 → bf16 (stored as uint16)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32).copy()
    # round-to-nearest-even: add 0x7FFF + lsb of upper 16 bits.
    lsb = (bits >> 16) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    bits = bits + bias
    return (bits >> 16).astype(np.uint16)


def _bf16_to_f32(x: np.ndarray) -> np.ndarray:
    """bf16 (stored as uint16) → f32 with low 16 mantissa bits zeroed."""
    x = np.ascontiguousarray(x, dtype=np.uint16)
    bits = x.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32).copy()


# ─── Numpy reference: softmax ───────────────────────────────────────


def _softmax_reference(scores: np.ndarray) -> np.ndarray:
    """Byte-equal numpy reference for bert_mini_attention_softmax.

    Args:
        scores: (NUM_ROWS, PAD) f32 — pre-scaled attention scores
            with tail elements [M..PAD-1] set to a large negative
            sentinel so they don't contribute to max / sum.

    Returns:
        (NUM_ROWS, PAD) f32 — softmax(scores) with bf16 round-trip
        applied to match silicon storage type. The rounded result is
        what the silicon kernel produces; cast to bf16 storage via
        :func:`_f32_to_bf16` for byte-equal compare.
    """
    log2e = np.float32(1.4426950408889634)
    out = np.empty_like(scores, dtype=np.float32)
    for r in range(scores.shape[0]):
        row = scores[r].astype(np.float32)
        scaled = row * log2e
        scaled = _bf16_to_f32(_f32_to_bf16(scaled))   # bf16 round-trip
        row_max = scaled.max()
        row_max = _bf16_to_f32(_f32_to_bf16(np.array([row_max])))[0]
        e = np.exp2(scaled - row_max)
        e = _bf16_to_f32(_f32_to_bf16(e))             # bf16 round-trip
        s = e.sum()
        inv_s = _bf16_to_f32(_f32_to_bf16(np.array([1.0 / s])))[0]
        out[r] = _bf16_to_f32(_f32_to_bf16(e * inv_s))
    return out


def _layer_norm_reference(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Byte-equal numpy reference for bert_mini_layer_norm.

    All inputs in f32 (host side); the reference applies the same
    bf16 round-trip after each reduction step that the silicon
    kernel does internally so the byte-equal compare is meaningful
    after :func:`_f32_to_bf16` packing.

    Args:
        x:     (M, HIDDEN) f32 — row-major activations.
        gamma: (HIDDEN,)  f32 — per-channel scale.
        beta:  (HIDDEN,)  f32 — per-channel bias.

    Returns:
        (M, HIDDEN) f32 — LayerNorm(x).
    """
    epsilon = np.float32(1e-5)
    cols = x.shape[1]
    out = np.empty_like(x, dtype=np.float32)
    for r in range(x.shape[0]):
        row = x[r].astype(np.float32)
        row_bf = _bf16_to_f32(_f32_to_bf16(row))
        s = row_bf.sum()
        sq = (row_bf * row_bf).sum()
        mean = s / cols
        variance = sq / cols - mean * mean
        inv_std = 1.0 / np.sqrt(variance + epsilon)
        # Cast intermediate back through bf16 so the per-channel mul
        # picks up the same mantissa as silicon.
        mean_bf = _bf16_to_f32(_f32_to_bf16(np.array([mean])))[0]
        inv_std_bf = _bf16_to_f32(_f32_to_bf16(np.array([inv_std])))[0]
        diff = _bf16_to_f32(_f32_to_bf16(row_bf - mean_bf))
        norm = _bf16_to_f32(_f32_to_bf16(diff * inv_std_bf))
        scaled = _bf16_to_f32(_f32_to_bf16(norm * gamma))
        out[r] = _bf16_to_f32(_f32_to_bf16(scaled + beta))
    return out


# ─── Op classes + entry-point functions ─────────────────────────────


class _BertMiniAttentionSoftmax(NpuOp):
    """``bert_mini_attention_softmax`` — per-row softmax of attention scores."""

    name = "bert_mini_attention_softmax"

    def __call__(
        self,
        *,
        scores: np.ndarray,        # (NUM_ROWS, PAD) f32; tail pre-filled with sentinel
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_mini_attention_softmax(scores, artifacts_dir=artifacts_dir)


def bert_mini_attention_softmax(
    scores: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """Run per-row softmax over (NUM_ROWS, PAD) bf16 attention scores.

    NUM_ROWS = M * NUM_HEADS = 47 * 4 = 188 for BERT-mini step 0.3.
    PAD = 64 (M=47 padded up to bf16-vector multiple).

    Caller is responsible for:
      * Multiplying raw scores by 1/sqrt(head_dim) BEFORE passing in.
      * Filling tail elements [M..PAD-1] with -65000.0 sentinel.

    Returns:
        (NUM_ROWS, PAD) f32 softmax output (bf16 round-trip applied).
    """
    if scores.dtype != np.float32:
        scores = scores.astype(np.float32)
    if scores.ndim != 2 or scores.shape[1] % 32 != 0:
        raise ValueError(
            f"softmax: scores must be (NUM_ROWS, PAD) f32 with PAD%32==0, "
            f"got {scores.shape}"
        )

    reference = _softmax_reference(scores)
    artifacts_path = Path(artifacts_dir) if artifacts_dir is not None else (
        _default_artifacts_dir("attention_softmax")
    )
    if not _has_silicon_artifacts(artifacts_path):
        return reference

    NUM_ROWS, PAD = scores.shape
    in_bf16 = _f32_to_bf16(scores)
    in_bytes = _round4(NUM_ROWS * PAD * 2)
    out_bytes = _round4(NUM_ROWS * PAD * 2)
    in_payload = in_bf16.tobytes() + bytes(in_bytes - len(in_bf16.tobytes()))
    # softmax kernel ABI (one input, one output): arg3=x, arg4=y.
    raw, _, _, _ = default_backend().run_xclbin(
        xclbin=artifacts_path / "final.xclbin",
        insts=artifacts_path / "insts.bin",
        kernel_name="MLIR_AIE",
        in_buffers=[(in_payload, 3)],
        out_size=out_bytes,
        out_arg_index=4,
    )
    out_bf16 = np.frombuffer(raw, dtype=np.uint16, count=NUM_ROWS * PAD)
    return _bf16_to_f32(out_bf16.reshape(NUM_ROWS, PAD).copy())


class _BertMiniLayerNorm(NpuOp):
    """``bert_mini_layer_norm`` — per-row LN over hidden dim."""

    name = "bert_mini_layer_norm"

    def __call__(
        self,
        *,
        x: np.ndarray,             # (M, HIDDEN) f32
        gamma: np.ndarray,         # (HIDDEN,) f32
        beta: np.ndarray,          # (HIDDEN,) f32
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_mini_layer_norm(x, gamma, beta, artifacts_dir=artifacts_dir)


def bert_mini_layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """Run per-row LayerNorm over (M, HIDDEN) f32 activations.

    Args:
        x:     (M, HIDDEN) f32 row-major.
        gamma: (HIDDEN,)  f32 per-channel scale.
        beta:  (HIDDEN,)  f32 per-channel bias.

    Returns:
        (M, HIDDEN) f32 LayerNorm output (bf16 round-trip applied).
    """
    for arr, name in [(x, "x"), (gamma, "gamma"), (beta, "beta")]:
        if arr.dtype != np.float32:
            raise TypeError(f"layer_norm: {name} must be f32, got {arr.dtype}")
    if x.ndim != 2 or x.shape[1] != _HIDDEN:
        raise ValueError(
            f"layer_norm: x must be (M, {_HIDDEN}) f32, got {x.shape}"
        )
    if gamma.shape != (_HIDDEN,) or beta.shape != (_HIDDEN,):
        raise ValueError(
            f"layer_norm: gamma/beta must be ({_HIDDEN},), got "
            f"gamma={gamma.shape} beta={beta.shape}"
        )

    reference = _layer_norm_reference(x, gamma, beta)
    artifacts_path = Path(artifacts_dir) if artifacts_dir is not None else (
        _default_artifacts_dir("layer_norm")
    )
    if not _has_silicon_artifacts(artifacts_path):
        return reference

    M, H = x.shape
    in_bf16 = _f32_to_bf16(x)
    gamma_bf16 = _f32_to_bf16(gamma)
    beta_bf16 = _f32_to_bf16(beta)
    gb_bf16 = np.concatenate([gamma_bf16, beta_bf16], axis=0)

    in_bytes = _round4(M * H * 2)
    gb_bytes = _round4(2 * H * 2)
    out_bytes = _round4(M * H * 2)
    in_payload = in_bf16.tobytes() + bytes(in_bytes - in_bf16.nbytes)
    gb_payload = gb_bf16.tobytes() + bytes(gb_bytes - gb_bf16.nbytes)

    raw, _, _, _ = default_backend().run_xclbin(
        xclbin=artifacts_path / "final.xclbin",
        insts=artifacts_path / "insts.bin",
        kernel_name="MLIR_AIE",
        in_buffers=[
            (in_payload, _ARG_X),
            (gb_payload, _ARG_GB),
        ],
        out_size=out_bytes,
        out_arg_index=_ARG_Y,
    )
    out_bf16 = np.frombuffer(raw, dtype=np.uint16, count=M * H)
    return _bf16_to_f32(out_bf16.reshape(M, H).copy())


# ─── GELU (step 0.3b SCOPE-4) ────────────────────────────────────────


def _gelu_reference(x: np.ndarray) -> np.ndarray:
    """Byte-equal numpy reference for bert_mini_gelu.

    Tanh-approximation GELU with bf16 round-trip on the result so the
    reference matches silicon storage type.

        gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

    Args:
        x: (M, HIDDEN) f32 — FFN expansion-layer activation in float.

    Returns:
        (M, HIDDEN) f32 — GELU(x) with bf16 round-trip applied.
    """
    sqrt_2_over_pi = np.float32(0.79788456)
    kBeta = np.float32(0.044715)
    out = np.empty_like(x, dtype=np.float32)
    for r in range(x.shape[0]):
        row = x[r].astype(np.float32)
        x3 = row * row * row
        inner = sqrt_2_over_pi * (row + kBeta * x3)
        t = np.tanh(inner).astype(np.float32)
        result = np.float32(0.5) * row * (np.float32(1.0) + t)
        out[r] = _bf16_to_f32(_f32_to_bf16(result))
    return out


class _BertMiniGelu(NpuOp):
    """``bert_mini_gelu`` — per-row tanh-approximation GELU over (M, HIDDEN)."""

    name = "bert_mini_gelu"

    def __call__(
        self,
        *,
        x: np.ndarray,
        artifacts_dir: Path | str | None = None,
    ) -> np.ndarray:
        return bert_mini_gelu(x, artifacts_dir=artifacts_dir)


def bert_mini_gelu(
    x: np.ndarray,
    *,
    artifacts_dir: Path | str | None = None,
) -> np.ndarray:
    """Run per-row tanh-GELU over (M, HIDDEN) f32 activations.

    Default shape (BERT-mini step 0.3b): M=47, HIDDEN=1024 (= 4×256
    FFN expansion). HIDDEN must be a multiple of 16 (the upstream
    bf16 GELU vector width).

    Args:
        x: (M, HIDDEN) f32 row-major activations.

    Returns:
        (M, HIDDEN) f32 GELU output (bf16 round-trip applied).
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if x.ndim != 2 or x.shape[1] % 16 != 0:
        raise ValueError(
            f"gelu: x must be (M, HIDDEN) f32 with HIDDEN%16==0, got {x.shape}"
        )

    reference = _gelu_reference(x)
    artifacts_path = Path(artifacts_dir) if artifacts_dir is not None else (
        _default_artifacts_dir("gelu")
    )
    if not _has_silicon_artifacts(artifacts_path):
        return reference

    M, H = x.shape
    in_bf16 = _f32_to_bf16(x)
    in_bytes = _round4(M * H * 2)
    out_bytes = _round4(M * H * 2)
    in_payload = in_bf16.tobytes() + bytes(in_bytes - in_bf16.nbytes)

    # GELU ABI mirrors softmax: arg 3 = input, arg 4 = output.
    raw, _, _, _ = default_backend().run_xclbin(
        xclbin=artifacts_path / "final.xclbin",
        insts=artifacts_path / "insts.bin",
        kernel_name="MLIR_AIE",
        in_buffers=[(in_payload, 3)],
        out_size=out_bytes,
        out_arg_index=4,
    )
    out_bf16 = np.frombuffer(raw, dtype=np.uint16, count=M * H)
    return _bf16_to_f32(out_bf16.reshape(M, H).copy())


# ─── Op registration ─────────────────────────────────────────────────

register_npu_op("bert_mini_attention_softmax", _BertMiniAttentionSoftmax())
register_npu_op("bert_mini_layer_norm", _BertMiniLayerNorm())
register_npu_op("bert_mini_gelu", _BertMiniGelu())


# ─── Step 0.3c re-exports ─────────────────────────────────────────────
# The BertMiniBlock host-side composition class lives in block.py to
# keep import-time cheap (no torch / transformers needed for the kernel
# entry points). Re-expose here for convenient `from bionpu.kernels.
# scoring.bert_mini_block import BertMiniBlock`.
from bionpu.kernels.scoring.bert_mini_block.block import (  # noqa: E402
    BertMiniBlock,
    BertMiniBlockWeights,
)

__all__ = [
    "bert_mini_attention_softmax",
    "bert_mini_layer_norm",
    "bert_mini_gelu",
    "BertMiniBlock",
    "BertMiniBlockWeights",
]
