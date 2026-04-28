# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# bert_mini_block.block — host-side composition of one BERT-mini layer
#                         on AIE2P silicon.
#
# Phase 0 step 0.3c (PRD-dnabert-epi v0.1.5 §1.5): all eight kernels for
# one BERT-mini block were silicon-validated in step 0.3b. This module
# orchestrates them into a single end-to-end transformer block forward
# pass (3× Q/K/V projections → per-head Q·Kᵀ → softmax → per-head
# scores·V → output projection → residual+LN1 → ffn1 → GELU → ffn2 →
# residual+LN2). 21 silicon dispatches per block; in-process pyxrt only.
#
# No new silicon work. No `npu_silicon_lock` wrapping (in-process pyxrt
# does not need it per CLAUDE.md / PRD step 0.3c brief).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bionpu.kernels.scoring.bert_int8_matmul import (
    bert_int8_matmul_qkvo_h256,
    bert_int8_matmul_ffn1_h256,
    bert_int8_matmul_ffn2_h256,
    bert_int8_matmul_qkt,
    bert_int8_matmul_sv,
)
from bionpu.kernels.scoring.bert_mini_block import (
    bert_mini_attention_softmax,
    bert_mini_layer_norm,
    bert_mini_gelu,
)


# ─── BERT-mini hyperparameters (pinned per PRD §1.2) ─────────────────
_M = 47          # token-pair sequence length
_HIDDEN = 256
_NUM_HEADS = 4
_HEAD_DIM = 64   # _HIDDEN / _NUM_HEADS
_FFN_DIM = 1024  # 4 * _HIDDEN
_SOFTMAX_PAD = 64  # M=47 padded to bf16-vector multiple

_NEG_SENTINEL_BF16 = np.float32(-65000.0)


# ─── Quantization helpers (per-tensor sym INT8) ──────────────────────
#
# The quantize.py passport machinery returns per-output-channel weight
# scales + per-tensor activation scales. For step 0.3c composition, we
# want a self-contained pipeline: given a dict of FP32 weights from a
# BertLayer, produce per-Linear (W_int8, scales_combined_per_oc, x_scale,
# y_scale) tuples and run the silicon path through them.
#
# fused-scale convention matches bionpu.kernels.scoring.bert_int8_matmul:
#   y_int8 = sat( (x_int8 @ w_int8.T).i32 * scale_combined ),
#   scale_combined[oc] = (scale_x * scale_w[oc]) / scale_y.
#
# Bias is added in FP32 *after* dequantising y_int8 (bias is small,
# kept in FP for accuracy). The trailing scale_combined entry stays
# zero for all current kernel ABIs.


def _quantise_per_channel_sym_int8(W_fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric INT8 quantisation of a 2D weight."""
    W_fp = np.ascontiguousarray(W_fp, dtype=np.float32)
    # Per-output-channel max-abs scaling along the input axis.
    scale = np.maximum(np.abs(W_fp).max(axis=1), 1e-12) / 127.0
    W_int8 = np.clip(np.round(W_fp / scale[:, None]), -128, 127).astype(np.int8)
    return W_int8, scale.astype(np.float32)


def _quantise_per_tensor_sym_int8(x_fp: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric INT8 quantisation of an activation."""
    x_fp = np.ascontiguousarray(x_fp, dtype=np.float32)
    m = float(np.maximum(np.abs(x_fp).max(), 1e-12))
    scale = m / 127.0
    x_int8 = np.clip(np.round(x_fp / scale), -128, 127).astype(np.int8)
    return x_int8, float(scale)


def _build_scales_combined(
    scale_x: float,
    scale_w_per_oc: np.ndarray,
    scale_y: float,
    n: int,
) -> np.ndarray:
    """Pack (N+1,) float32 scales_combined for the matmul kernel."""
    if scale_y <= 0.0:
        scale_y = 1.0
    out = np.empty((n + 1,), dtype=np.float32)
    out[:n] = (scale_x * scale_w_per_oc) / scale_y
    out[n] = 0.0  # trailing bias slot reserved
    return out


# ─── BertMiniBlock ───────────────────────────────────────────────────


@dataclass
class BertMiniBlockWeights:
    """FP32 weights + biases for one BERT-mini transformer layer.

    Keys mirror HuggingFace BertLayer state_dict naming so the caller
    can pass `dict(layer.state_dict())` directly; we re-shape and
    quantise on construction.
    """
    # Self-attention sublayer (BertAttention.self.{query,key,value})
    q_w: np.ndarray  # (256, 256)
    q_b: np.ndarray  # (256,)
    k_w: np.ndarray
    k_b: np.ndarray
    v_w: np.ndarray
    v_b: np.ndarray
    # Output projection (BertSelfOutput.dense) + residual+LN1
    o_w: np.ndarray  # (256, 256)
    o_b: np.ndarray
    ln1_g: np.ndarray  # (256,)
    ln1_b: np.ndarray
    # FFN sublayer (BertIntermediate + BertOutput)
    ffn1_w: np.ndarray  # (1024, 256)
    ffn1_b: np.ndarray  # (1024,)
    ffn2_w: np.ndarray  # (256, 1024)
    ffn2_b: np.ndarray  # (256,)
    ln2_g: np.ndarray
    ln2_b: np.ndarray

    @classmethod
    def from_hf_layer_state_dict(cls, sd: dict[str, Any]) -> BertMiniBlockWeights:
        """Construct from a HuggingFace BertLayer state_dict."""
        def _np(t):
            try:
                return t.detach().to("cpu").numpy().astype(np.float32)
            except AttributeError:
                return np.asarray(t, dtype=np.float32)
        return cls(
            q_w=_np(sd["attention.self.query.weight"]),
            q_b=_np(sd["attention.self.query.bias"]),
            k_w=_np(sd["attention.self.key.weight"]),
            k_b=_np(sd["attention.self.key.bias"]),
            v_w=_np(sd["attention.self.value.weight"]),
            v_b=_np(sd["attention.self.value.bias"]),
            o_w=_np(sd["attention.output.dense.weight"]),
            o_b=_np(sd["attention.output.dense.bias"]),
            ln1_g=_np(sd["attention.output.LayerNorm.weight"]),
            ln1_b=_np(sd["attention.output.LayerNorm.bias"]),
            ffn1_w=_np(sd["intermediate.dense.weight"]),
            ffn1_b=_np(sd["intermediate.dense.bias"]),
            ffn2_w=_np(sd["output.dense.weight"]),
            ffn2_b=_np(sd["output.dense.bias"]),
            ln2_g=_np(sd["output.LayerNorm.weight"]),
            ln2_b=_np(sd["output.LayerNorm.bias"]),
        )


@dataclass
class _PreQuantisedLinear:
    """One Linear's INT8 weights + per-channel scale + FP32 bias."""
    w_int8: np.ndarray
    scale_w: np.ndarray  # (N,) float32
    bias: np.ndarray     # (N,) float32

    @classmethod
    def from_fp(cls, W_fp: np.ndarray, b_fp: np.ndarray) -> _PreQuantisedLinear:
        w_int8, scale_w = _quantise_per_channel_sym_int8(W_fp)
        return cls(w_int8=w_int8, scale_w=scale_w, bias=b_fp.astype(np.float32))


class BertMiniBlock:
    """End-to-end BERT-mini transformer-layer dispatch on AIE2P.

    Constructor builds an INT8-quantised weight passport from the FP32
    weights; ``forward(x)`` runs the silicon kernels in sequence and
    returns the FP32 layer output. Step 0.3c (PRD-dnabert-epi v0.1.5)
    of Phase 0 — composition only, no silicon authoring.

    Numerical design:
      * Each matmul (qkvo/qkt/sv/ffn) runs INT8 in / INT8 out with
        per-output-channel symmetric scales. The fused-scale collapses
        (scale_x * scale_w[oc] * (1 / scale_y_proxy)) into one float;
        host code de-quantises the INT8 output and adds the FP32 bias.
      * softmax / LN / GELU run FP32→bf16 silicon (per their kernel
        contracts).
      * Re-quantisation between matmul stages re-derives a per-tensor
        activation scale dynamically from the FP intermediate (cheaper
        than a separate calibration pass for Phase 0; trades a small
        amount of additional FP32 introspection on the host side).
    """

    def __init__(
        self,
        weights: BertMiniBlockWeights,
        *,
        artifacts_root: Path | str | None = None,
    ) -> None:
        self.weights = weights
        if artifacts_root is None:
            self._matmul_root = (
                Path(__file__).resolve().parents[1] / "bert_int8_matmul" / "build"
            )
            self._block_root = Path(__file__).resolve().parent / "build"
        else:
            self._matmul_root = Path(artifacts_root) / "bert_int8_matmul"
            self._block_root = Path(artifacts_root) / "bert_mini_block"
        self._artifacts = {
            "qkvo_h256": self._matmul_root / "qkvo_h256",
            "ffn1_h256": self._matmul_root / "ffn1_h256",
            "ffn2_h256": self._matmul_root / "ffn2_h256",
            "qkt": self._matmul_root / "qkt",
            "sv": self._matmul_root / "sv",
            "softmax": self._block_root / "softmax",
            "layer_norm": self._block_root / "layer_norm",
            "gelu": self._block_root / "gelu",
        }
        # Pre-quantise weights once at construction; biases stay FP32.
        self._q_q = _PreQuantisedLinear.from_fp(weights.q_w, weights.q_b)
        self._q_k = _PreQuantisedLinear.from_fp(weights.k_w, weights.k_b)
        self._q_v = _PreQuantisedLinear.from_fp(weights.v_w, weights.v_b)
        self._q_o = _PreQuantisedLinear.from_fp(weights.o_w, weights.o_b)
        self._q_ffn1 = _PreQuantisedLinear.from_fp(weights.ffn1_w, weights.ffn1_b)
        self._q_ffn2 = _PreQuantisedLinear.from_fp(weights.ffn2_w, weights.ffn2_b)

        # Per-op MSE breakdown collected during the most recent forward
        # pass; populated by ``forward(..., capture_mse=True)`` when the
        # caller passes a PyTorch reference dict.
        self.last_mse: dict[str, Any] = {}

    # ── Per-stage helpers ────────────────────────────────────────────

    def _matmul_qkvo(
        self,
        x_fp: np.ndarray,
        ql: _PreQuantisedLinear,
        variant: str,
    ) -> np.ndarray:
        """Run a (M, 256) @ (256, 256) projection on silicon, return FP32 (M, 256)."""
        x_int8, scale_x = _quantise_per_tensor_sym_int8(x_fp)
        # scale_y_proxy: pick a scale_y close to the expected output
        # magnitude so the INT8 saturation doesn't clip. Proxy by the
        # max-abs of the FP32 estimate (cheap on CPU, lossless).
        # We use the (W_fp @ x_fp) FP32 max as a one-shot estimate.
        # For step 0.3c this is accurate enough to keep clipping to <1%
        # of values in test runs; a real calibration would pin scale_y
        # at a corpus-derived percentile.
        # NOTE: bias is NOT folded into y here — applied post-dequant.
        y_fp_estimate_max = float(
            np.abs(x_fp.astype(np.float32) @ ql.w_int8.T.astype(np.float32) * ql.scale_w).max()
        )
        scale_y = max(y_fp_estimate_max, 1e-9) / 127.0
        sc_combined = _build_scales_combined(scale_x, ql.scale_w, scale_y, n=ql.w_int8.shape[0])
        y_int8 = bert_int8_matmul_qkvo_h256(
            x_int8,
            ql.w_int8,
            sc_combined,
            artifacts_dir=self._artifacts[variant],
        )
        # Dequantise and add FP32 bias.
        y_fp = y_int8.astype(np.float32) * scale_y + ql.bias[None, :]
        return y_fp

    def _matmul_ffn1(self, x_fp: np.ndarray) -> np.ndarray:
        x_int8, scale_x = _quantise_per_tensor_sym_int8(x_fp)
        y_fp_estimate_max = float(
            np.abs(
                x_fp.astype(np.float32)
                @ self._q_ffn1.w_int8.T.astype(np.float32) * self._q_ffn1.scale_w
            ).max()
        )
        scale_y = max(y_fp_estimate_max, 1e-9) / 127.0
        sc_combined = _build_scales_combined(
            scale_x, self._q_ffn1.scale_w, scale_y, n=self._q_ffn1.w_int8.shape[0]
        )
        y_int8 = bert_int8_matmul_ffn1_h256(
            x_int8,
            self._q_ffn1.w_int8,
            sc_combined,
            artifacts_dir=self._artifacts["ffn1_h256"],
        )
        return y_int8.astype(np.float32) * scale_y + self._q_ffn1.bias[None, :]

    def _matmul_ffn2(self, x_fp: np.ndarray) -> np.ndarray:
        x_int8, scale_x = _quantise_per_tensor_sym_int8(x_fp)
        y_fp_estimate_max = float(
            np.abs(
                x_fp.astype(np.float32)
                @ self._q_ffn2.w_int8.T.astype(np.float32) * self._q_ffn2.scale_w
            ).max()
        )
        scale_y = max(y_fp_estimate_max, 1e-9) / 127.0
        sc_combined = _build_scales_combined(
            scale_x, self._q_ffn2.scale_w, scale_y, n=self._q_ffn2.w_int8.shape[0]
        )
        y_int8 = bert_int8_matmul_ffn2_h256(
            x_int8,
            self._q_ffn2.w_int8,
            sc_combined,
            artifacts_dir=self._artifacts["ffn2_h256"],
        )
        return y_int8.astype(np.float32) * scale_y + self._q_ffn2.bias[None, :]

    def _matmul_qkt(self, q_h: np.ndarray, k_h: np.ndarray) -> np.ndarray:
        """Per-head Q · K^T matmul: (M, 64) × (M, 64)^T → (M, M)."""
        # qkt kernel ABI: x = (M, K=64), w = (N=47, K=64), output (M, N=47).
        # So for Q · K^T we set x = Q_h (M, 64), w = K_h (M, 64) — kernel
        # internally computes x @ w^T where w plays the role of "weights"
        # i.e. w[n,k] · x[m,k] summed over k = (Q · K^T)[m, n].
        x_int8, scale_x = _quantise_per_tensor_sym_int8(q_h)
        w_int8, scale_w_per = _quantise_per_channel_sym_int8(k_h)  # (N=M, K=64)
        # FP estimate for scale_y proxy.
        y_fp = q_h.astype(np.float32) @ k_h.astype(np.float32).T  # (M, M)
        scale_y = max(float(np.abs(y_fp).max()), 1e-9) / 127.0
        sc_combined = _build_scales_combined(scale_x, scale_w_per, scale_y, n=w_int8.shape[0])
        y_int8 = bert_int8_matmul_qkt(
            x_int8, w_int8, sc_combined, artifacts_dir=self._artifacts["qkt"]
        )
        return y_int8.astype(np.float32) * scale_y

    def _matmul_sv(self, scores_h: np.ndarray, v_h: np.ndarray) -> np.ndarray:
        """Per-head attn · V matmul: (M, M) × (M, 64) → (M, 64)."""
        # sv kernel ABI: x = (M, K=47), w = (N=64, K=47), output (M, N=64).
        # So for scores · V we need x = scores (M, M=47), w = V_h^T (head_dim=64, M=47).
        x_int8, scale_x = _quantise_per_tensor_sym_int8(scores_h)
        v_h_T = np.ascontiguousarray(v_h.T)  # (64, 47)
        w_int8, scale_w_per = _quantise_per_channel_sym_int8(v_h_T)  # (N=64, K=47)
        y_fp = scores_h.astype(np.float32) @ v_h.astype(np.float32)  # (M, 64)
        scale_y = max(float(np.abs(y_fp).max()), 1e-9) / 127.0
        sc_combined = _build_scales_combined(scale_x, scale_w_per, scale_y, n=w_int8.shape[0])
        y_int8 = bert_int8_matmul_sv(
            x_int8, w_int8, sc_combined, artifacts_dir=self._artifacts["sv"]
        )
        return y_int8.astype(np.float32) * scale_y

    # ── Block forward ────────────────────────────────────────────────

    def forward(
        self,
        x: np.ndarray,
        *,
        capture_mse: bool = False,
        torch_reference: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """End-to-end BERT-mini layer forward pass on AIE2P silicon.

        Args:
            x: (M, 256) float32 — token-major input embeddings.
            capture_mse: if True, fill ``self.last_mse`` with per-op
                MSE entries against ``torch_reference``.
            torch_reference: optional dict of intermediate FP32 outputs
                from the PyTorch reference. Keys: ``q_proj, k_proj,
                v_proj, qkt_per_head, softmax_per_head, sv_per_head,
                concat_proj, ln1, ffn1, gelu, ffn2, ln2``. When
                provided alongside ``capture_mse=True``, MSE per-op
                is recorded for the bisection report.

        Returns:
            (M, 256) float32 layer output.
        """
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if x.shape != (_M, _HIDDEN):
            raise ValueError(f"BertMiniBlock expects ({_M}, {_HIDDEN}), got {x.shape}")

        mse: dict[str, Any] = {}

        def _capture(name: str, ours: np.ndarray, key: str | None = None) -> None:
            if not capture_mse or torch_reference is None:
                return
            ref_key = key or name
            ref = torch_reference.get(ref_key)
            if ref is None:
                return
            mse[name] = float(((ours - ref.astype(np.float32)) ** 2).mean())

        # 1. Q/K/V projections (3 silicon dispatches).
        q_proj = self._matmul_qkvo(x, self._q_q, "qkvo_h256")  # (M, 256)
        _capture("q_proj", q_proj)
        k_proj = self._matmul_qkvo(x, self._q_k, "qkvo_h256")
        _capture("k_proj", k_proj)
        v_proj = self._matmul_qkvo(x, self._q_v, "qkvo_h256")
        _capture("v_proj", v_proj)

        # 2. Per-head Q/K/V split: (M, 256) → (M, 4, 64) → 4 × (M, 64).
        q_heads = q_proj.reshape(_M, _NUM_HEADS, _HEAD_DIM)
        k_heads = k_proj.reshape(_M, _NUM_HEADS, _HEAD_DIM)
        v_heads = v_proj.reshape(_M, _NUM_HEADS, _HEAD_DIM)

        # 3. Per-head Q·K^T (4 silicon dispatches), softmax (1 batched
        #    silicon dispatch over (4*M, PAD)), scores·V (4 dispatches).
        head_dim_inv_sqrt = np.float32(1.0 / np.sqrt(_HEAD_DIM))

        qkt_per_head: list[np.ndarray] = []
        for h in range(_NUM_HEADS):
            qh = np.ascontiguousarray(q_heads[:, h, :])  # (M, 64)
            kh = np.ascontiguousarray(k_heads[:, h, :])
            scores = self._matmul_qkt(qh, kh)            # (M, M)
            qkt_per_head.append(scores)
        if capture_mse and torch_reference is not None:
            ref_qkt = torch_reference.get("qkt_per_head")
            if ref_qkt is not None:
                mse["qkt_per_head"] = [
                    float(((qkt_per_head[h] - ref_qkt[h].astype(np.float32)) ** 2).mean())
                    for h in range(_NUM_HEADS)
                ]

        # Build the batched softmax input: shape (NUM_HEADS * M, PAD=64).
        #   row_h_m = scaled_scores_h_m, padded with sentinel beyond M.
        scores_batched = np.full(
            (_NUM_HEADS * _M, _SOFTMAX_PAD), _NEG_SENTINEL_BF16, dtype=np.float32
        )
        for h in range(_NUM_HEADS):
            scores_batched[h * _M : (h + 1) * _M, :_M] = (
                qkt_per_head[h] * head_dim_inv_sqrt
            )

        attn_batched = bert_mini_attention_softmax(
            scores_batched, artifacts_dir=self._artifacts["softmax"]
        )
        # Slice back to per-head (M, M) probabilities (drop pad cols).
        attn_per_head = [
            attn_batched[h * _M : (h + 1) * _M, :_M] for h in range(_NUM_HEADS)
        ]
        if capture_mse and torch_reference is not None:
            ref_sm = torch_reference.get("softmax_per_head")
            if ref_sm is not None:
                mse["softmax_per_head"] = [
                    float(((attn_per_head[h] - ref_sm[h].astype(np.float32)) ** 2).mean())
                    for h in range(_NUM_HEADS)
                ]

        sv_per_head: list[np.ndarray] = []
        for h in range(_NUM_HEADS):
            vh = np.ascontiguousarray(v_heads[:, h, :])  # (M, 64)
            sv_h = self._matmul_sv(attn_per_head[h], vh)  # (M, 64)
            sv_per_head.append(sv_h)
        if capture_mse and torch_reference is not None:
            ref_sv = torch_reference.get("sv_per_head")
            if ref_sv is not None:
                mse["sv_per_head"] = [
                    float(((sv_per_head[h] - ref_sv[h].astype(np.float32)) ** 2).mean())
                    for h in range(_NUM_HEADS)
                ]

        # 4. Concat heads → (M, 256).
        attn_concat = np.concatenate(sv_per_head, axis=1)  # (M, 256)

        # 5. Output projection (1 dispatch).
        attn_out = self._matmul_qkvo(attn_concat, self._q_o, "qkvo_h256")
        _capture("concat_proj", attn_out)

        # 6. Residual + LN1 (1 dispatch).
        residual1 = (x + attn_out).astype(np.float32)
        ln1 = bert_mini_layer_norm(
            residual1,
            self.weights.ln1_g.astype(np.float32),
            self.weights.ln1_b.astype(np.float32),
            artifacts_dir=self._artifacts["layer_norm"],
        )
        _capture("ln1", ln1)

        # 7. FFN1 (1 dispatch — host-side iterates 4 output groups).
        ffn1_out = self._matmul_ffn1(ln1)
        _capture("ffn1", ffn1_out)

        # 8. GELU (1 dispatch).
        gelu_out = bert_mini_gelu(
            ffn1_out.astype(np.float32),
            artifacts_dir=self._artifacts["gelu"],
        )
        _capture("gelu", gelu_out)

        # 9. FFN2 (1 dispatch — single output group at h=256).
        ffn2_out = self._matmul_ffn2(gelu_out)
        _capture("ffn2", ffn2_out)

        # 10. Residual + LN2 (1 dispatch).
        residual2 = (ln1 + ffn2_out).astype(np.float32)
        ln2 = bert_mini_layer_norm(
            residual2,
            self.weights.ln2_g.astype(np.float32),
            self.weights.ln2_b.astype(np.float32),
            artifacts_dir=self._artifacts["layer_norm"],
        )
        _capture("ln2", ln2)

        if capture_mse:
            self.last_mse = mse
        return ln2


__all__ = [
    "BertMiniBlock",
    "BertMiniBlockWeights",
]
