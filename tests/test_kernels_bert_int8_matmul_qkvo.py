"""bionpu.kernels.scoring.bert_int8_matmul (qkvo variant) — host-emulation tests.

GPL-3.0. (c) 2026 OpenSensor.

Locks in byte-equivalence of the qkvo emulation reference (M=47, K=768,
N=768) against a hand-computed numpy einsum. The silicon path will land
later in v0.4-rc; until then, the emulation path IS the byte-equivalent
reference for the silicon kernel.

The qkvo silicon kernel performs the same arithmetic as the head
specialization but at the BERT body's Q/K/V/O projection shape (N=768).
The IRON topology splits N across 4 compute tiles and K across 12
chunks; that's an implementation detail invisible to the host API as
long as the i32 accumulator + fused-scale semantics are preserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from bionpu.dispatch.npu import lookup_npu_op
from bionpu.kernels.scoring.bert_int8_matmul import (
    bert_int8_matmul_qkvo,
)


_M, _K, _N = 47, 768, 768


def _zero_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((_M, _K), dtype=np.int8)
    w = np.zeros((_N, _K), dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    return x, w, s


def test_qkvo_emulation_zero_inputs_yield_zero_output() -> None:
    """All-zero matmul gives all-zero output regardless of scales."""
    x, w, s = _zero_inputs()
    y = bert_int8_matmul_qkvo(x, w, s)
    assert y.shape == (_M, _N)
    assert y.dtype == np.int8
    assert (y == 0).all()


def test_qkvo_emulation_basic_correctness_against_einsum() -> None:
    """The emulation reference must match the documented arithmetic."""
    rng = np.random.default_rng(42)
    x = rng.integers(-128, 127, size=(_M, _K), dtype=np.int8)
    w = rng.integers(-128, 127, size=(_N, _K), dtype=np.int8)
    # Per-output combined scale chosen so the output stays in int8 range.
    # acc per cell is bounded above by 127*127*K = 12.4M, so 1e-5 keeps
    # |fy| < ~125, well within int8.
    s_vals = np.full((_N + 1,), 1e-5, dtype=np.float32)
    s_vals[_N] = 0.0  # bias slot unused
    s = s_vals

    y = bert_int8_matmul_qkvo(x, w, s)

    expected_acc = np.einsum("mk,nk->mn", x.astype(np.int32), w.astype(np.int32))
    expected = np.clip(
        np.round(expected_acc.astype(np.float32) * s[:_N]),
        -128, 127,
    ).astype(np.int8)
    assert (y == expected).all()


def test_qkvo_emulation_int8_saturation_floor() -> None:
    """Large-positive accumulators saturate to +127 across all 768 outputs."""
    x = np.full((_M, _K), 127, dtype=np.int8)
    w = np.full((_N, _K), 127, dtype=np.int8)
    # acc per cell = 127*127*768 = 12,386,304; * 1.0 → way above 127.
    s = np.ones((_N + 1,), dtype=np.float32)
    y = bert_int8_matmul_qkvo(x, w, s)
    assert (y == 127).all()


def test_qkvo_emulation_int8_saturation_ceiling() -> None:
    """Large-negative accumulators saturate to -128 across all 768 outputs."""
    x = np.full((_M, _K), 127, dtype=np.int8)
    w = np.full((_N, _K), -128, dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    y = bert_int8_matmul_qkvo(x, w, s)
    assert (y == -128).all()


def test_qkvo_emulation_per_channel_scale_isolation() -> None:
    """Each output channel must use ONLY its own combined scale, not its neighbour's.

    Constructs a w matrix where every output channel is identical so the i32
    accumulator is the same for every n; then chooses scales that make
    different output channels saturate to different INT8 values. If the
    kernel cross-contaminated scales, this test would fail.
    """
    x = np.full((_M, _K), 1, dtype=np.int8)
    w = np.full((_N, _K), 1, dtype=np.int8)  # acc = K = 768 for every (m, n)
    # First half of outputs scaled to ~30, second half to ~60.
    s = np.empty((_N + 1,), dtype=np.float32)
    s[: _N // 2] = 30.0 / 768.0   # round(30) = 30
    s[_N // 2 : _N] = 60.0 / 768.0  # round(60) = 60
    s[_N] = 0.0

    y = bert_int8_matmul_qkvo(x, w, s)

    assert y.shape == (_M, _N)
    assert (y[:, : _N // 2] == 30).all()
    assert (y[:, _N // 2 : _N] == 60).all()


def test_qkvo_emulation_rejects_dtype_mismatch() -> None:
    x = np.zeros((_M, _K), dtype=np.int16)  # wrong dtype
    w = np.zeros((_N, _K), dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    with pytest.raises(TypeError, match="must be int8"):
        bert_int8_matmul_qkvo(x, w, s)


def test_qkvo_emulation_rejects_shape_mismatch() -> None:
    x = np.zeros((_M, _K + 1), dtype=np.int8)  # K + 1 ≠ 768
    w = np.zeros((_N, _K), dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    with pytest.raises(ValueError, match="x shape mismatch"):
        bert_int8_matmul_qkvo(x, w, s)


def test_qkvo_emulation_rejects_wrong_n() -> None:
    """qkvo expects N=768 specifically; head's N=2 must be rejected."""
    x = np.zeros((_M, _K), dtype=np.int8)
    w = np.zeros((2, _K), dtype=np.int8)  # head shape (N=2) — should be rejected by qkvo
    s = np.ones((3,), dtype=np.float32)
    with pytest.raises(ValueError, match="w shape mismatch"):
        bert_int8_matmul_qkvo(x, w, s)


def test_qkvo_op_registered_in_npu_dispatcher() -> None:
    """The qkvo op should be discoverable via the bionpu.dispatch lookup."""
    op = lookup_npu_op("bert_int8_matmul_qkvo")
    assert op.name == "bert_int8_matmul_qkvo"
    # Calling via dispatcher matches direct call.
    x, w, s = _zero_inputs()
    y_via_dispatch = op(x=x, w=w, scales_combined=s)
    y_direct = bert_int8_matmul_qkvo(x, w, s)
    assert (y_via_dispatch == y_direct).all()


def test_qkvo_silicon_dispatch_unwired_raises_clearly(tmp_path) -> None:
    """When pretend-artifacts directory exists with empty xclbin, dispatch
    raises NotImplementedError with a clear message rather than silently
    falling back to emulation."""
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    (art_dir / "final.xclbin").write_bytes(b"")
    (art_dir / "insts.bin").write_bytes(b"")

    x, w, s = _zero_inputs()
    with pytest.raises(NotImplementedError, match="silicon dispatch not yet wired"):
        bert_int8_matmul_qkvo(x, w, s, artifacts_dir=art_dir)


def test_qkvo_emulation_short_M_works() -> None:
    """Emulation must accept M < 47 (e.g., DNABERT-Epi pre-computes only
    the CLS token = 1 row in some flows)."""
    rng = np.random.default_rng(0)
    short_m = 5
    x = rng.integers(-128, 127, size=(short_m, _K), dtype=np.int8)
    w = rng.integers(-128, 127, size=(_N, _K), dtype=np.int8)
    s = np.full((_N + 1,), 1e-6, dtype=np.float32)
    s[_N] = 0.0

    y = bert_int8_matmul_qkvo(x, w, s)
    assert y.shape == (short_m, _N)
    assert y.dtype == np.int8

    expected = np.clip(
        np.round(
            np.einsum("mk,nk->mn", x.astype(np.int32), w.astype(np.int32)).astype(np.float32)
            * s[:_N]
        ),
        -128, 127,
    ).astype(np.int8)
    assert (y == expected).all()
