"""bionpu.kernels.scoring.bert_int8_matmul — host-emulation smoke tests.

GPL-3.0. (c) 2026 OpenSensor.

Exercises the host-emulation reference path that the silicon
dispatch will be byte-equivalent to. v0.4-alpha ships only the
emulation; silicon dispatch lands in v0.4-beta.
"""

from __future__ import annotations

import numpy as np
import pytest

from bionpu.dispatch.npu import lookup_npu_op
from bionpu.kernels.scoring.bert_int8_matmul import (
    bert_int8_matmul_head,
)


_M, _K, _N = 47, 768, 2


def _zero_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((_M, _K), dtype=np.int8)
    w = np.zeros((_N, _K), dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    return x, w, s


def test_emulation_zero_inputs_yield_zero_output() -> None:
    """All-zero matmul gives all-zero output regardless of scales."""
    x, w, s = _zero_inputs()
    y = bert_int8_matmul_head(x, w, s)
    assert y.shape == (_M, _N)
    assert y.dtype == np.int8
    assert (y == 0).all()


def test_emulation_basic_correctness_against_einsum() -> None:
    """The emulation reference must match the documented arithmetic."""
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 127, size=(_M, _K), dtype=np.int8)
    w = rng.integers(-128, 127, size=(_N, _K), dtype=np.int8)
    # Per-output combined scale chosen so the output stays in int8 range.
    s = np.array([1e-4, 1e-4, 0.0], dtype=np.float32)

    y = bert_int8_matmul_head(x, w, s)

    expected_acc = np.einsum("mk,nk->mn", x.astype(np.int32), w.astype(np.int32))
    expected = np.clip(
        np.round(expected_acc.astype(np.float32) * s[:_N]),
        -128, 127,
    ).astype(np.int8)
    assert (y == expected).all()


def test_emulation_int8_saturation_floor() -> None:
    """Large-positive accumulators saturate to +127."""
    x = np.full((_M, _K), 127, dtype=np.int8)
    w = np.full((_N, _K), 127, dtype=np.int8)
    # acc per cell = 127*127*768 = 12,386,304; * 1.0 → way above 127.
    s = np.ones((_N + 1,), dtype=np.float32)
    y = bert_int8_matmul_head(x, w, s)
    assert (y == 127).all()


def test_emulation_int8_saturation_ceiling() -> None:
    """Large-negative accumulators saturate to -128."""
    x = np.full((_M, _K), 127, dtype=np.int8)
    w = np.full((_N, _K), -128, dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    y = bert_int8_matmul_head(x, w, s)
    assert (y == -128).all()


def test_emulation_rejects_dtype_mismatch() -> None:
    x = np.zeros((_M, _K), dtype=np.int16)  # wrong dtype
    w = np.zeros((_N, _K), dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    with pytest.raises(TypeError, match="must be int8"):
        bert_int8_matmul_head(x, w, s)


def test_emulation_rejects_shape_mismatch() -> None:
    x = np.zeros((_M, _K + 1), dtype=np.int8)  # K + 1 ≠ 768
    w = np.zeros((_N, _K), dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    with pytest.raises(ValueError, match="x shape mismatch"):
        bert_int8_matmul_head(x, w, s)


def test_op_registered_in_npu_dispatcher() -> None:
    """The op should be discoverable via the bionpu.dispatch lookup."""
    op = lookup_npu_op("bert_int8_matmul_head")
    assert op.name == "bert_int8_matmul_head"
    # Calling via dispatcher matches direct call.
    x, w, s = _zero_inputs()
    y_via_dispatch = op(x=x, w=w, scales_combined=s)
    y_direct = bert_int8_matmul_head(x, w, s)
    assert (y_via_dispatch == y_direct).all()


def test_silicon_dispatch_unwired_raises_clearly(tmp_path) -> None:
    """When pretend-artifacts directory exists with empty xclbin, dispatch
    raises NotImplementedError with a clear message rather than silently
    falling back to emulation."""
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    # Plant fake xclbin + insts so the function thinks artifacts are present.
    (art_dir / "final.xclbin").write_bytes(b"")
    (art_dir / "insts.bin").write_bytes(b"")

    x, w, s = _zero_inputs()
    with pytest.raises(NotImplementedError, match="silicon dispatch not yet wired"):
        bert_int8_matmul_head(x, w, s, artifacts_dir=art_dir)
