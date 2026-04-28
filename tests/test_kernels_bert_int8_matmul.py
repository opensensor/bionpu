"""bionpu.kernels.scoring.bert_int8_matmul — host-emulation smoke tests.

GPL-3.0. (c) 2026 OpenSensor.

Exercises the host-emulation reference path that the silicon
dispatch is byte-equivalent to.
"""

from __future__ import annotations

import numpy as np
import pytest

from bionpu.dispatch.npu import lookup_npu_op
from bionpu.kernels.scoring.bert_int8_matmul import (
    bert_int8_matmul_head,
    bert_int8_matmul_qkvo,
)


_M, _K, _N = 47, 768, 2


def _zero_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((_M, _K), dtype=np.int8)
    w = np.zeros((_N, _K), dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    return x, w, s


def _missing_artifacts(tmp_path):
    return tmp_path / "missing-artifacts"


def test_emulation_zero_inputs_yield_zero_output(tmp_path) -> None:
    """All-zero matmul gives all-zero output regardless of scales."""
    x, w, s = _zero_inputs()
    y = bert_int8_matmul_head(x, w, s, artifacts_dir=_missing_artifacts(tmp_path))
    assert y.shape == (_M, _N)
    assert y.dtype == np.int8
    assert (y == 0).all()


def test_emulation_basic_correctness_against_einsum(tmp_path) -> None:
    """The emulation reference must match the documented arithmetic."""
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 127, size=(_M, _K), dtype=np.int8)
    w = rng.integers(-128, 127, size=(_N, _K), dtype=np.int8)
    # Per-output combined scale chosen so the output stays in int8 range.
    s = np.array([1e-4, 1e-4, 0.0], dtype=np.float32)

    y = bert_int8_matmul_head(x, w, s, artifacts_dir=_missing_artifacts(tmp_path))

    expected_acc = np.einsum("mk,nk->mn", x.astype(np.int32), w.astype(np.int32))
    expected = np.clip(
        np.round(expected_acc.astype(np.float32) * s[:_N]),
        -128, 127,
    ).astype(np.int8)
    assert (y == expected).all()


def test_emulation_int8_saturation_floor(tmp_path) -> None:
    """Large-positive accumulators saturate to +127."""
    x = np.full((_M, _K), 127, dtype=np.int8)
    w = np.full((_N, _K), 127, dtype=np.int8)
    # acc per cell = 127*127*768 = 12,386,304; * 1.0 → way above 127.
    s = np.ones((_N + 1,), dtype=np.float32)
    y = bert_int8_matmul_head(x, w, s, artifacts_dir=_missing_artifacts(tmp_path))
    assert (y == 127).all()


def test_emulation_int8_saturation_ceiling(tmp_path) -> None:
    """Large-negative accumulators saturate to -128."""
    x = np.full((_M, _K), 127, dtype=np.int8)
    w = np.full((_N, _K), -128, dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    y = bert_int8_matmul_head(x, w, s, artifacts_dir=_missing_artifacts(tmp_path))
    assert (y == -128).all()


def test_emulation_rejects_dtype_mismatch(tmp_path) -> None:
    x = np.zeros((_M, _K), dtype=np.int16)  # wrong dtype
    w = np.zeros((_N, _K), dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    with pytest.raises(TypeError, match="must be int8"):
        bert_int8_matmul_head(x, w, s, artifacts_dir=_missing_artifacts(tmp_path))


def test_emulation_rejects_shape_mismatch(tmp_path) -> None:
    x = np.zeros((_M, _K + 1), dtype=np.int8)  # K + 1 ≠ 768
    w = np.zeros((_N, _K), dtype=np.int8)
    s = np.ones((_N + 1,), dtype=np.float32)
    with pytest.raises(ValueError, match="x shape mismatch"):
        bert_int8_matmul_head(x, w, s, artifacts_dir=_missing_artifacts(tmp_path))


def test_op_registered_in_npu_dispatcher(tmp_path) -> None:
    """The op should be discoverable via the bionpu.dispatch lookup."""
    op = lookup_npu_op("bert_int8_matmul_head")
    assert op.name == "bert_int8_matmul_head"
    # Calling via dispatcher matches direct call.
    x, w, s = _zero_inputs()
    artifacts_dir = _missing_artifacts(tmp_path)
    y_via_dispatch = op(x=x, w=w, scales_combined=s, artifacts_dir=artifacts_dir)
    y_direct = bert_int8_matmul_head(x, w, s, artifacts_dir=artifacts_dir)
    assert (y_via_dispatch == y_direct).all()


def test_head_silicon_dispatch_uses_xclbin_backend(monkeypatch, tmp_path) -> None:
    """Artifacts present now route through the pyxrt backend adapter."""
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    (art_dir / "final.xclbin").write_bytes(b"x")
    (art_dir / "insts.bin").write_bytes(b"i")

    x, w, s = _zero_inputs()
    calls = []

    class _Backend:
        def run_xclbin(self, **kwargs):
            calls.append(kwargs)
            return bytes(96), 1.0, 1.0, 1.0

    monkeypatch.setattr(
        "bionpu.kernels.scoring.bert_int8_matmul.default_backend",
        lambda: _Backend(),
    )

    y = bert_int8_matmul_head(x, w, s, artifacts_dir=art_dir)
    assert y.shape == (_M, _N)
    assert calls[0]["xclbin"] == art_dir / "final.xclbin"


def test_qkvo_silicon_dispatch_reads_row_major_output(monkeypatch, tmp_path) -> None:
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    (art_dir / "final.xclbin").write_bytes(b"x")
    (art_dir / "insts.bin").write_bytes(b"i")

    x = np.zeros((_M, 768), dtype=np.int8)
    w = np.zeros((768, 768), dtype=np.int8)
    s = np.ones((769,), dtype=np.float32)
    raw_arr = np.zeros((48, 768), dtype=np.int8)
    raw_arr[:_M, :192] = 1
    raw_arr[:_M, 192:384] = 2
    raw_arr[:_M, 384:576] = 3
    raw_arr[:_M, 576:] = 4
    raw = raw_arr.tobytes()

    class _Backend:
        def run_xclbin(self, **kwargs):
            return raw, 1.0, 1.0, 1.0

    monkeypatch.setattr(
        "bionpu.kernels.scoring.bert_int8_matmul.default_backend",
        lambda: _Backend(),
    )

    y = bert_int8_matmul_qkvo(x, w, s, artifacts_dir=art_dir)
    assert y.shape == (_M, 768)
    assert (y[:, :192] == 1).all()
    assert (y[:, 192:384] == 2).all()
    assert (y[:, 384:576] == 3).all()
    assert (y[:, 576:] == 4).all()
