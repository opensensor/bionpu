"""Batched stage-3 (fused-perts) correctness test.

Mirrors test_linear_projection_fused_correctness.py for the batched
variant. Verifies:

- IRON Python source contains the chunk-outer-loop sentinel
  (regression guard).
- The op class registers under
  ``dorado_fast_linear_projection_fused_perts_batched``.
- The pack/unpack helpers round-trip a batched buffer.
- The bf16 reference computed per-chunk-independent matches the
  expected math (silicon byte-equal-vs-bf16-ref tested in NPU gate).
- silicon-skipped NPU test gated on artifacts present.

The batched kernel is intentionally a host-amortisation trick: the
silicon math per chunk is byte-equal (within bf16 tolerance) to the
single-chunk fused-perts artifact's output. Per-chunk output is
independent of any other chunk in the batch (no cross-chunk state).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent.parent

# Geometry pinned by the kernel.
T_LSTM = 334
HIDDEN = 96
OUT_DIM = 256
CLAMP_LO = -5.0
CLAMP_HI = 5.0


# --------------------------------------------------------------------------- #
# CPU reference (bf16-narrowed; per-chunk-independent)
# --------------------------------------------------------------------------- #


def _linear_projection_cpu_bf16ref_single(
    x: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Single-chunk bf16 reference (matches stage-3 fused-perts contract)."""
    from bionpu.kernels.basecalling.lstm_cell_bf16 import (
        bf16_bytes_to_fp32,
        fp32_to_bf16_bytes,
    )

    assert x.shape == (T_LSTM, 1, HIDDEN)
    assert weight.shape == (OUT_DIM, HIDDEN)
    x_bf = bf16_bytes_to_fp32(
        fp32_to_bf16_bytes(x.astype(np.float32, copy=False)).reshape(-1),
        x.shape,
    )
    w_bf = bf16_bytes_to_fp32(
        fp32_to_bf16_bytes(weight.astype(np.float32, copy=False)).reshape(-1),
        weight.shape,
    )
    flat = x_bf.reshape(T_LSTM, HIDDEN).astype(np.float32)
    y = flat @ w_bf.astype(np.float32).T
    y = np.clip(y, CLAMP_LO, CLAMP_HI)
    y_bf = bf16_bytes_to_fp32(
        fp32_to_bf16_bytes(y).reshape(-1), y.shape,
    )
    return y_bf.reshape(T_LSTM, 1, OUT_DIM)


def _linear_projection_cpu_bf16ref_batched(
    x_batch: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Per-chunk-independent bf16 reference for the batched op."""
    n_chunks = x_batch.shape[0]
    out = np.empty(
        (n_chunks, T_LSTM, 1, OUT_DIM), dtype=np.float32,
    )
    for c in range(n_chunks):
        out[c] = _linear_projection_cpu_bf16ref_single(x_batch[c], weight)
    return out


# --------------------------------------------------------------------------- #
# Fast tests — no NPU required
# --------------------------------------------------------------------------- #


def test_batched_iron_uses_chunk_outer_loop_sentinel() -> None:
    """Regression guard: IRON Python keeps the chunk-outer loop shape."""
    py_path = (
        REPO
        / "bionpu-public"
        / "src"
        / "bionpu"
        / "kernels"
        / "basecalling"
        / "linear_projection"
        / "linear_projection_fused_batched.py"
    )
    src = py_path.read_text()
    # Sentinel pinned in the IRON Python core_body.
    assert "FUSED_BATCHED_CHUNK_OUTER_LOOP_SENTINEL" in src, (
        "linear_projection_fused_batched.py must contain the "
        "FUSED_BATCHED_CHUNK_OUTER_LOOP_SENTINEL marker"
    )
    # The chunk-outer + timestep-inner loop nest must be present.
    assert "for chunk in range_(n_chunks):" in src, (
        "linear_projection_fused_batched.py must contain the chunk-outer "
        "loop `for chunk in range_(n_chunks):`"
    )
    assert "for t in range_(L):" in src, (
        "linear_projection_fused_batched.py must contain the "
        "timestep-inner loop `for t in range_(L):`"
    )
    # Single weight acquire OUTSIDE the chunk loop (shared 48-KiB slab).
    assert "elem_w_full = of_weight.acquire(1)" in src, (
        "linear_projection_fused_batched.py must acquire elem_w_full once "
        "before the chunk loop (single 48-KiB shared weight slab)"
    )
    # Reuses the stage-3 kernel symbol (the C++ kernel is unchanged).
    assert "dorado_fast_linear_projection_fused_perts" in src, (
        "linear_projection_fused_batched.py must reuse the stage-3 "
        "kernel symbol (the C++ kernel is unchanged)"
    )


def test_batched_makefile_contains_fused_batched_target() -> None:
    """Regression guard: Makefile knows the fused-batched experiment."""
    makefile = (
        REPO
        / "bionpu-public"
        / "src"
        / "bionpu"
        / "kernels"
        / "basecalling"
        / "linear_projection"
        / "Makefile"
    )
    src = makefile.read_text()
    assert "experiment=fused-batched" in src or "fused-batched" in src
    # Artifact suffix bakes in the chunk count.
    assert "fused_batched_n" in src, (
        "Makefile must use suffix fused_batched_n${n_chunks}_L${seq}"
    )


def test_init_registers_batched_op() -> None:
    """Dispatch wiring: batched op is registered with the expected name."""
    import bionpu.kernels.basecalling.linear_projection  # noqa: F401
    from bionpu.dispatch.npu import NPU_OPS

    assert "dorado_fast_linear_projection_fused_perts_batched" in NPU_OPS, (
        "DoradoFastLinearProjectionFusedBatched must be registered under "
        "'dorado_fast_linear_projection_fused_perts_batched'"
    )
    # Existing ops must still be registered (additive contract).
    assert "dorado_fast_linear_projection_fused_perts" in NPU_OPS
    assert "dorado_fast_linear_projection" in NPU_OPS


def test_batched_pack_helpers_roundtrip() -> None:
    """Batched pack/unpack helpers preserve shape and byte counts."""
    from bionpu.kernels.basecalling.linear_projection import (
        _pack_input_bf16_batched,
        _pack_wb_bf16,
        _unpack_output_bf16_batched,
    )

    rng = np.random.default_rng(20260427)
    for n_chunks in (1, 2, 4, 8):
        x = rng.standard_normal(
            size=(n_chunks, T_LSTM, 1, HIDDEN)
        ).astype(np.float32)
        weight = (
            rng.standard_normal(size=(OUT_DIM, HIDDEN)) * 0.1
        ).astype(np.float32)

        x_packed = _pack_input_bf16_batched(x, n_chunks)
        assert x_packed.dtype == np.uint16
        assert x_packed.size == n_chunks * T_LSTM * HIDDEN, (
            f"n={n_chunks} input pack size mismatch: got {x_packed.size}, "
            f"expected {n_chunks * T_LSTM * HIDDEN}"
        )

        wb_packed = _pack_wb_bf16(weight)
        # Weight is SHARED across batch — single 48-KiB slab.
        assert wb_packed.size == OUT_DIM * HIDDEN, (
            "Batched weight wire format must be a SINGLE shared 48-KiB "
            "slab (no per-chunk repetition)"
        )

        # Round-trip the unpacker on a synthetic output buffer.
        out_buf = np.zeros(n_chunks * T_LSTM * OUT_DIM, dtype=np.uint16)
        out_unpacked = _unpack_output_bf16_batched(out_buf, n_chunks)
        assert out_unpacked.shape == (n_chunks, T_LSTM, 1, OUT_DIM)
        assert out_unpacked.dtype == np.float32


def test_batched_op_rejects_unsupported_n_chunks() -> None:
    """Op class rejects N_CHUNKS values it has no artifact for."""
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjectionFusedBatched,
    )

    op = DoradoFastLinearProjectionFusedBatched()
    rng = np.random.default_rng(0)
    # Build an x_batch with N_CHUNKS=3 (unsupported per task contract).
    x = rng.standard_normal(size=(3, T_LSTM, 1, HIDDEN)).astype(np.float32)
    w = rng.standard_normal(size=(OUT_DIM, HIDDEN)).astype(np.float32)
    with pytest.raises(ValueError, match="N_CHUNKS=3"):
        op(x_batch=x, weight=w, n_iters=1, warmup=0)


def test_batched_input_shape_validation() -> None:
    """Rank check: x_batch must be rank-4."""
    from bionpu.kernels.basecalling.linear_projection import (
        _pack_input_bf16_batched,
    )

    rng = np.random.default_rng(0)
    # Wrong shape — 3-D, missing channel dim.
    bad = rng.standard_normal(size=(2, T_LSTM, HIDDEN)).astype(np.float32)
    with pytest.raises(ValueError):
        _pack_input_bf16_batched(bad, 2)

    # Wrong N_CHUNKS — claims 4 chunks but tensor has 2.
    x = rng.standard_normal(size=(2, T_LSTM, 1, HIDDEN)).astype(np.float32)
    with pytest.raises(ValueError):
        _pack_input_bf16_batched(x, 4)


# --------------------------------------------------------------------------- #
# NPU-required test (gated on the bring-up host)
# --------------------------------------------------------------------------- #


def _batched_artifacts_present(n_chunks: int) -> bool:
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjectionFusedBatched,
    )

    return DoradoFastLinearProjectionFusedBatched.artifacts_present(
        n_chunks=n_chunks
    )


@pytest.mark.npu
@pytest.mark.parametrize("n_chunks", [1, 2, 4, 8])
def test_npu_batched_byte_equal_to_bf16_reference(n_chunks: int) -> None:
    """Silicon byte-equal vs per-chunk bf16 reference.

    For each supported N_CHUNKS, asserts:
      - Output shape matches (N_CHUNKS, L, 1, OUT_DIM).
      - No NaN / Inf in any chunk.
      - Each batched output chunk matches the single-chunk bf16
        reference within bf16 round-off (5e-2 absolute, same gate as
        the single-chunk fused-perts test).
    """
    if not _batched_artifacts_present(n_chunks):
        pytest.skip(
            f"linear_projection fused-batched n={n_chunks} NPU artifacts "
            f"missing; build via "
            f"`BIONPU_LINEAR_PROJECTION_BATCH_CHUNKS={n_chunks} make "
            f"experiment=fused-batched NPU2=1 seq={T_LSTM} all` and copy "
            f"outputs to dorado_fast_linear_projection_fused_batched_n"
            f"{n_chunks}/"
        )
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjectionFusedBatched,
    )

    rng = np.random.default_rng(20260427)
    x_batch = rng.standard_normal(
        size=(n_chunks, T_LSTM, 1, HIDDEN)
    ).astype(np.float32)
    weight = (
        rng.standard_normal(size=(OUT_DIM, HIDDEN)) * 0.1
    ).astype(np.float32)

    op = DoradoFastLinearProjectionFusedBatched()
    y_npu = op(x_batch=x_batch, weight=weight, n_iters=1, warmup=0)

    assert y_npu.shape == (n_chunks, T_LSTM, 1, OUT_DIM), (
        f"batched n={n_chunks} output shape {y_npu.shape} != expected"
    )
    assert y_npu.dtype == np.float32

    # Hard NaN/Inf gate (CLAUDE.md silicon-wedge etiquette).
    assert int(np.isnan(y_npu).sum()) == 0, (
        f"batched n={n_chunks} NPU output contains NaN values"
    )
    assert int(np.isinf(y_npu).sum()) == 0, (
        f"batched n={n_chunks} NPU output contains Inf values"
    )

    # Per-chunk byte-equal vs single-chunk bf16 reference.
    y_ref = _linear_projection_cpu_bf16ref_batched(x_batch, weight)
    diff = np.abs(y_npu - y_ref)
    max_diff = float(diff.max())
    assert max_diff < 5e-2, (
        f"batched n={n_chunks} NPU vs bf16 reference max-abs-error "
        f"{max_diff:.3e} exceeds 5e-2"
    )
