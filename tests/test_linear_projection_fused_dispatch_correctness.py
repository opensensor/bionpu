"""DESIGN-fusion stage 4 — linear_projection_fused_dispatch correctness.

Mirrors the structure of test_linear_projection_fused_correctness.py
but pins the stage-4 (fused-dispatch) artifact:

- Fast / non-NPU: source sentinel checks (regression-protect against
  accidental loss of the distinct stage-4 entry symbol or the
  inner-loop micro-opt).
- Fast / non-NPU: op registration + numpy reference parity.
- NPU-required: byte-equal-within-bf16-tolerance NPU output vs the
  CPU bf16-reference — runs only when artifacts are present.

Stage 4 collapses to stage 3's MLIR shape due to the IRON Kernel ABI
limitation (no objectfifo-handle args). See
linear_projection_fused_dispatch.py top-of-file rationale and
gaps.yaml#B1-fused-dispatch-kernel-fifo-walk for the full analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent.parent

# Geometry pinned by the kernel — must match
# bionpu/kernels/basecalling/linear_projection/__init__.py.
T_LSTM = 334
HIDDEN = 96
OUT_DIM = 256
OC_GROUP_SIZE = 64
N_OC_GROUPS = OUT_DIM // OC_GROUP_SIZE  # 4
CLAMP_LO = -5.0
CLAMP_HI = 5.0


# --------------------------------------------------------------------------- #
# CPU reference (bf16-narrowed)
# --------------------------------------------------------------------------- #


def _linear_projection_cpu_bf16ref(
    x: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """bf16 reference: narrow x and weight to bf16 first, then GEMM in
    FP32 accum, then clamp + bf16-narrow output. Mirrors the wire
    contract of the fused-dispatch kernel: bf16 multiplier inputs,
    accfloat accumulator, bf16 output narrowing.
    """
    from bionpu.kernels.basecalling.lstm_cell_bf16 import (
        bf16_bytes_to_fp32,
        fp32_to_bf16_bytes,
    )

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


# --------------------------------------------------------------------------- #
# Fast tests — no NPU required
# --------------------------------------------------------------------------- #


_KERNEL_DIR = (
    REPO
    / "bionpu-public"
    / "src"
    / "bionpu"
    / "kernels"
    / "basecalling"
    / "linear_projection"
)


def test_fused_dispatch_kernel_source_uses_sentinels() -> None:
    """Stage-4 regression guard: kernel keeps the distinct entry
    symbol and the stage-4 micro-opt sentinels.
    """
    cc_path = _KERNEL_DIR / "linear_projection_fused_dispatch.cc"
    src = cc_path.read_text()

    # Sentinel 1: the stage-4 single-call-per-dispatch marker.
    assert "FUSED_DISPATCH_SINGLE_CALL_PER_DISPATCH_SENTINEL" in src, (
        "linear_projection_fused_dispatch.cc must contain the "
        "FUSED_DISPATCH_SINGLE_CALL_PER_DISPATCH_SENTINEL comment "
        "(stage-4 marker)"
    )
    # Sentinel 2: the distinct entry symbol.
    assert "dorado_fast_linear_projection_fused_dispatch" in src, (
        "linear_projection_fused_dispatch.cc must export "
        "dorado_fast_linear_projection_fused_dispatch (stage-4 entry "
        "symbol missing)"
    )
    # Sentinel 3: bf16 weight pointer signature (precision contract).
    assert "bfloat16 *restrict W_full" in src, (
        "linear_projection_fused_dispatch.cc must take W_full as "
        "bfloat16* (precision contract: bf16 weights per "
        "DESIGN-fusion.md)"
    )
    # Sentinel 4: stage-4 micro-opt — aie::reduce_add for the per-OC
    # scalar reduction tail (vs stage 3's manual scalar sum).
    assert "aie::reduce_add(out_vec)" in src, (
        "linear_projection_fused_dispatch.cc must use aie::reduce_add "
        "for the per-OC reduction tail (stage-4 micro-opt; stage 3 "
        "uses a manual scalar sum)"
    )
    # Sentinel 5: precision contract — accfloat accumulator + bf16 mac.
    assert "aie::accum<accfloat, VEC>" in src, (
        "linear_projection_fused_dispatch.cc must use an accfloat "
        "accumulator (precision contract: FP32 accum, bf16 multiplier "
        "inputs)"
    )
    assert "aie::mac(acc, w_v, x_v)" in src, (
        "linear_projection_fused_dispatch.cc must use aie::mac with "
        "bf16 weight + bf16 input vectors (DESIGN-fusion.md "
        "precision contract)"
    )


def test_fused_dispatch_iron_uses_distinct_kernel_symbol() -> None:
    """Stage-4 regression guard: IRON Python references the stage-4
    kernel symbol + object file (NOT stage 3's).
    """
    py_path = _KERNEL_DIR / "linear_projection_fused_dispatch.py"
    src = py_path.read_text()

    assert '"dorado_fast_linear_projection_fused_dispatch"' in src, (
        "linear_projection_fused_dispatch.py must reference the "
        "stage-4 kernel symbol (dorado_fast_linear_projection_fused_dispatch)"
    )
    assert '"linear_projection_fused_dispatch.o"' in src, (
        "linear_projection_fused_dispatch.py must reference the "
        "stage-4 object file (linear_projection_fused_dispatch.o)"
    )
    # Weight depth=1 (acquire-once contract).
    assert 'name="weight_in", depth=1' in src, (
        "linear_projection_fused_dispatch.py must declare weight_in "
        "with depth=1 (stage-3 + stage-4 acquire-once contract)"
    )
    # The weight acquire must be OUTSIDE the timestep loop.
    assert "elem_w_full = of_weight.acquire(1)" in src, (
        "linear_projection_fused_dispatch.py must acquire elem_w_full "
        "once before the per-timestep loop"
    )


def test_init_registers_fused_dispatch_op() -> None:
    """Stage-4 dispatch wiring: fused-dispatch op is registered."""
    # Importing the package runs its __init__.py and registers the
    # per-group op, the stage-3 fused-perts op, AND the stage-4
    # fused-dispatch op via register_npu_op.
    import bionpu.kernels.basecalling.linear_projection  # noqa: F401
    from bionpu.dispatch.npu import NPU_OPS

    assert "dorado_fast_linear_projection_fused_dispatch" in NPU_OPS, (
        "DoradoFastLinearProjectionFusedDispatch must be registered "
        "under the 'dorado_fast_linear_projection_fused_dispatch' op "
        "name; the encoder dispatch consumes that key."
    )
    # Existing per-group + stage-3 ops MUST still be registered
    # (additive contract).
    assert "dorado_fast_linear_projection" in NPU_OPS, (
        "Existing per-group dorado_fast_linear_projection op must "
        "remain registered (stage-4 is additive)"
    )
    assert "dorado_fast_linear_projection_fused_perts" in NPU_OPS, (
        "Stage-3 dorado_fast_linear_projection_fused_perts op must "
        "remain registered (stage-4 is additive; do NOT replace "
        "stage 3)"
    )


def test_fused_dispatch_op_class_present() -> None:
    """Stage-4 op class exposes the expected interface."""
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjectionFusedDispatch,
    )

    assert (
        DoradoFastLinearProjectionFusedDispatch.name
        == "dorado_fast_linear_projection_fused_dispatch"
    )
    assert DoradoFastLinearProjectionFusedDispatch.INPUT_SHAPE == (
        T_LSTM, 1, HIDDEN
    )
    assert DoradoFastLinearProjectionFusedDispatch.OUTPUT_SHAPE == (
        T_LSTM, 1, OUT_DIM
    )
    assert DoradoFastLinearProjectionFusedDispatch.SEQ_LEN == T_LSTM


def test_fused_dispatch_pack_helpers_match_stage3() -> None:
    """Stage-4 wire format reuses stage-3 packers byte-for-byte."""
    from bionpu.kernels.basecalling.linear_projection import (
        _pack_input_bf16,
        _pack_wb_bf16,
        _unpack_output_bf16,
    )

    rng = np.random.default_rng(20260427)
    x = rng.standard_normal(size=(T_LSTM, 1, HIDDEN)).astype(np.float32)
    weight = (rng.standard_normal(size=(OUT_DIM, HIDDEN)) * 0.1).astype(
        np.float32
    )

    x_packed = _pack_input_bf16(x)
    assert x_packed.dtype == np.uint16
    assert x_packed.size == T_LSTM * HIDDEN

    wb_packed = _pack_wb_bf16(weight)
    assert wb_packed.dtype == np.uint16
    # Stage-4 wire format: ONE 48-KiB bf16 slab (same as stage 3).
    assert wb_packed.size == OUT_DIM * HIDDEN, (
        f"stage-4 wb packer must emit a single ({OUT_DIM} * {HIDDEN}) "
        f"bf16 slab; got size {wb_packed.size}"
    )

    out_bf = _unpack_output_bf16(
        np.zeros(T_LSTM * OUT_DIM, dtype=np.uint16),
    )
    assert out_bf.shape == (T_LSTM, 1, OUT_DIM)
    assert out_bf.dtype == np.float32


# --------------------------------------------------------------------------- #
# NPU-required test (gated on the bring-up host)
# --------------------------------------------------------------------------- #


def _fused_dispatch_artifacts_present() -> bool:
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjectionFusedDispatch,
    )

    return DoradoFastLinearProjectionFusedDispatch.artifacts_present()


@pytest.mark.npu
def test_npu_fused_dispatch_byte_equal_to_bf16_reference() -> None:
    """Stage-4 silicon validation hook (separate follow-up runs this).

    When the fused-dispatch artifact is present, runs the kernel and
    asserts the output matches the host-side bf16 reference within
    bf16 round-off (~5e-2 max-abs on clamped-[-5, 5] outputs). When
    the artifact isn't present (build-clean-only env), this test is
    skipped — matching the stage-3 test's behaviour.
    """
    if not _fused_dispatch_artifacts_present():
        pytest.skip(
            "linear_projection fused-dispatch NPU artifacts missing; "
            "build via "
            "`make experiment=fused-dispatch NPU2=1 seq=334 all` "
            "and copy outputs to "
            "src/bionpu/dispatch/_npu_artifacts/"
            "dorado_fast_linear_projection_fused_dispatch/"
        )
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjectionFusedDispatch,
    )

    rng = np.random.default_rng(20260427)
    x = rng.standard_normal(size=(T_LSTM, 1, HIDDEN)).astype(np.float32)
    weight = (rng.standard_normal(size=(OUT_DIM, HIDDEN)) * 0.1).astype(
        np.float32
    )

    op = DoradoFastLinearProjectionFusedDispatch()
    y_npu = op(x=x, weight=weight, n_iters=1, warmup=0)

    assert y_npu.shape == (T_LSTM, 1, OUT_DIM), (
        f"fused-dispatch output shape {y_npu.shape} != expected"
    )
    assert y_npu.dtype == np.float32

    assert int(np.isnan(y_npu).sum()) == 0, (
        "fused-dispatch NPU output contains NaN values"
    )
    assert int(np.isinf(y_npu).sum()) == 0, (
        "fused-dispatch NPU output contains Inf values"
    )

    y_ref = _linear_projection_cpu_bf16ref(x, weight)
    diff = np.abs(y_npu - y_ref)
    max_diff = float(diff.max())
    # bf16 mantissa = 8 bits; per-element drift on clamped-[-5, 5]
    # outputs sits below ~5e-2 in the worst case (saturation cancels
    # most error). Same gate as stage 3.
    assert max_diff < 5e-2, (
        f"fused-dispatch NPU vs bf16 reference max-abs-error "
        f"{max_diff:.3e} exceeds 5e-2"
    )
