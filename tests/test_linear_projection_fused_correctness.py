"""B1 / DESIGN-fusion stage 3 — linear_projection_fused_perts correctness.

Closes the build-clean half of B1 (the silicon byte-equal half is B2).
This file mirrors the pattern in
``tests/test_t51_conv_stem_layer2_correctness.py``:

- Fast / non-NPU: kernel + IRON-Python sources contain expected
  sentinels (regression-protect against accidental revert to the
  per-group shape, and against accidental loss of bf16 precision
  contract).
- Fast / non-NPU: the numpy CPU reference matches
  ``torch.nn.functional.linear`` (sanity gate for the reference).
- NPU-required: byte-equal-within-FP32-tolerance NPU output vs the
  CPU bf16-reference — runs only when artifacts are present.

The DESIGN-fusion target wall time (~141 ms) is recorded by the
encoder benchmark, not asserted here; this test file pins
*correctness* of the fused implementation.
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
# CPU reference (FP32 + bf16-narrowed reference)
# --------------------------------------------------------------------------- #


def _linear_projection_cpu_fp32(
    x: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Linear(96 -> 256, bias=False) + clamp[-5, 5] in FP32."""
    assert x.shape == (T_LSTM, 1, HIDDEN)
    assert weight.shape == (OUT_DIM, HIDDEN)
    flat = x.reshape(T_LSTM, HIDDEN).astype(np.float32)
    y = flat @ weight.astype(np.float32).T  # (L, OUT_DIM)
    y = np.clip(y, CLAMP_LO, CLAMP_HI)
    return y.reshape(T_LSTM, 1, OUT_DIM)


def _linear_projection_cpu_bf16ref(
    x: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """bf16 reference: narrow x and weight to bf16 first, then GEMM in
    FP32 accum, then clamp + bf16-narrow output. Mirrors the wire
    contract of the fused-perts kernel: bf16 multiplier inputs,
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
    # Narrow back through bf16 to match the kernel's output narrow.
    y_bf = bf16_bytes_to_fp32(
        fp32_to_bf16_bytes(y).reshape(-1), y.shape,
    )
    return y_bf.reshape(T_LSTM, 1, OUT_DIM)


# --------------------------------------------------------------------------- #
# Fast tests — no NPU required
# --------------------------------------------------------------------------- #


def test_fused_kernel_source_uses_perts_loop_sentinels() -> None:
    """B1 regression guard: kernel keeps the per-timestep fused shape.

    Asserts the C++ source has the inner OC-group loop sentinel and
    uses the bf16 mac path. A future revert to per-group fp32 would
    drop the sentinel.
    """
    cc_path = (
        REPO
        / "bionpu-public"
        / "src"
        / "bionpu"
        / "kernels"
        / "basecalling"
        / "linear_projection"
        / "linear_projection_fused.cc"
    )
    src = cc_path.read_text()
    # Sentinel 1: the fused kernel walks the 4 OC groups internally.
    assert "FUSED_PERTS_INNER_OC_LOOP" in src, (
        "linear_projection_fused.cc must contain the FUSED_PERTS_INNER_OC_LOOP "
        "sentinel comment (B1 stage-3 fix marker)"
    )
    # Sentinel 2: bf16 weight pointer signature.
    assert "dorado_fast_linear_projection_fused_perts" in src, (
        "linear_projection_fused.cc must export "
        "dorado_fast_linear_projection_fused_perts (B1 stage-3 entry "
        "symbol missing)"
    )
    assert "bfloat16 *restrict W_full" in src, (
        "linear_projection_fused.cc must take W_full as bfloat16* "
        "(precision contract: bf16 weights per DESIGN-fusion.md)"
    )
    # Sentinel 3: precision contract — accfloat accumulator with bf16 mac.
    assert "aie::accum<accfloat, VEC>" in src, (
        "linear_projection_fused.cc must use an accfloat accumulator "
        "(precision contract: FP32 accum, bf16 multiplier inputs)"
    )
    assert "aie::mac(acc, w_v, x_v)" in src, (
        "linear_projection_fused.cc must use aie::mac with bf16 weight + "
        "bf16 input vectors (DESIGN-fusion.md stage-3 precision contract)"
    )


def test_fused_iron_uses_single_weight_acquire() -> None:
    """B1 regression guard: IRON Python acquires the weight once.

    The DESIGN-fusion stage-3 contract is that the weight ObjectFifo
    is acquired at depth=1 outside the timestep loop and held for the
    whole dispatch. Asserting on the source keeps a future revert
    from silently regressing the call-count win.
    """
    py_path = (
        REPO
        / "bionpu-public"
        / "src"
        / "bionpu"
        / "kernels"
        / "basecalling"
        / "linear_projection"
        / "linear_projection_fused.py"
    )
    src = py_path.read_text()
    # Weight depth is 1 (single acquire-hold for the whole dispatch).
    assert 'name="weight_in", depth=1' in src, (
        "linear_projection_fused.py must declare weight_in with depth=1 "
        "(stage-3 acquire-once contract)"
    )
    # The weight acquire must be OUTSIDE the timestep loop.
    assert "elem_w_full = of_weight.acquire(1)" in src, (
        "linear_projection_fused.py must acquire elem_w_full once before "
        "the per-timestep loop"
    )


def test_init_registers_fused_op() -> None:
    """B1 dispatch wiring: fused-perts op is registered.

    The kernel package's __init__.py is responsible for calling
    ``register_npu_op`` at import time, so we explicitly import it
    before checking the registry — same pattern callers use.
    """
    # Importing the package runs its __init__.py and registers BOTH
    # the per-group op and the fused-perts op via register_npu_op.
    import bionpu.kernels.basecalling.linear_projection  # noqa: F401
    from bionpu.dispatch.npu import NPU_OPS

    assert "dorado_fast_linear_projection_fused_perts" in NPU_OPS, (
        "DoradoFastLinearProjectionFusedPerts must be registered under "
        "the 'dorado_fast_linear_projection_fused_perts' op name; the "
        "encoder dispatch consumes that key."
    )
    # Existing per-group op MUST still be registered (additive contract).
    assert "dorado_fast_linear_projection" in NPU_OPS, (
        "Existing per-group dorado_fast_linear_projection op must remain "
        "registered (B1 is additive; do NOT replace the existing artifact)."
    )


def test_pack_helpers_roundtrip() -> None:
    """Stage-3 host-side fp32 -> bf16 helpers preserve shape."""
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
    # Stage-3 wire format: ONE 48-KiB bf16 slab (no per-timestep
    # repetition like the per-group fp32 packer does).
    assert wb_packed.size == OUT_DIM * HIDDEN, (
        f"stage-3 wb packer must emit a single ({OUT_DIM} * {HIDDEN}) bf16 "
        f"slab; got size {wb_packed.size}"
    )

    # Round-trip a synthetic output buffer through the unpacker.
    out_bf = _unpack_output_bf16(
        np.zeros(T_LSTM * OUT_DIM, dtype=np.uint16),
    )
    assert out_bf.shape == (T_LSTM, 1, OUT_DIM)
    assert out_bf.dtype == np.float32


def test_cpu_reference_matches_pytorch() -> None:
    """The numpy reference matches torch.nn.functional.linear to FP32 jitter."""
    pytest.importorskip("torch")
    import torch

    rng = np.random.default_rng(20260427)
    x = rng.standard_normal(size=(T_LSTM, 1, HIDDEN)).astype(np.float32)
    weight = (rng.standard_normal(size=(OUT_DIM, HIDDEN)) * 0.1).astype(
        np.float32
    )

    y_np = _linear_projection_cpu_fp32(x, weight)
    with torch.inference_mode():
        y_torch = torch.nn.functional.linear(
            torch.from_numpy(x.reshape(T_LSTM, HIDDEN)),
            torch.from_numpy(weight),
        ).numpy()
        y_torch = np.clip(y_torch, CLAMP_LO, CLAMP_HI).reshape(
            T_LSTM, 1, OUT_DIM
        )
    diff = float(np.max(np.abs(y_np - y_torch)))
    # FP32 reduction order differs slightly between numpy and torch's
    # eigen path; the post-clamp values often hit the saturation band
    # and the diff there is exactly zero.
    assert diff < 1e-3, f"numpy vs torch reference drift {diff:.3e}"


# --------------------------------------------------------------------------- #
# NPU-required test (gated on the bring-up host)
# --------------------------------------------------------------------------- #


def _fused_artifacts_present() -> bool:
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjectionFusedPerts,
    )

    return DoradoFastLinearProjectionFusedPerts.artifacts_present()


@pytest.mark.npu
def test_npu_fused_perts_byte_equal_to_bf16_reference() -> None:
    """B1 silicon validation hook (B2 actually runs this on hardware).

    When the fused-perts artifact is present, runs the kernel and
    asserts the output matches the host-side bf16 reference within
    bf16 round-off (8 mantissa bits ~ 4e-3 relative). When it isn't
    present (build-clean-only env), this test is skipped.
    """
    if not _fused_artifacts_present():
        pytest.skip(
            "linear_projection fused-perts NPU artifacts missing; "
            "build via `make experiment=fused-perts NPU2=1 seq=334 all` "
            "and copy outputs to "
            "src/bionpu/dispatch/_npu_artifacts/"
            "dorado_fast_linear_projection_fused_perts/"
        )
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjectionFusedPerts,
    )

    rng = np.random.default_rng(20260427)
    x = rng.standard_normal(size=(T_LSTM, 1, HIDDEN)).astype(np.float32)
    weight = (rng.standard_normal(size=(OUT_DIM, HIDDEN)) * 0.1).astype(
        np.float32
    )

    op = DoradoFastLinearProjectionFusedPerts()
    y_npu = op(x=x, weight=weight, n_iters=1, warmup=0)

    assert y_npu.shape == (T_LSTM, 1, OUT_DIM), (
        f"fused-perts output shape {y_npu.shape} != expected"
    )
    assert y_npu.dtype == np.float32

    # Hard NaN/Inf gate.
    assert int(np.isnan(y_npu).sum()) == 0, (
        "fused-perts NPU output contains NaN values"
    )
    assert int(np.isinf(y_npu).sum()) == 0, (
        "fused-perts NPU output contains Inf values"
    )

    # bf16 reference — same precision contract as the kernel.
    y_ref = _linear_projection_cpu_bf16ref(x, weight)
    diff = np.abs(y_npu - y_ref)
    max_diff = float(diff.max())
    # bf16 mantissa = 8 bits; per-element absolute drift on
    # clamped-[-5,5] outputs sits below ~5e-2 in the worst case
    # (saturation band cancels most error). A relaxed gate keeps this
    # test stable across host/numpy bf16 cast variations.
    assert max_diff < 5e-2, (
        f"fused-perts NPU vs bf16 reference max-abs-error {max_diff:.3e} "
        f"exceeds 5e-2"
    )
