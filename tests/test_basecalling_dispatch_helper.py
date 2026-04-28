"""B3b unit tests — bionpu.kernels.basecalling.get_linear_projection_op.

Validates the env-var dispatch helper that the v0.2 encoder pipeline
will use to pick between the per-group and fused-perts artifacts.

These tests are NPU-free: they assert on the registered op's identity,
not on running it.
"""

from __future__ import annotations

import importlib

import pytest


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _import_helper_module():
    """Import (or re-import) the basecalling package fresh each call.

    The helper reads ``os.environ`` lazily inside
    :func:`get_linear_projection_op`, so a fresh import is not strictly
    required, but it keeps the test isolated from any module-level
    state we might add later.
    """
    import bionpu.kernels.basecalling as mod
    return importlib.reload(mod)


def _ensure_linear_projection_imported() -> None:
    """Force ``register_npu_op`` for both linear_projection variants."""
    import bionpu.kernels.basecalling.linear_projection  # noqa: F401


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_default_returns_per_group_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var set -> per-group op (silicon-conservative default)."""
    monkeypatch.delenv("BIONPU_DORADO_LINEAR_PROJECTION_VARIANT", raising=False)
    _ensure_linear_projection_imported()
    mod = _import_helper_module()
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjection,
    )

    op = mod.get_linear_projection_op()
    assert isinstance(op, DoradoFastLinearProjection), (
        f"default should be DoradoFastLinearProjection; got {type(op).__name__}"
    )
    assert op.name == "dorado_fast_linear_projection"


def test_env_per_group_returns_per_group_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``per_group`` -> per-group op."""
    monkeypatch.setenv(
        "BIONPU_DORADO_LINEAR_PROJECTION_VARIANT", "per_group"
    )
    _ensure_linear_projection_imported()
    mod = _import_helper_module()
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjection,
    )

    op = mod.get_linear_projection_op()
    assert isinstance(op, DoradoFastLinearProjection)


def test_env_fused_perts_returns_fused_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fused_perts`` -> fused-perts op (stage-3 / 58x silicon-validated)."""
    monkeypatch.setenv(
        "BIONPU_DORADO_LINEAR_PROJECTION_VARIANT", "fused_perts"
    )
    _ensure_linear_projection_imported()
    mod = _import_helper_module()
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjectionFusedPerts,
    )

    op = mod.get_linear_projection_op()
    assert isinstance(op, DoradoFastLinearProjectionFusedPerts), (
        f"fused_perts should be DoradoFastLinearProjectionFusedPerts; "
        f"got {type(op).__name__}"
    )
    assert op.name == "dorado_fast_linear_projection_fused_perts"


def test_invalid_variant_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown variant names raise a ``ValueError`` mentioning known set."""
    monkeypatch.setenv(
        "BIONPU_DORADO_LINEAR_PROJECTION_VARIANT", "not_a_real_variant"
    )
    _ensure_linear_projection_imported()
    mod = _import_helper_module()

    with pytest.raises(ValueError) as exc_info:
        mod.get_linear_projection_op()
    msg = str(exc_info.value)
    assert "BIONPU_DORADO_LINEAR_PROJECTION_VARIANT" in msg, (
        f"error message should reference the env var; got: {msg!r}"
    )
    assert "not_a_real_variant" in msg, (
        f"error message should echo the bad value; got: {msg!r}"
    )
    assert "per_group" in msg and "fused_perts" in msg, (
        f"error message should list known variants; got: {msg!r}"
    )


def test_both_ops_independently_importable_and_registered() -> None:
    """B3b additivity guard: both ops still in the registry."""
    _ensure_linear_projection_imported()
    from bionpu.dispatch.npu import NPU_OPS
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjection,
        DoradoFastLinearProjectionFusedPerts,
    )

    assert "dorado_fast_linear_projection" in NPU_OPS, (
        "per-group op missing from NPU_OPS; B3b helper is additive — it "
        "must not unregister the per-group artifact"
    )
    assert "dorado_fast_linear_projection_fused_perts" in NPU_OPS, (
        "fused-perts op missing from NPU_OPS; B3b helper requires both "
        "ops to be registered concurrently"
    )
    # Sanity: the registry entries are instances of the expected classes.
    assert isinstance(
        NPU_OPS["dorado_fast_linear_projection"], DoradoFastLinearProjection
    )
    assert isinstance(
        NPU_OPS["dorado_fast_linear_projection_fused_perts"],
        DoradoFastLinearProjectionFusedPerts,
    )


def test_module_exports_constants() -> None:
    """The package surfaces the env-var name + variant set for downstream."""
    import bionpu.kernels.basecalling as mod

    assert mod.LINEAR_PROJECTION_VARIANT_ENV == (
        "BIONPU_DORADO_LINEAR_PROJECTION_VARIANT"
    )
    assert mod.LINEAR_PROJECTION_DEFAULT_VARIANT == "per_group"
    assert set(mod.LINEAR_PROJECTION_VARIANTS) == {"per_group", "fused_perts"}
    assert (
        mod.LINEAR_PROJECTION_OP_NAMES["per_group"]
        == "dorado_fast_linear_projection"
    )
    assert (
        mod.LINEAR_PROJECTION_OP_NAMES["fused_perts"]
        == "dorado_fast_linear_projection_fused_perts"
    )
