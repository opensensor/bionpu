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

"""Basecalling-specific AIE kernels — populated by //.

Top-level package helpers for the (forthcoming) v0.2 encoder pipeline.

The encoder pipeline driver does NOT live in this package yet (it's
v0.2 scope per ``benchmarks/basecalling/run_pod5.sh`` line 12). When
it lands, it will dispatch the linear_projection op via
:func:`get_linear_projection_op`. Pre-landing the helper means cutover
is a one-line env-var flip.
"""

from __future__ import annotations

import os

LINEAR_PROJECTION_VARIANT_ENV = "BIONPU_DORADO_LINEAR_PROJECTION_VARIANT"
LINEAR_PROJECTION_VARIANTS: tuple[str, ...] = ("per_group", "fused_perts")
LINEAR_PROJECTION_DEFAULT_VARIANT = "per_group"

# Op-name keys in :data:`bionpu.dispatch.npu.NPU_OPS`. These MUST match
# the names used in
# ``bionpu.kernels.basecalling.linear_projection.__init__`` at
# ``register_npu_op`` time.
LINEAR_PROJECTION_OP_NAMES: dict[str, str] = {
    "per_group": "dorado_fast_linear_projection",
    "fused_perts": "dorado_fast_linear_projection_fused_perts",
}


def get_linear_projection_op(seq_len: int | None = None):
    """Return the linear_projection NpuOp picked by env var.

    ``BIONPU_DORADO_LINEAR_PROJECTION_VARIANT`` in
    ``{'per_group' (default for now), 'fused_perts'}`` selects the
    artifact. Defaults to ``'per_group'`` until the encoder pipeline is
    cut over. ``fused_perts`` is silicon-validated at 58x speedup with
    bf16 wire format (see ``results/basecalling/b-m6-fused-perts/``).

    Args:
        seq_len: optional production timestep count. ``None`` (default)
            returns the registry-resident short-shape op (L=334) for
            backward compatibility. A non-default value (e.g. 1667 for
            the production_long path) returns a freshly-constructed op
            of the chosen variant bound to that ``seq_len`` — the
            registry-resident instance is left untouched.

    Returns:
        The registered :class:`bionpu.dispatch.npu.NpuOp` instance for
        the chosen variant when ``seq_len is None``, or a fresh op
        instance bound to ``seq_len`` otherwise. The returned op
        exposes the same callable contract for both variants
        (``op(x=..., weight=..., n_iters=..., warmup=...)``) at the
        ``(seq_len, N=1, HIDDEN=96 -> OUT=256)`` shape. Wire-format
        precision differs (per-group: FP32 in/out; fused-perts:
        host-side bf16 narrow + bf16 unpack), but the Python-facing
        dtype is FP32 in both cases.

    Raises:
        ValueError: ``BIONPU_DORADO_LINEAR_PROJECTION_VARIANT`` is set
            to an unknown value.
        KeyError: the chosen variant's op is not registered (e.g. the
            ``linear_projection`` package has not been imported yet).
    """
    # Local import keeps the package init free of pyxrt-style heavy
    # imports for callers who only need module-level constants.
    from bionpu.dispatch.npu import NPU_OPS
    # Ensure the linear_projection package's register_npu_op calls
    # have run before we look anything up — importing it has the
    # side effect of populating NPU_OPS.
    import bionpu.kernels.basecalling.linear_projection as _lp  # noqa: F401

    variant = os.environ.get(
        LINEAR_PROJECTION_VARIANT_ENV, LINEAR_PROJECTION_DEFAULT_VARIANT
    )
    if variant not in LINEAR_PROJECTION_OP_NAMES:
        known = ", ".join(LINEAR_PROJECTION_VARIANTS)
        raise ValueError(
            f"{LINEAR_PROJECTION_VARIANT_ENV}={variant!r} is not a known "
            f"linear_projection variant; expected one of: {known}. "
            f"See bionpu/kernels/basecalling/linear_projection/POSTMORTEM-stage3.md "
            f"for variant selection guidance."
        )
    op_name = LINEAR_PROJECTION_OP_NAMES[variant]
    if op_name not in NPU_OPS:
        raise KeyError(
            f"linear_projection variant {variant!r} maps to op name "
            f"{op_name!r}, but it is not registered in NPU_OPS. Import "
            f"bionpu.kernels.basecalling.linear_projection before calling."
        )
    if seq_len is None:
        return NPU_OPS[op_name]
    if variant == "per_group":
        return _lp.DoradoFastLinearProjection(seq_len=int(seq_len))
    if variant == "fused_perts":
        return _lp.DoradoFastLinearProjectionFusedPerts(seq_len=int(seq_len))
    # Should be unreachable given the variant check above, but keep the
    # registry-resident instance as a safe fallback.
    return NPU_OPS[op_name]


__all__ = [
    "LINEAR_PROJECTION_DEFAULT_VARIANT",
    "LINEAR_PROJECTION_OP_NAMES",
    "LINEAR_PROJECTION_VARIANTS",
    "LINEAR_PROJECTION_VARIANT_ENV",
    "get_linear_projection_op",
]
