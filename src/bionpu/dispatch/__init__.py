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

"""Device dispatch (CPU / GPU / NPU) for bionpu — + .

Per  §4.1: a thin device-selection wrapper. Public surface:

- ``Device`` — `Literal["cpu", "gpu", "npu"]`.
- ``to_device(tensor, device)`` — moves a tensor to the named device.
  NPU branch is a **passthrough** (see ``devices.py``) because NPU
  kernels run on numpy arrays passed to XRT buffer-objects.
- ``ProfileTable`` — measured-latency JSON-backed lookup table populated
  by the bench harness and consulted by `dispatch()`.
- ``dispatch(op, *args, device=, override=, profile=)`` — runs ``op`` on a
  chosen device. Resolution order: ``override`` > ``device`` >
  ``profile.best_device(...)``. With none supplied, raises a clear
  `ValueError`.

  Two ``op`` calling conventions:

  1. ``op`` is a Python callable (e.g. a torch op): tensors in
     ``*args`` / ``**kwargs`` are moved to ``device`` and ``op`` is
     called. NPU device routes through :func:`to_device` as a
     passthrough — only useful for ops where the callable itself
     handles NPU (rare; mostly cpu/gpu paths).
  2. ``op`` is a string (e.g. ``"vector_scalar_mul"``): when ``device
     == "npu"`` the string is looked up in
     :data:`bionpu.dispatch.npu.NPU_OPS` and the registered
     :class:`NpuOp` runs with the keyword arguments. This is the v1
     contract for "the NPU runs this kernel" — pre-built xclbins +
     registered-op lookup table per  §4.1.

  :func:`bionpu.dispatch.npu.register_npu_op`.

- :class:`bionpu.dispatch.npu.NpuBackend` — the lazy-init runner for
  the NPU. Re-exported here so call-sites import from one place.

Out of scope for v1 (PRD §7): graph rewriting, lazy mode, op fusion.
Both basecalling (tracks/basecalling/) and CRISPR (tracks/crispr/)
dispatch through this module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from bionpu.dispatch.devices import DEVICES, Device, to_device
from bionpu.dispatch.npu import (
    NPU_OPS,
    NpuArtifactsMissingError,
    NpuBackend,
    NpuOp,
    NpuRunFailed,
    NpuRunResult,
    NpuVectorScalarMul,
    lookup_npu_op,
    register_npu_op,
)
from bionpu.dispatch.npu import (
    default_backend as default_npu_backend,
)
from bionpu.dispatch.profile import ProfileTable

__all__ = [
    "DEVICES",
    "Device",
    "NPU_OPS",
    "NpuArtifactsMissingError",
    "NpuBackend",
    "NpuOp",
    "NpuRunFailed",
    "NpuRunResult",
    "NpuVectorScalarMul",
    "ProfileTable",
    "default_npu_backend",
    "dispatch",
    "lookup_npu_op",
    "register_npu_op",
    "to_device",
]

def _move_args(
    args: tuple[Any, ...], kwargs: dict[str, Any], device: Device
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Move every Tensor in `args`/`kwargs` to `device`. Non-tensors pass through."""
    new_args = tuple(
        to_device(a, device) if isinstance(a, torch.Tensor) else a for a in args
    )
    new_kwargs = {
        k: (to_device(v, device) if isinstance(v, torch.Tensor) else v)
        for k, v in kwargs.items()
    }
    return new_args, new_kwargs

def _first_tensor_shape(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, ...]:
    """Return the shape of the first Tensor argument, for profile lookup."""
    for a in args:
        if isinstance(a, torch.Tensor):
            return tuple(a.shape)
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            return tuple(v.shape)
    # No tensor inputs: a scalar shape is a defensible default; the
    # profile table will simply miss and `best_device` will return
    # "cpu", which is the safe fallback.
    return ()

def dispatch(
    op: Callable[..., Any] | str,
    *args: Any,
    device: Device | None = None,
    override: Device | None = None,
    profile: ProfileTable | None = None,
    **kwargs: Any,
) -> Any:
    """Run an op on a chosen device.

    Two calling conventions:

    1. **Callable op**: ``op`` is any Python callable (e.g. a torch
       function). Tensor arguments (positional or keyword) are moved
       to ``device`` via :func:`to_device` and ``op(*args, **kwargs)``
       is called. Non-tensor arguments pass through.

    2. **Registered NPU op**: ``op`` is a string AND the chosen device
       is ``"npu"``. The string is looked up in :data:`NPU_OPS`
       (raises ``KeyError`` for unknown ops) and the registered
       :class:`NpuOp` is invoked with the keyword arguments. The v1
       contract per  §4.1: pre-built xclbins + a lookup
       table; no on-the-fly compile.

       For convenience, when ``op`` is a string AND the chosen device
       is ``"cpu"`` or ``"gpu"``, dispatch raises ``ValueError`` —
       string ops are an NPU-only construct in v1; cpu/gpu callers
       supply their own callables (numpy / torch). Cross-device
       parity tests therefore call ``dispatch(op="vector_scalar_mul",
       device="npu", ...)`` once and run their own numpy/torch
       reference for cpu/gpu, captured in the same harness window.

    Resolution order for the device (first match wins):

    1. ``override`` — manual override flag from PRD §4.1. Skips the
       profile lookup entirely.
    2. ``device`` — explicit device pin from the caller.
    3. ``profile.best_device(op_name, input_shape)`` — measured-latency
       lookup. Only consulted if `profile` is given AND neither
       ``override`` nor ``device`` was set.

    With none supplied, raises a clear ``ValueError``.

    Args:
        op: either a Python callable or a string keying into
            :data:`NPU_OPS`. See above.
        *args: positional arguments. For callable ``op``, tensors are
            moved to the chosen device. For string ``op``, positional
            args are not used (pass keyword args).
        device: explicit device pin.
        override: manual override (wins over ``device`` and
            ``profile``).
        profile: measured-latency table to consult when no device is
            given.
        **kwargs: keyword arguments. For callable ``op`` tensors are
            moved; for string ``op`` they pass straight to
            :class:`NpuOp.__call__`.

    Returns:
        Whatever the callable / NpuOp returns.

    Raises:
        ValueError: none of ``device``, ``override``, ``profile`` was
            supplied; or string ``op`` with cpu/gpu device.
        KeyError: string ``op`` not registered in :data:`NPU_OPS`.
        RuntimeError: chosen device resolves to ``"gpu"`` but CUDA
            isn't available on this host.
    """
    if override is not None:
        chosen: Device = override
    elif device is not None:
        chosen = device
    elif profile is not None:
        op_name = (
            op if isinstance(op, str) else getattr(op, "__name__", repr(op))
        )
        in_shape = _first_tensor_shape(args, kwargs)
        # v1 keeps it thin: we don't know the output shape until the op
        # runs. For the lookup we use input shape = output shape as a
        # crude key. The bench harness records with both shapes
        # known, so for a measured op this still hits.
        chosen = profile.best_device(
            op=op_name, input_shape=in_shape, output_shape=in_shape
        )
    else:
        raise ValueError(
            "dispatch requires device=, override=, or a profile table "
            "(profile=); none was supplied. See PRD-bio-on-xdna §4.1."
        )

    # String op: NPU-registry lookup path.
    if isinstance(op, str):
        if chosen != "npu":
            raise ValueError(
                f"dispatch(op={op!r}, device={chosen!r}): string ops are "
                f"NPU-only in v1. Use a callable for cpu/gpu paths and "
                f'dispatch(op="...", device="npu", ...) for the NPU '
                f"registered-op path. ( §4.1.)"
            )
        npu_op = lookup_npu_op(op)
        # The NpuOp contract is keyword-only — args are NOT forwarded.
        # If a caller passes positionals here it's almost certainly a
        # mistake; fail loud rather than silently dropping them.
        if args:
            raise TypeError(
                f"dispatch(op={op!r}): registered NPU ops accept "
                f"keyword arguments only (got {len(args)} positional)."
            )
        return npu_op(**kwargs)

    moved_args, moved_kwargs = _move_args(args, kwargs, chosen)
    return op(*moved_args, **moved_kwargs)
