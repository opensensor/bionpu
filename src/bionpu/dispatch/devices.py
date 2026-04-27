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

"""Device enum + tensor-placement helper for the dispatch layer.

Per umbrella PRD §4.1: "Tensor.to(device) where device is cpu | gpu | npu".

This module is the *thin* device-selection wrapper. There is no graph
rewriting and no lazy-mode here — those are explicitly out of scope for
v1 (PRD §7, "Dispatch layer over-engineered before it's needed").

NPU branch: NPU kernels operate on numpy
arrays passed to XRT buffer-objects, NOT on torch tensors. There is no
"move this torch tensor to the NPU" step — the kernel takes raw
host-memory arrays. Therefore :func:`to_device` on NPU is a
**passthrough**: it returns the input tensor unchanged. The actual
NPU-side allocation happens inside :mod:`bionpu.dispatch.npu` when a
registered op runs. Callers that want to run a real NPU kernel use
``dispatch(op="vector_scalar_mul", ..., device="npu")`` (the
registered-op path), not ``dispatch(some_torch_callable, t,
device="npu")`` (which would fall through here as a passthrough).
"""

from __future__ import annotations

import os
from typing import Literal

import torch

# A `Literal` (rather than an `enum.StrEnum`) keeps the wire format
# identical to JSON — the profile table serializes device names as plain
# strings, and call sites pass raw "cpu" / "gpu" / "npu".
Device = Literal["cpu", "gpu", "npu"]

# Public, ordered tuple of valid devices. Used both for validation and
# for `ProfileTable.best_device`'s deterministic tiebreak (gpu > cpu > npu).
DEVICES: tuple[Device, ...] = ("cpu", "gpu", "npu")

# When CUDA is genuinely unavailable, surface a clear actionable message
# rather than letting `tensor.to('cuda')` raise PyTorch's own error.
_CUDA_UNAVAILABLE_MESSAGE = (
    "CUDA not available; pin device='cpu' or set BIONPU_FORCE_CPU=1."
)

def _validate(device: str) -> Device:
    """Raise `ValueError` if `device` is not one of the three accepted names."""
    if device not in DEVICES:
        raise ValueError(
            f"unknown device {device!r}; expected one of {list(DEVICES)}"
        )
    return device  # type: ignore[return-value]

def to_device(tensor: torch.Tensor, device: Device) -> torch.Tensor:
    """Move `tensor` to the named logical device.

    - ``"cpu"``: ``tensor.to("cpu")``.
    - ``"gpu"``: ``tensor.to("cuda")`` if CUDA is available; otherwise
      raises a `RuntimeError` with a clear remediation hint. Honors the
      ``BIONPU_FORCE_CPU=1`` escape hatch by silently routing to CPU
      (useful in CI / on the bring-up agent's CPU-only laptop boot).
    - ``"npu"``: **passthrough** — returns ``tensor`` unchanged. NPU
      kernels operate on numpy arrays passed to XRT buffer-objects,
      not on torch tensors; there is no "move this torch tensor to
      the NPU" step. The real NPU-side allocation happens in
      :mod:`bionpu.dispatch.npu` when a registered op runs (see

    Args:
        tensor: any PyTorch tensor.
        device: one of ``"cpu"``, ``"gpu"``, ``"npu"``.

    Returns:
        A tensor on the requested device. May be the same object if the
        tensor is already on that device (PyTorch's `.to(...)` semantics).
        For ``"npu"`` the input is always returned unchanged.

    Raises:
        ValueError: `device` is not a valid `Device` literal.
        RuntimeError: `device == "gpu"` but CUDA isn't available and the
            `BIONPU_FORCE_CPU` escape hatch is not set.
    """
    dev = _validate(device)

    if dev == "cpu":
        return tensor.to("cpu")

    if dev == "gpu":
        if os.environ.get("BIONPU_FORCE_CPU") == "1":
            return tensor.to("cpu")
        if not torch.cuda.is_available():
            raise RuntimeError(_CUDA_UNAVAILABLE_MESSAGE)
        return tensor.to("cuda")

    return tensor

__all__ = ["DEVICES", "Device", "to_device"]
