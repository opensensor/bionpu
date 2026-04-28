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

"""Shared pyxrt-dispatch helpers for the basecalling kernel package.

The shape of the helper that linear_projection landed first
(``_resolve_dispatch_impl`` + ``_run_pyxrt_*``) is mostly
boilerplate. This module factors the boilerplate out so the rest of
the encoder ops (conv stem L1/L2/L3, lstm_cell, lstm_cell_bf16_acc)
can opt into the in-process path with a few lines each.

Why a separate module rather than a method on
:class:`bionpu.dispatch.npu.NpuBackend`? Each op already owns its
own arg-slot / payload layout and per-op env-var name; a free-function
helper keeps the call site readable and avoids smearing op-specific
ABI knowledge into the generic backend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from bionpu.dispatch.npu import default_backend


def resolve_dispatch_impl(impl: str | None, *, env_var: str) -> str:
    """Return the dispatch implementation: 'subprocess' (default) or 'pyxrt'.

    Priority: explicit ``impl`` arg, else env ``env_var``, else
    ``"subprocess"``. Raises :class:`ValueError` for unknown values.
    """
    chosen = (
        impl or os.environ.get(env_var, "subprocess")
    ).strip().lower()
    if chosen not in ("subprocess", "pyxrt"):
        raise ValueError(
            f"{env_var} / _impl must be one of 'subprocess' or 'pyxrt'; "
            f"got {chosen!r}."
        )
    return chosen


def run_pyxrt_with_buffers(
    *,
    xclbin_path: Path,
    insts_path: Path,
    in_buffers: Sequence[tuple[bytes, int]],
    out_size: int,
    out_arg_index: int,
    n_iters: int,
    warmup: int,
    kernel_name: str = "MLIR_AIE",
) -> tuple[bytes, float, float, float]:
    """Drive an xclbin in-process via pyxrt; thin wrapper over
    :meth:`NpuBackend.run_xclbin` to centralise argument hygiene.

    Returns ``(raw_output_bytes, avg_us, min_us, max_us)``.
    """
    backend = default_backend()
    return backend.run_xclbin(
        xclbin=xclbin_path,
        insts=insts_path,
        kernel_name=kernel_name,
        in_buffers=list(in_buffers),
        out_size=int(out_size),
        out_arg_index=int(out_arg_index),
        n_iters=int(n_iters),
        warmup=int(warmup),
    )


__all__ = [
    "resolve_dispatch_impl",
    "run_pyxrt_with_buffers",
]
