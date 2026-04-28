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

"""Genomics-specific AIE kernels — populated by //.

Per ``state/kmer_count_interface_contract.md`` (T1), this package hosts
the ``kmer_count`` kernel (3 registry entries — one per supported k —
with ``n_tiles`` as a constructor arg on the shared
:class:`BionpuKmerCount` op class). The :func:`get_kmer_count_op` helper
mirrors :func:`bionpu.kernels.basecalling.get_linear_projection_op` so
CLI (T16) and benchmark drivers (T15) consume a single helper rather
than constructing the op class directly.
"""

from __future__ import annotations

import os

KMER_COUNT_LAUNCH_CHUNKS_ENV = "BIONPU_KMER_COUNT_LAUNCH_CHUNKS"
KMER_COUNT_VALID_K: tuple[int, ...] = (15, 21, 31)
KMER_COUNT_VALID_N_TILES: tuple[int, ...] = (1, 2, 4, 8)
KMER_COUNT_DEFAULT_K = 21
KMER_COUNT_DEFAULT_N_TILES = 4


def _kmer_count_op_name(k: int) -> str:
    """Return the ``NPU_OPS`` registry key for k-mer counting at width ``k``.

    Per T1's registry naming pin (3 entries — one per supported k), the
    canonical name is ``bionpu_kmer_count_k{k}``.
    """
    return f"bionpu_kmer_count_k{int(k)}"


def get_kmer_count_op(
    k: int = KMER_COUNT_DEFAULT_K,
    n_tiles: int | None = None,
):
    """Return the k-mer counting :class:`NpuOp` for the requested ``(k, n_tiles)``.

    Per ``state/kmer_count_interface_contract.md`` (T1), there are 3
    ``NPU_OPS`` registry entries (one per supported k in ``{15, 21, 31}``)
    and the artifact-directory matrix is ``3 x 4`` — i.e. ``n_tiles`` is
    a constructor arg on the **shared** :class:`BionpuKmerCount` op
    class, not part of the registry name. This helper looks up
    ``bionpu_kmer_count_k{k}`` (forcing the kernel package to import so
    its ``register_npu_op`` calls run), then constructs a fresh
    ``BionpuKmerCount(k=k, n_tiles=n_tiles)`` instance bound to the
    requested fan-out.

    Args:
        k: Supported k-mer width. Must be in
            :data:`KMER_COUNT_VALID_K` (``{15, 21, 31}``). Defaults to
            ``21``.
        n_tiles: Tile fan-out. ``None`` (default) consults
            ``BIONPU_KMER_COUNT_LAUNCH_CHUNKS`` and falls back to
            :data:`KMER_COUNT_DEFAULT_N_TILES` (``4``). When the caller
            passes an explicit value, the env var is **not** consulted
            (caller wins).

    Returns:
        A freshly-constructed ``BionpuKmerCount(k=k, n_tiles=n_tiles)``
        instance. The registry-resident instance under the
        ``bionpu_kmer_count_k{k}`` key is left untouched (it carries the
        default ``n_tiles``).

    Raises:
        ValueError: ``k`` is not in :data:`KMER_COUNT_VALID_K` or
            ``n_tiles`` (after env-var resolution) is not in
            :data:`KMER_COUNT_VALID_N_TILES`.
        KeyError: the ``bionpu_kmer_count_k{k}`` op is not registered.
            Until T9 lands the ``BionpuKmerCount`` class, this is the
            expected outcome — callers that need the helper at T4 stage
            should catch :class:`KeyError`.
    """
    # Local imports keep the package init lightweight for callers who
    # only need module-level constants.
    from bionpu.dispatch.npu import NPU_OPS
    # Importing the kmer_count subpackage has the side effect of running
    # any ``register_npu_op`` calls. At T4 the module is an empty stub;
    # the import simply succeeds, and the NPU_OPS lookup below raises
    # KeyError. T9 populates the registry.
    import bionpu.kernels.genomics.kmer_count as _kc  # noqa: F401

    if int(k) not in KMER_COUNT_VALID_K:
        valid = ", ".join(str(x) for x in KMER_COUNT_VALID_K)
        raise ValueError(
            f"get_kmer_count_op: k={k!r} is not supported; expected one of "
            f"{{{valid}}}."
        )

    if n_tiles is None:
        env_val = os.environ.get(KMER_COUNT_LAUNCH_CHUNKS_ENV)
        if env_val is None or env_val == "":
            n_tiles = KMER_COUNT_DEFAULT_N_TILES
        else:
            try:
                n_tiles = int(env_val)
            except ValueError as exc:
                raise ValueError(
                    f"{KMER_COUNT_LAUNCH_CHUNKS_ENV}={env_val!r} is not a "
                    f"valid integer; expected one of "
                    f"{KMER_COUNT_VALID_N_TILES}."
                ) from exc

    if int(n_tiles) not in KMER_COUNT_VALID_N_TILES:
        valid = ", ".join(str(x) for x in KMER_COUNT_VALID_N_TILES)
        raise ValueError(
            f"get_kmer_count_op: n_tiles={n_tiles!r} is not supported; "
            f"expected one of {{{valid}}}."
        )

    op_name = _kmer_count_op_name(k)
    if op_name not in NPU_OPS:
        raise KeyError(
            f"k-mer count op {op_name!r} is not registered in NPU_OPS. "
            f"Import bionpu.kernels.genomics.kmer_count after T9 lands "
            f"the BionpuKmerCount class. At T4 stage the kmer_count "
            f"package is a stub and this lookup is expected to fail."
        )

    # The registry-resident instance is bound to the default n_tiles.
    # Construct a fresh instance bound to the caller-requested n_tiles
    # so the artifact-directory selection (k x n_tiles -> 12 dirs) is
    # honoured. Mirrors the basecalling helper's per-seq_len override
    # in :func:`get_linear_projection_op`.
    BionpuKmerCount = type(NPU_OPS[op_name])
    return BionpuKmerCount(k=int(k), n_tiles=int(n_tiles))


__all__ = [
    "KMER_COUNT_DEFAULT_K",
    "KMER_COUNT_DEFAULT_N_TILES",
    "KMER_COUNT_LAUNCH_CHUNKS_ENV",
    "KMER_COUNT_VALID_K",
    "KMER_COUNT_VALID_N_TILES",
    "get_kmer_count_op",
]
