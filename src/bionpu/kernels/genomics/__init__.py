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


# --------------------------------------------------------------------------- #
# Minimizer kernel helper (v0 — 2 registry entries: (k=15,w=10) + (k=21,w=11)).
# --------------------------------------------------------------------------- #

MINIMIZER_LAUNCH_CHUNKS_ENV = "BIONPU_MINIMIZER_LAUNCH_CHUNKS"
MINIMIZER_VALID_KW: tuple[tuple[int, int], ...] = ((15, 10), (21, 11))
MINIMIZER_VALID_N_TILES: tuple[int, ...] = (1, 2, 4, 8)
MINIMIZER_DEFAULT_K: int = 15
MINIMIZER_DEFAULT_W: int = 10
MINIMIZER_DEFAULT_N_TILES: int = 4


def _minimizer_op_name(k: int, w: int) -> str:
    """Return the ``NPU_OPS`` registry key for (k, w) minimizer extraction."""
    return f"bionpu_minimizer_k{int(k)}_w{int(w)}"


def get_minimizer_op(
    k: int = MINIMIZER_DEFAULT_K,
    w: int | None = None,
    n_tiles: int | None = None,
):
    """Return the minimizer :class:`NpuOp` for the requested ``(k, w, n_tiles)``.

    Mirrors :func:`get_kmer_count_op`. Two registry entries (one per
    supported ``(k, w)`` pinned pair) and a per-(k, w, n_tiles) artifact
    directory matrix.

    Args:
        k: minimizer ``k``. Must be a value in :data:`MINIMIZER_VALID_KW`
            (``15`` or ``21``). Defaults to ``15``.
        w: minimizer ``w``. ``None`` (default) selects the canonical
            partner from :data:`MINIMIZER_VALID_KW` (``10`` for ``k=15``;
            ``11`` for ``k=21``).
        n_tiles: tile fan-out. ``None`` (default) consults
            ``BIONPU_MINIMIZER_LAUNCH_CHUNKS`` and falls back to
            :data:`MINIMIZER_DEFAULT_N_TILES` (``4``).

    Returns:
        A freshly-constructed ``BionpuMinimizer(k, w, n_tiles)``.
    """
    from bionpu.dispatch.npu import NPU_OPS
    import bionpu.kernels.genomics.minimizer as _mz  # noqa: F401

    # Resolve w default from the (k, w) pinning table.
    if w is None:
        match = [pair for pair in MINIMIZER_VALID_KW if pair[0] == int(k)]
        if not match:
            valid = ", ".join(f"({a},{b})" for a, b in MINIMIZER_VALID_KW)
            raise ValueError(
                f"get_minimizer_op: k={k!r} has no canonical w; "
                f"pinned (k, w) ∈ {{{valid}}}."
            )
        w = match[0][1]

    if (int(k), int(w)) not in MINIMIZER_VALID_KW:
        valid = ", ".join(f"({a},{b})" for a, b in MINIMIZER_VALID_KW)
        raise ValueError(
            f"get_minimizer_op: (k, w)=({k!r}, {w!r}) not supported; "
            f"expected one of {{{valid}}}."
        )

    if n_tiles is None:
        env_val = os.environ.get(MINIMIZER_LAUNCH_CHUNKS_ENV)
        if env_val is None or env_val == "":
            n_tiles = MINIMIZER_DEFAULT_N_TILES
        else:
            try:
                n_tiles = int(env_val)
            except ValueError as exc:
                raise ValueError(
                    f"{MINIMIZER_LAUNCH_CHUNKS_ENV}={env_val!r} is not a "
                    f"valid integer; expected one of "
                    f"{MINIMIZER_VALID_N_TILES}."
                ) from exc

    if int(n_tiles) not in MINIMIZER_VALID_N_TILES:
        valid = ", ".join(str(x) for x in MINIMIZER_VALID_N_TILES)
        raise ValueError(
            f"get_minimizer_op: n_tiles={n_tiles!r} is not supported; "
            f"expected one of {{{valid}}}."
        )

    op_name = _minimizer_op_name(k, w)
    if op_name not in NPU_OPS:
        raise KeyError(
            f"minimizer op {op_name!r} is not registered in NPU_OPS. "
            f"Import bionpu.kernels.genomics.minimizer first."
        )

    BionpuMinimizer = type(NPU_OPS[op_name])
    return BionpuMinimizer(k=int(k), w=int(w), n_tiles=int(n_tiles))


# --------------------------------------------------------------------------- #
# Primer / adapter scan helper (v0).
# Three registry entries (one per pinned primer length P).
# --------------------------------------------------------------------------- #

PRIMER_SCAN_VALID_P: tuple[int, ...] = (13, 20, 25)
PRIMER_SCAN_VALID_N_TILES: tuple[int, ...] = (1, 2, 4, 8)
PRIMER_SCAN_DEFAULT_PRIMER: str = "AGATCGGAAGAGC"  # Illumina TruSeq P5, P=13
PRIMER_SCAN_DEFAULT_N_TILES: int = 4
CPG_ISLAND_VALID_N_TILES: tuple[int, ...] = (1, 2, 4, 8)
CPG_ISLAND_DEFAULT_N_TILES: int = 4


def _primer_scan_op_name(p: int) -> str:
    return f"bionpu_primer_scan_p{int(p)}"


def get_primer_scan_op(
    primer: str = PRIMER_SCAN_DEFAULT_PRIMER,
    n_tiles: int | None = None,
):
    """Return the primer/adapter-scan :class:`NpuOp` bound to ``primer``.

    Path B (runtime primer canonical): the primer canonical lives in
    the chunk header, so a single xclbin per primer length P handles
    any primer of length P at runtime. Three registry entries (one per
    pinned P).

    Args:
        primer: ACGT primer, length must be one of
            :data:`PRIMER_SCAN_VALID_P` (13, 20, or 25).
            Defaults to the Illumina TruSeq P5 adapter.
        n_tiles: tile fan-out. ``None`` (default) selects
            :data:`PRIMER_SCAN_DEFAULT_N_TILES` (4).

    Returns:
        A freshly-constructed ``BionpuPrimerScan(primer=primer,
        n_tiles=n_tiles)`` instance.
    """
    from bionpu.dispatch.npu import NPU_OPS
    import bionpu.kernels.genomics.primer_scan as _ps  # noqa: F401

    if not isinstance(primer, str) or not primer:
        raise ValueError(
            f"get_primer_scan_op: primer must be a non-empty ACGT string; "
            f"got {primer!r}"
        )
    p_len = len(primer)
    if p_len not in PRIMER_SCAN_VALID_P:
        valid = ", ".join(str(x) for x in PRIMER_SCAN_VALID_P)
        raise ValueError(
            f"get_primer_scan_op: primer length P={p_len} not in {{{valid}}}."
        )

    if n_tiles is None:
        n_tiles = PRIMER_SCAN_DEFAULT_N_TILES
    if int(n_tiles) not in PRIMER_SCAN_VALID_N_TILES:
        valid = ", ".join(str(x) for x in PRIMER_SCAN_VALID_N_TILES)
        raise ValueError(
            f"get_primer_scan_op: n_tiles={n_tiles!r} not in {{{valid}}}."
        )

    op_name = _primer_scan_op_name(p_len)
    if op_name not in NPU_OPS:
        raise KeyError(
            f"primer_scan op {op_name!r} is not registered in NPU_OPS. "
            f"Import bionpu.kernels.genomics.primer_scan first."
        )

    BionpuPrimerScan = type(NPU_OPS[op_name])
    return BionpuPrimerScan(primer=primer, n_tiles=int(n_tiles))


def get_cpg_island_op(
    n_tiles: int | None = None,
):
    """Return the CpG-island candidate scanner :class:`NpuOp`."""
    from bionpu.dispatch.npu import NPU_OPS
    import bionpu.kernels.genomics.cpg_island as _ci  # noqa: F401

    if n_tiles is None:
        n_tiles = CPG_ISLAND_DEFAULT_N_TILES
    if int(n_tiles) not in CPG_ISLAND_VALID_N_TILES:
        valid = ", ".join(str(x) for x in CPG_ISLAND_VALID_N_TILES)
        raise ValueError(
            f"get_cpg_island_op: n_tiles={n_tiles!r} not in {{{valid}}}."
        )

    op_name = "bionpu_cpg_island"
    if op_name not in NPU_OPS:
        raise KeyError(
            "cpg_island op is not registered in NPU_OPS. "
            "Import bionpu.kernels.genomics.cpg_island first."
        )

    BionpuCpgIsland = type(NPU_OPS[op_name])
    return BionpuCpgIsland(n_tiles=int(n_tiles))


# --------------------------------------------------------------------------- #
# Tandem repeat (STR) helper (v0 — single registry entry).
# --------------------------------------------------------------------------- #

TANDEM_REPEAT_VALID_N_TILES: tuple[int, ...] = (1, 2, 4, 8)
TANDEM_REPEAT_DEFAULT_N_TILES: int = 4


def get_tandem_repeat_op(
    n_tiles: int | None = None,
):
    """Return the short tandem repeat (STR) scanner :class:`NpuOp`."""
    from bionpu.dispatch.npu import NPU_OPS
    import bionpu.kernels.genomics.tandem_repeat as _tr  # noqa: F401

    if n_tiles is None:
        n_tiles = TANDEM_REPEAT_DEFAULT_N_TILES
    if int(n_tiles) not in TANDEM_REPEAT_VALID_N_TILES:
        valid = ", ".join(str(x) for x in TANDEM_REPEAT_VALID_N_TILES)
        raise ValueError(
            f"get_tandem_repeat_op: n_tiles={n_tiles!r} not in {{{valid}}}."
        )

    op_name = "bionpu_tandem_repeat"
    if op_name not in NPU_OPS:
        raise KeyError(
            "tandem_repeat op is not registered in NPU_OPS. "
            "Import bionpu.kernels.genomics.tandem_repeat first."
        )

    BionpuTandemRepeat = type(NPU_OPS[op_name])
    return BionpuTandemRepeat(n_tiles=int(n_tiles))


__all__ = [
    "CPG_ISLAND_DEFAULT_N_TILES",
    "CPG_ISLAND_VALID_N_TILES",
    "KMER_COUNT_DEFAULT_K",
    "KMER_COUNT_DEFAULT_N_TILES",
    "KMER_COUNT_LAUNCH_CHUNKS_ENV",
    "KMER_COUNT_VALID_K",
    "KMER_COUNT_VALID_N_TILES",
    "MINIMIZER_DEFAULT_K",
    "MINIMIZER_DEFAULT_N_TILES",
    "MINIMIZER_DEFAULT_W",
    "MINIMIZER_LAUNCH_CHUNKS_ENV",
    "MINIMIZER_VALID_KW",
    "MINIMIZER_VALID_N_TILES",
    "PRIMER_SCAN_DEFAULT_N_TILES",
    "PRIMER_SCAN_DEFAULT_PRIMER",
    "PRIMER_SCAN_VALID_N_TILES",
    "PRIMER_SCAN_VALID_P",
    "TANDEM_REPEAT_DEFAULT_N_TILES",
    "TANDEM_REPEAT_VALID_N_TILES",
    "get_kmer_count_op",
    "get_minimizer_op",
    "get_cpg_island_op",
    "get_primer_scan_op",
    "get_tandem_repeat_op",
]
