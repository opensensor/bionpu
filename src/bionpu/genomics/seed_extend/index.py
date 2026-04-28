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

"""Host-side reference index builder for seed-and-extend v0.

For minimap2-style seeding, the reference is preprocessed once into a
mapping ``canonical_minimizer -> list[(ref_pos, strand)]``. Later, each
query minimizer canonical is looked up to fetch all reference positions
where the same canonical occurs (these are "seed hits").

v0 design (per ``CLAUDE.md`` discussion in the PRD turn):

* Index is built **on host** (CPU) using the existing
  :func:`bionpu.data.minimizer_oracle.extract_minimizers` reference
  implementation. No NPU work is needed for the index step.
* Strand is **not** tracked explicitly because the oracle's
  ``extract_minimizers`` collapses forward + reverse-complement into a
  single canonical via ``min(fwd, rc)``. v0 therefore stores
  ``strand=0`` as a stable placeholder; v1 (when we move the kernel to
  emit ``(canonical, position, strand)`` triples) can fill it in.
* Storage: a Python ``dict[int, list[tuple[int, int]]]`` mapping
  canonical (uint64) to a list of ``(ref_pos_u32, strand)`` pairs. The
  pickled form is used as a disk cache for chr22-scale builds (~5 M
  entries; ~80 MB pickled).

Out of scope for v0:

* In-tile / in-memtile resident index (defer to v1; chr22 ~80 MB does
  not fit the 512 KiB memtile).
* Bloom filter / count-min "definitely no hit" pre-filter (v1).
* minimap2's exact bucket-hashing layout — we use a flat dict here.
"""

from __future__ import annotations

import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from bionpu.data.kmer_oracle import unpack_dna_2bit
from bionpu.data.minimizer_oracle import extract_minimizers


__all__ = [
    "MinimapIndex",
    "build_minimap2_index",
    "build_minimap2_index_from_packed",
    "load_index",
    "save_index",
]


class MinimapIndex:
    """Host-side reference minimizer index.

    Mirrors minimap2's ``mm_idx_t`` interface narrowly for v0: the only
    required operation is ``get(canonical) -> list[(ref_pos, strand)]``.
    """

    __slots__ = ("k", "w", "n_ref_bases", "_table", "build_wall_s")

    def __init__(
        self,
        *,
        k: int,
        w: int,
        n_ref_bases: int,
        table: dict[int, list[tuple[int, int]]] | None = None,
        build_wall_s: float = 0.0,
    ) -> None:
        self.k = int(k)
        self.w = int(w)
        self.n_ref_bases = int(n_ref_bases)
        self._table: dict[int, list[tuple[int, int]]] = (
            {} if table is None else table
        )
        self.build_wall_s = float(build_wall_s)

    # ----- accessors ------------------------------------------------- #

    def get(self, canonical: int) -> list[tuple[int, int]]:
        """Return all ``(ref_pos, strand)`` for ``canonical``.

        Returns an empty list when the canonical is not in the index
        (mirrors minimap2's ``mm_idx_get`` behaviour where ``*n = 0``
        signals "no hit").
        """
        return self._table.get(int(canonical), [])

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, canonical: object) -> bool:
        try:
            return int(canonical) in self._table  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def n_entries(self) -> int:
        """Total number of ``(canonical, ref_pos)`` rows (sum over all hits)."""
        return sum(len(v) for v in self._table.values())

    def items(self) -> Iterable[tuple[int, list[tuple[int, int]]]]:
        return self._table.items()

    def keys(self) -> Iterable[int]:
        return self._table.keys()


def _build_table_from_minimizers(
    minimizers: list[tuple[int, int]],
) -> dict[int, list[tuple[int, int]]]:
    """Group ``(canonical, ref_pos)`` minimizers by canonical key.

    Strand is stored as ``0`` (placeholder; v0 oracle is canonical-only).
    The per-canonical lists are kept in insertion order, which mirrors
    ``extract_minimizers``' position-ascending order — i.e. each list is
    sorted by ``ref_pos`` ascending.
    """
    table: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for canonical, ref_pos in minimizers:
        table[int(canonical)].append((int(ref_pos), 0))
    return dict(table)


def build_minimap2_index(
    ref_seq: str,
    *,
    k: int,
    w: int,
) -> MinimapIndex:
    """Build a reference minimizer index from an ACGT string.

    The implementation reuses
    :func:`bionpu.data.minimizer_oracle.extract_minimizers` so the
    canonical / window semantics are byte-equal-shaped to the silicon
    minimizer kernel (this is the same algorithm the NPU op
    silicon-validates against).

    Args:
        ref_seq: ACGT-only reference sequence (case-sensitive). Non-ACGT
            bases reset the rolling window per the oracle's contract.
        k: minimizer k. Must be in ``{15, 21}`` (oracle pinning).
        w: minimizer w. Must pair with k per the oracle pinning
            (``(15, 10)`` or ``(21, 11)``).

    Returns:
        :class:`MinimapIndex` with the reference's ``(canonical,
        ref_pos)`` entries grouped by canonical.
    """
    t0 = time.monotonic()
    minimizers = extract_minimizers(ref_seq, k=int(k), w=int(w))
    table = _build_table_from_minimizers(minimizers)
    wall = time.monotonic() - t0
    return MinimapIndex(
        k=k,
        w=w,
        n_ref_bases=len(ref_seq),
        table=table,
        build_wall_s=wall,
    )


def build_minimap2_index_from_packed(
    packed_ref: np.ndarray,
    *,
    n_bases: int,
    k: int,
    w: int,
) -> MinimapIndex:
    """Build the index from a packed-2-bit fixture (chr22 etc.).

    Convenience wrapper around :func:`build_minimap2_index`. Unpacks
    once via :func:`bionpu.data.kmer_oracle.unpack_dna_2bit` and feeds
    the ACGT string through.
    """
    if packed_ref.dtype != np.uint8:
        raise ValueError(
            f"packed_ref dtype must be uint8; got {packed_ref.dtype}"
        )
    ref_seq = unpack_dna_2bit(packed_ref, n_bases=int(n_bases))
    return build_minimap2_index(ref_seq, k=k, w=w)


# --------------------------------------------------------------------- #
# Disk cache (pickle).
# --------------------------------------------------------------------- #


def save_index(index: MinimapIndex, path: str | Path) -> None:
    """Pickle the index to disk for re-use across runs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(
            {
                "k": index.k,
                "w": index.w,
                "n_ref_bases": index.n_ref_bases,
                "table": index._table,  # noqa: SLF001 — intentional
                "build_wall_s": index.build_wall_s,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_index(path: str | Path) -> MinimapIndex:
    """Load a pickled index from disk."""
    p = Path(path)
    with open(p, "rb") as f:
        d = pickle.load(f)
    return MinimapIndex(
        k=d["k"],
        w=d["w"],
        n_ref_bases=d["n_ref_bases"],
        table=d["table"],
        build_wall_s=d.get("build_wall_s", 0.0),
    )
