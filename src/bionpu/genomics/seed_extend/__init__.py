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

"""minimap2-style seed extraction on AIE2P (v0).

Third silicon-validated CRISPR-shape primitive after ``kmer_count`` and
``minimizer``. v0 reuses the locked v0/v1 minimizer NPU artifacts
(``_npu_artifacts/bionpu_minimizer_k{k}_w{w}_n4/``) and adds host-side
glue for the reference-index build + per-query seed lookup.

Pipeline:

1. **Index build (CPU, one-time per reference)**:
   :func:`bionpu.genomics.seed_extend.index.build_minimap2_index`.
   For chr22 (~50 Mbp) the build wall is ~30 s on a single CPU core
   (oracle is pure Python — fine for v0; v1 can JIT or vectorise).
   The pickled cache is ~80 MB; cached at
   ``state/seed_extend_chr22_index_k{k}_w{w}.pickle``.

2. **Query extraction (NPU)**:
   :class:`bionpu.kernels.genomics.minimizer.BionpuMinimizer` produces
   ``[(canonical, query_pos), ...]`` from a packed-2-bit query. This is
   the same silicon op the v0/v1 minimizer kernel ships.

3. **Seed lookup (CPU, hash-table get)**:
   :func:`bionpu.genomics.seed_extend.lookup.query_to_seeds_from_minimizers`
   maps each query minimizer's canonical to the index's ``ref_pos``
   list, emitting one :class:`SeedHit` per match. Sorted by
   ``(ref_pos asc, query_pos asc)``.

The :class:`SeedExtractor` class is the user-facing entry point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from bionpu.genomics.seed_extend.index import (
    MinimapIndex,
    build_minimap2_index,
    build_minimap2_index_from_packed,
    load_index,
    save_index,
)
from bionpu.genomics.seed_extend.chain import (
    SeedChain,
    chain_seed_hits,
)
from bionpu.genomics.seed_extend.lookup import (
    SeedHit,
    query_to_seeds,
    query_to_seeds_from_minimizers,
)


__all__ = [
    "MinimapIndex",
    "SeedChain",
    "SeedExtractionResult",
    "SeedExtractor",
    "SeedHit",
    "build_minimap2_index",
    "build_minimap2_index_from_packed",
    "chain_seed_hits",
    "load_index",
    "query_to_seeds",
    "query_to_seeds_from_minimizers",
    "save_index",
]


@dataclass(frozen=True, slots=True)
class SeedExtractionResult:
    """End-to-end timing breakdown + the seed hits."""

    seeds: list[SeedHit]
    n_query_minimizers: int
    npu_wall_s: float
    lookup_wall_s: float
    used_npu: bool

    @property
    def total_wall_s(self) -> float:
        return self.npu_wall_s + self.lookup_wall_s


class SeedExtractor:
    """End-to-end seed-and-extend (seeding only) for v0.

    Wraps the v0 minimizer NPU op + a host-side reference index. The
    ``__call__`` entry point takes a packed-2-bit query and emits
    ``SeedExtractionResult``.

    v0 design notes:

    * Reference index is **immutable** after construction; reuse one
      :class:`SeedExtractor` per chr22 build to amortise the index cost.
    * NPU op is **lazily** constructed on first call so that
      ``SeedExtractor(index)`` works on machines without artifacts (the
      lookup-only path falls back to the oracle).
    * No extension / chaining / CIGAR — those are out of scope per the
      v0 PRD (defer to host CPU; AIE2P is a poor fit for banded SW).
    """

    def __init__(
        self,
        index: MinimapIndex | None = None,
        *,
        k: int | None = None,
        w: int | None = None,
        n_tiles: int = 4,
        prefer_npu: bool = True,
    ) -> None:
        if index is None and (k is None or w is None):
            raise ValueError(
                "SeedExtractor: either pass a prebuilt index or both k+w"
            )
        if index is not None:
            if k is not None and int(k) != index.k:
                raise ValueError(
                    f"SeedExtractor: k={k!r} disagrees with index.k={index.k}"
                )
            if w is not None and int(w) != index.w:
                raise ValueError(
                    f"SeedExtractor: w={w!r} disagrees with index.w={index.w}"
                )
            self.k = index.k
            self.w = index.w
        else:
            self.k = int(k)  # type: ignore[arg-type]
            self.w = int(w)  # type: ignore[arg-type]
        self._index: MinimapIndex | None = index
        self._n_tiles = int(n_tiles)
        self._prefer_npu = bool(prefer_npu)
        self._npu_op: Any | None = None  # lazy
        self.last_run: SeedExtractionResult | None = None

    # ----- index plumbing -------------------------------------------- #

    @property
    def index(self) -> MinimapIndex:
        if self._index is None:
            raise RuntimeError(
                "SeedExtractor has no index attached; pass one to the "
                "constructor or call .attach_index()."
            )
        return self._index

    def attach_index(self, index: MinimapIndex) -> None:
        if int(index.k) != self.k or int(index.w) != self.w:
            raise ValueError(
                f"index (k={index.k}, w={index.w}) disagrees with "
                f"extractor (k={self.k}, w={self.w})"
            )
        self._index = index

    # ----- NPU op plumbing ------------------------------------------- #

    def _resolve_npu_op(self):
        """Lazy-construct the minimizer NPU op the first time we need it."""
        if self._npu_op is not None:
            return self._npu_op
        if not self._prefer_npu:
            return None
        try:
            from bionpu.kernels.genomics.minimizer import BionpuMinimizer

            op = BionpuMinimizer(
                k=self.k, w=self.w, n_tiles=self._n_tiles
            )
            if not op.artifacts_present():
                return None
            self._npu_op = op
            return op
        except Exception:  # noqa: BLE001 — degrade to oracle path
            return None

    # ----- entry point ----------------------------------------------- #

    def __call__(
        self,
        *,
        packed_query: np.ndarray,
        n_bases: int,
        timeout_s: float = 600.0,
    ) -> SeedExtractionResult:
        """Run NPU minimizer extraction + host seed lookup for one query.

        Args:
            packed_query: uint8 1-D packed-2-bit query (per
                :func:`bionpu.data.kmer_oracle.pack_dna_2bit`).
            n_bases: number of bases packed (length of the unpacked
                query string). Used by the oracle fallback.
            timeout_s: subprocess timeout for the silicon path.

        Returns:
            :class:`SeedExtractionResult` with seeds + timing breakdown.
        """
        if self._index is None:
            raise RuntimeError(
                "SeedExtractor: no index attached; call attach_index() "
                "or pass one to the constructor."
            )

        op = self._resolve_npu_op()

        # ----- query minimizer extraction ----- #
        t_npu_0 = time.monotonic()
        if op is not None:
            mzs = op(packed_seq=packed_query, top_n=0, timeout_s=timeout_s)
            used_npu = True
        else:
            from bionpu.data.minimizer_oracle import extract_minimizers_packed

            mzs = extract_minimizers_packed(
                packed_query,
                n_bases=int(n_bases),
                k=self.k,
                w=self.w,
            )
            used_npu = False
        npu_wall = time.monotonic() - t_npu_0

        # ----- host-side seed lookup ----- #
        t_lk_0 = time.monotonic()
        seeds = query_to_seeds_from_minimizers(mzs, self._index)
        lookup_wall = time.monotonic() - t_lk_0

        result = SeedExtractionResult(
            seeds=seeds,
            n_query_minimizers=len(mzs),
            npu_wall_s=float(npu_wall),
            lookup_wall_s=float(lookup_wall),
            used_npu=used_npu,
        )
        self.last_run = result
        return result
