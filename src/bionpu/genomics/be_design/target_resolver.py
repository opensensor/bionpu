# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Track A v0 / v1 — gene-symbol target resolver for ``bionpu be design``.

Thin wrapper around :mod:`bionpu.data.genome_fetcher` that lets the BE
CLI accept ``--target SYMBOL --fasta hg38.fa`` (Mode A) in addition
to the existing ``--target-fasta`` (Mode C) path.

Public API
----------

* :func:`resolve_be_target` — symbol -> ``(name, sequence)``.

Strategy: identical to the CRISPR-design Mode A path, but exposed as
a separate function so the BE CLI's small-blast-radius edit doesn't
have to reach into ``bionpu.genomics.crispr_design`` (the v1 brief
explicitly forbids large refactors of crispr_design.py).
"""

from __future__ import annotations

from pathlib import Path

from bionpu.data.genome_fetcher import (
    GeneCoord,
    GeneSymbolNotFound,
    fetch_genomic_sequence,
    resolve_gene_symbol,
)

__all__ = [
    "GeneSymbolNotFound",
    "resolve_be_target",
]


def resolve_be_target(
    target: str,
    *,
    genome: str = "hg38",
    fasta_path: Path | str | None = None,
    flanks: int = 0,
) -> tuple[str, str, GeneCoord]:
    """Resolve a BE design target to ``(name, sequence, coord)``.

    Parameters
    ----------
    target:
        HGNC-ish gene symbol (e.g. ``"BRCA1"``). Case-insensitive.
    genome:
        Reference build label. Pass ``"hg38"`` (default) /
        ``"GRCh38"``. Anything else requires ``download_refgene_hg38``
        to have hydrated the user cache for that build.
    fasta_path:
        Path to the reference FASTA. If None, falls back to
        ``$BIONPU_GRCH38_FASTA`` then
        ``data_cache/genomes/grch38/hg38.fa``.
    flanks:
        Number of flanking bp to fetch on either side of the gene
        body. BE design typically wants 100-1000 bp of flank so the
        protospacer scan has context outside the gene span.

    Returns
    -------
    (name, sequence, coord)
        ``name`` is ``"{symbol}|{chrom}:{start}-{end}"``; ``sequence``
        is uppercase ACGT[N]; ``coord`` is the resolved
        :class:`GeneCoord`.

    Raises
    ------
    GeneSymbolNotFound
        Propagated from the underlying fetcher.
    ValueError
        If the FASTA cannot be located or the slice is out of range.
    """
    coord = resolve_gene_symbol(target, genome=genome)
    seq = fetch_genomic_sequence(
        coord, fasta_path=fasta_path, flanks=flanks, genome=genome
    )
    name = f"{coord.symbol}|{coord.chrom}:{coord.start}-{coord.end}"
    return name, seq, coord
