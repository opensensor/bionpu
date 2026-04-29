# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""UCSC-backed gene-symbol -> genomic-coordinate resolver.

This module replaces the 20-gene hardcoded ``_RESOLVE_GENE_TO_LOCUS``
whitelist that v1 of ``bionpu crispr design`` and ``bionpu be design``
shipped with, per PRD-guide-design v0.2 §1.1's "v2 deferral" gap.

Strategy: bundled refGene subset
--------------------------------

A curated subset of UCSC's hg38 ``refGene.txt.gz`` is shipped inside
the package at ``bionpu-public/data/refGene_hg38.tsv.gz`` (~7 KB,
~360 genes covering the 20-gene CRISPOR validation set + a curated
extension of well-studied disease/cancer/drug-target loci).

* On first call we decompress the subset into a module-level dict
  keyed by uppercase symbol. Subsequent calls hit that dict.
* If the requested gene is missing AND a network refresh is allowed,
  ``download_refgene_hg38`` can pull the upstream UCSC
  ``refGene.txt.gz`` (~50 MB) and rebuild the local cache. Network
  download is **opt-in** (controlled by the ``allow_network`` flag);
  v1's default is offline-only.
* Genomic sequence fetch reuses the existing in-tree FASTA reader
  (``bionpu.genomics.crispr_design.slice_chrom_from_fasta``); v1 does
  not introduce a 2bit-format dependency.

Public API
----------

* :class:`GeneCoord` — resolved gene record (1-based-inclusive, UCSC
  convention; downstream callers convert to 0-based as needed).
* :func:`resolve_gene_symbol` — symbol -> :class:`GeneCoord`.
* :func:`fetch_genomic_sequence` — coord -> uppercase ACGT slice from
  a local hg38 FASTA.
* :func:`download_refgene_hg38` — opt-in upstream refresh.
* :exc:`GeneSymbolNotFound` — raised when a symbol is unknown and
  network refresh is disabled (or the refresh failed).

CLI integrations
----------------

``bionpu.genomics.crispr_design.resolve_target`` calls this module's
``resolve_gene_symbol`` from inside its Mode A branch.
``bionpu.genomics.be_design.target_resolver`` exposes a thin wrapper
for the BE CLI's ``--target SYMBOL`` mode.

v2 deferrals (filed at the bottom of this module)
-------------------------------------------------

* HGNC alias resolution (``P53`` -> ``TP53`` etc.).
* Multi-organism (mm10, GRCh37 fallback).
* Polled UCSC DAS network refresh.
* True 2bit format reader (saves ~2.6 GB on disk vs FASTA).
"""

from __future__ import annotations

import gzip
import importlib.resources
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = [
    "GeneCoord",
    "GeneSymbolNotFound",
    "REFGENE_HG38_URL",
    "default_cache_dir",
    "download_refgene_hg38",
    "fetch_genomic_sequence",
    "iter_known_symbols",
    "load_refgene",
    "resolve_gene_symbol",
]

# UCSC mirror for the upstream refGene.txt.gz (used by
# `download_refgene_hg38`; not invoked by default).
REFGENE_HG38_URL = (
    "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refGene.txt.gz"
)
REFGENE_HG19_URL = (
    "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz"
)


# ---------------------------------------------------------------------------
# Public dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GeneCoord:
    """A resolved gene-symbol -> genomic-coordinate mapping.

    Coordinates follow the UCSC / NCBI convention: ``start`` and
    ``end`` are 1-based inclusive on the forward strand, regardless
    of ``strand``. Downstream callers (e.g. the in-process FASTA
    slicer) convert to 0-based half-open as needed.
    """

    symbol: str
    chrom: str
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    strand: str  # '+' or '-'
    refseq_id: str

    @property
    def length(self) -> int:
        """Span in bp (end - start + 1)."""
        return self.end - self.start + 1

    def as_zero_based_half_open(self) -> tuple[str, int, int]:
        """Convert to ``(chrom, start_0b, end_0b_exclusive)``."""
        return self.chrom, self.start - 1, self.end


class GeneSymbolNotFound(KeyError):
    """Raised when a gene symbol is not present in the resolver.

    The exception message includes a hint about how to either retry
    with ``allow_network=True`` (to pull the full refGene file) or
    pass an explicit ``--fasta`` + ``--target chr:start-end`` Mode B
    invocation.
    """


# ---------------------------------------------------------------------------
# Cache management.
# ---------------------------------------------------------------------------


def default_cache_dir() -> Path:
    """Return the default on-disk cache root.

    Honors ``$BIONPU_GENOME_CACHE`` if set; otherwise
    ``~/.bionpu/genome_cache``. The directory is created on demand.
    """
    env = os.environ.get("BIONPU_GENOME_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".bionpu" / "genome_cache"


# In-process cache: genome -> {symbol_upper -> GeneCoord}.
_REFGENE_CACHE: dict[str, dict[str, GeneCoord]] = {}


def _bundled_refgene_path(genome: str) -> Path | None:
    """Resolve the bundled refGene TSV that ships inside the package.

    Returns None if no bundle exists for the requested genome (which
    today is true for every genome other than ``hg38``).
    """
    if genome != "hg38":
        return None
    # Try installed package data first.
    try:
        ref = importlib.resources.files("bionpu") / ".." / ".." / ".." / "data" / "refGene_hg38.tsv.gz"
        candidate = Path(str(ref)).resolve()
        if candidate.is_file():
            return candidate
    except (TypeError, ModuleNotFoundError, FileNotFoundError):
        pass
    # Editable install / source-tree fallback: walk up from this file.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "refGene_hg38.tsv.gz"
        if candidate.is_file():
            return candidate
        # bionpu-public/ root
        candidate2 = parent / "bionpu-public" / "data" / "refGene_hg38.tsv.gz"
        if candidate2.is_file():
            return candidate2
    return None


def _user_cache_refgene_path(genome: str, cache_dir: Path | None) -> Path:
    """Resolve the user-cache refGene path for ``genome``."""
    base = cache_dir if cache_dir is not None else default_cache_dir()
    return base / f"refGene_{genome}.tsv"


def _parse_refgene_tsv(text: Iterable[str]) -> dict[str, GeneCoord]:
    """Parse a refGene TSV (our bundled format).

    Header row + comment lines (``#``) + empty lines are tolerated.
    Returns symbol(uppercase) -> :class:`GeneCoord`.
    """
    out: dict[str, GeneCoord] = {}
    seen_header = False
    for raw in text:
        line = raw.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if not seen_header and cols[0].strip().lower() == "symbol":
            seen_header = True
            continue
        if len(cols) < 6:
            # Skip malformed rows but don't crash; UCSC rows have
            # 16 cols and we only need a subset.
            continue
        symbol = cols[0].strip().upper()
        chrom = cols[1].strip()
        try:
            start = int(cols[2])
            end = int(cols[3])
        except ValueError:
            continue
        strand = cols[4].strip() or "+"
        refseq_id = cols[5].strip()
        # If the same symbol appears multiple times (UCSC's refGene
        # has one row per RefSeq accession), keep the FIRST entry
        # (the one with the longest span tends to come first after
        # we sort, but our bundled subset is already deduplicated).
        if symbol in out:
            continue
        out[symbol] = GeneCoord(
            symbol=symbol,
            chrom=chrom,
            start=start,
            end=end,
            strand=strand,
            refseq_id=refseq_id,
        )
    return out


def _parse_upstream_refgene(text: Iterable[str]) -> dict[str, GeneCoord]:
    """Parse UCSC's upstream ``refGene.txt`` format.

    Format (16 columns, no header): bin, name, chrom, strand, txStart,
    txEnd, cdsStart, cdsEnd, exonCount, exonStarts, exonEnds, score,
    name2, cdsStartStat, cdsEndStat, exonFrames.

    We pick out: name2 (HGNC symbol), chrom, strand, txStart, txEnd,
    name (RefSeq accession). Coordinates are 0-based half-open in
    UCSC; we promote to 1-based inclusive for :class:`GeneCoord`.

    For a given symbol, we keep the row with the longest span (best
    proxy for "the canonical transcript" without paywalled MANE).
    """
    spans: dict[str, GeneCoord] = {}
    spans_len: dict[str, int] = {}
    for raw in text:
        line = raw.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 13:
            continue
        try:
            tx_start_0b = int(cols[4])
            tx_end_0b_excl = int(cols[5])
        except ValueError:
            continue
        chrom = cols[2].strip()
        strand = cols[3].strip() or "+"
        refseq_id = cols[1].strip()
        symbol = cols[12].strip().upper()
        if not symbol or not chrom:
            continue
        # Drop alternate haplotypes / unplaced contigs to keep the
        # cache small and deterministic.
        if "_" in chrom or chrom.startswith("chrUn"):
            continue
        span = tx_end_0b_excl - tx_start_0b
        if symbol in spans_len and spans_len[symbol] >= span:
            continue
        spans[symbol] = GeneCoord(
            symbol=symbol,
            chrom=chrom,
            start=tx_start_0b + 1,  # 0-based -> 1-based inclusive
            end=tx_end_0b_excl,  # 0-based-excl -> 1-based-incl: same number
            strand=strand,
            refseq_id=refseq_id,
        )
        spans_len[symbol] = span
    return spans


def load_refgene(
    *,
    genome: str = "hg38",
    cache_dir: Path | None = None,
    allow_user_cache: bool = True,
) -> dict[str, GeneCoord]:
    """Load (and cache) the refGene table for ``genome``.

    Resolution order:

    1. In-process cache (already-loaded dict).
    2. User cache directory (``$BIONPU_GENOME_CACHE/refGene_{genome}.tsv``)
       — populated by an earlier ``download_refgene_hg38`` call.
    3. Bundled subset that ships inside the package (hg38 only).

    Returns the mapping; raises ``FileNotFoundError`` if no source
    can be located (e.g. caller asked for hg19 but never invoked
    ``download_refgene_hg38(genome="hg19")``).
    """
    if genome in _REFGENE_CACHE:
        return _REFGENE_CACHE[genome]

    # Step 2: user cache.
    if allow_user_cache:
        user_path = _user_cache_refgene_path(genome, cache_dir)
        if user_path.is_file():
            opener = gzip.open if user_path.suffix == ".gz" else open
            with opener(user_path, "rt", encoding="utf-8") as fh:
                # User-cache files written by `download_refgene_hg38`
                # use the upstream format; bundled files use ours.
                text = fh.read().splitlines()
            mapping = _parse_upstream_refgene(text)
            if not mapping:
                # Maybe it's the bundled-format TSV (manual paste).
                mapping = _parse_refgene_tsv(text)
            if mapping:
                _REFGENE_CACHE[genome] = mapping
                return mapping

    # Step 3: bundled subset.
    bundled = _bundled_refgene_path(genome)
    if bundled is not None:
        with gzip.open(bundled, "rt", encoding="utf-8") as fh:
            text = fh.read().splitlines()
        mapping = _parse_refgene_tsv(text)
        if mapping:
            _REFGENE_CACHE[genome] = mapping
            return mapping

    raise FileNotFoundError(
        f"no refGene source available for genome={genome!r}. "
        f"Tried in-process cache, user cache "
        f"({_user_cache_refgene_path(genome, cache_dir)}), and the "
        f"bundled subset. Call download_refgene_hg38("
        f"genome={genome!r}, allow_network=True) to populate the "
        f"user cache from UCSC."
    )


def iter_known_symbols(*, genome: str = "hg38") -> Iterable[str]:
    """Yield uppercase gene symbols known to the resolver."""
    return iter(load_refgene(genome=genome).keys())


# ---------------------------------------------------------------------------
# Public resolution API.
# ---------------------------------------------------------------------------


def resolve_gene_symbol(
    symbol: str,
    *,
    genome: str = "hg38",
    cache_dir: Path | None = None,
    allow_network: bool = False,
) -> GeneCoord:
    """Resolve a gene symbol to genomic coordinates.

    Parameters
    ----------
    symbol:
        HGNC-ish gene symbol. Case-insensitive (we uppercase). Aliases
        (``P53`` -> ``TP53``, ``HER2`` -> ``ERBB2``) are NOT resolved
        in v1 — see the v2 deferrals at the top of this module.
    genome:
        Reference build. v1 ships ``"hg38"`` (alias ``"GRCh38"``);
        ``"hg19"`` / ``"GRCh37"`` work only after a manual
        ``download_refgene_hg38(genome="hg19", allow_network=True)``.
    cache_dir:
        Override the on-disk cache root. Defaults to
        ``$BIONPU_GENOME_CACHE`` or ``~/.bionpu/genome_cache``.
    allow_network:
        If True and the symbol is missing from every local source,
        attempt to refresh the cache from UCSC. Default False (the
        bundled subset covers the v1 acceptance set).

    Returns
    -------
    GeneCoord

    Raises
    ------
    GeneSymbolNotFound
        If the symbol is unknown after every resolution step.
    """
    sym = symbol.strip().upper()
    if not sym:
        raise GeneSymbolNotFound(f"gene symbol must be non-empty; got {symbol!r}")
    # Normalize "GRCh38" -> "hg38" / "GRCh37" -> "hg19".
    genome_norm = _normalize_genome(genome)

    # Try the local cache first.
    try:
        table = load_refgene(genome=genome_norm, cache_dir=cache_dir)
        if sym in table:
            return table[sym]
    except FileNotFoundError:
        table = {}

    # Optional network refresh.
    if allow_network:
        download_refgene_hg38(
            genome=genome_norm,
            cache_dir=cache_dir,
            allow_network=True,
        )
        table = load_refgene(genome=genome_norm, cache_dir=cache_dir)
        if sym in table:
            return table[sym]

    known = sorted(table.keys())
    hint = ", ".join(known[:10]) + ("..." if len(known) > 10 else "")
    raise GeneSymbolNotFound(
        f"gene symbol {symbol!r} not found in {genome_norm} refGene "
        f"(searched {len(known)} known symbols). "
        f"Known sample: {hint}. "
        f"Hints: (a) check the casing/spelling; (b) HGNC aliases are "
        f"not resolved in v1 (deferred); (c) call "
        f"resolve_gene_symbol({symbol!r}, allow_network=True) to "
        f"refresh from UCSC; (d) pass an explicit "
        f"chr:start-end target instead."
    )


def _normalize_genome(genome: str) -> str:
    g = genome.strip().lower()
    if g in ("grch38", "hg38", "grch38.p14", "grch38.p13"):
        return "hg38"
    if g in ("grch37", "hg19"):
        return "hg19"
    return g


def fetch_genomic_sequence(
    coord: GeneCoord,
    *,
    fasta_path: Path | str | None = None,
    flanks: int = 0,
    genome: str = "hg38",
) -> str:
    """Fetch the genomic sequence for ``coord`` (with optional flanks).

    v1 reads from a local FASTA (the user's hg38.fa or equivalent).
    The 2bit-format path is a v2 deferral.

    Parameters
    ----------
    coord:
        A :class:`GeneCoord` (typically from ``resolve_gene_symbol``).
    fasta_path:
        Path to the reference FASTA. If None, we try
        ``$BIONPU_GRCH38_FASTA`` then ``data_cache/genomes/grch38/hg38.fa``.
    flanks:
        Number of additional bases to fetch on either side of the
        coordinate. Useful for guide-design pipelines that need 23 nt
        of 3' protospacer + PAM context (typical: 50-1000 bp).
    genome:
        Reference build label (informational; the actual reference
        is the FASTA you pass).

    Returns
    -------
    str
        Uppercase ACGT[N] string. Length == ``coord.length + 2 * flanks``
        unless the locus is near a contig edge (in which case
        ``slice_chrom_from_fasta`` raises a ``ValueError``).
    """
    # Lazy import to avoid a circular dep at import time.
    from bionpu.genomics.crispr_design import slice_chrom_from_fasta

    resolved_fasta = _resolve_fasta_path(fasta_path)
    zero_start = max(0, coord.start - 1 - flanks)
    zero_end = coord.end + flanks  # 1-based-incl + (excl flank) -> 0-based-excl
    return slice_chrom_from_fasta(
        fasta_path=resolved_fasta,
        chrom=coord.chrom,
        start=zero_start,
        end=zero_end,
    )


def _resolve_fasta_path(fasta_path: Path | str | None) -> Path:
    if fasta_path is not None:
        p = Path(fasta_path)
        if not p.is_file():
            raise ValueError(f"reference FASTA not found: {p}")
        return p
    env = os.environ.get("BIONPU_GRCH38_FASTA")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    candidate = Path("data_cache/genomes/grch38/hg38.fa")
    if candidate.is_file():
        return candidate
    raise ValueError(
        "no reference FASTA available. Pass --fasta <path> or set "
        "BIONPU_GRCH38_FASTA, or place the reference at "
        "data_cache/genomes/grch38/hg38.fa."
    )


# ---------------------------------------------------------------------------
# Network refresh (opt-in).
# ---------------------------------------------------------------------------


def download_refgene_hg38(
    *,
    genome: str = "hg38",
    cache_dir: Path | None = None,
    allow_network: bool = False,
    timeout_s: float = 60.0,
) -> Path:
    """Refresh the user cache from UCSC's upstream ``refGene.txt.gz``.

    This is **opt-in** — by default ``allow_network=False`` and the
    function raises immediately. The bundled subset covers the v1
    acceptance gates; users who need full coverage call this once per
    machine to hydrate ``$BIONPU_GENOME_CACHE``.

    Returns the local cache path; raises ``RuntimeError`` on network
    failure (with the URL, HTTP status, and reason in the message,
    per CLAUDE.md non-negotiables).
    """
    if not allow_network:
        raise PermissionError(
            "download_refgene_hg38 was called without "
            "allow_network=True. v1's default is offline-only "
            "(the bundled subset covers BRCA1/TP53/etc.); pass "
            "allow_network=True to opt in to the upstream "
            "refresh from UCSC."
        )
    genome_norm = _normalize_genome(genome)
    if genome_norm == "hg38":
        url = REFGENE_HG38_URL
    elif genome_norm == "hg19":
        url = REFGENE_HG19_URL
    else:
        raise ValueError(
            f"download_refgene_hg38: unsupported genome={genome!r} "
            f"(supported: hg38, hg19)."
        )

    base = cache_dir if cache_dir is not None else default_cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    dest_gz = base / f"refGene_{genome_norm}.txt.gz"
    dest_tsv = base / f"refGene_{genome_norm}.tsv"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bionpu/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"download_refgene_hg38: network refresh failed for "
            f"url={url!r}: {exc!r}. The bundled subset is still "
            f"available — fall back to the offline path."
        ) from exc

    dest_gz.write_bytes(data)
    # Decompress to plain TSV so subsequent loads are fast.
    with gzip.open(dest_gz, "rt", encoding="utf-8") as fh:
        text_lines = fh.read().splitlines()
    dest_tsv.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    # Reset in-process cache so the next load picks up the new file.
    _REFGENE_CACHE.pop(genome_norm, None)
    return dest_tsv
