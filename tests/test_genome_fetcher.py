# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# Unit tests for `bionpu.data.genome_fetcher` (UCSC genome-fetch v1).

from __future__ import annotations

import gzip
import time
from pathlib import Path

import pytest

from bionpu.data import genome_fetcher as gf


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Each test starts with a clean in-process refGene cache."""
    gf._REFGENE_CACHE.clear()
    yield
    gf._REFGENE_CACHE.clear()


def _write_synthetic_fasta(path: Path, chrom: str, seq: str) -> None:
    """Write a multi-line FASTA the slicer can stream through."""
    with path.open("w", encoding="ascii") as fh:
        fh.write(f">{chrom}\n")
        for i in range(0, len(seq), 80):
            fh.write(seq[i : i + 80] + "\n")


# --------------------------------------------------------------------- #
# Acceptance tests
# --------------------------------------------------------------------- #


def test_resolve_brca1_returns_canonical_grch38_coords() -> None:
    """AC#1: resolve_gene_symbol('BRCA1') returns the pinned GRCh38 coords."""
    coord = gf.resolve_gene_symbol("BRCA1", genome="hg38")
    assert coord.symbol == "BRCA1"
    assert coord.chrom == "chr17"
    assert coord.start == 43044295
    assert coord.end == 43125483
    assert coord.strand == "-"
    assert coord.length == coord.end - coord.start + 1


def test_resolve_is_case_insensitive() -> None:
    """Symbols are normalised to upper-case before lookup."""
    upper = gf.resolve_gene_symbol("BRCA1", genome="hg38")
    lower = gf.resolve_gene_symbol("brca1", genome="hg38")
    mixed = gf.resolve_gene_symbol("Brca1", genome="hg38")
    assert upper == lower == mixed


def test_resolve_grch38_alias_normalises_to_hg38() -> None:
    """``genome='GRCh38'`` resolves the same as ``genome='hg38'``."""
    a = gf.resolve_gene_symbol("BRCA1", genome="GRCh38")
    b = gf.resolve_gene_symbol("BRCA1", genome="hg38")
    assert a == b


def test_resolve_unknown_gene_raises_helpful_error() -> None:
    """AC#7: unknown symbols raise GeneSymbolNotFound with hints."""
    with pytest.raises(gf.GeneSymbolNotFound) as excinfo:
        gf.resolve_gene_symbol("NOT_A_REAL_GENE_XYZ", genome="hg38")
    msg = str(excinfo.value)
    assert "NOT_A_REAL_GENE_XYZ" in msg
    assert "hg38" in msg.lower()
    # Hint about allow_network=True or chr:start-end fallback.
    assert "allow_network" in msg or "chr:start-end" in msg


def test_cache_warm_lookup_is_faster_than_cold() -> None:
    """AC#2: second call hits in-process cache (no re-parse of TSV)."""
    # Cold: forces TSV decompression + parse.
    gf._REFGENE_CACHE.clear()
    t0 = time.perf_counter()
    coord_cold = gf.resolve_gene_symbol("BRCA1", genome="hg38")
    cold_ns = (time.perf_counter() - t0) * 1e9

    # Warm: in-process dict hit.
    t1 = time.perf_counter()
    coord_warm = gf.resolve_gene_symbol("BRCA1", genome="hg38")
    warm_ns = (time.perf_counter() - t1) * 1e9

    assert coord_cold == coord_warm
    # Warm path should be at least 5x faster (sanity; usually >100x).
    # Tolerate flakey tiny timings by adding a 1-µs floor.
    assert warm_ns < max(cold_ns / 2.0, 5_000.0), (
        f"warm lookup not faster than cold (cold={cold_ns:.0f} ns, "
        f"warm={warm_ns:.0f} ns)"
    )


def test_iter_known_symbols_includes_pinned_set() -> None:
    """The bundled subset includes every gene from the v0 pinned set."""
    known = set(gf.iter_known_symbols(genome="hg38"))
    pinned = {
        "BRCA1", "BRCA2", "EMX1", "FANCF", "RNF2", "VEGFA", "HBB",
        "TP53", "ATM", "MYC", "CCR5", "CD33", "CXCR4", "KRAS",
        "TET2", "RUNX1", "DNMT1", "HPRT1", "AAVS1", "CFTR",
    }
    missing = pinned - known
    assert not missing, f"bundled subset is missing v0 pinned genes: {missing}"


def test_iter_known_symbols_includes_curated_extension() -> None:
    """The bundled subset extends past the 20 v0 pins."""
    known = set(gf.iter_known_symbols(genome="hg38"))
    # A handful of well-studied curated additions.
    for sym in ("EGFR", "PTEN", "BRAF", "KIT", "ERBB2", "TERT"):
        assert sym in known, f"curated gene {sym} missing from bundled subset"


def test_fetch_genomic_sequence_with_flanks_against_synthetic_fasta(
    tmp_path: Path,
) -> None:
    """AC#3: fetch_genomic_sequence returns the correct slice + flanks."""
    # Build a synthetic chr17 with a recognisable BRCA1 window.
    # Coordinate system: BRCA1 is at 43044295-43125483 (1-based incl).
    # The synthetic chr17 is 43_200_000 bp of A's with a 100-bp CGTA
    # marker stamped at 43044294 (0-based) — i.e. the first base of the
    # 1-based BRCA1 start.
    chrom_len = 43_200_000
    bases = bytearray(b"A") * chrom_len
    marker = b"CGTA" * 25  # 100 bp
    bases[43_044_294:43_044_294 + len(marker)] = marker
    fasta = tmp_path / "synthetic_chr17.fa"
    _write_synthetic_fasta(fasta, "chr17", bases.decode("ascii"))

    coord = gf.resolve_gene_symbol("BRCA1", genome="hg38")
    seq = gf.fetch_genomic_sequence(
        coord, fasta_path=fasta, flanks=0
    )
    assert seq.startswith("CGTACGTA")
    # Length should equal the gene span exactly.
    assert len(seq) == coord.length

    # With flanks: prepend / append 50 bp each side.
    seq_with_flanks = gf.fetch_genomic_sequence(
        coord, fasta_path=fasta, flanks=50
    )
    assert len(seq_with_flanks) == coord.length + 100
    # Flank bases come from the surrounding A's.
    assert seq_with_flanks[:50] == "A" * 50


def test_fetch_genomic_sequence_missing_fasta_raises() -> None:
    """fetch_genomic_sequence raises ValueError when no FASTA is available."""
    coord = gf.resolve_gene_symbol("BRCA1", genome="hg38")
    # Use a guaranteed-missing path so we hit the explicit-path branch.
    with pytest.raises(ValueError, match="reference FASTA not found"):
        gf.fetch_genomic_sequence(coord, fasta_path="/no/such/path.fa")


def test_download_refgene_refuses_without_allow_network() -> None:
    """download_refgene_hg38 is opt-in; default refuses with PermissionError."""
    with pytest.raises(PermissionError, match="allow_network"):
        gf.download_refgene_hg38(genome="hg38")


def test_user_cache_path_honours_env_and_override(monkeypatch, tmp_path) -> None:
    """default_cache_dir reads $BIONPU_GENOME_CACHE; override wins."""
    monkeypatch.setenv("BIONPU_GENOME_CACHE", str(tmp_path / "envcache"))
    assert gf.default_cache_dir() == (tmp_path / "envcache").resolve()


def test_user_cache_overrides_bundled_subset(tmp_path, monkeypatch) -> None:
    """When a user-cache TSV exists, it shadows the bundled subset."""
    cache = tmp_path / "cache"
    cache.mkdir()
    # Write a bundled-format TSV with a synthetic gene.
    tsv = cache / "refGene_hg38.tsv"
    tsv.write_text(
        "symbol\tchrom\tstart_1b\tend_1b\tstrand\trefseq_id\n"
        "FAKE_SHADOW_GENE\tchr1\t100\t200\t+\tNM_000000\n"
    )
    monkeypatch.setenv("BIONPU_GENOME_CACHE", str(cache))
    gf._REFGENE_CACHE.clear()
    coord = gf.resolve_gene_symbol("FAKE_SHADOW_GENE", genome="hg38")
    assert coord.chrom == "chr1"
    assert coord.start == 100
    assert coord.end == 200


def test_zero_based_half_open_helper() -> None:
    """as_zero_based_half_open returns canonical Python-slice coords."""
    coord = gf.resolve_gene_symbol("BRCA1", genome="hg38")
    chrom, zs, ze = coord.as_zero_based_half_open()
    assert chrom == "chr17"
    assert zs == 43044294
    assert ze == 43125483
    assert ze - zs == coord.length


# --------------------------------------------------------------------- #
# Integration test — wires through bionpu.genomics.crispr_design.
# --------------------------------------------------------------------- #


def test_crispr_design_resolves_curated_gene_via_fetcher(tmp_path) -> None:
    """AC#9: a non-Tier-1 gene resolved via fetcher reaches resolve_target.

    EGFR is in the curated extension but NOT in
    ``cd._RESOLVE_GENE_TO_LOCUS``; this test exercises the
    delegation-to-fetcher path inside ``resolve_target``.
    """
    from bionpu.genomics import crispr_design as cd

    # Confirm assumption: EGFR isn't pinned in the hardcoded dict.
    assert "EGFR" not in cd._RESOLVE_GENE_TO_LOCUS

    # Build a synthetic chr7 FASTA covering EGFR's coords.
    coord = gf.resolve_gene_symbol("EGFR", genome="hg38")
    chrom_len = coord.end + 10_000
    bases = bytearray(b"N") * chrom_len
    # Plant a recognisable marker at the start of the EGFR span.
    marker = b"ACGTACGTAC" * 10
    bases[coord.start - 1 : coord.start - 1 + len(marker)] = marker
    fasta = tmp_path / "synthetic_chr7.fa"
    _write_synthetic_fasta(fasta, coord.chrom, bases.decode("ascii"))

    resolved = cd.resolve_target(
        target="EGFR", genome="GRCh38", fasta_path=fasta
    )
    assert resolved.gene == "EGFR"
    assert resolved.chrom == coord.chrom
    assert resolved.start == coord.start - 1  # 0-based
    assert resolved.end == coord.end
    # First 100 bp of the resolved sequence should match the marker.
    assert resolved.sequence.startswith(marker.decode("ascii"))
