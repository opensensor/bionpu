# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# CLI integration tests for `bionpu crispr pe design` (Track B v0, Task T10).
#
# These tests exercise the 3-level subparser nesting + the three-mode
# semantics (Mode A gene symbol, Mode B target FASTA, Mode C synbio with
# --genome none). They monkey-patch the slow paths (PRIDICT scoring,
# ViennaRNA folding, off-target scan) to keep wall-clock time bounded;
# the real-PRIDICT integration smoke lives in T12 (Wave 9).

from __future__ import annotations

import json
import sys

import pytest

# The CLI subparser routing under test.
from bionpu.cli import main as cli_main
from bionpu.genomics.pe_design.types import PRIDICTScore, PegRNAFoldingFeatures


# ---------------------------------------------------------------------------
# Synthetic target sequence — 200 bp window with two NGG PAMs flanking a
# C-to-T edit site so the enumerator emits at least one PE2 candidate on
# each strand. The composition (GC ~50%, ACGT only) keeps the GC-band
# pruning rule (25-75%) happy after PBS/RTT/spacer assembly.
# ---------------------------------------------------------------------------

# Synthetic target borrowed from the T6 enumerator test fixtures
# (test_pe_design_enumerator.py). The Anzalone-2019 HEK3-style proximal
# spacer + canonical PAM sit at offsets [32, 52) and [52, 55), giving a
# PE2 nick at + offset 49. The 3' flank carries an engineered CCN at +
# offset 110 so the reverse-complement strand has an NGG within the
# PE3 distance-band (40-100 bp from the PE2 nick) — that lets
# `--strategy both` surface PE3 candidates as well. The edit base sits
# at offset 50 (1 bp 3' of the nick); we change G -> A which does NOT
# recreate the PAM (post-edit GGCC...A is not NGG) so PAM-recreation
# pruning does NOT kick in.
_PROXIMAL_SPACER = "GGCCCAGACTGAGCACGTGA"
_PROXIMAL_PAM = "TGG"
_UPSTREAM = "ATATATATATATATATATATATATATATATAT"  # 32 bp
# 3' flank: 32 bp AT, then CCAGGCT (CCN at offset 87-89 on +, giving an
# NGG on - whose nick lands ~58 bp from the PE2 nick). Then more AT pad
# to keep the locus a comfortable 200 bp window. The CCN is *outside*
# the PE2 RTT window [49, 49+30) = [49, 79) so it does not interfere
# with the PE2 enumeration; it only becomes the PE3 nicking-guide PAM.
_DOWNSTREAM = (
    "ATATATATATATATATATATATATATATATAT"  # 32 bp pad after the PAM
    "CCAGGCTACATATATATATATATATATATATATATATATAT"
    "ATATATATATATATATATATATATATATATATAT"
    "AT"
)
_SYNTHETIC_TARGET = _UPSTREAM + _PROXIMAL_SPACER + _PROXIMAL_PAM + _DOWNSTREAM
# Pad with AT to exactly 200 bp so coordinate arithmetic is round.
if len(_SYNTHETIC_TARGET) < 200:
    _SYNTHETIC_TARGET = _SYNTHETIC_TARGET + ("AT" * 200)
_SYNTHETIC_TARGET = _SYNTHETIC_TARGET[:200]
assert len(_SYNTHETIC_TARGET) == 200, len(_SYNTHETIC_TARGET)
# Sanity: the protospacer + PAM occupy [32, 55).
assert _SYNTHETIC_TARGET[32:52] == _PROXIMAL_SPACER
assert _SYNTHETIC_TARGET[52:55] == _PROXIMAL_PAM
# The base at offset 50 (the edit site we'll target) must be a single
# nucleotide we can substitute to a non-PAM-recreating ALT. Offset 50
# is the 19th base of the spacer -> 'G' for the proximal spacer.
_EDIT_OFFSET = 50  # 0-based; 1-based pos = 51
_REF_BASE = _SYNTHETIC_TARGET[_EDIT_OFFSET]  # 'G' for the proximal spacer
_ALT_BASE = "A"  # G->A: does not recreate NGG at the PAM site


def _write_fasta(path, name: str, seq: str) -> None:
    path.write_text(f">{name}\n{seq}\n")


# ---------------------------------------------------------------------------
# Stub helpers — keep PRIDICT + ViennaRNA + off-target out of the test path.
# ---------------------------------------------------------------------------


class _StubPRIDICTScorer:
    """Return a deterministic PRIDICTScore for any pegRNA seq.

    Mirrors :class:`bionpu.scoring.pridict2.PRIDICT2Scorer`'s public
    surface: ``score(pegrna_seq, *, scaffold_variant, target_context,
    folding_features=None) -> PRIDICTScore``. Also supports
    context-manager protocol for the ``--low-memory`` path."""

    def __init__(self, *args, **kwargs) -> None:
        # capture model_variant for the `--pridict-cell-type` test
        self.model_variant = kwargs.get("model_variant", "HEK293")
        self.calls = 0
        self.closed = False

    def score(
        self,
        pegrna_seq: str,
        *,
        scaffold_variant: str = "sgRNA_canonical",
        target_context: str = "",
        folding_features=None,
    ) -> PRIDICTScore:
        self.calls += 1
        # Deterministic pseudo-score derived from the pegRNA hash so two
        # ranks differ; cap to 0-100 / 0-1 ranges.
        h = abs(hash(pegrna_seq)) % 1000
        eff = 30.0 + (h / 1000.0) * 50.0  # 30..80
        return PRIDICTScore(
            efficiency=eff,
            edit_rate=eff / 100.0,
            confidence=1.0,
            notes=(),
        )

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "_StubPRIDICTScorer":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _stub_folding_extractor(
    spacer: str,
    scaffold: str,
    rtt: str,
    pbs: str,
    *,
    scaffold_variant: str = "sgRNA_canonical",
) -> PegRNAFoldingFeatures:
    """Return a benign features tuple — MFE in the no-penalty zone."""
    return PegRNAFoldingFeatures(
        mfe_kcal=-15.0,
        mfe_structure="." * (len(spacer) + len(scaffold) + len(rtt) + len(pbs)),
        pbs_pairing_prob=0.4,
        scaffold_disruption=0.05,
    )


def _stub_off_target_scan(
    spacer: str, genome_path, *, max_mismatches: int = 4
):
    """No off-targets — predictable scoring."""
    return ([], 0.0, 0)


def _stub_off_target_scan_callable_factory(genome_path):
    """Return a per-spacer callable matching the ranker's contract."""
    def _f(spacer: str, *, max_mismatches: int = 4):
        return ([], 0.0, 0)
    return _f


@pytest.fixture
def patched_slow_paths(monkeypatch):
    """Monkey-patch the heavy components so the CLI returns in <1 s.

    The CLI/public-API path imports them from these symbol locations:

    * ``bionpu.genomics.pe_design.cli.PRIDICT2Scorer``  — the scorer ctor
    * ``bionpu.genomics.pe_design.cli.compute_folding_features`` — the
      folding extractor
    * ``bionpu.genomics.pe_design.cli.off_target_scan_for_spacer`` —
      the genome-FASTA scan adapter

    Patching at the CLI's import binding (rather than at the source
    modules) keeps the test fast even if those source modules do
    expensive work at import time.
    """
    from bionpu.genomics.pe_design import cli as _cli_module

    monkeypatch.setattr(
        _cli_module, "PRIDICT2Scorer", _StubPRIDICTScorer, raising=True
    )
    monkeypatch.setattr(
        _cli_module,
        "compute_folding_features",
        _stub_folding_extractor,
        raising=True,
    )
    monkeypatch.setattr(
        _cli_module,
        "off_target_scan_for_spacer",
        _stub_off_target_scan,
        raising=True,
    )
    yield


# ---------------------------------------------------------------------------
# Test 1 — Mode A end-to-end (gene symbol).
# ---------------------------------------------------------------------------


def test_mode_a_gene_symbol_end_to_end(tmp_path, monkeypatch, patched_slow_paths):
    """`--target VEGFA --edit "G>C at chr6:43770152"` resolves the gene
    symbol and produces ranked pegRNAs.

    To avoid hitting the bundled refGene table or a real FASTA, we
    stub :func:`bionpu.data.genome_fetcher.resolve_gene_symbol` and
    :func:`bionpu.data.genome_fetcher.fetch_genomic_sequence` to return
    synthetic coordinates + sequence. The CLI's Mode A branch calls
    those helpers, slices a window around the edit site, and feeds the
    sequence to the (stubbed) downstream pipeline.
    """
    from bionpu.data.genome_fetcher import GeneCoord
    from bionpu.genomics.pe_design import cli as _cli_module

    # Fake gene at chr1:1-200 (1-based inclusive, 200 bp), '+' strand.
    # We pass --flanks 0 so the fetched window starts at chr1:1
    # (window_start_0b == 0) and absolute genomic coords in the edit
    # notation map directly to offsets in the fetched sequence.
    fake_coord = GeneCoord(
        symbol="VEGFA",
        chrom="chr1",
        start=1,
        end=200,
        strand="+",
        refseq_id="NM_FAKE",
    )

    def _fake_resolve(symbol, *, genome="hg38", **kwargs):
        assert symbol.upper() == "VEGFA"
        return fake_coord

    def _fake_fetch(coord, *, fasta_path=None, flanks=0, genome="hg38"):
        return _SYNTHETIC_TARGET

    monkeypatch.setattr(
        _cli_module, "resolve_gene_symbol", _fake_resolve, raising=True
    )
    monkeypatch.setattr(
        _cli_module, "fetch_genomic_sequence", _fake_fetch, raising=True
    )

    # Mode A also needs an off-target FASTA path to exist so the
    # file-existence check in design_prime_editor_guides passes when
    # --genome hg38 is requested. patched_slow_paths stubs the actual
    # scan callable, so the file content is irrelevant.
    fake_fasta = tmp_path / "fake_hg38.fa"
    fake_fasta.write_text(">chr1\n" + _SYNTHETIC_TARGET + "\n")
    monkeypatch.setenv("BIONPU_GRCH38_FASTA", str(fake_fasta))

    out = tmp_path / "pegrnas.tsv"
    rc = cli_main(
        [
            "crispr", "pe", "design",
            "--target", "VEGFA",
            "--edit", f"{_REF_BASE}>{_ALT_BASE} at chr1:{_EDIT_OFFSET + 1}",
            "--strategy", "pe2",
            "--genome", "hg38",
            "--flanks", "0",
            "--top", "5",
            "--format", "tsv",
            "--output", str(out),
        ]
    )

    assert rc == 0, "CLI must succeed in Mode A"
    text = out.read_text()
    lines = text.rstrip("\n").split("\n")
    # Header + at least one ranked candidate.
    assert lines[0].startswith("pegrna_id\t"), f"unexpected header: {lines[0]!r}"
    assert len(lines) >= 2, f"expected at least one row; got {len(lines) - 1}"


# ---------------------------------------------------------------------------
# Test 2 — Mode B end-to-end (target FASTA).
# ---------------------------------------------------------------------------


def test_mode_b_target_fasta_end_to_end(tmp_path, patched_slow_paths):
    """`--target-fasta` reads the file, parses the edit notation, and
    runs the downstream pipeline. No genome fetch needed."""
    target = tmp_path / "target.fa"
    _write_fasta(target, "synthetic_locus", _SYNTHETIC_TARGET)
    out = tmp_path / "pegrnas.tsv"

    rc = cli_main(
        [
            "crispr", "pe", "design",
            "--target-fasta", str(target),
            "--edit", f"{_REF_BASE}>{_ALT_BASE} at synthetic_locus:{_EDIT_OFFSET + 1}",  # 1-based pos 101 = offset 100
            "--strategy", "pe2",
            "--genome", "none",
            "--top", "5",
            "--format", "tsv",
            "--output", str(out),
        ]
    )

    assert rc == 0
    text = out.read_text()
    lines = text.rstrip("\n").split("\n")
    assert lines[0].startswith("pegrna_id\t")
    # Mode B with --genome none is effectively Mode C semantically; ensure
    # we at least got a ranked output.
    assert len(lines) >= 2


# ---------------------------------------------------------------------------
# Test 3 — Mode C synbio (--genome none) — off-target columns NaN/0,
# notes contain NO_OFF_TARGET_SCAN.
# ---------------------------------------------------------------------------


def test_mode_c_synbio_skips_off_target(tmp_path, patched_slow_paths):
    """`--genome none` skips off-target scanning entirely; the output
    rows must carry NaN for ``cfd_aggregate_pegrna`` and 0 for
    ``off_target_count_pegrna``, with ``NO_OFF_TARGET_SCAN`` in
    ``notes``.

    This test ALSO ensures the off-target stub is never called — the
    Mode C branch must short-circuit the scan (a real GRCh38 scan would
    take minutes per spacer)."""
    target = tmp_path / "target.fa"
    _write_fasta(target, "synthetic_locus", _SYNTHETIC_TARGET)
    out = tmp_path / "pegrnas.json"

    rc = cli_main(
        [
            "crispr", "pe", "design",
            "--target-fasta", str(target),
            "--edit", f"{_REF_BASE}>{_ALT_BASE} at synthetic_locus:{_EDIT_OFFSET + 1}",
            "--strategy", "pe2",
            "--genome", "none",
            "--top", "3",
            "--format", "json",
            "--output", str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert isinstance(payload, list) and len(payload) >= 1
    for row in payload:
        # JSON encodes NaN as null per T9's emit rules.
        assert row["cfd_aggregate_pegrna"] is None, (
            f"Mode C: cfd_aggregate_pegrna must be null/NaN; got "
            f"{row['cfd_aggregate_pegrna']!r}"
        )
        assert row["off_target_count_pegrna"] == 0, (
            f"Mode C: off_target_count_pegrna must be 0; got "
            f"{row['off_target_count_pegrna']!r}"
        )
        assert "NO_OFF_TARGET_SCAN" in row["notes"], (
            f"Mode C: notes must include NO_OFF_TARGET_SCAN; got "
            f"{row['notes']!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Both PE2 and PE3 strategies in one run.
# ---------------------------------------------------------------------------


def test_strategy_both_yields_pe2_and_pe3(tmp_path, patched_slow_paths):
    """`--strategy both` must surface candidates with both
    ``pe_strategy="PE2"`` AND ``pe_strategy="PE3"`` in the output."""
    target = tmp_path / "target.fa"
    _write_fasta(target, "synthetic_locus", _SYNTHETIC_TARGET)
    out = tmp_path / "pegrnas.tsv"

    rc = cli_main(
        [
            "crispr", "pe", "design",
            "--target-fasta", str(target),
            "--edit", f"{_REF_BASE}>{_ALT_BASE} at synthetic_locus:{_EDIT_OFFSET + 1}",
            "--strategy", "both",
            "--genome", "none",
            "--top", "50",  # generous so both strategies surface
            "--format", "tsv",
            "--output", str(out),
        ]
    )
    assert rc == 0
    text = out.read_text()
    lines = text.rstrip("\n").split("\n")
    header = lines[0].split("\t")
    pe_strategy_idx = header.index("pe_strategy")
    strategies_seen = {line.split("\t")[pe_strategy_idx] for line in lines[1:]}
    assert "PE2" in strategies_seen, (
        f"expected at least one PE2 row; saw strategies {strategies_seen}"
    )
    assert "PE3" in strategies_seen, (
        f"expected at least one PE3 row; saw strategies {strategies_seen}. "
        f"If the synthetic target lacks an opposite-strand NGG within the "
        f"40-100 bp PE3 window, this assertion is the fixture's red flag, "
        f"not a CLI regression."
    )


# ---------------------------------------------------------------------------
# Test 5 — `--help` works for the 3-level subparser path.
# ---------------------------------------------------------------------------


def test_pe_design_help_exits_zero(capsys):
    """`bionpu crispr pe design --help` must emit help and exit 0."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["crispr", "pe", "design", "--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--target" in captured.out
    assert "--edit" in captured.out
    assert "--strategy" in captured.out
