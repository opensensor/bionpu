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

"""Track B v0 — Edit-specifier parser (Task T2 of
``track-b-pegrna-design-plan.md``).

This module accepts two notation families and emits a single
:class:`bionpu.genomics.pe_design.types.EditSpec` record in genome
``+``-strand orientation:

* **Simple notation** (free-form, ergonomic for the CLI):

  * ``"C>T at chr1:100"`` — single-base substitution at 1-based position
    100 on chr1.
  * ``"AAA>GGG at chr1:100..102"`` — multi-base delins.
  * ``"insAGT at chr1:100"`` — pure insertion BEFORE position 100.
  * ``"del chr1:100..105"`` — pure deletion (inclusive 1-based range).

* **HGVS notation** (the standard the upstream literature speaks):

  * ``"chr1:g.100C>T"`` — genomic flavor; no transcript needed.
  * ``"chr1:g.100_102delCGTinsAAA"`` — genomic delins.
  * ``"chr1:g.100dupC"`` — genomic duplication (encoded as an
    insertion of ``C`` immediately AFTER position 100).
  * ``"chr1:g.100_105del"`` — genomic deletion.
  * ``"NM_007294.4:c.5266dupC"`` — transcript-relative; resolves
    through :mod:`bionpu.data.genome_fetcher`'s bundled refGene subset
    and applies a strand-aware complement when the transcript's gene
    is on the minus strand.

For minus-strand transcripts, the HGVS alt allele is described in
transcript orientation (which equals the minus strand). To emit
:class:`EditSpec` in genome ``+``-strand orientation the parser
complements the alt allele AND records ``strand="-"`` so downstream
consumers (T6 enumerator, T9 TSV) can re-orient as needed.

v1 limitations (filed as follow-ons; see plan §T2 acceptance note)
------------------------------------------------------------------

* Transcript ``c.NNN`` -> genomic-coord resolution is best-effort —
  the bundled refGene TSV ships only the transcript span (txStart /
  txEnd), NOT the per-exon CDS map. v1 treats ``c.NNN`` as a 1-based
  offset from the transcript's CDS start surrogate (which we proxy
  with the transcript's 5' end on the gene's strand). This is correct
  for the strand-flip + complement test but does NOT give exact
  genomic coordinates for genes with introns — the v1 enumerator
  flags the resulting :class:`EditSpec` for downstream review.
* Lookup-by-RefSeq-ID (``NM_007294.4`` -> ``BRCA1`` symbol) is also
  best-effort: the parser walks the loaded refGene table looking for
  a row whose ``refseq_id`` field starts with the supplied accession
  (version stripped). A dedicated
  ``resolve_gene_symbol_from_refseq_id`` is filed as a v1 follow-on
  in :mod:`bionpu.data.genome_fetcher`.
* HGVS structural variants (``inv``, ``[..]`` allele combos,
  positional uncertainty ``(..)``) are NOT supported in v0 and raise
  :class:`UnsupportedHGVSVariant`.
"""

from __future__ import annotations

import re
from pathlib import Path

from bionpu.genomics.pe_design.pegrna_constants import MAX_EDIT_LENGTH_BP
from bionpu.genomics.pe_design.types import EditSpec

__all__ = [
    "parse_edit_spec",
    "RefMismatchError",
    "EditTooLargeError",
    "UnsupportedHGVSVariant",
]


# --------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------- #


class RefMismatchError(ValueError):
    """The ``ref`` allele in the notation does not match the FASTA at
    the resolved coordinates. Indicates a stale notation, a wrong
    genome build, or a coordinate-system off-by-one."""


class EditTooLargeError(ValueError):
    """The ref or alt allele exceeds
    :data:`bionpu.genomics.pe_design.pegrna_constants.MAX_EDIT_LENGTH_BP`.
    PRIDICT 2.0 (Mathis 2024) is trained on edits up to ~50 bp; larger
    edits are out of distribution and rejected at parse time."""


class UnsupportedHGVSVariant(ValueError):
    """The HGVS form (e.g. inversion, structural variant, allele combo)
    is not in the v0 supported subset. Filed as a v1 deferral; see
    plan §T2 + this module's docstring for the supported set."""


# --------------------------------------------------------------------- #
# Constants for IUPAC + complementing.
# --------------------------------------------------------------------- #


_DNA_ALPHABET = frozenset("ACGTUacgtu")
_COMPLEMENT_TABLE = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")


def _normalise_dna(seq: str) -> str:
    """Uppercase + replace U->T (HGVS-on-mRNA may carry U)."""
    s = seq.upper().replace("U", "T")
    if not all(c in "ACGTN" for c in s):
        raise ValueError(f"non-DNA character in allele {seq!r}")
    return s


def _complement(seq: str) -> str:
    """Watson-Crick complement (NOT reverse). Used for the minus-strand
    HGVS alt-allele flip; the caller decides whether to also reverse."""
    return seq.translate(_COMPLEMENT_TABLE).upper()


def _reverse_complement(seq: str) -> str:
    return _complement(seq)[::-1]


# --------------------------------------------------------------------- #
# Length-cap enforcement (used by all parse paths before we return).
# --------------------------------------------------------------------- #


def _enforce_length_cap(ref_seq: str, alt_seq: str, notation: str) -> None:
    if len(ref_seq) > MAX_EDIT_LENGTH_BP or len(alt_seq) > MAX_EDIT_LENGTH_BP:
        raise EditTooLargeError(
            f"edit length exceeds MAX_EDIT_LENGTH_BP="
            f"{MAX_EDIT_LENGTH_BP} (ref={len(ref_seq)}, "
            f"alt={len(alt_seq)}). PRIDICT 2.0 (Mathis 2024) is "
            f"trained on edits up to ~50 bp; larger edits are out "
            f"of distribution. Notation: {notation!r}."
        )


# --------------------------------------------------------------------- #
# Reference-allele validation (optional; only when genome_path is given).
# --------------------------------------------------------------------- #


def _validate_ref_against_genome(
    *,
    chrom: str,
    start_0b: int,
    end_0b_excl: int,
    expected_ref: str,
    genome_path: Path,
    notation: str,
) -> None:
    """Read the reference at the resolved coords and assert it matches
    ``expected_ref``. Pure insertions (``expected_ref == ""``) skip the
    check (there's nothing to validate)."""
    if not expected_ref:
        return
    # Lazy import to mirror genome_fetcher's own pattern (avoids any
    # circular-import surprise during module init).
    from bionpu.genomics.crispr_design import slice_chrom_from_fasta

    actual = slice_chrom_from_fasta(
        fasta_path=genome_path,
        chrom=chrom,
        start=start_0b,
        end=end_0b_excl,
    ).upper()
    if actual != expected_ref.upper():
        raise RefMismatchError(
            f"ref allele mismatch at {chrom}:{start_0b + 1}-"
            f"{end_0b_excl}: notation says {expected_ref!r} but "
            f"FASTA says {actual!r}. Notation: {notation!r}. "
            f"Possible causes: stale notation, wrong genome build, "
            f"off-by-one in the coordinate, or the FASTA is mismatched "
            f"to the notation's coordinate system."
        )


# --------------------------------------------------------------------- #
# Simple-notation parser
# --------------------------------------------------------------------- #
#
# Grammar (whitespace-tolerant):
#   <ref>><alt> at <chrom>:<pos>           (single-base substitution)
#   <ref>><alt> at <chrom>:<start>..<end>  (multi-base delins)
#   ins<alt>   at <chrom>:<pos>            (pure insertion BEFORE pos)
#   del        <chrom>:<start>..<end>      (pure deletion, inclusive)
# 1-based inclusive coordinates throughout (UCSC convention).


_RE_SIMPLE_SUB_SINGLE = re.compile(
    r"""^\s*
        (?P<ref>[ACGTUacgtu]+)\s*>\s*(?P<alt>[ACGTUacgtu]+)
        \s+at\s+
        (?P<chrom>[A-Za-z0-9_]+):(?P<pos>\d+)
        \s*$""",
    re.VERBOSE,
)

_RE_SIMPLE_SUB_RANGE = re.compile(
    r"""^\s*
        (?P<ref>[ACGTUacgtu]+)\s*>\s*(?P<alt>[ACGTUacgtu]+)
        \s+at\s+
        (?P<chrom>[A-Za-z0-9_]+):(?P<start>\d+)\.\.(?P<end>\d+)
        \s*$""",
    re.VERBOSE,
)

_RE_SIMPLE_INS = re.compile(
    r"""^\s*ins(?P<alt>[ACGTUacgtu]+)
        \s+at\s+
        (?P<chrom>[A-Za-z0-9_]+):(?P<pos>\d+)
        \s*$""",
    re.VERBOSE,
)

_RE_SIMPLE_DEL = re.compile(
    r"""^\s*del\s+
        (?P<chrom>[A-Za-z0-9_]+):(?P<start>\d+)\.\.(?P<end>\d+)
        \s*$""",
    re.VERBOSE,
)


def _parse_simple(notation: str, *, genome_path: Path | None) -> EditSpec:
    """Try each simple-notation regex in turn. Raises ``ValueError`` if
    no pattern matches (the caller maps that to a clean error)."""
    s = notation.strip()

    # Substitution, single position (e.g. "C>T at chr1:100").
    m = _RE_SIMPLE_SUB_SINGLE.match(s)
    if m:
        ref = _normalise_dna(m["ref"])
        alt = _normalise_dna(m["alt"])
        chrom = m["chrom"]
        pos_1b = int(m["pos"])
        if len(ref) != 1:
            raise ValueError(
                f"single-position simple substitution requires a 1-bp "
                f"ref allele; got {len(ref)} bp in {notation!r}. Use "
                f"the range form ``REF>ALT at chrom:start..end`` for "
                f"multi-base substitutions."
            )
        start_0b = pos_1b - 1
        end_0b_excl = pos_1b
        edit_type = "substitution" if len(ref) == len(alt) else "substitution"
        spec = EditSpec(
            chrom=chrom,
            start=start_0b,
            end=end_0b_excl,
            ref_seq=ref,
            alt_seq=alt,
            edit_type=edit_type,
            notation_used=notation,
            strand="+",
        )
        _enforce_length_cap(ref, alt, notation)
        if genome_path is not None:
            _validate_ref_against_genome(
                chrom=chrom,
                start_0b=start_0b,
                end_0b_excl=end_0b_excl,
                expected_ref=ref,
                genome_path=genome_path,
                notation=notation,
            )
        return spec

    # Substitution, range (e.g. "CGT>AAA at chr1:100..102").
    m = _RE_SIMPLE_SUB_RANGE.match(s)
    if m:
        ref = _normalise_dna(m["ref"])
        alt = _normalise_dna(m["alt"])
        chrom = m["chrom"]
        start_1b = int(m["start"])
        end_1b = int(m["end"])
        if start_1b > end_1b:
            raise ValueError(
                f"simple substitution range start > end: "
                f"{start_1b}..{end_1b} in {notation!r}"
            )
        if len(ref) != end_1b - start_1b + 1:
            raise ValueError(
                f"simple substitution: ref length {len(ref)} does not "
                f"match coordinate span {end_1b - start_1b + 1} in "
                f"{notation!r}"
            )
        start_0b = start_1b - 1
        end_0b_excl = end_1b
        spec = EditSpec(
            chrom=chrom,
            start=start_0b,
            end=end_0b_excl,
            ref_seq=ref,
            alt_seq=alt,
            edit_type="substitution",
            notation_used=notation,
            strand="+",
        )
        _enforce_length_cap(ref, alt, notation)
        if genome_path is not None:
            _validate_ref_against_genome(
                chrom=chrom,
                start_0b=start_0b,
                end_0b_excl=end_0b_excl,
                expected_ref=ref,
                genome_path=genome_path,
                notation=notation,
            )
        return spec

    # Insertion (e.g. "insAGT at chr1:100").
    m = _RE_SIMPLE_INS.match(s)
    if m:
        alt = _normalise_dna(m["alt"])
        chrom = m["chrom"]
        pos_1b = int(m["pos"])
        # Convention: insertion lands BEFORE position pos_1b in 1-based
        # space; that's a zero-width range at 0-based (pos_1b - 1, pos_1b - 1).
        start_0b = pos_1b - 1
        end_0b_excl = pos_1b - 1
        spec = EditSpec(
            chrom=chrom,
            start=start_0b,
            end=end_0b_excl,
            ref_seq="",
            alt_seq=alt,
            edit_type="insertion",
            notation_used=notation,
            strand="+",
        )
        _enforce_length_cap("", alt, notation)
        # Pure insertion: nothing to validate against the FASTA.
        return spec

    # Deletion (e.g. "del chr1:100..105").
    m = _RE_SIMPLE_DEL.match(s)
    if m:
        chrom = m["chrom"]
        start_1b = int(m["start"])
        end_1b = int(m["end"])
        if start_1b > end_1b:
            raise ValueError(
                f"simple deletion range start > end: "
                f"{start_1b}..{end_1b} in {notation!r}"
            )
        start_0b = start_1b - 1
        end_0b_excl = end_1b
        # Resolve the ref allele from the FASTA if available; else
        # leave ref_seq empty and the downstream consumers can read it
        # themselves.
        ref_seq = ""
        if genome_path is not None:
            from bionpu.genomics.crispr_design import slice_chrom_from_fasta

            ref_seq = slice_chrom_from_fasta(
                fasta_path=genome_path,
                chrom=chrom,
                start=start_0b,
                end=end_0b_excl,
            ).upper()
        spec = EditSpec(
            chrom=chrom,
            start=start_0b,
            end=end_0b_excl,
            ref_seq=ref_seq,
            alt_seq="",
            edit_type="deletion",
            notation_used=notation,
            strand="+",
        )
        # Length cap on ref only (alt is empty).
        _enforce_length_cap(
            ref_seq if ref_seq else "X" * (end_0b_excl - start_0b),
            "",
            notation,
        )
        return spec

    raise ValueError(
        f"could not parse simple-notation edit specifier {notation!r}. "
        f"Supported forms: 'REF>ALT at chrom:pos', "
        f"'REF>ALT at chrom:start..end', 'insSEQ at chrom:pos', "
        f"'del chrom:start..end'."
    )


# --------------------------------------------------------------------- #
# HGVS parser (genomic + transcript flavors)
# --------------------------------------------------------------------- #
#
# Supported subset:
#   <chrom>:g.<pos><ref>><alt>                         (genomic sub)
#   <chrom>:g.<start>_<end>del<seq>?ins<seq>           (genomic delins)
#   <chrom>:g.<start>_<end>del<seq>?                   (genomic del)
#   <chrom>:g.<pos>dup<seq>                            (genomic dup)
#   <transcript>:c.<pos><ref>><alt>                    (transcript sub)
#   <transcript>:c.<start>_<end>del<seq>?ins<seq>      (transcript delins)
#   <transcript>:c.<start>_<end>del<seq>?              (transcript del)
#   <transcript>:c.<pos>dup<seq>                       (transcript dup)
#
# Inversions, allele combos, and positional uncertainty raise
# UnsupportedHGVSVariant.


_RE_HGVS_PREFIX = re.compile(
    r"""^\s*
        (?:(?P<acc>[A-Za-z0-9_.]+):)?
        (?P<flavor>[gcm])\.
        (?P<rest>.*\S)
        \s*$""",
    re.VERBOSE,
)

_RE_HGVS_SUB = re.compile(
    r"""^(?P<pos>\d+)
         (?P<ref>[ACGTUacgtu])>(?P<alt>[ACGTUacgtu])$""",
    re.VERBOSE,
)

_RE_HGVS_DELINS = re.compile(
    r"""^(?P<start>\d+)(?:_(?P<end>\d+))?
         del(?P<delseq>[ACGTUacgtu]*)
         ins(?P<insseq>[ACGTUacgtu]+)$""",
    re.VERBOSE,
)

_RE_HGVS_DEL = re.compile(
    r"""^(?P<start>\d+)(?:_(?P<end>\d+))?
         del(?P<delseq>[ACGTUacgtu]*)$""",
    re.VERBOSE,
)

_RE_HGVS_DUP = re.compile(
    r"""^(?P<start>\d+)(?:_(?P<end>\d+))?
         dup(?P<dupseq>[ACGTUacgtu]*)$""",
    re.VERBOSE,
)

_RE_HGVS_INV = re.compile(
    r"""^(?P<start>\d+)_(?P<end>\d+)inv""",
    re.VERBOSE,
)


def _looks_like_hgvs(notation: str) -> bool:
    """Heuristic: an HGVS spec contains ``g.``, ``c.``, or ``m.`` as a
    flavor prefix (optionally preceded by a transcript/genome accession).
    The ``X.YY`` accession dot does NOT count."""
    return bool(_RE_HGVS_PREFIX.match(notation.strip()))


def _resolve_transcript_to_coord(accession: str, *, genome: str):
    """Best-effort RefSeq-ID -> :class:`GeneCoord` lookup.

    Strips the version suffix (``NM_007294.4`` -> ``NM_007294``) and
    walks the loaded refGene table for a row whose ``refseq_id`` begins
    with that prefix. Filed v1 follow-on: a dedicated
    ``resolve_gene_symbol_from_refseq_id`` lives in genome_fetcher
    proper; today we walk the dict (~360 entries) in O(N) which is
    fine for v0.
    """
    from bionpu.data.genome_fetcher import GeneSymbolNotFound, load_refgene

    acc_root = accession.split(".")[0]
    table = load_refgene(genome=genome)
    for symbol, coord in table.items():
        if coord.refseq_id == acc_root:
            return coord
    # No exact hit; emit a clean error consistent with the rest of the
    # genome_fetcher API surface.
    raise GeneSymbolNotFound(
        f"RefSeq accession {accession!r} (root={acc_root!r}) is not "
        f"present in the bundled refGene_{genome}.tsv subset. v1 "
        f"deferral: a dedicated resolve_gene_symbol_from_refseq_id "
        f"is filed; today we walk the loaded refGene table for a "
        f"matching refseq_id prefix. To work around: use the "
        f"genomic-coord HGVS flavor (chrom:g.NNN...) directly."
    )


def _transcript_position_to_genomic(
    pos_1b_txn: int,
    *,
    coord,
) -> int:
    """Best-effort transcript-pos -> genomic-pos mapper (v1).

    The bundled refGene TSV ships only the transcript span (txStart /
    txEnd); we don't have the per-exon CDS map. v1 treats ``c.NNN`` as
    a 1-based offset from the transcript's 5' end on the gene's strand
    (i.e. introns + UTRs are NOT skipped). This is correct for the
    strand-flip + complement test that BRCA1 ``c.5266dupC`` exercises,
    but it's NOT a faithful c.-coordinate map. Filed as a v1 follow-on
    in this module's docstring.

    Returns a 1-based inclusive genomic position on the ``+`` strand.
    """
    if coord.strand == "+":
        # 5' end of the transcript on +/genomic = coord.start.
        return coord.start + pos_1b_txn - 1
    # Minus strand: the transcript's 5' end maps to the gene's 3' end
    # on +/genomic, i.e. coord.end. Offset moves toward smaller
    # genomic positions.
    return coord.end - pos_1b_txn + 1


def _parse_hgvs(notation: str, *, genome: str, genome_path: Path | None) -> EditSpec:
    """Parse the HGVS subset and emit a +-strand EditSpec."""
    s = notation.strip()
    m = _RE_HGVS_PREFIX.match(s)
    if not m:
        raise ValueError(
            f"HGVS notation must contain a 'g.', 'c.', or 'm.' "
            f"flavor prefix; got {notation!r}."
        )
    accession = m["acc"] or ""
    flavor = m["flavor"]
    rest = m["rest"]

    # Reject unsupported forms early.
    if _RE_HGVS_INV.match(rest):
        raise UnsupportedHGVSVariant(
            f"HGVS inversion ('inv') is not supported in v0; got "
            f"{notation!r}. Filed as v1 deferral."
        )
    if "[" in rest or "]" in rest or "(" in rest or ")" in rest:
        raise UnsupportedHGVSVariant(
            f"HGVS allele combos ('[..]') and positional-uncertainty "
            f"forms ('(..)') are not supported in v0; got {notation!r}."
        )

    # --- Resolve the (chrom, strand, anchor) per flavor. --- #

    if flavor == "g":
        # Accession (if present) must be a chrom literal (e.g. "chr1").
        if accession.lower().startswith("nm_") or accession.lower().startswith(
            "nr_"
        ) or accession.lower().startswith("nc_"):
            # NC_000017.11 etc. is technically a genomic accession but
            # we don't ship the contig-accession map; punt to v1.
            raise UnsupportedHGVSVariant(
                f"genomic-accession HGVS (e.g. NC_000017.11:g....) is "
                f"not supported in v0; please use the chr<N>:g.... "
                f"flavor. Got {notation!r}."
            )
        chrom = accession or "chr1"  # sane default for chrom-less input
        strand = "+"
        coord_anchor = None
    elif flavor == "c":
        if not accession:
            raise ValueError(
                f"transcript-relative HGVS ('c.') requires a transcript "
                f"accession prefix (e.g. 'NM_007294.4:c.5266dupC'); "
                f"got {notation!r}."
            )
        coord_anchor = _resolve_transcript_to_coord(
            accession, genome=genome
        )
        chrom = coord_anchor.chrom
        strand = coord_anchor.strand
    elif flavor == "m":
        # Mitochondrial HGVS — supported as a thin pass-through where
        # we treat positions as genomic 1-based coords on chrM, +.
        chrom = "chrM"
        strand = "+"
        coord_anchor = None
    else:  # pragma: no cover -- regex guarantees one of g|c|m
        raise ValueError(f"unknown HGVS flavor {flavor!r}")

    # Helper: map a 1-based position in HGVS-coords to 1-based genomic
    # +-strand position.
    def hgvs_pos_to_genomic_1b(p: int) -> int:
        if coord_anchor is None:
            return p
        return _transcript_position_to_genomic(p, coord=coord_anchor)

    # --- Try each variant form. --- #

    # Substitution (e.g. "100C>T").
    m_sub = _RE_HGVS_SUB.match(rest)
    if m_sub:
        pos_1b = int(m_sub["pos"])
        ref = _normalise_dna(m_sub["ref"])
        alt = _normalise_dna(m_sub["alt"])
        gpos_1b = hgvs_pos_to_genomic_1b(pos_1b)
        # For minus-strand transcripts, complement BOTH alleles (reflect
        # them onto the genomic + strand). The genomic coordinate is a
        # single base, so no reverse needed for length-1 alleles.
        if strand == "-":
            ref = _complement(ref)
            alt = _complement(alt)
        start_0b = gpos_1b - 1
        end_0b_excl = gpos_1b
        spec = EditSpec(
            chrom=chrom,
            start=start_0b,
            end=end_0b_excl,
            ref_seq=ref,
            alt_seq=alt,
            edit_type="substitution",
            notation_used=notation,
            strand=strand,
        )
        _enforce_length_cap(ref, alt, notation)
        if genome_path is not None and strand == "+":
            _validate_ref_against_genome(
                chrom=chrom,
                start_0b=start_0b,
                end_0b_excl=end_0b_excl,
                expected_ref=ref,
                genome_path=genome_path,
                notation=notation,
            )
        return spec

    # Delins (e.g. "100_102delCGTinsAAA" or "100delCinsTT").
    m_di = _RE_HGVS_DELINS.match(rest)
    if m_di:
        start_hgvs = int(m_di["start"])
        end_hgvs = int(m_di["end"]) if m_di["end"] else start_hgvs
        delseq = _normalise_dna(m_di["delseq"]) if m_di["delseq"] else ""
        insseq = _normalise_dna(m_di["insseq"])
        # For a transcript on the - strand, swap start and end after
        # the position flip so the +-strand range is well-formed
        # (start_genomic <= end_genomic).
        gstart_1b = hgvs_pos_to_genomic_1b(start_hgvs)
        gend_1b = hgvs_pos_to_genomic_1b(end_hgvs)
        if strand == "-":
            gstart_1b, gend_1b = min(gstart_1b, gend_1b), max(
                gstart_1b, gend_1b
            )
            # Complement (and reverse for multi-base alleles) the
            # alleles to land on the + strand.
            if delseq:
                delseq = _reverse_complement(delseq)
            insseq = _reverse_complement(insseq)
        ref = delseq
        alt = insseq
        start_0b = gstart_1b - 1
        end_0b_excl = gend_1b
        spec = EditSpec(
            chrom=chrom,
            start=start_0b,
            end=end_0b_excl,
            ref_seq=ref,
            alt_seq=alt,
            edit_type="substitution",
            notation_used=notation,
            strand=strand,
        )
        _enforce_length_cap(ref, alt, notation)
        if genome_path is not None and ref and strand == "+":
            _validate_ref_against_genome(
                chrom=chrom,
                start_0b=start_0b,
                end_0b_excl=end_0b_excl,
                expected_ref=ref,
                genome_path=genome_path,
                notation=notation,
            )
        return spec

    # Pure deletion (e.g. "100_105del" or "100delC").
    m_del = _RE_HGVS_DEL.match(rest)
    if m_del:
        start_hgvs = int(m_del["start"])
        end_hgvs = int(m_del["end"]) if m_del["end"] else start_hgvs
        delseq = _normalise_dna(m_del["delseq"]) if m_del["delseq"] else ""
        gstart_1b = hgvs_pos_to_genomic_1b(start_hgvs)
        gend_1b = hgvs_pos_to_genomic_1b(end_hgvs)
        if strand == "-":
            gstart_1b, gend_1b = min(gstart_1b, gend_1b), max(
                gstart_1b, gend_1b
            )
            if delseq:
                delseq = _reverse_complement(delseq)
        # If delseq wasn't provided, try to fill from the FASTA.
        ref = delseq
        start_0b = gstart_1b - 1
        end_0b_excl = gend_1b
        if not ref and genome_path is not None and strand == "+":
            from bionpu.genomics.crispr_design import slice_chrom_from_fasta

            ref = slice_chrom_from_fasta(
                fasta_path=genome_path,
                chrom=chrom,
                start=start_0b,
                end=end_0b_excl,
            ).upper()
        spec = EditSpec(
            chrom=chrom,
            start=start_0b,
            end=end_0b_excl,
            ref_seq=ref,
            alt_seq="",
            edit_type="deletion",
            notation_used=notation,
            strand=strand,
        )
        _enforce_length_cap(
            ref if ref else "X" * (end_0b_excl - start_0b),
            "",
            notation,
        )
        if ref and genome_path is not None and strand == "+":
            _validate_ref_against_genome(
                chrom=chrom,
                start_0b=start_0b,
                end_0b_excl=end_0b_excl,
                expected_ref=ref,
                genome_path=genome_path,
                notation=notation,
            )
        return spec

    # Duplication (e.g. "100dupC" or "100_102dupCGT").
    m_dup = _RE_HGVS_DUP.match(rest)
    if m_dup:
        start_hgvs = int(m_dup["start"])
        end_hgvs = int(m_dup["end"]) if m_dup["end"] else start_hgvs
        dupseq = _normalise_dna(m_dup["dupseq"]) if m_dup["dupseq"] else ""
        gstart_1b = hgvs_pos_to_genomic_1b(start_hgvs)
        gend_1b = hgvs_pos_to_genomic_1b(end_hgvs)
        if strand == "-":
            gstart_1b, gend_1b = min(gstart_1b, gend_1b), max(
                gstart_1b, gend_1b
            )
            if dupseq:
                dupseq = _reverse_complement(dupseq)
        # If dupseq is empty, try to infer from the FASTA.
        if not dupseq and genome_path is not None and strand == "+":
            from bionpu.genomics.crispr_design import slice_chrom_from_fasta

            dupseq = slice_chrom_from_fasta(
                fasta_path=genome_path,
                chrom=chrom,
                start=gstart_1b - 1,
                end=gend_1b,
            ).upper()
        # HGVS dup is encoded as an insertion of dupseq immediately
        # AFTER the END of the duplicated range (3' on +/genomic, 3'
        # on the transcript strand for c.dup).
        # 0-based zero-width range at (gend_1b, gend_1b).
        start_0b = gend_1b
        end_0b_excl = gend_1b
        spec = EditSpec(
            chrom=chrom,
            start=start_0b,
            end=end_0b_excl,
            ref_seq="",
            alt_seq=dupseq,
            edit_type="insertion",
            notation_used=notation,
            strand=strand,
        )
        _enforce_length_cap("", dupseq if dupseq else "X", notation)
        return spec

    raise ValueError(
        f"could not parse HGVS notation {notation!r} (flavor={flavor}). "
        f"Supported forms: substitution (NNN[ref]>[alt]), delins "
        f"(NNN_MMMdelXinsY), deletion (NNN_MMMdel[X]), and duplication "
        f"(NNNdup[X]). Inversions and structural variants are filed as "
        f"v1 deferrals."
    )


# --------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------- #


def parse_edit_spec(
    notation: str,
    *,
    genome_path: Path | str | None = None,
    genome: str = "hg38",
) -> EditSpec:
    """Parse an edit specifier string into an :class:`EditSpec`.

    Parameters
    ----------
    notation:
        Either simple notation (``"C>T at chr1:100"``,
        ``"insAGT at chr1:100"``, ``"del chr1:100..105"``) or HGVS
        notation (``"chr1:g.100C>T"``, ``"NM_007294.4:c.5266dupC"``).
        Case-insensitive on the allele letters; coordinates are 1-based
        inclusive (UCSC convention).
    genome_path:
        Optional path to a local hg38 FASTA. When supplied we validate
        the ref allele against the FASTA and fail with
        :class:`RefMismatchError` on disagreement. For HGVS specs on
        minus-strand transcripts, ref-validation is currently skipped
        (the transcript-pos -> genomic-pos map is best-effort in v1;
        see the module docstring).
    genome:
        Reference build. v1 ships ``"hg38"`` (alias ``"GRCh38"``).
        Forwarded to :mod:`bionpu.data.genome_fetcher`.

    Returns
    -------
    EditSpec
        In genome ``+``-strand orientation. ``strand`` carries the
        transcript strand for HGVS-resolved specs; ``"+"`` for direct
        genomic-coord specs.

    Raises
    ------
    ValueError
        Notation is unparseable.
    RefMismatchError
        ``genome_path`` was supplied and the FASTA disagrees with the
        notation's ref allele.
    EditTooLargeError
        Ref or alt allele exceeds
        :data:`pegrna_constants.MAX_EDIT_LENGTH_BP`.
    UnsupportedHGVSVariant
        HGVS form (inversion / structural variant / allele combo) is
        not in the v0 supported subset.
    """
    if not isinstance(notation, str):
        raise TypeError(
            f"parse_edit_spec: notation must be str, got "
            f"{type(notation).__name__}"
        )
    if not notation.strip():
        raise ValueError("parse_edit_spec: notation must be non-empty")

    if genome_path is not None and not isinstance(genome_path, Path):
        genome_path = Path(genome_path)

    if _looks_like_hgvs(notation):
        return _parse_hgvs(notation, genome=genome, genome_path=genome_path)
    return _parse_simple(notation, genome_path=genome_path)
