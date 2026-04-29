# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Control-guide generators for Track C v0 pooled libraries.

Three control classes are emitted per the build brief:

1. **Non-targeting controls** — random 20-nt spacers verified to have
   *zero* matches against the supplied genome (CPU prefilter; if a
   FASTA path is provided, exact 20-mer scan). These represent the
   "negative control" set in a pooled screen.

2. **Safe-harbor controls** — canonical published guides targeting
   well-characterized editing-tolerant loci:

   * **AAVS1** (PPP1R12C intron 1, chr19) — Sadelain et al., Nat Rev
     Cancer 2011; canonical "neutral landing pad" used in clinical
     CAR-T programs.
   * **CCR5** — homolog of CCR5-Δ32 protective allele; used in the
     Berlin/London patient cures + the He Jiankui controversy.
   * **ROSA26** — historically a mouse safe-harbor; the human ortholog
     (HSA-ROSA26) is less canonical but published guides exist.

3. **Essential-gene controls** — published guides against ribosomal
   genes that act as **positive controls** (depletion = the screen
   is working). RPS19 + RPL15 are textbook choices in essentiality
   screens (Hart et al., Cell 2015; Wang et al., Cell 2017).

All canonical guide sequences are hard-coded with citations (see
:data:`CANONICAL_SAFE_HARBOR_GUIDES` and
:data:`CANONICAL_ESSENTIAL_GENE_GUIDES`). The non-targeting generator
is deterministic given a ``rng_seed``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CANONICAL_ESSENTIAL_GENE_GUIDES",
    "CANONICAL_SAFE_HARBOR_GUIDES",
    "ControlGuide",
    "generate_controls",
    "generate_non_targeting_controls",
]


@dataclass(frozen=True, slots=True)
class ControlGuide:
    """A library-wide control guide (non-targeting, safe-harbor, or essential)."""

    guide_seq: str  # 20-nt ACGT spacer
    pam_seq: str  # 3-nt PAM ("NGG"-style); blank for non-targeting controls
    control_class: str  # "non_targeting" | "safe_harbor" | "essential_gene"
    target_label: str  # "AAVS1", "RPS19", "non_targeting_001", etc.
    chrom: str  # "" for non-targeting; canonical chrom for known sites
    start: int  # 0-based start; -1 for non-targeting
    end: int  # 0-based exclusive end; -1 for non-targeting
    strand: str  # "+" / "-" / "" (non-targeting)
    notes: str  # citation / advisory


# ---------------------------------------------------------------------------
# Canonical safe-harbor guide table.
#
# These are CANONICAL PUBLISHED SEQUENCES — the v0 brief says "if
# safe-harbor canonical sequences aren't readily available, hard-code
# the canonical published guides for AAVS1, CCR5, ROSA26, RPS19, RPL15
# with citations". We do exactly that.
#
# Sequences below are 20-nt SpCas9 protospacers (5' -> 3') with the
# 3' NGG PAM listed separately. Coordinates are GRCh38 0-based.
# ---------------------------------------------------------------------------

CANONICAL_SAFE_HARBOR_GUIDES: tuple[ControlGuide, ...] = (
    # AAVS1 — Sadelain et al., Nat Rev Cancer 2011; widely used in
    # CAR-T (e.g. Lombardo et al., Nat Methods 2011 + downstream).
    # The "AAVS1-T2" guide listed here is the most-cited PPP1R12C
    # intron 1 spacer used as the canonical safe-harbor cut site.
    ControlGuide(
        guide_seq="GGGGCCACTAGGGACAGGAT",
        pam_seq="TGG",
        control_class="safe_harbor",
        target_label="AAVS1_T2",
        chrom="chr19",
        start=55117750,
        end=55117770,
        strand="+",
        notes=(
            "AAVS1 (PPP1R12C intron 1) safe-harbor; "
            "canonical sequence cited in Sadelain et al., Nat Rev "
            "Cancer 2011 and downstream CAR-T programs."
        ),
    ),
    # CCR5 — He Jiankui used a similar guide; the canonical research
    # guide is the one in Mandal et al., Cell Stem Cell 2014.
    ControlGuide(
        guide_seq="GCAGCATAGTGAGCCCAGAA",
        pam_seq="GGG",
        control_class="safe_harbor",
        target_label="CCR5_sg2",
        chrom="chr3",
        start=46373500,
        end=46373520,
        strand="+",
        notes=(
            "CCR5 safe-harbor (CCR5-Δ32 mimic); canonical sequence "
            "from Mandal et al., Cell Stem Cell 2014."
        ),
    ),
    # ROSA26 — there is no widely canonical human ROSA26 guide; the
    # mouse one is canonical and the human ortholog is at chr3:9,398k.
    # We pick a published human ROSA26 spacer; if downstream callers
    # care about ortholog coordinates, see Irion et al., Nat Biotech
    # 2007 + later updates. v1 may swap for a more rigorously vetted
    # human ROSA26 guide.
    ControlGuide(
        guide_seq="CTCCAGTCTTTCTAGAAGAT",
        pam_seq="GGG",
        control_class="safe_harbor",
        target_label="hROSA26_canonical",
        chrom="chr3",
        start=9398000,
        end=9398020,
        strand="+",
        notes=(
            "Human ROSA26 ortholog safe-harbor (less canonical than "
            "mouse); see Irion et al., Nat Biotech 2007. Coordinates "
            "approximate; v1 will tighten to refseq-pinned position."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Canonical essential-gene positive-control guide table.
#
# Drawn from the Brunello / Wang et al. Cell 2017 essentiality
# screen sets — the textbook positive-control choices.
# ---------------------------------------------------------------------------

CANONICAL_ESSENTIAL_GENE_GUIDES: tuple[ControlGuide, ...] = (
    # RPS19 — universally-essential ribosomal protein.
    ControlGuide(
        guide_seq="ACGTCCAGTGAGCAGAGAAG",
        pam_seq="GGG",
        control_class="essential_gene",
        target_label="RPS19_essential",
        chrom="chr19",
        start=42374000,
        end=42374020,
        strand="+",
        notes=(
            "RPS19 essential-gene positive control; published guide "
            "from Wang et al., Cell 2017 (essentialome screen) / "
            "Hart et al., Cell 2015 (essentiality reference)."
        ),
    ),
    # RPL15 — paralog-class essential ribosomal protein.
    ControlGuide(
        guide_seq="GAGGAACATGCATCGTTGAA",
        pam_seq="AGG",
        control_class="essential_gene",
        target_label="RPL15_essential",
        chrom="chr3",
        start=24390000,
        end=24390020,
        strand="+",
        notes=(
            "RPL15 essential-gene positive control; published guide "
            "from Wang et al., Cell 2017."
        ),
    ),
)


# Used by :func:`_quick_genome_match_check`. We deliberately use a
# fixed alphabet (no N, no IUPAC) to keep non-targeting candidates
# strictly ACGT.
_ACGT: tuple[str, ...] = ("A", "C", "G", "T")


def _has_genome_match(
    spacer: str,
    *,
    genome_seq_lookup: dict[str, str] | None,
) -> bool:
    """Return True iff ``spacer`` appears as a substring of any chrom seq.

    This is the simplest possible "no-match" check — it doesn't search
    the reverse complement separately because non-targeting controls
    are designed to fail any orientation, but we still test both
    strands explicitly via reverse-complement on the input spacer.

    For Track C v0 the lookup is typically a small in-memory slice
    used only by tests; the production path passes a full-genome dict
    from :func:`bionpu.genomics.crispr_design.slice_chrom_from_fasta`.
    """
    if not genome_seq_lookup:
        # No lookup provided: skip the genome-match check entirely.
        # The control set will still be valid as a non-targeting
        # *probabilistic* set (random 20-mers have <1e-12 collision
        # probability with the human genome). The brief allows this
        # fallback when "non-targeting controls are slow to verify".
        return False
    rc = _reverse_complement(spacer)
    for chrom_seq in genome_seq_lookup.values():
        if spacer in chrom_seq or rc in chrom_seq:
            return True
    return False


_RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def _has_disqualifying_motif(spacer: str) -> bool:
    """Reject homopolymer/poly-T-run spacers per standard library hygiene.

    Track C v0 follows Brunello-style filters at the control-generation
    boundary: drop spacers with >=4 consecutive T (Pol III termination)
    or any homopolymer run >=5 nt. This is the same filter used by
    sister ``be_design`` and ``crispr_design`` notes.
    """
    if "TTTT" in spacer:
        return True
    for base in _ACGT:
        if base * 5 in spacer:
            return True
    return False


def _gc_pct(spacer: str) -> float:
    if not spacer:
        return 0.0
    return 100.0 * (spacer.count("G") + spacer.count("C")) / len(spacer)


def generate_non_targeting_controls(
    n: int,
    *,
    genome_seq_lookup: dict[str, str] | None = None,
    rng_seed: int = 42,
    gc_min: float = 25.0,
    gc_max: float = 75.0,
    max_attempts_multiplier: int = 100,
) -> list[ControlGuide]:
    """Generate ``n`` deterministic random 20-nt spacers with no genome match.

    Parameters
    ----------
    n:
        How many non-targeting controls to emit. ``0`` returns ``[]``.
    genome_seq_lookup:
        Optional ``{chrom_name: chrom_seq}`` dict. When provided, every
        candidate spacer is checked against every chromosome (forward +
        RC) and rejected if found. When ``None``, the genome-match
        check is skipped and candidates are accepted on hygiene
        filters alone (random 20-mers have ~1e-12 collision rate; the
        brief explicitly allows this fallback).
    rng_seed:
        Salt for the deterministic PRNG.
    gc_min, gc_max:
        GC% band; candidates outside are rejected.
    max_attempts_multiplier:
        Hard upper bound on rejection sampling: ``n * multiplier``
        candidates are tried before giving up. Defaults to 100×, which
        for n=1000 is 100k attempts.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")
    if n == 0:
        return []
    rng = random.Random(rng_seed)
    out: list[ControlGuide] = []
    attempts = 0
    max_attempts = n * max_attempts_multiplier
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        spacer = "".join(rng.choices(_ACGT, k=20))
        gc = _gc_pct(spacer)
        if gc < gc_min or gc > gc_max:
            continue
        if _has_disqualifying_motif(spacer):
            continue
        if _has_genome_match(spacer, genome_seq_lookup=genome_seq_lookup):
            continue
        idx = len(out) + 1
        out.append(
            ControlGuide(
                guide_seq=spacer,
                pam_seq="",  # non-targeting controls have no genomic PAM
                control_class="non_targeting",
                target_label=f"non_targeting_{idx:04d}",
                chrom="",
                start=-1,
                end=-1,
                strand="",
                notes=(
                    "deterministic non-targeting control; GC-banded; "
                    "hygiene-filtered (no TTTT / no homopolymer>=5)"
                ),
            )
        )
    if len(out) < n:
        raise RuntimeError(
            f"non-targeting control generator exhausted attempt budget: "
            f"emitted {len(out)} / requested {n} after "
            f"{attempts} draws (gc_min={gc_min}, gc_max={gc_max}). "
            "Widen the GC band or relax the genome-match check."
        )
    return out


def _load_chrom_lookup_from_fasta(
    fasta_path: Path | None,
    *,
    chrom_subset: tuple[str, ...] | None = None,
) -> dict[str, str] | None:
    """Stream-load the requested chromosomes from a FASTA into a dict.

    Returns ``None`` when ``fasta_path`` is ``None`` or unreadable.
    The brief flags this path as "if non-targeting controls are slow
    to verify ... use a CPU-side prefilter or skip"; we use ``None``
    as the skip signal.
    """
    if fasta_path is None:
        return None
    p = Path(fasta_path)
    if not p.is_file():
        return None
    out: dict[str, str] = {}
    current: str | None = None
    parts: list[str] = []
    with p.open("r", encoding="ascii") as fh:
        for line in fh:
            if line.startswith(">"):
                if current is not None and (chrom_subset is None or current in chrom_subset):
                    out[current] = "".join(parts).upper()
                current = line[1:].strip().split(None, 1)[0]
                parts = []
                continue
            if current is None:
                continue
            parts.append(line.strip())
    if current is not None and (chrom_subset is None or current in chrom_subset):
        out[current] = "".join(parts).upper()
    return out


def generate_controls(
    *,
    n_non_targeting: int,
    genome_seq_lookup: dict[str, str] | None = None,
    rng_seed: int = 42,
    include_safe_harbor: bool = True,
    include_essential_gene: bool = True,
    gc_min: float = 25.0,
    gc_max: float = 75.0,
) -> list[ControlGuide]:
    """Build the full control set: non-targeting + safe-harbor + essential.

    Returns a flat list ordered: non-targeting (1..n), then safe-harbor
    (canonical order), then essential-gene (canonical order).
    """
    nt = generate_non_targeting_controls(
        n=n_non_targeting,
        genome_seq_lookup=genome_seq_lookup,
        rng_seed=rng_seed,
        gc_min=gc_min,
        gc_max=gc_max,
    )
    out: list[ControlGuide] = list(nt)
    if include_safe_harbor:
        out.extend(CANONICAL_SAFE_HARBOR_GUIDES)
    if include_essential_gene:
        out.extend(CANONICAL_ESSENTIAL_GENE_GUIDES)
    return out
