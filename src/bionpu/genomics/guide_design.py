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

"""Host-side CRISPR guide enumeration and cheap sequence filters.

This module is the CPU front-end for guide-design workflows: enumerate
PAM-valid spacer candidates from a target sequence, compute stable
per-guide metadata, and apply inexpensive filters before off-target
prefiltering or neural scoring.

The implementation intentionally mirrors the oracle scanner's strand
convention:

* ``+`` candidates use ``seq[i : i + spacer_len + pam_len]`` directly.
* ``-`` candidates use the reverse-complement of each forward window.
* ``window_start`` is the 0-based start of the forward-strand genomic
  window. For ``-`` candidates this is still the forward coordinate of
  the reverse-complemented window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from bionpu.data.kmer_oracle import canonical_kmer_2bit, kmer_mask

__all__ = [
    "GuideCandidate",
    "GuideFilter",
    "canonical_guide_key",
    "enumerate_guides",
    "gc_fraction",
    "has_low_complexity",
    "homopolymer_run",
    "matches_pam",
    "n_fraction",
    "reverse_complement",
]


_BASE_TO_2BIT = {"A": 0, "C": 1, "G": 2, "T": 3}
_VALID_BASES = frozenset("ACGT")
_RC_TABLE = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
_IUPAC = {
    "A": frozenset("A"),
    "C": frozenset("C"),
    "G": frozenset("G"),
    "T": frozenset("T"),
    "N": frozenset("ACGT"),
    "R": frozenset("AG"),
    "Y": frozenset("CT"),
    "S": frozenset("GC"),
    "W": frozenset("AT"),
    "K": frozenset("GT"),
    "M": frozenset("AC"),
    "B": frozenset("CGT"),
    "D": frozenset("AGT"),
    "H": frozenset("ACT"),
    "V": frozenset("ACG"),
}


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of an A/C/G/T/N sequence."""
    return seq.translate(_RC_TABLE)[::-1].upper()


def matches_pam(pam: str, template: str) -> bool:
    """Return true when ``pam`` matches an IUPAC PAM template.

    ``N`` and the common degenerate IUPAC symbols are accepted in the
    template. Ambiguous bases in the genomic PAM are not treated as
    wildcards, so data-side ``N`` is rejected unless the template's
    allowed set explicitly contains ``N`` (the built-in templates do not).
    """
    pam = pam.upper()
    template = template.upper()
    if len(pam) != len(template):
        raise ValueError(
            f"PAM length mismatch: observed {len(pam)} vs template {len(template)}"
        )
    for base, code in zip(pam, template, strict=True):
        allowed = _IUPAC.get(code)
        if allowed is None:
            raise ValueError(f"unsupported PAM template code {code!r}")
        if base not in allowed:
            return False
    return True


def gc_fraction(seq: str) -> float:
    """Return GC fraction over A/C/G/T bases; ``N`` is ignored."""
    seq = seq.upper()
    acgt = [c for c in seq if c in _VALID_BASES]
    if not acgt:
        return 0.0
    return sum(1 for c in acgt if c in {"G", "C"}) / len(acgt)


def n_fraction(seq: str) -> float:
    """Return the fraction of sequence characters that are ``N``."""
    if not seq:
        return 0.0
    seq = seq.upper()
    return seq.count("N") / len(seq)


def homopolymer_run(seq: str) -> int:
    """Return the longest same-base run in ``seq``."""
    if not seq:
        return 0
    best = 1
    cur = 1
    prev = seq[0].upper()
    for raw in seq[1:]:
        c = raw.upper()
        if c == prev:
            cur += 1
        else:
            best = max(best, cur)
            prev = c
            cur = 1
    return max(best, cur)


def _shannon_entropy(seq: str) -> float:
    counts: dict[str, int] = {}
    for c in seq.upper():
        if c in _VALID_BASES:
            counts[c] = counts.get(c, 0) + 1
    n = sum(counts.values())
    if n == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def has_low_complexity(seq: str, *, min_entropy_bits: float = 1.2) -> bool:
    """Return true when nucleotide entropy is below ``min_entropy_bits``."""
    return _shannon_entropy(seq) < float(min_entropy_bits)


def canonical_guide_key(spacer: str) -> int:
    """Return the canonical 2-bit key for an A/C/G/T spacer.

    This is the same ``min(forward, reverse_complement)`` key used by
    the k-mer toolkit, generalized to the guide spacer length.
    """
    spacer = spacer.upper()
    if not spacer or len(spacer) > 32:
        raise ValueError("spacer length must be in 1..32")
    key = 0
    for i, c in enumerate(spacer):
        v = _BASE_TO_2BIT.get(c)
        if v is None:
            raise ValueError(f"non-ACGT base {c!r} at spacer position {i}")
        key = (key << 2) | v
    return canonical_kmer_2bit(key & kmer_mask(len(spacer)), len(spacer))


@dataclass(frozen=True, slots=True)
class GuideFilter:
    """Cheap host-side guide filters.

    Defaults are intentionally conservative for SpCas9 guide design:
    20-nt spacers, 40-70% GC, no data-side ``N``, and no homopolymer
    run longer than 4.
    """

    spacer_len: int = 20
    min_gc: float = 0.40
    max_gc: float = 0.70
    max_n_fraction: float = 0.0
    max_homopolymer: int = 4
    min_entropy_bits: float = 1.2
    drop_failed: bool = True


@dataclass(frozen=True, slots=True)
class GuideCandidate:
    """One PAM-valid guide candidate with filter metadata."""

    guide_id: str
    spacer: str
    pam: str
    chrom: str
    window_start: int
    spacer_start: int
    strand: str
    canonical_key: int
    gc: float
    n_frac: float
    max_homopolymer: int
    low_complexity: bool
    passes_filters: bool
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)


def _candidate_rejections(spacer: str, filt: GuideFilter) -> tuple[str, ...]:
    reasons: list[str] = []
    gc = gc_fraction(spacer)
    n_frac = n_fraction(spacer)
    max_hp = homopolymer_run(spacer)
    if n_frac > filt.max_n_fraction:
        reasons.append("n_content")
    if gc < filt.min_gc:
        reasons.append("low_gc")
    if gc > filt.max_gc:
        reasons.append("high_gc")
    if max_hp > filt.max_homopolymer:
        reasons.append("homopolymer")
    if has_low_complexity(spacer, min_entropy_bits=filt.min_entropy_bits):
        reasons.append("low_complexity")
    return tuple(reasons)


def enumerate_guides(
    seq: str,
    *,
    chrom: str = "target",
    offset: int = 0,
    pam_templates: Sequence[str] = ("NGG",),
    guide_filter: GuideFilter | None = None,
    include_reverse: bool = True,
) -> list[GuideCandidate]:
    """Enumerate PAM-valid guide candidates from ``seq``.

    Args:
        seq: target DNA sequence. Lowercase is accepted and normalized.
        chrom: contig/region label carried into output records.
        offset: genomic offset added to reported positions.
        pam_templates: one or more IUPAC PAM templates, e.g. ``("NGG",
            "NAG")``. All templates must have the same length.
        guide_filter: filter config. ``None`` uses :class:`GuideFilter`
            defaults.
        include_reverse: when true, enumerate both strands.

    Returns:
        Candidate records sorted by ``(window_start, strand, guide_id)``.
    """
    filt = guide_filter or GuideFilter()
    if filt.spacer_len <= 0 or filt.spacer_len > 32:
        raise ValueError("spacer_len must be in 1..32")
    if not pam_templates:
        raise ValueError("at least one PAM template is required")

    templates = tuple(t.upper() for t in pam_templates)
    pam_len = len(templates[0])
    if pam_len == 0:
        raise ValueError("PAM template must be non-empty")
    if any(len(t) != pam_len for t in templates):
        raise ValueError("all PAM templates must have the same length")

    seq = seq.upper()
    window_len = filt.spacer_len + pam_len
    n = len(seq)
    if n < window_len:
        return []

    out: list[GuideCandidate] = []

    def emit(oriented_window: str, window_start: int, strand: str) -> None:
        spacer = oriented_window[: filt.spacer_len]
        pam = oriented_window[filt.spacer_len :]
        if not _VALID_BASES.issuperset(spacer):
            return
        if not any(matches_pam(pam, template) for template in templates):
            return
        reasons = _candidate_rejections(spacer, filt)
        if reasons and filt.drop_failed:
            return
        abs_window_start = int(offset) + window_start
        if strand == "+":
            spacer_start = abs_window_start
        else:
            spacer_start = abs_window_start + pam_len
        guide_id = f"{chrom}:{abs_window_start}:{strand}:{spacer}"
        out.append(
            GuideCandidate(
                guide_id=guide_id,
                spacer=spacer,
                pam=pam,
                chrom=chrom,
                window_start=abs_window_start,
                spacer_start=spacer_start,
                strand=strand,
                canonical_key=canonical_guide_key(spacer),
                gc=gc_fraction(spacer),
                n_frac=n_fraction(spacer),
                max_homopolymer=homopolymer_run(spacer),
                low_complexity=has_low_complexity(
                    spacer, min_entropy_bits=filt.min_entropy_bits
                ),
                passes_filters=not reasons,
                rejection_reasons=reasons,
            )
        )

    for start in range(n - window_len + 1):
        window = seq[start : start + window_len]
        emit(window, start, "+")
        if include_reverse:
            emit(reverse_complement(window), start, "-")

    out.sort(key=lambda c: (c.window_start, c.strand, c.guide_id))
    return out
