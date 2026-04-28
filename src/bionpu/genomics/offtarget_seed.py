# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Host-side CRISPR off-target seed prefilter.

This module intentionally stays small and CPU-only. It builds oriented
seed candidates for CRISPR protospacers so downstream dynamic
programming/scoring can spend time only on locations whose PAM-proximal
guide seed is compatible with a reference window.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "GuideRecord",
    "OffTargetSeedCandidate",
    "encode_seed_2bit",
    "prefilter_offtargets",
    "reverse_complement",
    "seed_mismatch_positions",
]

_BASE_TO_2BIT: Final[dict[str, int]] = {"A": 0, "C": 1, "G": 2, "T": 3}
_RC_TRANS: Final[dict[int, int]] = str.maketrans("ACGTacgt", "TGCAtgca")
_IUPAC_PAM: Final[dict[str, frozenset[str]]] = {
    "A": frozenset({"A"}),
    "C": frozenset({"C"}),
    "G": frozenset({"G"}),
    "T": frozenset({"T"}),
    "N": frozenset({"A", "C", "G", "T"}),
    "R": frozenset({"A", "G"}),
    "Y": frozenset({"C", "T"}),
    "S": frozenset({"G", "C"}),
    "W": frozenset({"A", "T"}),
    "K": frozenset({"G", "T"}),
    "M": frozenset({"A", "C"}),
    "B": frozenset({"C", "G", "T"}),
    "D": frozenset({"A", "G", "T"}),
    "H": frozenset({"A", "C", "T"}),
    "V": frozenset({"A", "C", "G"}),
}


@dataclass(frozen=True, slots=True)
class GuideRecord:
    """Guide identifier and protospacer sequence."""

    guide_id: str
    sequence: str


@dataclass(frozen=True, slots=True)
class OffTargetSeedCandidate:
    """A seed-compatible CRISPR off-target candidate.

    ``position`` is the 0-indexed start of the genomic protospacer on
    the forward reference coordinate system. For ``strand == "-"``, the
    candidate target sequence is the reverse complement of
    ``reference[position:position + guide_length]``.
    """

    guide_id: str
    ref_name: str
    position: int
    strand: str
    seed: str
    seed_key: int
    target_seed: str
    target_seed_key: int
    seed_length: int
    seed_mismatches: int
    seed_mismatch_positions: tuple[int, ...]
    seed_hit_count: int
    pam: str | None = None


@dataclass(frozen=True, slots=True)
class _ReferenceSeedHit:
    ref_name: str
    position: int
    strand: str
    target_seed: str
    target_seed_key: int
    pam: str | None


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of an ACGT/IUPAC string."""

    return seq.translate(_RC_TRANS)[::-1].upper()


def encode_seed_2bit(seed: str) -> int | None:
    """Encode an ACGT seed as an MSB-first 2-bit integer.

    Returns ``None`` when the seed contains any non-ACGT base. That
    keeps N-containing guide/reference windows out of the prefilter
    without silently coercing ambiguous sequence to a concrete base.
    """

    key = 0
    for base in seed.upper():
        value = _BASE_TO_2BIT.get(base)
        if value is None:
            return None
        key = (key << 2) | value
    return key


def seed_mismatch_positions(a: str, b: str) -> tuple[int, ...]:
    """Return 0-indexed mismatch positions between two equal-length seeds."""

    if len(a) != len(b):
        raise ValueError(f"seed lengths differ: {len(a)} != {len(b)}")
    return tuple(i for i, (x, y) in enumerate(zip(a.upper(), b.upper())) if x != y)


def prefilter_offtargets(
    guides: Mapping[str, str] | Iterable[GuideRecord | tuple[str, str] | str],
    references: Mapping[str, str] | str,
    *,
    seed_length: int = 12,
    max_seed_mismatches: int = 0,
    pam: str | Sequence[str] | None = "NGG",
    pam_aware: bool = True,
    guide_length: int | None = None,
) -> list[OffTargetSeedCandidate]:
    """Return deterministic seed-compatible CRISPR off-target candidates.

    The seed is the PAM-proximal suffix of each guide. Reference windows
    are scanned on both strands. With ``pam_aware=True`` and the default
    ``pam="NGG"``, plus-strand candidates require an ``NGG`` immediately
    downstream of the protospacer; minus-strand candidates require the
    reverse-complement-oriented ``NGG`` immediately upstream. ``pam``
    may also be a sequence such as ``("NGG", "NAG")``.
    """

    if seed_length <= 0:
        raise ValueError(f"seed_length must be positive; got {seed_length}")
    if max_seed_mismatches < 0:
        raise ValueError(
            f"max_seed_mismatches must be non-negative; got {max_seed_mismatches}"
        )

    guide_records = _normalise_guides(guides)
    reference_records = _normalise_references(references)
    if not guide_records or not reference_records:
        return []

    effective_guide_length = guide_length
    if effective_guide_length is None:
        lengths = {len(g.sequence) for g in guide_records}
        if len(lengths) != 1:
            raise ValueError(
                "guide_length is required when guides have different lengths"
            )
        effective_guide_length = lengths.pop()
    if effective_guide_length < seed_length:
        raise ValueError(
            "guide_length must be at least seed_length; "
            f"got guide_length={effective_guide_length}, seed_length={seed_length}"
        )

    pam_patterns = _normalise_pam_patterns(pam)
    ref_index: dict[int, list[_ReferenceSeedHit]] = {}
    for ref_name, ref_seq in reference_records:
        for hit in _iter_reference_seed_hits(
            ref_name=ref_name,
            ref_seq=ref_seq.upper(),
            guide_length=effective_guide_length,
            seed_length=seed_length,
            pams=pam_patterns,
            pam_aware=pam_aware,
        ):
            ref_index.setdefault(hit.target_seed_key, []).append(hit)

    out: list[OffTargetSeedCandidate] = []
    for guide in guide_records:
        guide_seq = guide.sequence.upper()
        if len(guide_seq) != effective_guide_length:
            raise ValueError(
                f"guide {guide.guide_id!r} has length {len(guide_seq)}; "
                f"expected {effective_guide_length}"
            )
        if not _is_acgt(guide_seq):
            continue
        seed = guide_seq[-seed_length:]
        seed_key = encode_seed_2bit(seed)
        if seed_key is None:
            continue

        hit_groups = (
            (ref_index.get(seed_key, []),)
            if max_seed_mismatches == 0
            else ref_index.values()
        )
        for hits in hit_groups:
            if not hits:
                continue
            mismatch_positions = seed_mismatch_positions(seed, hits[0].target_seed)
            if len(mismatch_positions) > max_seed_mismatches:
                continue
            for hit in hits:
                mismatch_positions = seed_mismatch_positions(seed, hit.target_seed)
                if len(mismatch_positions) > max_seed_mismatches:
                    continue
                out.append(
                    OffTargetSeedCandidate(
                        guide_id=guide.guide_id,
                        ref_name=hit.ref_name,
                        position=hit.position,
                        strand=hit.strand,
                        seed=seed,
                        seed_key=seed_key,
                        target_seed=hit.target_seed,
                        target_seed_key=hit.target_seed_key,
                        seed_length=seed_length,
                        seed_mismatches=len(mismatch_positions),
                        seed_mismatch_positions=mismatch_positions,
                        seed_hit_count=len(hits),
                        pam=hit.pam,
                    )
                )

    out.sort(
        key=lambda c: (
            c.guide_id,
            c.ref_name,
            c.position,
            c.strand,
            c.target_seed,
        )
    )
    return out


def _normalise_guides(
    guides: Mapping[str, str] | Iterable[GuideRecord | tuple[str, str] | str],
) -> list[GuideRecord]:
    if isinstance(guides, Mapping):
        return [GuideRecord(str(k), str(v)) for k, v in guides.items()]

    out: list[GuideRecord] = []
    for i, guide in enumerate(guides):
        if isinstance(guide, GuideRecord):
            out.append(guide)
        elif isinstance(guide, str):
            out.append(GuideRecord(f"guide_{i}", guide))
        else:
            guide_id, sequence = guide
            out.append(GuideRecord(str(guide_id), str(sequence)))
    return out


def _normalise_references(references: Mapping[str, str] | str) -> list[tuple[str, str]]:
    if isinstance(references, str):
        return [("ref", references)]
    return [(str(k), str(v)) for k, v in references.items()]


def _normalise_pam_patterns(pam: str | Sequence[str] | None) -> tuple[str, ...]:
    if pam is None:
        return ()
    if isinstance(pam, str):
        patterns = (pam.upper(),)
    else:
        patterns = tuple(str(p).upper() for p in pam)
    if any(not p for p in patterns):
        raise ValueError("PAM patterns must be non-empty")
    if len({len(p) for p in patterns}) > 1:
        raise ValueError("all PAM patterns must have the same length")
    return patterns


def _iter_reference_seed_hits(
    *,
    ref_name: str,
    ref_seq: str,
    guide_length: int,
    seed_length: int,
    pams: tuple[str, ...],
    pam_aware: bool,
) -> Iterable[_ReferenceSeedHit]:
    n = len(ref_seq)
    pam_length = len(pams[0]) if pams else 0
    last_protospacer_start = n - guide_length
    if last_protospacer_start < 0:
        return

    for pos in range(last_protospacer_start + 1):
        protospacer = ref_seq[pos : pos + guide_length]
        if not _is_acgt(protospacer):
            continue

        plus_seed = protospacer[-seed_length:]
        plus_key = encode_seed_2bit(plus_seed)
        if plus_key is not None and _has_pam(
            ref_seq=ref_seq,
            pos=pos,
            strand="+",
            guide_length=guide_length,
            pams=pams,
            pam_length=pam_length,
            pam_aware=pam_aware,
        ):
            plus_pam = (
                ref_seq[pos + guide_length : pos + guide_length + pam_length]
                if pam_aware and pams
                else None
            )
            yield _ReferenceSeedHit(ref_name, pos, "+", plus_seed, plus_key, plus_pam)

        minus_oriented = reverse_complement(protospacer)
        minus_seed = minus_oriented[-seed_length:]
        minus_key = encode_seed_2bit(minus_seed)
        if minus_key is not None and _has_pam(
            ref_seq=ref_seq,
            pos=pos,
            strand="-",
            guide_length=guide_length,
            pams=pams,
            pam_length=pam_length,
            pam_aware=pam_aware,
        ):
            minus_pam = (
                reverse_complement(ref_seq[pos - pam_length : pos])
                if pam_aware and pams
                else None
            )
            yield _ReferenceSeedHit(
                ref_name, pos, "-", minus_seed, minus_key, minus_pam
            )


def _has_pam(
    *,
    ref_seq: str,
    pos: int,
    strand: str,
    guide_length: int,
    pams: tuple[str, ...],
    pam_length: int,
    pam_aware: bool,
) -> bool:
    if not pam_aware or not pams:
        return True
    if strand == "+":
        start = pos + guide_length
        observed = ref_seq[start : start + pam_length]
    else:
        start = pos - pam_length
        if start < 0:
            return False
        observed = reverse_complement(ref_seq[start:pos])
    return any(_pam_matches(pam, observed) for pam in pams)


def _pam_matches(pattern: str, observed: str) -> bool:
    if len(pattern) != len(observed):
        return False
    for pat, obs in zip(pattern.upper(), observed.upper()):
        allowed = _IUPAC_PAM.get(pat)
        if allowed is None:
            raise ValueError(f"unsupported PAM code {pat!r} in {pattern!r}")
        if obs not in allowed:
            return False
    return True


def _is_acgt(seq: str) -> bool:
    return all(base in _BASE_TO_2BIT for base in seq.upper())
