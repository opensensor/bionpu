# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Tests for the methylation-context scanner oracle."""

from __future__ import annotations

from bionpu.data.kmer_oracle import pack_dna_2bit
from bionpu.data.methylation_context_oracle import (
    MethylationContextHit,
    find_methylation_contexts,
    find_methylation_contexts_packed,
)


def test_methylation_contexts_emit_both_strands() -> None:
    seq = "ACGCCAGCTGG"

    assert find_methylation_contexts(seq) == [
        MethylationContextHit(1, "+", "CG", "CG"),
        MethylationContextHit(2, "-", "CG", "CG"),
        MethylationContextHit(3, "+", "CHH", "CCA"),
        MethylationContextHit(4, "+", "CHG", "CAG"),
        MethylationContextHit(6, "-", "CHG", "CTG"),
        MethylationContextHit(7, "+", "CHG", "CTG"),
        MethylationContextHit(9, "-", "CHG", "CAG"),
        MethylationContextHit(10, "-", "CHH", "CCA"),
    ]


def test_methylation_contexts_skip_boundaries_and_ambiguous_bases() -> None:
    assert find_methylation_contexts("") == []
    assert find_methylation_contexts("C") == []
    assert find_methylation_contexts("CA") == []
    assert find_methylation_contexts("NGN") == []
    assert find_methylation_contexts("TAAG") == [
        MethylationContextHit(3, "-", "CHH", "CTT")
    ]


def test_methylation_contexts_are_case_insensitive() -> None:
    assert find_methylation_contexts("acg") == [
        MethylationContextHit(1, "+", "CG", "CG"),
        MethylationContextHit(2, "-", "CG", "CG"),
    ]


def test_methylation_contexts_packed_matches_string_oracle() -> None:
    seq = "ACGCCAGCTGG"
    packed = pack_dna_2bit(seq)

    assert find_methylation_contexts_packed(
        packed, n_bases=len(seq)
    ) == find_methylation_contexts(seq)
