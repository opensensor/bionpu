# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Track B v0 — Tests for pegRNA design scaffold/RTT/PBS constants and shared types.

Covers Task T3 acceptance criteria from
``track-b-pegrna-design-plan.md``:

* Constants integrity (no typos; expected names exported)
* All scaffold sequences are ACGU-only (RNA bases) — except optional
  ``cr772`` which may be a documented v1 TODO if the canonical
  sequence wasn't recoverable.
* Length-range constants are ``int``-typed and form non-empty inclusive
  ranges.
* ``RankedPegRNA`` carries every field the T9 TSV schema lists.
"""

from __future__ import annotations

import dataclasses
from typing import get_type_hints

import pytest


# ---------- constants module ---------- #


def test_pegrna_constants_module_exports_all_named_constants():
    """T3 must export the four scaffold variants + four length ranges +
    edit-length cap by the names the plan pins."""
    from bionpu.genomics.pe_design import pegrna_constants as pc

    expected_names = {
        # scaffolds
        "SGRNA_CANONICAL",
        "EVOPREQ1",
        "TEVOPREQ1",
        "CR772",
        "SCAFFOLD_VARIANTS",
        # PBS / RTT / nick-distance ranges
        "PBS_LENGTH_MIN",
        "PBS_LENGTH_MAX",
        "RTT_LENGTH_MIN",
        "RTT_LENGTH_MAX",
        "NICK_DISTANCE_MIN",
        "NICK_DISTANCE_MAX",
        # edit cap
        "MAX_EDIT_LENGTH_BP",
    }
    actual = set(dir(pc))
    missing = expected_names - actual
    assert not missing, f"pegrna_constants is missing expected names: {missing}"


def test_scaffold_sequences_are_rna_acgu_only():
    """Every scaffold sequence we publish must be ACGU-only RNA.

    ``cr772`` may legitimately be ``None`` in v0 if the canonical Yan
    2024 sequence is not yet pinned (documented as a v1 gap). All other
    scaffolds must have a concrete RNA sequence.
    """
    from bionpu.genomics.pe_design import pegrna_constants as pc

    rna_alphabet = set("ACGU")
    required_scaffolds = ["SGRNA_CANONICAL", "EVOPREQ1", "TEVOPREQ1"]
    for name in required_scaffolds:
        seq = getattr(pc, name)
        assert isinstance(seq, str), f"{name} must be a str RNA sequence"
        assert len(seq) > 0, f"{name} must be non-empty"
        bad = set(seq) - rna_alphabet
        assert not bad, f"{name} contains non-ACGU bases: {bad}"

    # cr772 may be None (v1 TODO) or a valid RNA string
    cr772 = pc.CR772
    if cr772 is not None:
        assert isinstance(cr772, str)
        bad = set(cr772) - rna_alphabet
        assert not bad, f"CR772 contains non-ACGU bases: {bad}"


def test_length_ranges_are_ints_and_form_valid_inclusive_ranges():
    """All length/distance constants must be ints with min < max."""
    from bionpu.genomics.pe_design import pegrna_constants as pc

    pairs = [
        ("PBS_LENGTH_MIN", "PBS_LENGTH_MAX", 8, 15),
        ("RTT_LENGTH_MIN", "RTT_LENGTH_MAX", 10, 30),
        ("NICK_DISTANCE_MIN", "NICK_DISTANCE_MAX", 40, 100),
    ]
    for lo_name, hi_name, exp_lo, exp_hi in pairs:
        lo = getattr(pc, lo_name)
        hi = getattr(pc, hi_name)
        assert isinstance(lo, int), f"{lo_name} must be int, got {type(lo)}"
        assert isinstance(hi, int), f"{hi_name} must be int, got {type(hi)}"
        assert lo < hi, f"{lo_name}={lo} must be strictly less than {hi_name}={hi}"
        assert lo == exp_lo, f"{lo_name} pinned to {exp_lo}, got {lo}"
        assert hi == exp_hi, f"{hi_name} pinned to {exp_hi}, got {hi}"

    assert isinstance(pc.MAX_EDIT_LENGTH_BP, int)
    assert pc.MAX_EDIT_LENGTH_BP == 50, "Mathis 2024 PRIDICT 2.0 trained cap"


def test_scaffold_variants_dict_indexes_all_named_scaffolds():
    """``SCAFFOLD_VARIANTS`` must be a name -> sequence mapping covering
    the four pinned variants."""
    from bionpu.genomics.pe_design import pegrna_constants as pc

    assert isinstance(pc.SCAFFOLD_VARIANTS, dict)
    expected_keys = {"sgRNA_canonical", "evopreQ1", "tevopreQ1", "cr772"}
    assert set(pc.SCAFFOLD_VARIANTS.keys()) == expected_keys
    # every value is either a str RNA or None (cr772 v1 TODO)
    rna_alphabet = set("ACGU")
    for key, seq in pc.SCAFFOLD_VARIANTS.items():
        if seq is None:
            assert key == "cr772", f"only cr772 may be None; got {key}=None"
            continue
        assert isinstance(seq, str)
        bad = set(seq) - rna_alphabet
        assert not bad, f"{key} has non-ACGU bases: {bad}"


def test_constants_module_has_citation_docstrings():
    """Every scaffold constant must have a citation in its module-level
    documentation.

    We assert the module docstring mentions the canonical citations so
    readers can trace each scaffold to its publication.
    """
    from bionpu.genomics.pe_design import pegrna_constants as pc

    doc = (pc.__doc__ or "")
    for citation_token in ("Anzalone", "Nelson", "Mathis"):
        assert citation_token in doc, (
            f"pegrna_constants module docstring must cite '{citation_token}'"
        )


# ---------- types module ---------- #


def test_types_module_exports_all_seven_named_types():
    """T3's types.py must export the seven named types Wave 2/3/etc.
    consume."""
    from bionpu.genomics.pe_design import types as t

    expected = [
        "EditSpec",
        "PegRNAFoldingFeatures",
        "PRIDICTScore",
        "PegRNACandidate",
        "PE3PegRNACandidate",
        "OffTargetSite",
        "RankedPegRNA",
    ]
    for name in expected:
        assert hasattr(t, name), f"types.py missing {name}"


def test_pe3_pegrna_candidate_extends_pegrna_candidate():
    """PE3PegRNACandidate must subclass PegRNACandidate so PE3 ranking
    code can treat it polymorphically."""
    from bionpu.genomics.pe_design.types import (
        PE3PegRNACandidate,
        PegRNACandidate,
    )

    assert issubclass(PE3PegRNACandidate, PegRNACandidate)


def test_ranked_pegrna_has_all_t9_tsv_schema_fields():
    """The ``RankedPegRNA`` dataclass must carry every column T9 emits.

    Per plan §T9 TSV columns (verbatim ordering); we check by name only,
    not by order — T9 owns the ordering.
    """
    from bionpu.genomics.pe_design.types import RankedPegRNA

    expected_fields = {
        # identity / edit
        "pegrna_id",
        "edit_notation",
        "edit_position",
        "edit_type",
        # spacer / pegRNA composition
        "spacer_strand",
        "spacer_seq",
        "pam_seq",
        "scaffold_variant",
        "pbs_seq",
        "pbs_length",
        "rtt_seq",
        "rtt_length",
        "rt_product_seq",
        "nick_site",
        "full_pegrna_rna_seq",
        # PE strategy + nicking sgRNA (PE3 only; nullable)
        "pe_strategy",
        "nicking_spacer",
        "nicking_pam",
        "nicking_distance",
        # PRIDICT
        "pridict_efficiency",
        "pridict_edit_rate",
        "pridict_confidence",
        # folding
        "mfe_kcal",
        "scaffold_disruption",
        "pbs_pairing_prob",
        # off-target
        "cfd_aggregate_pegrna",
        "off_target_count_pegrna",
        "cfd_aggregate_nicking",
        "off_target_count_nicking",
        # composite + ranking + notes
        "composite_pridict",
        "rank",
        "notes",
    }
    actual = {f.name for f in dataclasses.fields(RankedPegRNA)}
    missing = expected_fields - actual
    extra = actual - expected_fields
    assert not missing, f"RankedPegRNA missing TSV-schema fields: {sorted(missing)}"
    # extras are allowed (additive), but flag them for visibility.
    if extra:
        # not a failure — just record. Plan §T9 allows additive fields.
        pass


def test_edit_spec_carries_required_fields():
    """EditSpec must expose the fields T2 promises."""
    from bionpu.genomics.pe_design.types import EditSpec

    fields = {f.name for f in dataclasses.fields(EditSpec)}
    expected = {
        "chrom",
        "start",
        "end",
        "ref_seq",
        "alt_seq",
        "edit_type",
        "notation_used",
        "strand",
    }
    missing = expected - fields
    assert not missing, f"EditSpec missing required fields: {sorted(missing)}"


def test_pegrna_candidate_carries_required_fields():
    """PegRNACandidate must expose the fields T6 promises."""
    from bionpu.genomics.pe_design.types import PegRNACandidate

    fields = {f.name for f in dataclasses.fields(PegRNACandidate)}
    expected = {
        "spacer_seq",
        "pam_seq",
        "scaffold_variant",
        "pbs_seq",
        "pbs_length",
        "rtt_seq",
        "rtt_length",
        "nick_site",
        "full_pegrna_rna_seq",
        "edit_position_in_rtt",
        "strategy",
        "strand",
        "rt_product_seq",
        "chrom",
    }
    missing = expected - fields
    assert not missing, f"PegRNACandidate missing required fields: {sorted(missing)}"


def test_namedtuple_types_have_expected_fields():
    """Folding features, PRIDICT score, and OffTargetSite are
    NamedTuples — verify their field schemas."""
    from bionpu.genomics.pe_design.types import (
        OffTargetSite,
        PegRNAFoldingFeatures,
        PRIDICTScore,
    )

    assert PegRNAFoldingFeatures._fields == (
        "mfe_kcal",
        "mfe_structure",
        "pbs_pairing_prob",
        "scaffold_disruption",
    )
    assert PRIDICTScore._fields == (
        "efficiency",
        "edit_rate",
        "confidence",
        "notes",
    )
    assert OffTargetSite._fields == (
        "chrom",
        "pos",
        "strand",
        "mismatches",
        "cfd_score",
    )


def test_package_init_exposes_empty_all():
    """Per plan: T3 ships an ``__init__.py`` skeleton with empty
    ``__all__`` so later tasks can extend without race."""
    import bionpu.genomics.pe_design as pkg

    assert hasattr(pkg, "__all__")
    assert isinstance(pkg.__all__, list)
    assert pkg.__all__ == [], (
        "T3 must ship pe_design/__init__.py with empty __all__; "
        "downstream tasks (T6/T10/etc.) extend it later."
    )
