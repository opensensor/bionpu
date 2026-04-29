# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""IUPAC PAM specs for the supported Cas9 + base editor variants.

Per ``PRDs/PRD-crispr-state-of-the-art-roadmap.md`` §3.1 Phase 1 the
v0 deliverable is SpCas9 wild-type + ABE7.10 + BE4max + SpCas9-NG.
The PAM specs cover the broader Cas9 zoo (SpRY, SaCas9-KKH) so the
ranker can be exercised on every variant the silicon kernel supports;
the BE-variant table is intentionally narrow (Phase 1 scope).

Activity-window conventions (per the PRD):

* CBE (cytidine deaminase): nt 4-8 from the PAM-distal end (1-indexed),
  i.e. positions [3..7] (0-indexed) of a 20-nt protospacer counted from
  the 5' end. Target base = ``C``.
* ABE (adenosine deaminase): nt 4-7 from the PAM-distal end (1-indexed),
  i.e. positions [3..6] (0-indexed). Target base = ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BE_VARIANTS",
    "CAS9_VARIANTS",
    "BaseEditorSpec",
    "Cas9PamSpec",
    "get_be_spec",
    "get_cas9_spec",
]


@dataclass(frozen=True)
class Cas9PamSpec:
    """A Cas9 variant + its IUPAC PAM template."""

    name: str
    pam_iupac: str
    notes: str = ""


@dataclass(frozen=True)
class BaseEditorSpec:
    """A base editor variant + its activity window + target base."""

    name: str
    target_base: str  # "C" for CBE, "A" for ABE
    window_lo_pam_distal_1idx: int  # inclusive, 1-indexed (e.g. 4)
    window_hi_pam_distal_1idx: int  # inclusive, 1-indexed (e.g. 8 for CBE)
    notes: str = ""


# ----- Cas9 PAM zoo ----- #
#
# v0 silicon kernel is shape-correct for every entry below; the ranker's
# v0 SCOPE only asserts on entries flagged as `phase1=True` in the
# v0-cleared list. Phase 2 expansion is mostly a ranker / docs task —
# the silicon kernel does not need to be rebuilt to accept new variants.

CAS9_VARIANTS: dict[str, Cas9PamSpec] = {
    "wt": Cas9PamSpec(
        name="SpCas9 (wild-type)",
        pam_iupac="NGG",
        notes="Canonical SpCas9 NGG PAM; reference for ABE7.10 / BE4max.",
    ),
    "NG": Cas9PamSpec(
        name="SpCas9-NG",
        pam_iupac="NG",
        notes=(
            "SpCas9-NG (Nishimasu 2018) — relaxed NG PAM. v0 stores the "
            "2-base IUPAC literally; the kernel honors pam_length so this "
            "shares the same xclbin as NGG."
        ),
    ),
    "SpRY": Cas9PamSpec(
        name="SpRY",
        pam_iupac="NRN",
        notes=(
            "Walton 2020 SpRY near-PAMless variant. NRN covers ~2/4 of "
            "all genomic positions; useful for BE site selection in "
            "PAM-poor regions."
        ),
    ),
    "SaCas9-KKH": Cas9PamSpec(
        name="SaCas9-KKH",
        pam_iupac="NNNRRT",
        notes=(
            "Kleinstiver 2015 SaCas9-KKH variant; smaller delivery "
            "package, longer PAM. Phase 2 expansion target."
        ),
    ),
}

#: Set of Cas9 variants in the v0 SCOPE acceptance gate (per PRD §3.1
#: Phase 1). Other entries in :data:`CAS9_VARIANTS` are silicon-supported
#: but not in the v0 cleared list.
PHASE1_CAS9_VARIANTS: frozenset[str] = frozenset({"wt", "NG"})


# ----- Base editor zoo ----- #

BE_VARIANTS: dict[str, BaseEditorSpec] = {
    "BE4max": BaseEditorSpec(
        name="BE4max",
        target_base="C",
        window_lo_pam_distal_1idx=4,
        window_hi_pam_distal_1idx=8,
        notes=(
            "Optimized cytidine base editor (Koblan 2018). Activity "
            "window nt 4-8 from PAM-distal; target C->T."
        ),
    ),
    "ABE7.10": BaseEditorSpec(
        name="ABE7.10",
        target_base="A",
        window_lo_pam_distal_1idx=4,
        window_hi_pam_distal_1idx=7,
        notes=(
            "ABE7.10 (Gaudelli 2017). Activity window nt 4-7 from "
            "PAM-distal; target A->G."
        ),
    ),
    "ABE8e": BaseEditorSpec(
        name="ABE8e",
        target_base="A",
        window_lo_pam_distal_1idx=4,
        window_hi_pam_distal_1idx=8,
        notes=(
            "ABE8e (Richter 2020) — wider activity window than 7.10. "
            "Phase 2 expansion target."
        ),
    ),
}

#: Set of BE variants in the v0 SCOPE acceptance gate (per PRD §3.1
#: Phase 1).
PHASE1_BE_VARIANTS: frozenset[str] = frozenset({"BE4max", "ABE7.10"})


def get_cas9_spec(name: str) -> Cas9PamSpec:
    """Lookup a Cas9 PAM spec by name, raising ValueError on miss."""
    spec = CAS9_VARIANTS.get(name)
    if spec is None:
        valid = ", ".join(sorted(CAS9_VARIANTS))
        raise ValueError(
            f"unknown Cas9 variant {name!r}; valid: {{{valid}}}"
        )
    return spec


def get_be_spec(name: str) -> BaseEditorSpec:
    """Lookup a BE spec by name, raising ValueError on miss."""
    spec = BE_VARIANTS.get(name)
    if spec is None:
        valid = ", ".join(sorted(BE_VARIANTS))
        raise ValueError(
            f"unknown BE variant {name!r}; valid: {{{valid}}}"
        )
    return spec
