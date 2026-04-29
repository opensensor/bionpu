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

"""Track B v0 — Published pegRNA scaffolds + PBS/RTT/nick-distance ranges.

All constants in this module are pinned with citations to the original
publications. v0 covers four scaffolds and the PRIDICT 2.0 default
search ranges.

Citations
---------
* **Anzalone 2019** — Anzalone, A.V., Randolph, P.B., Davis, J.R.,
  Sousa, A.A., Koblan, L.W., Levy, J.M., Chen, P.J., Wilson, C.,
  Newby, G.A., Raguram, A., Liu, D.R. *Search-and-replace genome
  editing without double-strand breaks or donor DNA*. Nature **576**,
  149–157 (2019). doi:10.1038/s41586-019-1711-4. The original PE2
  scaffold (``sgRNA_canonical``); PRIDICT 2.0 was trained exclusively
  on this scaffold.
* **Nelson 2022** — Nelson, J.W., Randolph, P.B., Shen, S.P., Everette,
  K.A., Chen, P.J., Anzalone, A.V., An, M., Newby, G.A., Chen, J.C.,
  Hsu, A., Liu, D.R. *Engineered pegRNAs improve prime editing
  efficiency*. Nature Biotechnology **40**, 402–410 (2022).
  doi:10.1038/s41587-021-01039-7. Source of ``evopreQ1`` and
  ``tevopreQ1`` engineered 3' motifs.
* **Yan 2024** — Yan, J. et al. extended-scaffold pegRNA work
  ("cr772" family). The canonical published cr772 sequence is not
  reliably reproduced from memory; T3 ships the slot with ``CR772 =
  None`` plus a v1 TODO so the enumerator stays correct rather than
  emitting a fabricated sequence (see plan §T3 acceptance note).
* **Mathis 2024** — Mathis, N. et al. *PRIDICT 2.0*. Nature
  Biotechnology (2024). Default ``pbs_search_range = (8, 17)``; v0
  pins ``[8, 15]`` per plan §T3 to align with PE2/PE3 canonical
  ranges. The ``MAX_EDIT_LENGTH_BP = 50`` cap reflects the trained
  edit-length range.
"""

from __future__ import annotations

__all__ = [
    # scaffolds
    "SGRNA_CANONICAL",
    "EVOPREQ1",
    "TEVOPREQ1",
    "CR772",
    "SCAFFOLD_VARIANTS",
    # PBS / RTT / nick distance
    "PBS_LENGTH_MIN",
    "PBS_LENGTH_MAX",
    "RTT_LENGTH_MIN",
    "RTT_LENGTH_MAX",
    "NICK_DISTANCE_MIN",
    "NICK_DISTANCE_MAX",
    # edit cap
    "MAX_EDIT_LENGTH_BP",
]


# ----------------------------------------------------------------------
# Scaffold sequences (RNA, 5' -> 3', ACGU-only)
# ----------------------------------------------------------------------
#
# The canonical SpCas9 sgRNA scaffold from Anzalone 2019 (Nature, "Search-
# and-replace genome editing without double-strand breaks or donor DNA",
# Supplementary Methods + Figure 1c). This is the scaffold inserted
# between the spacer (5') and the 3' extension (RTT + PBS) in PE2/PE3
# pegRNAs and is the only scaffold PRIDICT 2.0 was trained on.
#
# Sequence reproduces the canonical 80-nt SpCas9 scaffold (tracrRNA
# fusion + tetraloop) used throughout the prime-editing literature.
SGRNA_CANONICAL: str = (
    "GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUUU"
)


# evopreQ1: Anzalone-scaffold + 3' evopreQ1 pseudoknot motif from Nelson
# 2022 (Nat Biotechnol). The motif is appended to the 3' end of the PBS
# (i.e. it sits 3' of the pegRNA proper) and stabilises the 3' extension
# against exonuclease degradation, improving editing efficiency
# 3-5x in HEK293 reporter assays. The published 3' motif is a 42-nt
# pseudoknot derived from the prequeosine-1 riboswitch.
#
# Note: in production pegRNA assembly the 3' motif is concatenated
# AFTER the PBS, so SCAFFOLD_VARIANTS holds the canonical scaffold
# sequence here for ``evopreQ1`` in v0; T6 enumerator handles the 3'
# motif appendage as a separate concern. This keeps the constant module
# focused on the canonical 80-nt scaffold body and lets the enumerator
# decide motif placement.
EVOPREQ1: str = SGRNA_CANONICAL


# tevopreQ1: Nelson 2022 — "trimmed" evopreQ1, a shorter pseudoknot motif
# that yields equivalent or improved editing efficiency vs evopreQ1 with
# fewer total nucleotides. Same scaffold body; the 3' motif difference
# is handled at enumeration time (see EVOPREQ1 note).
TEVOPREQ1: str = SGRNA_CANONICAL


# cr772: Yan 2024 — extended-scaffold prime-editing variant. The
# canonical published sequence is not reproduced here from memory to
# avoid shipping a fabricated sequence. v0 ships the slot as ``None``
# with an explicit v1 gap to look up the sequence from the upstream
# paper / repository before enabling enumerator support.
#
# TODO(track-b-v1): canonical cr772 sequence — look up from Yan 2024
# (the precise journal + supplementary table needs verification against
# the published paper). Once recovered, replace ``None`` with the RNA
# sequence; T6 enumerator already routes through SCAFFOLD_VARIANTS so
# no enumerator change is required.
CR772: str | None = None


# Variant registry. T6 enumerator looks up scaffolds by name from this
# mapping; T9 TSV emits the variant name as a column.
SCAFFOLD_VARIANTS: dict[str, str | None] = {
    "sgRNA_canonical": SGRNA_CANONICAL,
    "evopreQ1": EVOPREQ1,
    "tevopreQ1": TEVOPREQ1,
    "cr772": CR772,
}


# ----------------------------------------------------------------------
# PBS / RTT / nick-distance ranges
# ----------------------------------------------------------------------
#
# PBS (primer binding site) length range. Mathis 2024 (PRIDICT 2.0)
# default search range is 8-17 nt; v0 pins 8-15 per plan §T3 to align
# with PE2/PE3 canonical ranges.
PBS_LENGTH_MIN: int = 8
PBS_LENGTH_MAX: int = 15

# RTT (reverse-transcriptase template) length range. PE2/PE3 canonical
# range is 10-30 nt; v0 enumerates this whole range.
RTT_LENGTH_MIN: int = 10
RTT_LENGTH_MAX: int = 30

# PE3 nicking-sgRNA distance range from the PE2 nick site, on the
# OPPOSITE strand. Anzalone 2019 + Mathis 2024 PRIDICT 2.0 default.
NICK_DISTANCE_MIN: int = 40
NICK_DISTANCE_MAX: int = 100


# ----------------------------------------------------------------------
# Edit-length cap (Mathis 2024)
# ----------------------------------------------------------------------
#
# PRIDICT 2.0 was trained on edits up to ~50 bp. T2 edit-spec enforces
# this cap at parse time; T6 enumerator inherits the cap implicitly via
# the parsed EditSpec.
MAX_EDIT_LENGTH_BP: int = 50
